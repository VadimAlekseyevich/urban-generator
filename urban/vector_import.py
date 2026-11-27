"""GeoJSON WGS84 → та же метрическая Terrain, без промежуточного TIFF."""

from hashlib import sha256
import json
from math import ceil, floor, isfinite
from numbers import Real

import numpy as np
from rasterio.errors import RasterioError
from rasterio.features import bounds, rasterize
from rasterio.transform import from_origin
from rasterio.warp import transform_geom

from urban.io import TerrainReadError, local_metric_crs, rasterize_classes
from urban.model import (
    ClassCode, CoordinateMode, MAX_INPUT_BYTES, MAX_RASTER_SIDE, Terrain,
    VectorSource, validate_pixel_size_m,
)


VECTOR_IMPORT_VERSION = 1
ROLE_TYPES = {
    "boundary": {"Polygon", "MultiPolygon"},
    "road": {"LineString", "MultiLineString"},
    "building": {"Polygon", "MultiPolygon"},
    "exclusion": {"Polygon", "MultiPolygon"},
}
ROLE_CODES = {"road": ClassCode.EXISTING_ROAD, "building": ClassCode.EXISTING_BUILDING,
              "exclusion": ClassCode.BLOCKED}


def _fail(message: str) -> None:
    raise TerrainReadError(message)


def _position(position) -> None:
    if (not isinstance(position, list) or len(position) != 2
            or any(isinstance(v, bool) or not isinstance(v, Real) or not isfinite(v) for v in position)
            or not -180 < position[0] < 180 or not -80 <= position[1] <= 84):
        _fail("Нужны двумерные координаты WGS84 [долгота, широта] в пределах UTM: (-180,180), [-80,84].")


def _line(line, *, ring=False) -> None:
    if not isinstance(line, list) or len(line) < (4 if ring else 2):
        _fail("Пустая или неполная геометрия: линия требует 2 точки, кольцо — 4.")
    for position in line:
        _position(position)
    if ring and line[0] != line[-1]:
        _fail("Кольца Polygon должны быть замкнуты: последняя точка совпадает с первой.")
    if any(a == b for a, b in zip(line, line[1:])):
        _fail("В геометрии есть нулевой отрезок. Удалите повторяющиеся соседние точки.")


def _intersects(a, b, starts, ends):
    """Пересечение отрезков, включая касание (координаты около локального нуля)."""
    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    c1, c2 = cross(b - a, starts - a), cross(b - a, ends - a)
    c3, c4 = cross(ends - starts, a - starts), cross(ends - starts, b - starts)
    overlap = np.all(np.maximum(np.minimum(a, b), np.minimum(starts, ends))
                     <= np.minimum(np.maximum(a, b), np.maximum(starts, ends)), axis=-1)
    return overlap & (c1 * c2 <= 0) & (c3 * c4 <= 0)


def _inside(point, ring):
    a, b = ring[:-1], ring[1:]
    crossing = (a[:, 1] > point[1]) != (b[:, 1] > point[1])
    a, b = a[crossing], b[crossing]
    return bool(np.count_nonzero(point[0] < a[:, 0] +
                (point[1] - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])) % 2)


def _polygon(polygon) -> None:
    if not isinstance(polygon, list) or not polygon:
        _fail("Polygon должен содержать внешнее кольцо.")
    # Ограничиваем квадратичную проверку пересечений до растеризации.
    if sum(len(r) if isinstance(r, list) else 0 for r in polygon) > 2048:
        _fail("В одном полигоне больше 2048 точек. Упростите геометрию в GIS.")
    rings = []
    for line in polygon:
        _line(line, ring=True)
        ring = np.asarray(line, dtype=float) - np.asarray(polygon[0][0], dtype=float)
        a, b = ring[:-1], ring[1:]
        area = np.sum(a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1])
        if area == 0:
            _fail("Полигон имеет нулевую площадь или самопересечение. Исправьте геометрию в GIS.")
        for i in range(len(a)):
            js = np.arange(i + 2, len(a))
            if i == 0:
                js = js[js != len(a) - 1]
            if np.any(_intersects(a[i], b[i], a[js], b[js])):
                _fail("Самопересечение кольца Polygon. Исправьте геометрию в GIS.")
        for previous in rings:
            if any(np.any(_intersects(x, y, previous[:-1], previous[1:])) for x, y in zip(a, b)):
                _fail("Кольца полигона пересекаются или касаются. Исправьте отверстия в GIS.")
        if rings and (not _inside(ring[0], rings[0]) or
                      any(_inside(ring[0], hole) or _inside(hole[0], ring) for hole in rings[1:])):
            _fail("Отверстие должно лежать внутри внешнего кольца и вне других отверстий.")
        rings.append(ring)


def _geometry(geometry, role) -> None:
    if not isinstance(geometry, dict) or geometry.get("type") not in ROLE_TYPES[role]:
        _fail(f"Роль {role}: нужны геометрии {', '.join(sorted(ROLE_TYPES[role]))}. Проверьте поле role.")
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or not coordinates:
        _fail("Геометрия пуста. Добавьте координаты объекта.")
    kind = geometry["type"]
    if kind == "LineString":
        _line(coordinates)
    elif kind == "MultiLineString":
        for line in coordinates:
            _line(line)
    elif kind == "Polygon":
        _polygon(coordinates)
    else:
        for polygon in coordinates:
            _polygon(polygon)


def _parse(data: bytes) -> dict:
    if not isinstance(data, bytes) or not data:
        _fail("Загрузите один непустой GeoJSON FeatureCollection в WGS84.")
    if len(data) > MAX_INPUT_BYTES:
        _fail("GeoJSON превышает 8 МиБ. Обрежьте участок или упростите геометрию.")
    try:
        # Запрещаем повторные ключи: иначе role/координаты имеют неоднозначный смысл.
        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Повторный ключ JSON")
                result[key] = value
            return result
        document = json.loads(data.decode("utf-8-sig"), object_pairs_hook=unique_keys,
                              parse_constant=lambda value: _fail(f"Недопустимое число JSON: {value}."))
    except (ValueError, UnicodeError, RecursionError) as error:
        raise TerrainReadError("Не удалось прочитать GeoJSON. Нужен корректный UTF-8 JSON без повторных ключей и NaN.") from error
    if (not isinstance(document, dict) or document.get("type") != "FeatureCollection"
            or not isinstance(document.get("features"), list) or not document["features"]):
        _fail("Нужен непустой GeoJSON FeatureCollection с массивом features.")
    if "crs" in document:
        _fail("Удалите устаревшее поле crs: поддерживается GeoJSON WGS84, порядок [долгота, широта].")
    return document


def read_geojson(data: bytes, pixel_size_m: float, *, source_name: str = "input.geojson") -> Terrain:
    """Строгие роли; одна граница; UTM и сетка, привязанная к кратным шагу метрам.

    Включение в boundary (включая отверстия) по центру ячейки. Предметные
    слои растеризуются all_touched, затем обрезаются по маске исследования.
    Оригинал хранится побайтно; отсутствующие ID назначаются по порядку.
    """
    try:
        h = validate_pixel_size_m(pixel_size_m)
        document = _parse(data)
        boundary = None
        manifest, seen, features = [], set(), []
        for index, feature in enumerate(document["features"], 1):
            if not isinstance(feature, dict) or feature.get("type") != "Feature":
                _fail(f"Объект {index}: нужен GeoJSON Feature.")
            properties = feature.get("properties")
            role = properties.get("role") if isinstance(properties, dict) else None
            if not isinstance(role, str) or role not in ROLE_TYPES:
                _fail(f"Объект {index}: неизвестная роль. Задайте properties.role: boundary, road, building или exclusion.")
            geometry = feature.get("geometry")
            _geometry(geometry, role)
            identifier = feature.get("id", f"feature-{index}")
            if isinstance(identifier, bool) or not isinstance(identifier, (str, int)) or identifier == "":
                _fail(f"Объект {index}: id должен быть непустой строкой или целым числом.")
            if str(identifier) in seen:
                _fail(f"Повторный id {identifier}. Назначьте уникальные идентификаторы объектов.")
            seen.add(str(identifier))
            manifest.append({"id": identifier, "role": role, "source": properties.get("source", source_name)})
            features.append((role, geometry))
            if role == "boundary":
                if boundary is not None:
                    _fail("Нужна ровно одна роль boundary. Объедините части в MultiPolygon.")
                boundary = geometry
        if boundary is None:
            _fail("Нет границы исследования. Добавьте Polygon/MultiPolygon с role=boundary.")
        west, south, east, north = bounds(boundary)
        if east - west > 6 or north - south > 6:
            _fail("Слишком большой охват для местной UTM. Обрежьте GeoJSON до небольшого участка города.")
        crs = local_metric_crs((west + east) / 2, (south + north) / 2)
        projected_boundary = transform_geom("EPSG:4326", crs, boundary, precision=-1)
        west, south, east, north = bounds(projected_boundary)
        left, top = floor(west / h) * h, ceil(north / h) * h
        width, height = ceil((east - left) / h), ceil((top - south) / h)
        if not (1 <= width <= MAX_RASTER_SIDE and 1 <= height <= MAX_RASTER_SIDE):
            _fail(f"При шаге {h:g} м охват требует {width}×{height} ячеек; предел 256×256. "
                  "Уменьшите boundary или увеличьте шаг до 10 м.")
        affine = from_origin(left, top, h, h)
        mask = rasterize([projected_boundary], out_shape=(height, width), transform=affine,
                         fill=0, default_value=1, all_touched=False, dtype="uint8").astype(bool)
        if not mask.any():
            _fail("Граница не покрывает ни одного центра ячейки. Уточните boundary или уменьшите шаг.")
        shapes = {code: [] for code in ROLE_CODES.values()}
        for role, geometry in features:
            if role != "boundary":
                shapes[ROLE_CODES[role]].append(transform_geom("EPSG:4326", crs, geometry, precision=-1))
        classes = rasterize_classes(shapes, mask.shape, affine)
        if not np.any(mask & (classes == ClassCode.EXISTING_ROAD)):
            _fail("Нет исходной дороги внутри boundary после обрезки и приоритетов. "
                  "Добавьте LineString с role=road; проверьте перекрытие запретами и зданиями.")
        metadata = {"format": "GeoJSON", "import_version": VECTOR_IMPORT_VERSION,
                    "source_name": source_name, "source_sha256": sha256(data).hexdigest(),
                    "source_crs": "EPSG:4326", "features": manifest,
                    "boundary_rule": "cell_center", "layers_all_touched": True,
                    "priority": ["building", "exclusion", "road", "free"]}
        return Terrain(classes=classes, valid_mask=mask, pixel_size_m=h,
                       coordinate_mode=CoordinateMode.GEOREFERENCED,
                       input_sha256=sha256(data).hexdigest(), crs=crs, transform=affine,
                       vector_source=VectorSource(source_name, data, json.dumps(metadata, ensure_ascii=False)))
    except TerrainReadError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError, RasterioError, RecursionError) as error:
        raise TerrainReadError("Не удалось подготовить GeoJSON. Проверьте геометрии и шаг 5–10 м.") from error
