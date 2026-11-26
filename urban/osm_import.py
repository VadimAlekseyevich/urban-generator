"""Локальная подготовка тематического GeoTIFF из OpenStreetMap PBF.

Дополнительный пакет osmium нужен только для импорта. Ядро и Streamlit
по-прежнему работают с TIFF, без osmium и доступа к сети.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from math import floor, isfinite
from pathlib import Path

import numpy as np
from rasterio.crs import CRS
from rasterio.features import bounds, rasterize
from rasterio.io import MemoryFile
from rasterio.transform import array_bounds, from_origin
from rasterio.warp import transform, transform_bounds, transform_geom

from urban.io import read_terrain
from urban.model import ClassCode, MAX_RASTER_SIDE, validate_pixel_size_m


ATTRIBUTION = "© OpenStreetMap contributors, https://www.openstreetmap.org/copyright"
IMPORT_VERSION = 1
ROAD_TYPES = frozenset({
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "living_street", "service", "road",
})
BLOCKED_LANDUSE = frozenset({
    "forest", "grass", "meadow", "farmland", "farmyard", "orchard", "vineyard",
    "cemetery", "recreation_ground", "allotments", "industrial", "commercial",
    "retail", "railway", "military", "quarry", "landfill", "reservoir", "basin",
    "construction", "religious", "education",
})
BLOCKED_NATURAL = frozenset({
    "water", "wetland", "wood", "scrub", "heath", "grassland", "sand",
    "beach", "mud", "bare_rock", "glacier",
})
BLOCKED_LEISURE = frozenset({
    "park", "garden", "nature_reserve", "pitch", "playground", "sports_centre",
    "stadium", "golf_course", "recreation_ground", "swimming_pool",
})
BLOCKED_AMENITY = frozenset({
    "school", "kindergarten", "college", "university", "hospital", "clinic",
    "grave_yard", "place_of_worship", "parking", "fire_station",
})
KEYS = ("building", "building:part", "highway", "waterway", "railway",
        "natural", "landuse", "leisure", "amenity", "boundary", "aeroway",
        "man_made", "power")


@dataclass(frozen=True, slots=True)
class Window:
    crs: CRS
    transform: object
    size: int
    pixel_size_m: float
    wgs84_bounds: tuple[float, float, float, float]


def make_window(lon: float, lat: float, size: int = 256, pixel_size_m: float = 10) -> Window:
    """Квадрат в местной UTM с заданным географическим центром."""
    if not (isfinite(lon) and isfinite(lat) and -180 < lon < 180 and -80 <= lat <= 84):
        raise ValueError("Центр должен быть в пределах UTM: долгота (-180,180), широта [-80,84].")
    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= MAX_RASTER_SIDE:
        raise ValueError("Размер карты должен быть целым от 1 до 256.")
    h = validate_pixel_size_m(pixel_size_m)
    zone = min(60, floor((lon + 180) / 6) + 1)
    # Специальные зоны UTM для Норвегии и Шпицбергена.
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone = 32
    if 72 <= lat < 84:
        for west, east, special in ((0, 9, 31), (9, 21, 33), (21, 33, 35), (33, 42, 37)):
            if west <= lon < east:
                zone = special
    crs = CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)
    xs, ys = transform("EPSG:4326", crs, [lon], [lat])
    half = size * h / 2
    affine = from_origin(xs[0] - half, ys[0] + half, h, h)
    bbox = transform_bounds(crs, "EPSG:4326", *array_bounds(size, size, affine), densify_pts=21)
    if bbox[0] >= bbox[2]:
        raise ValueError("Участок пересекает линию перемены дат; выберите другой центр.")
    return Window(crs, affine, size, h, bbox)


def classify(tags, kind: str) -> ClassCode | None:
    """Консервативные, явно описанные правила версии 1."""
    if kind == "area":
        if any(tags.get(key) not in (None, "", "no") for key in ("building", "building:part")):
            return ClassCode.EXISTING_BUILDING
        if (tags.get("landuse") in BLOCKED_LANDUSE or tags.get("natural") in BLOCKED_NATURAL
                or tags.get("leisure") in BLOCKED_LEISURE or tags.get("amenity") in BLOCKED_AMENITY
                or tags.get("boundary") in {"protected_area", "national_park"}
                or tags.get("waterway") == "riverbank" or tags.get("aeroway") in {"aerodrome", "apron", "runway", "taxiway"}
                or tags.get("power") in {"substation", "plant"} or tags.get("man_made") == "wastewater_plant"):
            return ClassCode.BLOCKED
    if kind == "way":
        highway = tags.get("highway")
        if highway and highway != "no":
            # Эстакады/тоннели не образуют надёжных связей в плоской сетке.
            if (highway in ROAD_TYPES and tags.get("access") not in {"no", "private"}
                    and tags.get("bridge") in (None, "", "no") and tags.get("tunnel") in (None, "", "no")
                    and tags.get("layer") in (None, "", "0")):
                return ClassCode.EXISTING_ROAD
            return ClassCode.BLOCKED
        if (tags.get("waterway") in {"river", "stream", "canal", "drain", "ditch"}
                or tags.get("railway") in {"rail", "light_rail", "tram", "narrow_gauge", "subway"}
                or tags.get("aeroway") in {"runway", "taxiway"}):
            return ClassCode.BLOCKED
    if kind == "node" and tags.get("amenity") in BLOCKED_AMENITY:
        return ClassCode.BLOCKED
    return None


def _overlaps(a, b) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _rasterize(shapes: dict, window: Window) -> np.ndarray:
    # Приоритет: здания > известные запреты > дороги > неизвестное пространство.
    # all_touched сохраняет даже узкие объекты в пределах грубой сетки.
    classes = np.zeros((window.size, window.size), dtype=np.uint8)
    for code in (ClassCode.EXISTING_ROAD, ClassCode.BLOCKED, ClassCode.EXISTING_BUILDING):
        if shapes[code]:
            mask = rasterize(shapes[code], out_shape=classes.shape, transform=window.transform,
                             fill=0, default_value=1, all_touched=True, dtype="uint8")
            classes[mask != 0] = code
    return classes


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_pbf(path: Path, window: Window, *, index: str = "flex_mem") -> tuple[bytes, dict]:
    """Сканирует PBF локально, хранит геометрии только выбранного окна.

    Индекс координат узлов и сборщик multipolygon могут использовать RAM
    всего регионального файла. Время чтения зависит от его размера.
    """
    import osmium

    path = Path(path)
    if not path.is_file() or not path.name.lower().endswith(".osm.pbf"):
        raise ValueError("Укажите существующий файл .osm.pbf.")
    if index not in osmium.index.map_types():
        raise ValueError("Неподдерживаемый индекс osmium: " + index)
    source_hash = _hash_file(path)
    factory = osmium.geom.GeoJSONFactory()
    shapes = {code: [] for code in (ClassCode.EXISTING_ROAD, ClassCode.BLOCKED, ClassCode.EXISTING_BUILDING)}
    invalid = 0
    processor = (osmium.FileProcessor(str(path)).with_locations(index).with_areas()
                 .with_filter(osmium.filter.KeyFilter(*KEYS)))
    for obj in processor:
        kind = obj.type_str()
        if kind not in ("n", "w", "a"):
            continue
        code = classify(obj.tags, {"n": "node", "w": "way", "a": "area"}[kind])
        if code is None:
            continue
        try:
            if kind == "n":
                geom = json.loads(factory.create_point(obj))
            elif kind == "w":
                geom = json.loads(factory.create_linestring(obj))
            else:
                geom = json.loads(factory.create_multipolygon(obj))
            if not _overlaps(bounds(geom), window.wgs84_bounds):
                continue
            projected = transform_geom("EPSG:4326", window.crs, geom)
            shapes[code].append(projected)
        except (RuntimeError, ValueError, osmium.InvalidLocationError):
            invalid += 1
    classes = _rasterize(shapes, window)
    metadata = {
        "import_version": IMPORT_VERSION, "source_name": path.name,
        "source_sha256": source_hash, "attribution": ATTRIBUTION,
        "bounds_wgs84": list(window.wgs84_bounds), "crs": window.crs.to_string(),
        "transform": list(window.transform)[:6], "size": window.size,
        "pixel_size_m": window.pixel_size_m,
        "features_in_window": {str(int(code)): len(items) for code, items in shapes.items()},
        "invalid_geometries_in_scan": invalid,
        "class_cells": {str(code): int(np.count_nonzero(classes == code)) for code in range(4)},
        "warning": "Код 0: неизвестные или незанятые по OSM ячейки, не подтверждённая доступность строительства.",
    }
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", height=window.size, width=window.size, count=1,
                         dtype="uint8", crs=window.crs, transform=window.transform,
                         nodata=255, compress="deflate") as dataset:
            dataset.write(classes, 1)
            dataset.update_tags(OSM_IMPORT=json.dumps(metadata, ensure_ascii=False),
                                ATTRIBUTION=ATTRIBUTION)
        data = memory.read()
    # Та же проверка, что в UI, включая наличие исходной дороги.
    read_terrain(data)
    return data, metadata
