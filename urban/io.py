"""Чтение одной тематической карты TIFF из байтов, без файлов на диске."""

from hashlib import sha256
import json
from math import floor, isfinite
import warnings

import numpy as np
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.errors import NotGeoreferencedWarning, RasterioError
from rasterio.io import DatasetReader, MemoryFile
from rasterio.transform import Affine

from urban.model import (
    INPUT_CLASS_CODES,
    MAX_INPUT_BYTES,
    MAX_RASTER_SIDE,
    ClassCode,
    CoordinateMode,
    Terrain,
    validate_georeference,
    validate_pixel_size_m,
)


class TerrainReadError(ValueError):
    """Ошибка входа с сообщением, пригодным для показа в интерфейсе."""


def local_metric_crs(lon: float, lat: float) -> CRS:
    """Местная UTM, общая для векторных источников и OSM."""
    if not (isfinite(lon) and isfinite(lat) and -180 < lon < 180 and -80 <= lat <= 84):
        raise ValueError("Центр должен быть в пределах UTM: долгота (-180,180), широта [-80,84].")
    zone = min(60, floor((lon + 180) / 6) + 1)
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone = 32
    if 72 <= lat < 84:
        for west, east, special in ((0, 9, 31), (9, 21, 33), (21, 33, 35), (33, 42, 37)):
            if west <= lon < east:
                zone = special
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def rasterize_classes(shapes: dict, shape: tuple[int, int], transform: Affine) -> np.ndarray:
    """Единый приоритет OSM/GeoJSON: здания > запреты > дороги > свободное.

    all_touched сохраняет узкие объекты на грубой сетке; растеризация
    не исправляет разрывы дорожной топологии.
    """
    classes = np.zeros(shape, dtype=np.uint8)
    for code in (ClassCode.EXISTING_ROAD, ClassCode.BLOCKED, ClassCode.EXISTING_BUILDING):
        if shapes[code]:
            mask = rasterize(shapes[code], out_shape=shape, transform=transform,
                             fill=0, default_value=1, all_touched=True, dtype="uint8")
            classes[mask != 0] = code
    return classes


def read_terrain(
    data: bytes,
    local_scale: float | None = None,
    *,
    discard_georeferencing: bool = False,
) -> Terrain:
    """Прочитать TIFF согласно §2 DESIGN.md.

    local_scale явно выбирает локальный масштаб в метрах. Если у файла
    есть любая привязка, её сброс дополнительно требует
    discard_georeferencing=True. Исходный SHA-256 относится к байтам файла,
    а маска объединяет внутреннюю маску TIFF, объявленный NoData и код 255.
    """
    if not isinstance(data, bytes) or not data:
        raise TerrainReadError("Передайте байты одного непустого файла TIFF.")
    if len(data) > MAX_INPUT_BYTES:
        raise TerrainReadError("Файл превышает 8 МиБ. Сохраните меньший фрагмент территории.")
    if data[:4] not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        raise TerrainReadError("Неизвестный формат. Загрузите одноканальную карту классов TIFF.")
    if not isinstance(discard_georeferencing, bool):
        raise TerrainReadError("Сброс привязки должен быть явно задан как True или False.")

    try:
        # Отсутствие привязки — поддерживаемый режим; вместо предупреждения
        # GDAL ниже выдаётся требование подтвердить локальный масштаб.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", NotGeoreferencedWarning)
            with MemoryFile(data) as memory, memory.open() as dataset:
                _validate_header(dataset)
                missing_georeference = any(
                    issubclass(item.category, NotGeoreferencedWarning) for item in caught
                )
                mode, h = _coordinate_settings(
                    dataset, local_scale, discard_georeferencing, missing_georeference,
                )

                classes = dataset.read(1)
                valid_mask = dataset.read_masks(1) > 0
                if dataset.nodata is not None:
                    # Внутренняя маска GDAL может иметь приоритет над NoData;
                    # контракт требует пересечения всех источников покрытия.
                    valid_mask &= classes != dataset.nodata
                valid_mask &= classes != ClassCode.NODATA

                if not valid_mask.any():
                    raise TerrainReadError("Маска покрытия пуста. Добавьте ячейки исследуемой территории.")
                values = classes[valid_mask]
                invalid = np.unique(values[~np.isin(values, tuple(INPUT_CLASS_CODES))])
                if invalid.size:
                    shown = ", ".join(str(value) for value in invalid[:5])
                    raise TerrainReadError(
                        f"Коды вне легенды: {shown}. Используйте 0–3 и 255; "
                        "подготовьте тематическую карту вместо снимка или высот."
                    )
                if not np.any(values == ClassCode.EXISTING_ROAD):
                    raise TerrainReadError(
                        "Нет исходной дороги внутри маски. Добавьте дорожные ячейки с кодом 2."
                    )
                osm_import_json = dataset.tags().get("OSM_IMPORT")
                if osm_import_json is not None:
                    try:
                        metadata = json.loads(osm_import_json)
                        if not isinstance(metadata, dict):
                            raise ValueError("Нужен объект JSON")
                        osm_import_json = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
                    except (ValueError, TypeError, RecursionError) as error:
                        raise TerrainReadError("Повреждены метаданные OSM_IMPORT. Повторите импорт исходного PBF.") from error
                return Terrain(
                    classes=classes, valid_mask=valid_mask, pixel_size_m=h,
                    coordinate_mode=mode, input_sha256=sha256(data).hexdigest(),
                    crs=dataset.crs if mode == CoordinateMode.GEOREFERENCED else None,
                    transform=dataset.transform if mode == CoordinateMode.GEOREFERENCED else None,
                    osm_import_json=osm_import_json,
                )
    except TerrainReadError:
        raise
    except (RasterioError, OSError, ValueError) as error:
        raise TerrainReadError(
            "Не удалось прочитать или декодировать TIFF. "
            "Проверьте файл и повторно экспортируйте одноканальную карту классов."
        ) from error


def _validate_header(dataset: DatasetReader) -> None:
    if dataset.driver != "GTiff" or dataset.subdatasets:
        raise TerrainReadError("Нужен один растр TIFF. Экспортируйте одну карту без дополнительных страниц.")
    if not (1 <= dataset.width <= MAX_RASTER_SIDE and 1 <= dataset.height <= MAX_RASTER_SIDE):
        raise TerrainReadError("Размер растра должен быть от 1 до 256 ячеек по стороне. Обрежьте участок.")
    if dataset.count != 1:
        raise TerrainReadError("Нужен один канал с кодами классов. RGB-снимок следует заменить тематической картой.")
    if not np.issubdtype(np.dtype(dataset.dtypes[0]), np.integer):
        raise TerrainReadError("Нужен целочисленный тип пикселей. Экспортируйте коды классов как UInt8.")
    if dataset.nodata in (0, 1, 2, 3):
        raise TerrainReadError("NoData совпадает с кодом 0–3. Перенесите NoData в 255 и экспортируйте TIFF заново.")


def _coordinate_settings(
    dataset: DatasetReader, local_scale: float | None,
    discard_georeferencing: bool, missing_georeference: bool,
) -> tuple[CoordinateMode, float]:
    has_crs = dataset.crs is not None
    # Одинаковая identity-матрица бывает и явной привязкой, и результатом
    # её отсутствия. Во втором случае Rasterio выдаёт NotGeoreferencedWarning.
    has_transform = dataset.transform != Affine.identity() or (
        not has_crs and not missing_georeference
    )
    has_other_georeference = bool(dataset.gcps[0]) or dataset.rpcs is not None
    has_any_georeference = has_crs or has_transform or has_other_georeference
    if discard_georeferencing:
        if local_scale is None:
            raise TerrainReadError("При сбросе привязки укажите локальный масштаб 5–10 м/пиксель.")
    elif has_any_georeference:
        if not has_crs or not has_transform or has_other_georeference:
            raise TerrainReadError(
                "Привязка неполная или задана через GCP/RPC. Исправьте CRS и affine-привязку "
                "либо явно выберите локальный режим со сбросом привязки "
                "и масштабом 5–10 м/пиксель."
            )
        if local_scale is not None:
            raise TerrainReadError(
                "У файла есть геопривязка. Для локального режима явно сбросьте её "
                "и подтвердите масштаб; иначе используйте геопривязку файла."
            )
        try:
            h = dataset.transform.a
            validate_georeference(dataset.crs, dataset.transform, h)
        except ValueError as error:
            raise TerrainReadError(
                f"{error}. Перепроецируйте в местную метрическую CRS с квадратным пикселем "
                "5–10 м без поворота, методом ближайшего соседа."
            ) from error
        return CoordinateMode.GEOREFERENCED, h
    elif local_scale is None:
        raise TerrainReadError("У TIFF нет геопривязки. Выберите локальный режим и масштаб 5–10 м/пиксель.")

    try:
        return CoordinateMode.LOCAL, validate_pixel_size_m(local_scale)
    except ValueError as error:
        raise TerrainReadError(f"{error}. Укажите подтверждённый локальный масштаб.") from error
