"""Общие модели, единицы и коэффициенты MVP (§2–4 DESIGN.md)."""

from dataclasses import dataclass, field, fields
from enum import IntEnum, StrEnum
from math import isfinite
from numbers import Integral, Real
import re

import numpy as np
from numpy.typing import NDArray
from rasterio.crs import CRS
from rasterio.transform import Affine


class ClassCode(IntEnum):
    FREE = 0
    BLOCKED = 1
    EXISTING_ROAD = 2
    EXISTING_BUILDING = 3
    NEW_ROAD = 4
    NEW_BUILDING = 5
    NEW_SCHOOL = 6
    NODATA = 255


INPUT_CLASS_CODES = frozenset({
    ClassCode.FREE, ClassCode.BLOCKED, ClassCode.EXISTING_ROAD,
    ClassCode.EXISTING_BUILDING, ClassCode.NODATA,
})
PIXEL_SIZE_M_RANGE = (5.0, 10.0)
MAX_RASTER_SIDE = 256
MAX_INPUT_BYTES = 8 * 1024 * 1024
GEOREFERENCE_REL_TOL = 1e-6

HOUSE_SIDE_CELLS = 2
USEFUL_AREA_FRACTION = 0.8
AREA_PER_RESIDENT_M2 = 30.0
SCHOOLCHILD_FRACTION = 0.12

ROAD_PROXIMITY_WEIGHT = 0.7
BARRIER_CLEARANCE_WEIGHT = 0.3
ROAD_DISTANCE_SCALE_M = 300.0
BARRIER_DISTANCE_SCALE_M = 50.0


class CoordinateMode(StrEnum):
    LOCAL = "local"
    GEOREFERENCED = "georeferenced"


def validate_pixel_size_m(value: float) -> float:
    """Проверка масштаба, общая для заголовка файла и модели."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not PIXEL_SIZE_M_RANGE[0] <= value <= PIXEL_SIZE_M_RANGE[1]
    ):
        raise ValueError("Шаг растра должен быть от 5 до 10 м/пиксель")
    return float(value)


def validate_georeference(crs: CRS | None, transform: Affine | None, pixel_size_m: float) -> None:
    """Проверить метрическую сетку до чтения пикселей или создания Terrain."""
    if not isinstance(crs, CRS) or not crs.is_projected:
        raise ValueError("Нужна проецированная CRS с единицей метр")
    if crs.linear_units_factor[1] != 1.0:
        raise ValueError("Единица CRS должна быть метр")
    if not isinstance(transform, Affine) or not all(isfinite(v) for v in transform):
        raise ValueError("Нужна конечная affine-привязка GeoTIFF")
    h = validate_pixel_size_m(pixel_size_m)
    tolerance = h * GEOREFERENCE_REL_TOL
    if (
        abs(transform.a - h) > tolerance
        or abs(transform.e + h) > tolerance
        or abs(transform.b) > tolerance
        or abs(transform.d) > tolerance
    ):
        raise ValueError("Нужен квадратный пиксель без поворота: восток по столбцам, юг по строкам")


@dataclass(frozen=True, slots=True)
class Scenario:
    """Параметры запуска; metadata полей доступны форме и генератору.

    Границы включены. Счётчики и seed — целые числа, предельный путь —
    конечное число метров. Булевы значения параметрами не считаются.
    """

    target_population: int = field(
        default=600, metadata={"min": 1, "max": 20_000, "unit": "чел."}
    )
    floors: int = field(default=3, metadata={"min": 1, "max": 9, "unit": "этажей"})
    street_spacing_cells: int = field(
        default=8, metadata={"min": 5, "max": 20, "unit": "ячеек"}
    )
    school_budget: int = field(default=1, metadata={"min": 0, "max": 3, "unit": "шт."})
    school_capacity: int = field(
        default=80, metadata={"min": 1, "max": 500, "unit": "мест"}
    )
    max_school_distance_m: float = field(
        default=400.0, metadata={"min": 50, "max": 2_000, "unit": "м"}
    )
    seed: int = field(default=42, metadata={"min": 0, "max": 2**32 - 1, "unit": ""})

    def __post_init__(self) -> None:
        for parameter in fields(self):
            value = getattr(self, parameter.name)
            expected_type = Real if parameter.name == "max_school_distance_m" else Integral
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, expected_type):
                raise ValueError(f"{parameter.name}: недопустимый тип параметра")
            lower, upper = parameter.metadata["min"], parameter.metadata["max"]
            if not lower <= value <= upper:
                raise ValueError(f"{parameter.name}: значение должно быть от {lower} до {upper}")
            # Стандартные скаляры удобны для дальнейшего экспорта в JSON.
            normalized = float(value) if expected_type is Real else int(value)
            object.__setattr__(self, parameter.name, normalized)


@dataclass(frozen=True, slots=True, eq=False)
class Terrain:
    """Исходная территория в памяти, не изменяемая генератором.

    classes — двумерные исходные коды; valid_mask=True означает включение
    в исследование. Код 255 всегда исключён. Исключённые ячейки нормализуются
    в 255, входные массивы копируются и защищаются от записи.

    pixel_size_m измеряется в метрах, input_sha256 — хеш байтов входного файла.
    В локальном режиме CRS отсутствует, transform строится от левого нижнего
    угла (0, 0). Геопривязанный режим сохраняет исходные CRS и transform.
    Чтение TIFF и проверка его заголовка выполняются загрузчиком.
    """

    classes: NDArray[np.uint8]
    valid_mask: NDArray[np.bool_]
    pixel_size_m: float
    coordinate_mode: CoordinateMode
    input_sha256: str
    crs: CRS | None = None
    transform: Affine | None = None

    def __post_init__(self) -> None:
        classes = np.asarray(self.classes)
        mask = np.asarray(self.valid_mask)
        if classes.ndim != 2 or not all(1 <= size <= MAX_RASTER_SIDE for size in classes.shape):
            raise ValueError("Территория должна быть двумерной, от 1 до 256 ячеек по стороне")
        if not np.issubdtype(classes.dtype, np.integer):
            raise ValueError("Коды классов должны быть целочисленными")
        if mask.shape != classes.shape or mask.dtype != np.bool_:
            raise ValueError("Маска должна быть булевой и совпадать с размером растра")
        mask = mask & (classes != ClassCode.NODATA)
        if not mask.any():
            raise ValueError("Маска покрытия пуста")
        if not np.isin(classes[mask], tuple(INPUT_CLASS_CODES)).all():
            raise ValueError("В территории есть коды вне входной легенды 0–3 и 255")
        if not np.any(mask & (classes == ClassCode.EXISTING_ROAD)):
            raise ValueError("Нужна хотя бы одна исходная дорожная ячейка")

        h = validate_pixel_size_m(self.pixel_size_m)
        object.__setattr__(self, "pixel_size_m", h)
        if not isinstance(self.input_sha256, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", self.input_sha256
        ):
            raise ValueError("Хеш входного файла должен быть SHA-256 из 64 шестнадцатеричных знаков")
        object.__setattr__(self, "input_sha256", self.input_sha256.lower())

        try:
            mode = CoordinateMode(self.coordinate_mode)
        except (ValueError, TypeError) as error:
            raise ValueError("Неизвестный режим координат") from error
        object.__setattr__(self, "coordinate_mode", mode)
        if mode == CoordinateMode.LOCAL:
            if self.crs is not None or self.transform is not None:
                raise ValueError("В локальном режиме явно сбросьте исходную CRS и привязку")
            object.__setattr__(self, "transform", Affine(h, 0, 0, 0, -h, classes.shape[0] * h))
        else:
            validate_georeference(self.crs, self.transform, h)

        normalized = np.where(mask, classes, ClassCode.NODATA).astype(np.uint8)
        normalized.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "classes", normalized)
        object.__setattr__(self, "valid_mask", mask)

    @property
    def cell_area_m2(self) -> float:
        return self.pixel_size_m**2


@dataclass(frozen=True, slots=True, eq=False)
class Suitability:
    """Данные первого шага генерации на исходной сетке.

    Расстояния между центрами ячеек измеряются в метрах. scores содержит
    балл S на free_mask и -inf вне неё: такие ячейки исключены из ранжирования.
    Массивы, возвращённые calculate_suitability, защищены от записи.
    """

    free_mask: NDArray[np.bool_]
    road_distance_m: NDArray[np.float64]
    barrier_distance_m: NDArray[np.float64]
    scores: NDArray[np.float64]
