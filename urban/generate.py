"""Стадии генерации на неизменяемой исходной территории."""

import numpy as np
from scipy.ndimage import binary_propagation, distance_transform_edt, generate_binary_structure

from urban.model import (
    BARRIER_CLEARANCE_WEIGHT,
    BARRIER_DISTANCE_SCALE_M,
    ROAD_DISTANCE_SCALE_M,
    ROAD_PROXIMITY_WEIGHT,
    ClassCode,
    RoadNetwork,
    Scenario,
    Suitability,
    Terrain,
)


def calculate_suitability(terrain: Terrain) -> Suitability:
    """Свободная маска и S из §4, шаг 1 DESIGN.md; без случайности.

    Расстояние до дорог евклидово и может пересекать запреты. Достижимость
    участков определяется при росте улиц. Для расстояния до
    запретов за пределами растра добавляется рамка из исключённых ячеек:
    ближайший центр рамки находится в h метрах от центра граничной ячейки.
    Существующая застройка исключена из free_mask, но не служит фактором b.
    """
    classes = terrain.classes
    free_mask = terrain.valid_mask & (classes == ClassCode.FREE)
    roads = terrain.valid_mask & (classes == ClassCode.EXISTING_ROAD)
    road_distance_m = distance_transform_edt(~roads, sampling=terrain.pixel_size_m)

    non_barrier_mask = terrain.valid_mask & (classes != ClassCode.BLOCKED)
    padded = np.pad(non_barrier_mask, 1, mode="constant", constant_values=False)
    barrier_distance_m = distance_transform_edt(padded, sampling=terrain.pixel_size_m)[1:-1, 1:-1].copy()

    scores = (
        ROAD_PROXIMITY_WEIGHT * np.maximum(0, 1 - road_distance_m / ROAD_DISTANCE_SCALE_M)
        + BARRIER_CLEARANCE_WEIGHT * np.minimum(1, barrier_distance_m / BARRIER_DISTANCE_SCALE_M)
    )
    scores[~free_mask] = -np.inf
    for array in (free_mask, road_distance_m, barrier_distance_m, scores):
        array.setflags(write=False)
    return Suitability(free_mask, road_distance_m, barrier_distance_m, scores)


def generate_roads(terrain: Terrain, scenario: Scenario) -> RoadNetwork:
    """Вырастить улицы по §4, шаг 2 DESIGN.md от всех исходных дорог.

    Локальный PCG64 получает seed явно из сценария. Оси разрешены только
    на исходной свободной земле. Четырёхсвязное распространение идёт
    по старым дорогам и разрешённым осям; изолированные оси исключаются.
    На результат влияют только территория, шаг улиц и seed.
    """
    suitability = calculate_suitability(terrain)
    spacing = scenario.street_spacing_cells
    rng = np.random.Generator(np.random.PCG64(scenario.seed))
    row_offset, column_offset = (int(value) for value in rng.integers(0, spacing, size=2))
    rows, columns = np.indices(terrain.classes.shape)
    axes = (rows % spacing == row_offset) | (columns % spacing == column_offset)
    permitted_axes = axes & suitability.free_mask
    existing_roads = terrain.valid_mask & (terrain.classes == ClassCode.EXISTING_ROAD)
    road_mask = binary_propagation(
        existing_roads,
        structure=generate_binary_structure(2, 1),
        mask=existing_roads | permitted_axes,
    )
    new_roads = road_mask & permitted_axes
    road_mask.setflags(write=False)
    new_roads.setflags(write=False)
    return RoadNetwork(suitability, new_roads, road_mask, (row_offset, column_offset))
