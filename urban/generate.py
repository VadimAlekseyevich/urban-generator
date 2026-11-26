"""Стадии генерации на неизменяемой исходной территории."""

import numpy as np
from scipy.ndimage import distance_transform_edt

from urban.model import (
    BARRIER_CLEARANCE_WEIGHT,
    BARRIER_DISTANCE_SCALE_M,
    ROAD_DISTANCE_SCALE_M,
    ROAD_PROXIMITY_WEIGHT,
    ClassCode,
    Suitability,
    Terrain,
)


def calculate_suitability(terrain: Terrain) -> Suitability:
    """Свободная маска и S из §4, шаг 1 DESIGN.md; без случайности.

    Расстояние до дорог евклидово и может пересекать запреты. Достижимость
    по сети будет проверяться на следующей стадии. Для расстояния до
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
