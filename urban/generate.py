"""Стадии генерации на неизменяемой исходной территории."""

from math import floor

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import binary_dilation, binary_propagation, distance_transform_edt, generate_binary_structure

from urban.model import (
    BARRIER_CLEARANCE_WEIGHT,
    BARRIER_DISTANCE_SCALE_M,
    HOUSE_SIDE_CELLS,
    ROAD_DISTANCE_SCALE_M,
    ROAD_PROXIMITY_WEIGHT,
    SCHOOL_SECTOR_COLUMNS,
    SCHOOL_SECTOR_ROWS,
    Building,
    BuildingPlan,
    ClassCode,
    RoadNetwork,
    Scenario,
    SchoolReserve,
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
    if terrain.development_mask is not None:
        free_mask &= terrain.development_mask
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


def reserve_school_sites(roads: RoadNetwork) -> SchoolReserve:
    """Резерв площадок по §4, шаг 3 DESIGN.md, независимо от бюджета.

    Каждая площадка свободна после роста улиц и соседствует с достигнутой
    дорогой по стороне. Строки делятся на 3 группы, столбцы — на 4;
    при остатке первые группы на одну ячейку шире (np.array_split).
    В каждом секторе максимум S выбирается в построчном порядке.
    """
    near_road = binary_dilation(roads.road_mask, structure=generate_binary_structure(2, 1))
    candidates = roads.suitability.free_mask & ~roads.road_mask & near_road
    height, width = candidates.shape
    row_groups = np.array_split(np.arange(height), SCHOOL_SECTOR_ROWS)
    column_groups = np.array_split(np.arange(width), SCHOOL_SECTOR_COLUMNS)
    mask = np.zeros_like(candidates)
    coordinates = []
    for rows in row_groups:
        for columns in column_groups:
            if not rows.size or not columns.size:
                continue
            row_start, column_start = int(rows[0]), int(columns[0])
            sector = np.s_[row_start:int(rows[-1]) + 1, column_start:int(columns[-1]) + 1]
            positions = np.argwhere(candidates[sector])
            if not positions.size:
                continue
            scores = roads.suitability.scores[sector]
            best = int(np.argmax(scores[positions[:, 0], positions[:, 1]]))
            row, column = positions[best]
            coordinate = (row_start + int(row), column_start + int(column))
            coordinates.append(coordinate)
            mask[coordinate] = True
    mask.setflags(write=False)
    return SchoolReserve(tuple(sorted(coordinates)), mask)


def generate_buildings(terrain: Terrain, scenario: Scenario) -> BuildingPlan:
    """Сквозной расчёт улиц, резерва и новых домов без запуска интерфейса."""
    roads = generate_roads(terrain, scenario)
    reserve = reserve_school_sites(roads)
    return place_buildings(terrain, scenario, roads, reserve)


def place_buildings(
    terrain: Terrain, scenario: Scenario, roads: RoadNetwork, reserve: SchoolReserve,
) -> BuildingPlan:
    """Размещение окон 2×2 по §4, шаги 4–5 DESIGN.md.

    Кандидаты изначально упорядочены по строке и столбцу. Перестановка
    PCG64(seed) и стабильная сортировка по убыванию среднего S задают
    порядок равных кандидатов. Занятость проверяется после каждого дома.
    """
    side = HOUSE_SIDE_CELLS
    ids = np.zeros(terrain.classes.shape, dtype=np.int32)
    buildings = []
    available = roads.suitability.free_mask & ~roads.road_mask & ~reserve.mask
    if min(available.shape) >= side:
        near_road = binary_dilation(roads.road_mask, structure=generate_binary_structure(2, 1))
        free_windows = sliding_window_view(available, (side, side)).all(axis=(-2, -1))
        road_windows = sliding_window_view(near_road, (side, side)).any(axis=(-2, -1))
        candidates = np.argwhere(free_windows & road_windows)
        score_windows = sliding_window_view(roads.suitability.scores, (side, side))
        scores = score_windows[candidates[:, 0], candidates[:, 1]].mean(axis=(-2, -1))
        rng = np.random.Generator(np.random.PCG64(scenario.seed))
        shuffled = rng.permutation(len(candidates))
        order = shuffled[np.argsort(-scores[shuffled], kind="stable")]
        area = side**2 * terrain.cell_area_m2
        residents = floor(area * scenario.floors * scenario.demographic_profile.useful_area_fraction / scenario.demographic_profile.area_per_resident_m2 + 0.5)
        demand = floor(scenario.demographic_profile.schoolchild_fraction * residents + 0.5)
        population = 0
        for index in order:
            row, column = (int(value) for value in candidates[index])
            footprint = np.s_[row:row + side, column:column + side]
            if not available[footprint].all():
                continue
            house_entry, road_entry = _building_entry(row, column, roads.road_mask)
            building_id = len(buildings) + 1
            available[footprint] = False
            ids[footprint] = building_id
            buildings.append(Building(
                id=building_id, row=row, column=column,
                house_entry=house_entry, road_entry=road_entry, floors=scenario.floors,
                footprint_area_m2=area, population=residents, school_demand=demand,
            ))
            population += residents
            if population >= scenario.target_population:
                break
    ids.setflags(write=False)
    return BuildingPlan(terrain, scenario, roads, reserve, tuple(buildings), ids)


def _building_entry(row: int, column: int, road_mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """Первая по координатам пара ячеек дома и дороги с общей стороной."""
    height, width = road_mask.shape
    for r in range(row, row + HOUSE_SIDE_CELLS):
        for c in range(column, column + HOUSE_SIDE_CELLS):
            for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                neighbour = (r + dr, c + dc)
                if 0 <= neighbour[0] < height and 0 <= neighbour[1] < width and road_mask[neighbour]:
                    return (r, c), neighbour
    raise ValueError("У дома отсутствует вход к дорожной сети")
