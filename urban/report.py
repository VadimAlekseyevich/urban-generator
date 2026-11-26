"""Карты, независимая проверка результата и итоговые метрики."""

from collections import deque
from dataclasses import dataclass
from math import floor
from numbers import Integral

import numpy as np
from scipy.ndimage import label
from PIL import Image, ImageDraw

from urban.model import (
    AREA_PER_RESIDENT_M2, HOUSE_SIDE_CELLS, SCHOOLCHILD_FRACTION, USEFUL_AREA_FRACTION,
    BuildingPlan, ClassCode, RoadNetwork, Terrain,
)
from urban.services import SchoolAllocation


# Подписи и цвета общие для карты и легенды интерфейса.
INPUT_LEGEND = {
    ClassCode.FREE: ("Свободная территория", "#e5e7eb"),
    ClassCode.BLOCKED: ("Запрет, в том числе вода", "#2563eb"),
    ClassCode.EXISTING_ROAD: ("Существующая дорога", "#475569"),
    ClassCode.EXISTING_BUILDING: ("Существующая застройка", "#b45309"),
    ClassCode.NODATA: ("Вне исследования / NoData", "#111827"),
}

ROAD_LEGEND = {**INPUT_LEGEND, ClassCode.NEW_ROAD: ("Новая улица", "#ef4444")}
BUILDING_LEGEND = {**ROAD_LEGEND, ClassCode.NEW_BUILDING: ("Новый дом", "#16a34a")}
SCHOOL_LEGEND = {**BUILDING_LEGEND, ClassCode.NEW_SCHOOL: ("Новая школа", "#9333ea")}


def terrain_image(terrain: Terrain) -> Image.Image:
    """Предпросмотр с целочисленным увеличением без смешивания классов."""
    return _classes_image(terrain.classes, INPUT_LEGEND)


def road_project_classes(terrain: Terrain, roads: RoadNetwork) -> np.ndarray:
    """Наложить новые улицы на копию исходного растра."""
    project = terrain.classes.copy()
    project[roads.new_roads] = ClassCode.NEW_ROAD
    project.setflags(write=False)
    return project


def road_project_image(terrain: Terrain, roads: RoadNetwork) -> Image.Image:
    return _classes_image(road_project_classes(terrain, roads), ROAD_LEGEND)


def building_project_classes(plan: BuildingPlan) -> np.ndarray:
    """Проект с домами на отдельной копии исходного растра."""
    project = plan.terrain.classes.copy()
    project[plan.roads.new_roads] = ClassCode.NEW_ROAD
    project[plan.building_ids > 0] = ClassCode.NEW_BUILDING
    project.setflags(write=False)
    return project


def building_project_image(plan: BuildingPlan) -> Image.Image:
    return _classes_image(building_project_classes(plan), BUILDING_LEGEND)


def building_rows(plan: BuildingPlan) -> list[dict]:
    return [{
        "ID": building.id, "Строка": building.row, "Столбец": building.column,
        "Этажей": building.floors, "Площадь пятна, м²": building.footprint_area_m2,
        "Жителей, чел.": building.population,
    } for building in plan.buildings]


def _classes_image(classes: np.ndarray, legend: dict) -> Image.Image:
    rgb = np.zeros((*classes.shape, 3), dtype=np.uint8)
    for code, (_, color) in legend.items():
        rgb[classes == code] = tuple(bytes.fromhex(color[1:]))
    image = Image.fromarray(rgb)
    factor = max(1, 768 // max(image.size))
    return image.resize((image.width * factor, image.height * factor), Image.Resampling.NEAREST)


def terrain_areas(terrain: Terrain) -> list[dict]:
    """NoData имеет число ячеек, но исключён из площади исследования."""
    rows = []
    for code, (label, _) in INPUT_LEGEND.items():
        count = int(np.count_nonzero(terrain.classes == code))
        rows.append({
            "Код": int(code), "Класс": label, "Ячеек": count,
            "Площадь, м²": count * terrain.cell_area_m2 if code != ClassCode.NODATA else None,
        })
    return rows


def school_project_classes(plan: BuildingPlan, allocation: SchoolAllocation) -> np.ndarray:
    """Отмечены только открытые школы, незанятый резерв остаётся свободным."""
    project = building_project_classes(plan).copy()
    for school in allocation.schools:
        project[school.site.coordinate] = ClassCode.NEW_SCHOOL
    project.setflags(write=False)
    return project


def school_project_image(plan: BuildingPlan, allocation: SchoolAllocation) -> Image.Image:
    """Тот же масштаб, что у исходной карты; номера — только на предпросмотре."""
    image = _classes_image(school_project_classes(plan, allocation), SCHOOL_LEGEND)
    factor = image.width // plan.terrain.classes.shape[1]
    draw = ImageDraw.Draw(image)
    for school in allocation.schools:
        row, column = school.site.coordinate
        label = str(school.id)
        left, top, right, bottom = draw.textbbox((0, 0), label, stroke_width=1)
        x = min((column + 1) * factor + 2, image.width - right - 1)
        y = max(-top, min(row * factor, image.height - bottom - 1))
        # На правом краю подпись слева от площадки, чтобы не закрыть её.
        if x < (column + 1) * factor:
            x = max(-left, column * factor - right - 2)
        draw.text((x, y), label, fill="white", stroke_width=1, stroke_fill="#111827")
    return image


def school_rows(allocation: SchoolAllocation) -> list[dict]:
    return [{
        "ID школы": school.id, "Строка": school.site.coordinate[0],
        "Столбец": school.site.coordinate[1], "Вместимость, мест": school.capacity,
        "Занято, мест": school.served,
        "Загрузка, %": round(100 * school.served / school.capacity, 2),
        "Макс. назначенный путь, м": school.max_assigned_distance_m,
    } for school in allocation.schools]


def school_deficit_reasons(allocation: SchoolAllocation) -> tuple[int, int]:
    """Остатки спроса без достижимой школы и при наличии достижимой школы."""
    no_route = sum(b.remaining for b in allocation.buildings if not b.reachable_school)
    no_capacity = sum(b.remaining for b in allocation.buildings if b.reachable_school)
    return no_route, no_capacity


class ReportValidationError(ValueError):
    """Готовый проект нарушает жёсткое ограничение; выдача запрещена."""


@dataclass(frozen=True, slots=True)
class ResultMetrics:
    violations_m2: float
    disconnected_road_cells: int
    new_road_cells: int
    new_road_area_m2: float
    new_buildings: int
    building_area_m2: float
    developed_area_m2: float
    developed_percent: float | None
    target_population: int
    population: int
    population_delta: int
    population_shortfall: int
    school_demand: int
    school_served: int
    school_deficit: int
    school_coverage_percent: float | None
    opened_schools: int
    no_route_places: int
    no_capacity_places: int


METRIC_UNITS = {
    "violations_m2": "м²", "disconnected_road_cells": "ячеек",
    "new_road_cells": "ячеек", "new_road_area_m2": "м²",
    "new_buildings": "шт.", "building_area_m2": "м²",
    "developed_area_m2": "м²", "developed_percent": "%",
    "target_population": "чел.", "population": "чел.",
    "population_delta": "чел.", "population_shortfall": "чел.",
    "school_demand": "мест", "school_served": "мест", "school_deficit": "мест",
    "school_coverage_percent": "%", "opened_schools": "шт.",
    "no_route_places": "мест", "no_capacity_places": "мест",
}


@dataclass(frozen=True, slots=True, eq=False)
class VerifiedReport:
    """Один согласованный результат для UI и будущей выгрузки."""
    plan: BuildingPlan
    allocation: SchoolAllocation
    classes: np.ndarray
    metrics: ResultMetrics


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportValidationError(message)


def _inside(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    return (len(cell) == 2 and all(isinstance(x, Integral) for x in cell)
            and 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1])


def _adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _road_steps(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Независимый BFS по четырём сторонам, без графа и матрицы генератора."""
    steps = np.full(mask.shape, -1, dtype=np.int32)
    steps[start] = 0
    queue = deque([start])
    height, width = mask.shape
    while queue:
        r, c = queue.popleft()
        distance = int(steps[r, c]) + 1
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and steps[nr, nc] < 0:
                steps[nr, nc] = distance
                queue.append((nr, nc))
    return steps


def validate_result(
    plan: BuildingPlan, allocation: SchoolAllocation, classes: np.ndarray | None = None,
) -> VerifiedReport:
    """Пересчитать ограничения и метрики из исходника, слоёв и назначений.

    При переданном classes проверяется также готовый растр (например, перед
    экспортом). Ни матрица Дейкстры, ни суммы генератора не служат источником
    истинных расстояний, демографии и баланса мест.
    """
    source = plan.terrain.classes
    shape = source.shape
    scenario = plan.scenario
    h = plan.terrain.pixel_size_m
    area = plan.terrain.cell_area_m2
    free = source == ClassCode.FREE
    original_roads = source == ClassCode.EXISTING_ROAD
    new_roads = plan.roads.new_roads
    ids = plan.building_ids
    _require(new_roads.shape == shape and new_roads.dtype == np.bool_, "Некорректный слой новых дорог")
    _require(ids.shape == shape and np.issubdtype(ids.dtype, np.integer), "Некорректный слой ID домов")
    if classes is not None:
        _require(classes.shape == shape and np.issubdtype(classes.dtype, np.integer), "Некорректный итоговый растр")
        _require(np.array_equal(classes[~free], source[~free]), "Изменены исходные запрещённые ячейки")
    _require(not np.any((new_roads | (ids != 0)) & ~free), "Изменены исходные запрещённые ячейки")
    _require(not np.any(new_roads & (ids != 0)), "Пересечение дорог и домов")
    road_mask = original_roads | new_roads
    _require(np.array_equal(plan.roads.road_mask, road_mask), "Несогласованный слой дорог")
    components, _ = label(road_mask)
    disconnected = int(np.count_nonzero(new_roads & ~np.isin(components, np.unique(components[original_roads]))))
    _require(disconnected == 0, "Новые дороги не связаны с исходной дорогой")

    reserve = plan.reserve.mask
    _require(reserve.shape == shape and reserve.dtype == np.bool_, "Некорректный школьный резерв")
    reserve_expected = np.zeros(shape, dtype=bool)
    for cell in plan.reserve.coordinates:
        _require(_inside(cell, shape) and free[cell] and not road_mask[cell], "Недопустимый школьный резерв")
        _require(not reserve_expected[cell], "Повтор школьной площадки")
        reserve_expected[cell] = True
    _require(np.array_equal(reserve, reserve_expected), "Несогласованный школьный резерв")
    _require(not np.any(reserve & (ids != 0)), "Дом занимает школьный резерв")

    expected_ids = np.zeros(shape, dtype=np.int32)
    homes = {}
    demands = {}
    population = 0
    side = HOUSE_SIDE_CELLS
    for building in plan.buildings:
        bid = building.id
        r, c = building.row, building.column
        _require(isinstance(bid, Integral) and 0 < bid <= np.iinfo(np.int32).max and bid not in homes,
                 "Некорректный или повторный ID дома")
        _require(_inside((r, c), shape) and r + side <= shape[0] and c + side <= shape[1],
                 "Дом выходит за границы карты")
        footprint = np.s_[r:r + side, c:c + side]
        _require(not expected_ids[footprint].any(), "Пересечение домов")
        expected_ids[footprint] = bid
        entry, road = building.house_entry, building.road_entry
        _require(_inside(entry, shape) and _inside(road, shape) and
                 r <= entry[0] < r + side and c <= entry[1] < c + side and
                 road_mask[road] and _adjacent(entry, road), "Некорректный вход дома")
        footprint_area = side**2 * area
        _require(building.floors == scenario.floors and building.footprint_area_m2 == footprint_area,
                 "Некорректная площадь или этажность дома")
        residents = floor(footprint_area * scenario.floors * USEFUL_AREA_FRACTION / AREA_PER_RESIDENT_M2 + 0.5)
        demand = floor(SCHOOLCHILD_FRACTION * residents + 0.5)
        _require(building.population == residents and building.school_demand == demand,
                 "Некорректная демография дома")
        population += residents
        homes[bid] = building
        demands[bid] = demand
    _require(np.array_equal(ids, expected_ids), "ID домов не соответствуют пятнам застройки")

    served = dict.fromkeys(homes, 0)
    reachable = dict.fromkeys(homes, False)
    school_cells = set()
    school_ids = set()
    _require(len(allocation.schools) <= scenario.school_budget, "Превышен бюджет школ")
    for school in allocation.schools:
        cell, entry = school.site.coordinate, school.site.road_entry
        _require(_inside(cell, shape) and reserve[cell] and ids[cell] == 0 and
                 cell not in school_cells, "Некорректное размещение школы")
        _require(_inside(entry, shape) and road_mask[entry] and _adjacent(cell, entry),
                 "Некорректный вход школы")
        _require(isinstance(school.id, Integral) and school.id > 0 and school.id not in school_ids,
                 "Некорректный ID школы")
        _require(school.capacity == scenario.school_capacity, "Некорректная вместимость школы")
        school_cells.add(cell)
        school_ids.add(school.id)
        steps = _road_steps(road_mask, entry)
        distances = {}
        for bid, building in homes.items():
            n = int(steps[building.road_entry])
            if n >= 0 and (n + 2) * h <= scenario.max_school_distance_m:
                distances[bid] = (n + 2) * h
                reachable[bid] = True
        used = 0
        assigned = set()
        for assignment in school.assignments:
            bid = assignment.building_id
            _require(bid in distances and bid not in assigned, "Недопустимое или повторное назначение")
            _require(assignment.distance_m == distances[bid], "Некорректная длина назначенного пути")
            places = assignment.places
            _require(isinstance(places, Integral) and not isinstance(places, (bool, np.bool_)) and places > 0,
                     "Некорректное число назначенных мест")
            assigned.add(bid)
            used += int(places)
            served[bid] += int(places)
        _require(0 < used <= school.capacity, "Превышена вместимость школы или отсутствуют назначения")
    _require(all(served[bid] <= demands[bid] for bid in homes), "Назначения превышают спрос дома")
    summaries = {b.building_id: b for b in allocation.buildings}
    _require(len(summaries) == len(allocation.buildings) and summaries.keys() == homes.keys(),
             "Несогласованный список домов в распределении")
    for bid, summary in summaries.items():
        _require(summary.demand == demands[bid] and summary.served == served[bid] and
                 summary.reachable_school == reachable[bid], "Несогласованный баланс мест дома")

    expected = source.copy()
    expected[new_roads] = ClassCode.NEW_ROAD
    expected[ids > 0] = ClassCode.NEW_BUILDING
    for cell in school_cells:
        expected[cell] = ClassCode.NEW_SCHOOL
    if classes is not None:
        _require(np.array_equal(classes, expected), "Итоговый растр не соответствует слоям проекта")
    expected.setflags(write=False)
    demand_total = sum(demands.values())
    served_total = sum(served.values())
    deficit = demand_total - served_total
    no_route = sum(demands[bid] - served[bid] for bid in homes if not reachable[bid])
    no_capacity = sum(demands[bid] - served[bid] for bid in homes if reachable[bid])
    _require(no_route + no_capacity == deficit, "Нарушен баланс дефицита")
    road_count = int(new_roads.sum())
    house_cells = int(np.count_nonzero(ids))
    developed = road_count + house_cells + len(school_cells)
    free_count = int(free.sum())
    delta = population - scenario.target_population
    metrics = ResultMetrics(
        violations_m2=0.0, disconnected_road_cells=disconnected,
        new_road_cells=road_count, new_road_area_m2=road_count * area,
        new_buildings=len(homes), building_area_m2=house_cells * area,
        developed_area_m2=developed * area,
        developed_percent=100 * developed / free_count if free_count else None,
        target_population=scenario.target_population, population=population,
        population_delta=delta, population_shortfall=max(0, -delta),
        school_demand=demand_total, school_served=served_total, school_deficit=deficit,
        school_coverage_percent=100 * served_total / demand_total if demand_total else None,
        opened_schools=len(school_cells), no_route_places=no_route, no_capacity_places=no_capacity,
    )
    return VerifiedReport(plan, allocation, expected, metrics)
