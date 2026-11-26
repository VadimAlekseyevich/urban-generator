"""Дорожная доступность и распределение школьных мест (§4, шаг 6 DESIGN.md)."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from urban.model import BuildingPlan, Scenario

Coordinate = tuple[int, int]


@dataclass(frozen=True, slots=True)
class SchoolSite:
    coordinate: Coordinate
    road_entry: Coordinate


@dataclass(frozen=True, slots=True, eq=False)
class SchoolPaths:
    """Строки матрицы — площадки, столбцы — дома в порядке BuildingPlan."""

    sites: tuple[SchoolSite, ...]
    building_ids: tuple[int, ...]
    distances_m: NDArray[np.float64]

    def reachable(self, max_distance_m: float) -> NDArray[np.bool_]:
        """Порог включён; бесконечный путь никогда не допустим."""
        return np.isfinite(self.distances_m) & (self.distances_m <= max_distance_m)


def build_road_graph(road_mask: NDArray[np.bool_], h: float) -> tuple[csr_matrix, NDArray[np.int32]]:
    """Компактный граф всех дорожных ячеек; -1 вне дороги.

    Каждое ребро по общей стороне добавляется в обе стороны с весом h.
    Изолированные дорожные ячейки остаются вершинами.
    """
    node_ids = np.full(road_mask.shape, -1, dtype=np.int32)
    count = int(np.count_nonzero(road_mask))
    node_ids[road_mask] = np.arange(count, dtype=np.int32)
    horizontal = road_mask[:, :-1] & road_mask[:, 1:]
    vertical = road_mask[:-1, :] & road_mask[1:, :]
    start = np.concatenate((node_ids[:, :-1][horizontal], node_ids[:-1, :][vertical]))
    end = np.concatenate((node_ids[:, 1:][horizontal], node_ids[1:, :][vertical]))
    graph = csr_matrix(
        (np.full(2 * len(start), h, dtype=float),
         (np.concatenate((start, end)), np.concatenate((end, start)))),
        shape=(count, count),
    )
    node_ids.setflags(write=False)
    return graph, node_ids


def school_road_entry(coordinate: Coordinate, road_mask: NDArray[np.bool_]) -> Coordinate:
    """Первая по строке и столбцу соседняя дорожная ячейка."""
    row, column = coordinate
    height, width = road_mask.shape
    for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
        r, c = row + dr, column + dc
        if 0 <= r < height and 0 <= c < width and road_mask[r, c]:
            return r, c
    raise ValueError("У школьной площадки отсутствует вход к дорожной сети")


def calculate_school_paths(plan: BuildingPlan) -> SchoolPaths:
    """Дейкстра от входов площадок, плюс два подхода по h метров.

    Расчёт не зависит от бюджета, вместимости и порога сценария.
    Входы домов используются без повторного выбора; исходные слои неизменны.
    """
    road_mask = plan.roads.road_mask
    h = plan.terrain.pixel_size_m
    sites = tuple(SchoolSite(c, school_road_entry(c, road_mask)) for c in plan.reserve.coordinates)
    building_ids = tuple(b.id for b in plan.buildings)
    distances = np.empty((len(sites), len(building_ids)), dtype=float)
    if sites and building_ids:
        graph, node_ids = build_road_graph(road_mask, h)
        sources = np.array([node_ids[s.road_entry] for s in sites], dtype=np.int32)
        targets = np.array([node_ids[b.road_entry] for b in plan.buildings], dtype=np.int32)
        if np.any(targets < 0):
            raise ValueError("Вход дома отсутствует в дорожной сети")
        network_distances = dijkstra(graph, directed=False, indices=sources)
        # Восстанавливаем целое число рёбер, чтобы не накопить погрешность
        # сложения дробного h на пути, равном порогу.
        distances = (np.rint(network_distances[:, targets] / h) + 2) * h
    distances.setflags(write=False)
    return SchoolPaths(sites, building_ids, distances)


@dataclass(frozen=True, slots=True)
class SchoolAssignment:
    """Назначенные ребёнку/детям дома места в одной школе."""

    building_id: int
    places: int
    distance_m: float


@dataclass(frozen=True, slots=True)
class OpenSchool:
    id: int
    site: SchoolSite
    capacity: int
    assignments: tuple[SchoolAssignment, ...]

    @property
    def served(self) -> int:
        return sum(a.places for a in self.assignments)

    @property
    def max_assigned_distance_m(self) -> float:
        return max(a.distance_m for a in self.assignments)


@dataclass(frozen=True, slots=True)
class BuildingService:
    building_id: int
    demand: int
    served: int
    reachable_school: bool

    @property
    def remaining(self) -> int:
        return self.demand - self.served


@dataclass(frozen=True, slots=True, eq=False)
class SchoolAllocation:
    paths: SchoolPaths
    schools: tuple[OpenSchool, ...]
    buildings: tuple[BuildingService, ...]

    @property
    def demand(self) -> int:
        return sum(b.demand for b in self.buildings)

    @property
    def served(self) -> int:
        return sum(b.served for b in self.buildings)

    @property
    def deficit(self) -> int:
        return self.demand - self.served


def allocate_school_places(
    paths: SchoolPaths, demands: tuple[int, ...], scenario: Scenario,
) -> SchoolAllocation:
    """Жадный выбор по остаточному спросу, затем раздача мест по пути и ID.

    demands соответствует столбцам paths.distances_m. Ничьи между площадками
    разрешаются координатами, между домами — ID. Школы нумеруются в порядке
    открытия. Дома могут получать места в нескольких школах, но не сверх
    своего спроса. Площадки с нулевым выигрышем не открываются.
    """
    if len(demands) != len(paths.building_ids) or any(d < 0 for d in demands):
        raise ValueError("Спрос должен соответствовать домам и быть неотрицательным")
    if paths.distances_m.shape != (len(paths.sites), len(demands)):
        raise ValueError("Размер матрицы путей не соответствует площадкам и домам")

    remaining = np.array(demands, dtype=np.int64)
    reachable = paths.reachable(scenario.max_school_distance_m)
    schools = []
    opened = set()
    reachable_opened = np.zeros(len(demands), dtype=bool)
    for _ in range(scenario.school_budget):
        gains = np.minimum(scenario.school_capacity, reachable @ remaining)
        candidates = [i for i, gain in enumerate(gains) if i not in opened and gain > 0]
        if not candidates:
            break
        i = min(candidates, key=lambda j: (-gains[j], paths.sites[j].coordinate))
        opened.add(i)
        reachable_opened |= reachable[i]
        available = scenario.school_capacity
        assignments = []
        order = sorted(
            np.flatnonzero(reachable[i] & (remaining > 0)),
            key=lambda j: (paths.distances_m[i, j], paths.building_ids[j]),
        )
        for j in order:
            places = min(available, int(remaining[j]))
            assignments.append(SchoolAssignment(
                paths.building_ids[j], places, float(paths.distances_m[i, j]),
            ))
            remaining[j] -= places
            available -= places
            if available == 0:
                break
        schools.append(OpenSchool(
            len(schools) + 1, paths.sites[i], scenario.school_capacity, tuple(assignments),
        ))

    buildings = tuple(
        BuildingService(bid, int(demand), int(demand - left), bool(can_reach))
        for bid, demand, left, can_reach in zip(
            paths.building_ids, demands, remaining, reachable_opened, strict=True,
        )
    )
    return SchoolAllocation(paths, tuple(schools), buildings)


def calculate_school_services(plan: BuildingPlan) -> SchoolAllocation:
    """Рассчитать школы для готовой застройки, не меняя улицы и дома."""
    paths = calculate_school_paths(plan)
    demands = tuple(b.school_demand for b in plan.buildings)
    return allocate_school_places(paths, demands, plan.scenario)
