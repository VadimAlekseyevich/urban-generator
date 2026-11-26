"""Дорожные пути до школьных площадок (§4, шаг 6 DESIGN.md)."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from urban.model import BuildingPlan

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
