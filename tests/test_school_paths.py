"""Проверки сетевой доступности школьных площадок."""

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest
from scipy.sparse import issparse

from urban.generate import calculate_suitability, place_buildings
from urban.model import CoordinateMode, RoadNetwork, Scenario, SchoolReserve, Terrain
from urban.services import build_road_graph, calculate_school_paths, school_road_entry


def u_plan(h=10):
    # Дорога в форме U; школа (0, 1), дорожный вход дома (1, 4).
    classes = np.ones((7, 7), dtype=np.uint8)
    classes[1:6, 1] = classes[1:6, 4] = classes[5, 1:5] = 2
    classes[0, 1] = 0
    classes[0:2, 5:7] = 0
    terrain = Terrain(classes, np.ones_like(classes, dtype=bool), h,
                      CoordinateMode.LOCAL, sha256(classes.tobytes()).hexdigest())
    roads = RoadNetwork(calculate_suitability(terrain), np.zeros_like(classes, dtype=bool),
                        classes == 2, (0, 0))
    mask = np.zeros_like(classes, dtype=bool)
    mask[0, 1] = True
    return place_buildings(terrain, Scenario(target_population=1), roads, SchoolReserve(((0, 1),), mask))


@pytest.mark.parametrize("h", [5, 5.1, 7.5, 10])
def test_u_path_and_inclusive_threshold(h):
    plan = u_plan(h)
    before = plan.roads.road_mask.copy()
    paths = calculate_school_paths(plan)
    assert paths.sites[0].road_entry == (1, 1)
    assert plan.buildings[0].road_entry == (1, 4)
    assert paths.building_ids == (plan.buildings[0].id,)
    # 4 + 3 + 4 дорожных рёбер и два подхода.
    assert paths.distances_m.shape == (1, 1)
    assert paths.distances_m[0, 0] == 13 * h
    assert not paths.reachable(13 * h - 0.01)[0, 0]
    assert paths.reachable(13 * h)[0, 0]
    np.testing.assert_array_equal(plan.roads.road_mask, before)
    assert not paths.distances_m.flags.writeable


def test_diagonal_is_not_an_edge_and_isolated_nodes_are_kept():
    mask = np.array([[True, False, False], [False, True, True]])
    graph, ids = build_road_graph(mask, 10)
    assert issparse(graph)
    assert graph.shape == (3, 3)
    assert graph.nnz == 2
    assert graph[ids[0, 0], ids[1, 1]] == 0
    assert graph[ids[1, 1], ids[1, 2]] == graph[ids[1, 2], ids[1, 1]] == 10
    assert ids[0, 1] == -1


def test_disconnected_road_entries_have_infinite_distance():
    plan = u_plan()
    mask = plan.roads.road_mask.copy()
    mask[5, 2:4] = False
    plan = replace(plan, roads=replace(plan.roads, road_mask=mask))
    paths = calculate_school_paths(plan)
    assert np.isinf(paths.distances_m[0, 0])
    assert not paths.reachable(2000)[0, 0]


def test_school_entry_uses_coordinate_order():
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    assert school_road_entry((1, 1), mask) == (0, 1)
    assert school_road_entry((0, 0), mask) == (0, 1)
    with pytest.raises(ValueError, match="вход"):
        school_road_entry((1, 1), np.zeros_like(mask))


def test_empty_houses_or_sites_preserve_matrix_shape():
    plan = u_plan()
    assert calculate_school_paths(replace(plan, buildings=())).distances_m.shape == (1, 0)
    empty = SchoolReserve((), np.zeros_like(plan.reserve.mask))
    assert calculate_school_paths(replace(plan, reserve=empty)).distances_m.shape == (0, 1)
    assert calculate_school_paths(replace(plan, reserve=empty, buildings=())).distances_m.shape == (0, 0)


def test_same_road_entry_still_requires_two_approaches():
    classes = np.array([[0, 2, 0, 0], [1, 1, 0, 0]], dtype=np.uint8)
    terrain = Terrain(classes, np.ones_like(classes, dtype=bool), 10,
                      CoordinateMode.LOCAL, sha256(classes.tobytes()).hexdigest())
    roads = RoadNetwork(calculate_suitability(terrain), np.zeros_like(classes, dtype=bool),
                        classes == 2, (0, 0))
    mask = np.zeros_like(classes, dtype=bool)
    mask[0, 0] = True
    plan = place_buildings(terrain, Scenario(target_population=1), roads, SchoolReserve(((0, 0),), mask))
    paths = calculate_school_paths(plan)
    assert paths.sites[0].road_entry == plan.buildings[0].road_entry == (0, 1)
    assert paths.distances_m[0, 0] == 20


@pytest.mark.parametrize("threshold, expected_served", [(129.99, 0), (130.0, 4)])
def test_u_threshold_controls_actual_assignments(threshold, expected_served):
    from math import dist
    from urban.report import validate_result
    from urban.services import calculate_school_services

    plan = u_plan()
    plan = replace(plan, scenario=replace(plan.scenario, max_school_distance_m=threshold))
    school_cell = plan.reserve.coordinates[0]
    house_cell = plan.buildings[0].house_entry
    direct = dist(school_cell, house_cell) * plan.terrain.pixel_size_m
    assert direct < threshold
    allocation = calculate_school_services(plan)
    report = validate_result(plan, allocation)
    assert allocation.paths.distances_m[0, 0] == 130
    assert report.metrics.school_served == expected_served
    assert report.metrics.school_deficit == 4 - expected_served
    if expected_served:
        assert allocation.schools[0].assignments[0].distance_m == 130
    else:
        assert not allocation.schools
