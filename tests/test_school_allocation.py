"""Приёмка жадного распределения школьных мест (§4, шаг 6)."""

from dataclasses import replace

import numpy as np
import pytest

from examples.create_demo import DEMO_PATH
from urban.generate import generate_buildings
from urban.io import read_terrain
from urban.model import Scenario
from urban.services import SchoolPaths, SchoolSite, allocate_school_places, calculate_school_services


def allocate(distances, demands=(4, 4), *, ids=(1, 2), coordinates=None, **params):
    if coordinates is None:
        coordinates = tuple((0, i) for i in range(len(distances)))
    sites = tuple(SchoolSite(c, (1, c[1])) for c in coordinates)
    matrix = np.asarray(distances, dtype=float).reshape(len(sites), len(ids))
    paths = SchoolPaths(sites, ids, matrix)
    scenario = Scenario(school_capacity=5, **params)
    return allocate_school_places(paths, demands, scenario)


def test_capacity_partial_assignment_and_inclusive_threshold():
    result = allocate([[50, 100]], max_school_distance_m=100)
    assert (result.demand, result.served, result.deficit) == (8, 5, 3)
    school, = result.schools
    assert school.served == school.capacity == 5
    assert school.max_assigned_distance_m == 100
    assert [(a.building_id, a.places) for a in school.assignments] == [(1, 4), (2, 1)]
    assert [b.remaining for b in result.buildings] == [0, 3]
    assert all(b.reachable_school for b in result.buildings)


def test_zero_budget_keeps_all_demand_unserved():
    result = allocate([[50, 100]], school_budget=0)
    assert not result.schools
    assert (result.demand, result.served, result.deficit) == (8, 0, 8)
    assert all(not b.reachable_school for b in result.buildings)


def test_greedy_gain_uses_residual_demand_and_stops_without_gain():
    # Первая площадка имеет выигрыш 4, вторая и третья 5;
    # после открытия второй третья теряет выигрыш, первая сохраняет 3.
    result = allocate([[50, np.inf], [100, 50], [100, 50]], school_budget=3)
    assert [s.site.coordinate for s in result.schools] == [(0, 1), (0, 0)]
    assert [s.served for s in result.schools] == [5, 3]
    assert result.served == 8
    assert result.deficit == 0
    assert [b.remaining for b in result.buildings] == [0, 0]
    for b in result.buildings:
        assert sum(a.places for s in result.schools for a in s.assignments
                   if a.building_id == b.building_id) == b.demand


def test_ties_by_site_coordinate_and_house_id_not_input_order():
    result = allocate([[50, 50], [50, 50]], ids=(9, 2),
                      coordinates=((3, 4), (1, 4)))
    school, = result.schools
    assert school.site.coordinate == (1, 4)
    assert [(a.building_id, a.places) for a in school.assignments] == [(2, 4), (9, 1)]


@pytest.mark.parametrize("distances, demands", [
    ([[401, np.inf]], (4, 4)),
    ([[50, 100]], (0, 0)),
    ([], (4, 4)),
])
def test_zero_gain_does_not_open_school(distances, demands):
    result = allocate(distances, demands, school_budget=3)
    assert result.schools == ()
    assert result.served == 0
    assert result.deficit == sum(demands)


def test_open_school_reachability_distinguishes_capacity_from_distance():
    result = allocate([[50, 100, np.inf]], demands=(4, 4, 4), ids=(1, 2, 3))
    assert [(b.remaining, b.reachable_school) for b in result.buildings] == [
        (0, True), (3, True), (4, False),
    ]


def test_empty_buildings():
    result = allocate([[]], (), ids=(), school_budget=3)
    assert result.schools == result.buildings == ()
    assert result.demand == result.served == result.deficit == 0


def test_demo_budget_and_reproducibility():
    terrain = read_terrain(DEMO_PATH.read_bytes())
    plan = generate_buildings(terrain, Scenario())
    before_ids = plan.building_ids.copy()
    before_roads = plan.roads.road_mask.copy()
    before_source = terrain.classes.copy()
    one = calculate_school_services(plan)
    repeat = calculate_school_services(plan)
    zero_plan = generate_buildings(terrain, replace(plan.scenario, school_budget=0))
    zero = calculate_school_services(zero_plan)
    assert one.schools == repeat.schools
    assert one.buildings == repeat.buildings
    assert one.demand == zero.demand == plan.school_demand
    assert one.served > 0 == zero.served
    assert one.deficit == one.demand - one.served
    assert len(one.schools) == 1
    assert one.schools[0].site.coordinate in plan.reserve.coordinates
    assert all(a.distance_m <= plan.scenario.max_school_distance_m
               for s in one.schools for a in s.assignments)
    assert all(s.served <= s.capacity for s in one.schools)
    assert plan.buildings == zero_plan.buildings
    np.testing.assert_array_equal(plan.building_ids, zero_plan.building_ids)
    np.testing.assert_array_equal(plan.building_ids, before_ids)
    np.testing.assert_array_equal(plan.roads.road_mask, before_roads)
    np.testing.assert_array_equal(plan.roads.road_mask, zero_plan.roads.road_mask)
    np.testing.assert_array_equal(terrain.classes, before_source)
