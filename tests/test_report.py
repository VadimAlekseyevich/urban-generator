"""Независимая приёмка готового результата (§4, шаг 7; §5)."""

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from examples.create_demo import DEMO_PATH
from urban.generate import calculate_suitability, generate_buildings, place_buildings
from urban.io import read_terrain
from urban.model import CoordinateMode, RoadNetwork, Scenario, SchoolReserve, Terrain
from urban.report import METRIC_UNITS, ReportValidationError, school_project_classes, validate_result
from urban.services import calculate_school_services


def terrain_from(classes):
    classes = np.asarray(classes, dtype=np.uint8)
    return Terrain(classes, np.ones_like(classes, dtype=bool), 10, CoordinateMode.LOCAL,
                   sha256(classes.tobytes()).hexdigest())


@pytest.fixture
def demo():
    plan = generate_buildings(read_terrain(DEMO_PATH.read_bytes()), Scenario())
    return plan, calculate_school_services(plan)


def two_houses():
    classes = np.ones((4, 7), dtype=np.uint8)
    classes[2, :] = 2
    classes[:2, :2] = classes[:2, 3:5] = 0
    classes[3, 2] = 0
    terrain = terrain_from(classes)
    roads = RoadNetwork(calculate_suitability(terrain), np.zeros_like(classes, dtype=bool),
                        classes == 2, (0, 0))
    reserve = np.zeros_like(classes, dtype=bool)
    reserve[3, 2] = True
    plan = place_buildings(terrain, Scenario(target_population=64, school_capacity=5),
                           roads, SchoolReserve(((3, 2),), reserve))
    return plan, calculate_school_services(plan)


def test_demo_validates_all_metrics(demo):
    plan, allocation = demo
    before = plan.terrain.classes.copy()
    report = validate_result(plan, allocation, school_project_classes(plan, allocation))
    m = report.metrics
    assert m.violations_m2 == m.disconnected_road_cells == 0
    assert m.population == 608 and m.population_delta == 8
    assert m.population_shortfall == 0
    assert m.new_buildings == 19 and m.building_area_m2 == 7600
    assert m.school_demand == 76
    assert m.school_served == allocation.served
    assert m.no_route_places + m.no_capacity_places == m.school_deficit
    assert m.developed_area_m2 == np.count_nonzero(np.isin(report.classes, [4, 5, 6])) * 100
    assert m.developed_percent == 100 * np.count_nonzero(np.isin(report.classes, [4, 5, 6])) / np.count_nonzero(before == 0)
    assert not report.classes.flags.writeable
    np.testing.assert_array_equal(plan.terrain.classes, before)
    assert set(METRIC_UNITS) == set(m.__dataclass_fields__)


def test_water_corruption_rejected(demo):
    plan, allocation = demo
    classes = school_project_classes(plan, allocation).copy()
    cell = tuple(np.argwhere(plan.terrain.classes == 1)[0])
    classes[cell] = 5
    with pytest.raises(ReportValidationError, match="запрещённые"):
        validate_result(plan, allocation, classes)
    ids = plan.building_ids.copy()
    ids[cell] = 1
    with pytest.raises(ReportValidationError, match="запрещённые"):
        validate_result(replace(plan, building_ids=ids), allocation)


def test_disconnected_road_rejected(demo):
    plan, allocation = demo
    new_roads = plan.roads.new_roads.copy()
    mask = plan.roads.road_mask.copy()
    # Свободная ячейка на изолированном берегу демо.
    cell = tuple(np.argwhere((plan.terrain.classes == 0) & (np.indices(mask.shape)[1] > 65))[0])
    new_roads[cell] = mask[cell] = True
    roads = replace(plan.roads, new_roads=new_roads, road_mask=mask)
    with pytest.raises(ReportValidationError, match="не связаны"):
        validate_result(replace(plan, roads=roads), allocation)


def test_over_capacity_rejected():
    plan, allocation = two_houses()
    school, = allocation.schools
    assignment = replace(school.assignments[0], places=6)
    bad = replace(allocation, schools=(replace(school, assignments=(assignment,)),))
    with pytest.raises(ReportValidationError, match="вместимость"):
        validate_result(plan, bad)


def test_wrong_path_rejected():
    plan, allocation = two_houses()
    school, = allocation.schools
    assignment = replace(school.assignments[0], distance_m=20)
    assert school.assignments[0].distance_m != 20
    bad = replace(allocation, schools=(replace(school, assignments=(assignment,)),))
    with pytest.raises(ReportValidationError, match="длина"):
        validate_result(plan, bad)


def test_two_houses_coverage_and_deficit():
    plan, allocation = two_houses()
    report = validate_result(plan, allocation)
    m = report.metrics
    assert m.school_demand == 8 and m.school_served == 5
    assert m.school_deficit == 3 and m.school_coverage_percent == 62.5
    assert (m.no_route_places, m.no_capacity_places) == (0, 3)


def test_missing_house_cells_and_wrong_demography_rejected(demo):
    plan, allocation = demo
    ids = plan.building_ids.copy()
    ids[plan.buildings[0].house_entry] = 0
    with pytest.raises(ReportValidationError, match="ID домов"):
        validate_result(replace(plan, building_ids=ids), allocation)
    bad_building = replace(plan.buildings[0], population=999)
    with pytest.raises(ReportValidationError, match="демография"):
        validate_result(replace(plan, buildings=(bad_building, *plan.buildings[1:])), allocation)


def test_empty_denominators():
    terrain = terrain_from([[2, 3], [1, 255]])
    plan = generate_buildings(terrain, Scenario())
    m = validate_result(plan, calculate_school_services(plan)).metrics
    assert m.developed_percent is None and m.school_coverage_percent is None
    assert m.population == m.school_demand == m.school_deficit == 0
    assert m.population_shortfall == 600
