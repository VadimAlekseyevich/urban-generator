"""Приёмка домов: вместимость, отбор, входы, остановка и воспроизводимость."""

from dataclasses import replace
from hashlib import sha256
from math import floor

import numpy as np
import pytest

from examples.create_demo import DEMO_PATH
from urban.generate import calculate_suitability, generate_buildings, place_buildings
from urban.io import read_terrain
from urban.model import CoordinateMode, RoadNetwork, Scenario, SchoolReserve, Terrain


def placement_input(classes, *, h=10, scores=None, reserved=()):
    classes = np.asarray(classes, dtype=np.uint8)
    terrain = Terrain(
        classes=classes, valid_mask=np.ones_like(classes, dtype=bool), pixel_size_m=h,
        coordinate_mode=CoordinateMode.LOCAL, input_sha256=sha256(classes.tobytes()).hexdigest(),
    )
    suitability = calculate_suitability(terrain)
    if scores is not None:
        suitability = replace(suitability, scores=np.asarray(scores, dtype=float))
    roads = RoadNetwork(suitability, np.zeros_like(classes, dtype=bool), classes == 2, (0, 0))
    mask = np.zeros_like(classes, dtype=bool)
    for coordinate in reserved:
        mask[coordinate] = True
    return terrain, roads, SchoolReserve(tuple(reserved), mask)


def place(inputs, scenario=None):
    terrain, roads, reserve = inputs
    return place_buildings(terrain, scenario or Scenario(), roads, reserve)


def test_one_house_has_specified_area_population_demand_and_entry() -> None:
    result = place(placement_input([[2, 0, 0], [2, 0, 0]]), Scenario(target_population=1))
    assert len(result.buildings) == 1
    building = result.buildings[0]
    assert (building.id, building.row, building.column, building.floors) == (1, 0, 1, 3)
    assert building.footprint_area_m2 == 400
    assert building.population == result.population == 32
    assert building.school_demand == result.school_demand == 4
    assert building.house_entry == (0, 1)
    assert building.road_entry == (0, 0)
    assert result.stop_reason == "target_reached"
    np.testing.assert_array_equal(result.building_ids, [[0, 1, 1], [0, 1, 1]])


@pytest.mark.parametrize("h, floors, population, demand", [(5, 1, 3, 0), (10, 1, 11, 1), (10, 9, 96, 12)])
def test_scale_floors_and_half_up_rounding(h, floors, population, demand) -> None:
    result = place(placement_input([[2, 0, 0], [2, 0, 0]], h=h), Scenario(target_population=1, floors=floors))
    building = result.buildings[0]
    assert building.footprint_area_m2 == 4 * h**2
    assert building.population == population
    assert building.school_demand == demand


def test_entry_is_lexicographically_first_house_road_pair() -> None:
    result = place(placement_input([[2, 2, 2], [2, 0, 0], [2, 0, 0]]))
    assert result.buildings[0].house_entry == (1, 1)
    assert result.buildings[0].road_entry == (0, 1)


@pytest.mark.parametrize("classes", [[[2]], [[2, 0]], [[2], [0]], [[2, 3], [1, 255]]])
def test_no_space_returns_zero_growth(classes) -> None:
    result = place(placement_input(classes))
    assert result.buildings == ()
    assert result.population == result.school_demand == 0
    assert not result.building_ids.any()
    assert result.stop_reason == "candidates_exhausted"


@pytest.mark.parametrize("obstacle", [1, 2, 3, 255])
def test_every_footprint_cell_must_be_free(obstacle: int) -> None:
    result = place(placement_input([[2, 0, obstacle], [2, 0, 0]]))
    assert not result.buildings


def test_reserved_cell_blocks_entire_house_window() -> None:
    result = place(placement_input([[2, 0, 0], [2, 0, 0]], reserved=((1, 2),)))
    assert result.population == 0
    assert result.reserve.mask[1, 2]


def test_new_street_is_not_available_for_a_house() -> None:
    terrain, roads, reserve = placement_input([[2, 0, 0], [2, 0, 0]])
    new_roads = np.array([[False, False, True], [False, False, False]])
    roads = replace(roads, new_roads=new_roads, road_mask=roads.road_mask | new_roads)
    assert not place_buildings(terrain, Scenario(), roads, reserve).buildings


def test_diagonal_road_contact_is_insufficient() -> None:
    result = place(placement_input([[2, 1, 1], [1, 0, 0], [1, 0, 0]]))
    assert not result.buildings


def test_highest_mean_score_wins_and_ties_follow_pcg64_seed() -> None:
    classes = np.zeros((3, 8), dtype=np.uint8)
    classes[0] = 2
    equal = place(placement_input(classes, scores=np.ones((3, 8))), Scenario(target_population=1))
    # Семь окон в построчном порядке; PCG64(42) начинает перестановку с индекса 3.
    assert (equal.buildings[0].row, equal.buildings[0].column) == (1, 3)
    scores = np.full((3, 8), 0.1)
    scores[1:, 4:6] = 0.9
    preferred = place(placement_input(classes, scores=scores), Scenario(target_population=1))
    assert (preferred.buildings[0].row, preferred.buildings[0].column) == (1, 4)


def test_overlapping_candidates_are_rechecked_and_demand_is_summed_per_house() -> None:
    classes = np.zeros((3, 16), dtype=np.uint8)
    classes[0] = 2
    result = place(placement_input(classes, scores=np.ones((3, 16))), Scenario(target_population=20_000))
    assert len(result.buildings) >= 4
    assert result.stop_reason == "candidates_exhausted"
    assert set(np.unique(result.building_ids)) == {0, *(b.id for b in result.buildings)}
    for building in result.buildings:
        assert np.count_nonzero(result.building_ids == building.id) == 4
        assert np.all(result.building_ids[building.row:building.row + 2, building.column:building.column + 2] == building.id)
    assert result.population == 32 * len(result.buildings)
    assert result.school_demand == 4 * len(result.buildings)
    assert result.school_demand != floor(0.12 * result.population + 0.5)


@pytest.mark.parametrize("target", [1, 32, 33, 600])
def test_demo_reaches_target_with_less_than_one_house_of_excess(target: int) -> None:
    terrain = read_terrain(DEMO_PATH.read_bytes())
    result = generate_buildings(terrain, Scenario(target_population=target))
    assert result.stop_reason == "target_reached"
    assert 0 <= result.population - target < 32
    assert len(result.buildings) == (target + 31) // 32


def test_demo_repeat_preserves_ids_entries_and_source_layers() -> None:
    terrain = read_terrain(DEMO_PATH.read_bytes())
    classes_before, mask_before = terrain.classes.copy(), terrain.valid_mask.copy()
    first = generate_buildings(terrain, Scenario())
    snapshots = [first.roads.road_mask.copy(), first.reserve.mask.copy(), first.roads.suitability.scores.copy()]
    second = generate_buildings(terrain, Scenario())
    assert first.buildings == second.buildings
    np.testing.assert_array_equal(first.building_ids, second.building_ids)
    assert [b.id for b in first.buildings] == list(range(1, len(first.buildings) + 1))
    assert not first.building_ids[:, 56:].any()
    occupied = first.building_ids > 0
    assert np.all(terrain.classes[occupied] == 0)
    assert not np.any(occupied & first.roads.road_mask)
    assert not np.any(occupied & first.reserve.mask)
    for building in first.buildings:
        assert first.building_ids[building.house_entry] == building.id
        assert first.roads.road_mask[building.road_entry]
        assert sum(abs(a - b) for a, b in zip(building.house_entry, building.road_entry)) == 1
    place_buildings(terrain, Scenario(), first.roads, first.reserve)
    for actual, before in zip((first.roads.road_mask, first.reserve.mask, first.roads.suitability.scores), snapshots):
        np.testing.assert_array_equal(actual, before)
    np.testing.assert_array_equal(terrain.classes, classes_before)
    np.testing.assert_array_equal(terrain.valid_mask, mask_before)
    assert not np.shares_memory(first.building_ids, terrain.classes)
    with pytest.raises(ValueError, match="read-only"):
        first.building_ids[0, 0] = 1


def test_budget_does_not_change_houses_but_floors_change_population_and_stopping() -> None:
    terrain = read_terrain(DEMO_PATH.read_bytes())
    baseline = generate_buildings(terrain, Scenario())
    without_school = generate_buildings(terrain, Scenario(school_budget=0, school_capacity=500))
    assert baseline.buildings == without_school.buildings
    np.testing.assert_array_equal(baseline.building_ids, without_school.building_ids)
    taller = generate_buildings(terrain, Scenario(floors=9))
    assert taller.population == 96 * len(taller.buildings)
    assert len(taller.buildings) < len(baseline.buildings)
    assert (taller.buildings[0].row, taller.buildings[0].column) == (baseline.buildings[0].row, baseline.buildings[0].column)
