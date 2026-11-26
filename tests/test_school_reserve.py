"""Резерв школ: сектора, ранжирование, допустимость и неизменность входа."""

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest
from scipy.ndimage import binary_dilation, generate_binary_structure

from examples.create_demo import DEMO_PATH
from urban.generate import calculate_suitability, generate_roads, reserve_school_sites
from urban.io import read_terrain
from urban.model import ClassCode, CoordinateMode, RoadNetwork, Scenario, Terrain


def controlled_network(classes, scores=None) -> RoadNetwork:
    """Исходные дороги без роста: контрольные входы выбора площадок."""
    classes = np.array(classes, dtype=np.uint8)
    terrain = Terrain(
        classes=classes, valid_mask=np.ones_like(classes, dtype=bool), pixel_size_m=10,
        coordinate_mode=CoordinateMode.LOCAL, input_sha256=sha256(classes.tobytes()).hexdigest(),
    )
    suitability = calculate_suitability(terrain)
    if scores is not None:
        suitability = replace(suitability, scores=np.asarray(scores, dtype=float))
    return RoadNetwork(
        suitability=suitability, new_roads=np.zeros_like(classes, dtype=bool),
        road_mask=classes == ClassCode.EXISTING_ROAD, axis_offsets=(0, 0),
    )


def test_equal_scores_use_coordinate_order_and_empty_sectors_are_skipped() -> None:
    classes = np.zeros((6, 8), dtype=np.uint8)
    classes[2, :] = ClassCode.EXISTING_ROAD
    reserve = reserve_school_sites(controlled_network(classes, np.ones((6, 8))))
    # Нижний ряд секторов не содержит соседей дороги.
    assert reserve.coordinates == tuple((r, c) for r in (1, 3) for c in (0, 2, 4, 6))
    assert reserve.mask.sum() == 8
    assert not reserve.mask[4:, :].any()


def test_highest_score_wins_before_coordinate_order() -> None:
    classes = np.zeros((6, 8), dtype=np.uint8)
    classes[2, :] = ClassCode.EXISTING_ROAD
    scores = np.zeros((6, 8))
    scores[1, 0], scores[1, 1] = 0.5, 0.9
    reserve = reserve_school_sites(controlled_network(classes, scores))
    assert (1, 1) in reserve.coordinates
    assert (1, 0) not in reserve.coordinates
    assert reserve.mask[1, 1]


def test_all_twelve_sectors_can_have_one_site() -> None:
    classes = np.zeros((6, 8), dtype=np.uint8)
    classes[1::2, :] = ClassCode.EXISTING_ROAD
    reserve = reserve_school_sites(controlled_network(classes, np.ones((6, 8))))
    assert reserve.coordinates == tuple((r, c) for r in (0, 2, 4) for c in (0, 2, 4, 6))
    assert len(reserve.coordinates) == reserve.mask.sum() == 12


def test_uneven_sectors_put_remainder_in_first_groups() -> None:
    classes = np.zeros((7, 10), dtype=np.uint8)
    classes[[1, 4, 6], :] = ClassCode.EXISTING_ROAD
    reserve = reserve_school_sites(controlled_network(classes, np.ones((7, 10))))
    # Строки: 3+2+2, столбцы: 3+3+2+2.
    assert reserve.coordinates == tuple((r, c) for r in (0, 3, 5) for c in (0, 3, 6, 8))


@pytest.mark.parametrize("classes, expected", [
    ([[2]], ()),
    ([[2, 0]], ((0, 1),)),
    ([[2], [0]], ((1, 0),)),
    ([[2, 0, 0], [0, 0, 0]], ((0, 1), (1, 0))),
    ([[2, 1], [1, 0]], ()),  # Диагональ не является доступом к дороге.
])
def test_maps_smaller_than_sector_grid(classes, expected: tuple) -> None:
    roads = controlled_network(classes)
    reserve = reserve_school_sites(roads)
    assert reserve.coordinates == expected
    assert reserve.mask.shape == np.shape(classes)
    assert reserve.mask.sum() == len(expected)


@pytest.mark.parametrize("classes", [
    [[2, 3], [1, 255]],
    [[1, 1, 1], [1, 2, 1], [0, 1, 0]],
])
def test_no_eligible_site_returns_empty_reserve(classes) -> None:
    reserve = reserve_school_sites(controlled_network(classes))
    assert reserve.coordinates == ()
    assert not reserve.mask.any()


def test_roads_buildings_water_and_nodata_cannot_be_reserved_even_with_high_scores() -> None:
    classes = np.array([[1, 255, 3, 0], [2, 2, 2, 2]], dtype=np.uint8)
    scores = np.array([[1, 1, 1, 0.1], [1, 1, 1, 1]])
    reserve = reserve_school_sites(controlled_network(classes, scores))
    assert reserve.coordinates == ((0, 3),)


def test_new_road_cells_are_excluded_from_formerly_free_land() -> None:
    roads = controlled_network([[2, 0, 0], [0, 0, 0]], np.ones((2, 3)))
    new_roads = np.array([[False, True, True], [False, False, False]])
    roads = replace(roads, new_roads=new_roads, road_mask=roads.road_mask | new_roads)
    reserve = reserve_school_sites(roads)
    assert reserve.coordinates == ((1, 0), (1, 1), (1, 2))
    assert not np.any(reserve.mask & roads.road_mask)


@pytest.mark.parametrize("budget", [0, 1, 2, 3])
def test_school_budget_does_not_change_demo_reserve(budget: int) -> None:
    terrain = read_terrain(DEMO_PATH.read_bytes())
    baseline = reserve_school_sites(generate_roads(terrain, Scenario()))
    scenario = Scenario(school_budget=budget, school_capacity=500, max_school_distance_m=2_000)
    reserve = reserve_school_sites(generate_roads(terrain, scenario))
    assert reserve.coordinates == baseline.coordinates
    np.testing.assert_array_equal(reserve.mask, baseline.mask)


def test_demo_sites_are_free_by_reached_roads_and_leave_input_layers_unchanged() -> None:
    terrain = read_terrain(DEMO_PATH.read_bytes())
    roads = generate_roads(terrain, Scenario())
    inputs = [terrain.classes, terrain.valid_mask, roads.new_roads, roads.road_mask,
              roads.suitability.free_mask, roads.suitability.scores]
    snapshots = [array.copy() for array in inputs]
    reserve = reserve_school_sites(roads)
    again = reserve_school_sites(roads)
    assert 0 < len(reserve.coordinates) <= 12
    assert reserve.coordinates == tuple(sorted(set(reserve.coordinates)))
    assert reserve.coordinates == again.coordinates
    np.testing.assert_array_equal(reserve.mask, again.mask)
    assert reserve.mask.sum() == len(reserve.coordinates)
    assert not reserve.mask[:, 56:].any()
    assert np.all(terrain.classes[reserve.mask] == ClassCode.FREE)
    assert not np.any(reserve.mask & roads.road_mask)
    near = binary_dilation(roads.road_mask, structure=generate_binary_structure(2, 1))
    assert np.all(near[reserve.mask])
    near_old = binary_dilation(terrain.classes == ClassCode.EXISTING_ROAD,
                              structure=generate_binary_structure(2, 1))
    assert np.any(reserve.mask & ~near_old)  # Учитываются и новые улицы.
    sectors = [(r // 32, c // 24) for r, c in reserve.coordinates]
    assert len(sectors) == len(set(sectors))
    for r, c in reserve.coordinates:
        row_start, column_start = r // 32 * 32, c // 24 * 24
        eligible = (roads.suitability.free_mask & ~roads.road_mask & near)[
            row_start:row_start + 32, column_start:column_start + 24
        ]
        scores = roads.suitability.scores[row_start:row_start + 32, column_start:column_start + 24]
        assert roads.suitability.scores[r, c] == np.max(scores[eligible])
    for array, before in zip(inputs, snapshots):
        np.testing.assert_array_equal(array, before)
        assert not np.shares_memory(reserve.mask, array)
    with pytest.raises(ValueError, match="read-only"):
        reserve.mask[0, 0] = True
