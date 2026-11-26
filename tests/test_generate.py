"""Пригодность и улицы: расстояния, связность и сохранность исходных слоёв."""

from collections import deque
from dataclasses import replace
from hashlib import sha256
from math import dist, sqrt

import numpy as np
import pytest
from scipy.ndimage import generate_binary_structure, label

from examples.create_demo import DEMO_PATH
from urban.generate import calculate_suitability, generate_roads
from urban.io import read_terrain
from urban.model import ClassCode, CoordinateMode, Scenario, Terrain


def terrain_from(classes, *, scale=10, mask=None) -> Terrain:
    classes = np.array(classes, dtype=np.uint8)
    return Terrain(
        classes=classes,
        valid_mask=np.ones_like(classes, dtype=bool) if mask is None else mask,
        pixel_size_m=scale, coordinate_mode=CoordinateMode.LOCAL,
        input_sha256=sha256(classes.tobytes()).hexdigest(),
    )


@pytest.mark.parametrize("h", [5, 7.5, 10])
def test_hand_calculated_score_uses_metres_and_ignores_existing_house_as_barrier(h: float) -> None:
    classes = np.zeros((7, 7), dtype=np.uint8)
    classes[3, 0] = ClassCode.EXISTING_ROAD
    classes[1, 3] = ClassCode.BLOCKED
    classes[3, 2] = ClassCode.EXISTING_BUILDING
    result = calculate_suitability(terrain_from(classes, scale=h))
    # Ячейка (3, 3): дорога в трёх шагах, вода в двух, рамка в четырёх.
    assert result.road_distance_m[3, 3] == pytest.approx(3 * h)
    assert result.barrier_distance_m[3, 3] == pytest.approx(2 * h)
    expected = 0.7 * (1 - 3 * h / 300) + 0.3 * (2 * h / 50)
    assert result.scores[3, 3] == pytest.approx(expected)
    if h == 10:
        assert result.scores[3, 3] == pytest.approx(0.75)


def test_distance_is_euclidean_including_diagonals() -> None:
    classes = np.zeros((7, 7), dtype=np.uint8)
    classes[2, 2] = ClassCode.EXISTING_ROAD
    result = calculate_suitability(terrain_from(classes))
    assert result.road_distance_m[3, 3] == pytest.approx(10 * sqrt(2))
    assert result.barrier_distance_m[3, 3] == 40
    assert result.scores[3, 3] == pytest.approx(0.7 * (1 - 10 * sqrt(2) / 300) + 0.3 * 0.8)


@pytest.mark.parametrize("barrier", [ClassCode.BLOCKED, ClassCode.NODATA])
def test_water_and_nodata_limit_clearance(barrier: ClassCode) -> None:
    classes = np.zeros((7, 7), dtype=np.uint8)
    classes[0, 0] = ClassCode.EXISTING_ROAD
    classes[3, 4] = barrier
    result = calculate_suitability(terrain_from(classes))
    assert result.barrier_distance_m[3, 3] == 10
    assert not result.free_mask[3, 4]
    assert np.isneginf(result.scores[3, 4])


def test_mask_hole_is_excluded_and_limits_clearance() -> None:
    classes = np.zeros((7, 7), dtype=np.uint8)
    classes[0, 0] = ClassCode.EXISTING_ROAD
    mask = np.ones_like(classes, dtype=bool)
    mask[3, 4] = False
    result = calculate_suitability(terrain_from(classes, mask=mask))
    assert not result.free_mask[3, 4]
    assert np.isneginf(result.scores[3, 4])
    assert result.barrier_distance_m[3, 3] == 10


def test_all_four_sides_have_an_excluded_outer_frame() -> None:
    classes = np.zeros((7, 7), dtype=np.uint8)
    classes[3, 3] = ClassCode.EXISTING_ROAD
    result = calculate_suitability(terrain_from(classes))
    for position in ((0, 3), (6, 3), (3, 0), (3, 6)):
        assert result.barrier_distance_m[position] == 10
    assert result.barrier_distance_m[3, 3] == 40


def test_occupied_and_forbidden_cells_are_not_candidates() -> None:
    terrain = terrain_from([[0, 1, 2], [3, 255, 0]])
    result = calculate_suitability(terrain)
    np.testing.assert_array_equal(result.free_mask, [[True, False, False], [False, False, True]])
    assert np.isneginf(result.scores[~result.free_mask]).all()
    assert np.isfinite(result.scores[result.free_mask]).all()


def test_distance_factors_are_clipped_at_their_normalization_scales() -> None:
    classes = np.zeros((128, 128), dtype=np.uint8)
    classes[64, 0] = ClassCode.EXISTING_ROAD
    result = calculate_suitability(terrain_from(classes))
    assert result.road_distance_m[64, 64] == 640
    assert result.barrier_distance_m[64, 64] == 640
    assert result.scores[64, 64] == pytest.approx(0.3)
    assert np.all(result.scores[result.free_mask] >= 0)
    assert np.all(result.scores[result.free_mask] <= 1)


@pytest.mark.parametrize("classes", [[[2]], [[2, 3], [1, 255]]])
def test_terrain_without_free_land_returns_empty_candidates(classes) -> None:
    result = calculate_suitability(terrain_from(classes))
    assert not result.free_mask.any()
    assert np.isneginf(result.scores).all()
    assert np.isfinite(result.road_distance_m).all()
    assert np.isfinite(result.barrier_distance_m).all()


@pytest.mark.parametrize("shape", [(1, 7), (7, 1)])
def test_single_row_or_column_has_frame_at_one_pixel(shape: tuple[int, int]) -> None:
    classes = np.zeros(shape, dtype=np.uint8)
    classes[0, 0] = ClassCode.EXISTING_ROAD
    result = calculate_suitability(terrain_from(classes))
    np.testing.assert_array_equal(result.barrier_distance_m, np.full(shape, 10.0))


def test_all_distances_match_independent_small_grid_calculation() -> None:
    terrain = terrain_from([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 1, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    result = calculate_suitability(terrain)
    roads = [(1, 1), (3, 6)]
    barriers = [(1, 4), (3, 3)]
    # Независимый перебор расстояний до двух объектов и центров рамки.
    barriers += [(row, column) for row in (-1, 6) for column in range(-1, 9)]
    barriers += [(row, column) for column in (-1, 8) for row in range(6)]
    for row, column in np.ndindex(terrain.classes.shape):
        d = min(dist((row, column), road) for road in roads) * 10
        b = min(dist((row, column), barrier) for barrier in barriers) * 10
        assert result.road_distance_m[row, column] == pytest.approx(d)
        assert result.barrier_distance_m[row, column] == pytest.approx(b)
        if terrain.classes[row, column] == ClassCode.FREE:
            expected = 0.7 * max(0, 1 - d / 300) + 0.3 * min(1, b / 50)
            assert result.scores[row, column] == pytest.approx(expected)


def test_demo_source_is_unchanged_and_stage_outputs_are_repeatable() -> None:
    terrain = read_terrain(DEMO_PATH.read_bytes())
    before_classes = terrain.classes.copy()
    before_mask = terrain.valid_mask.copy()
    first = calculate_suitability(terrain)
    second = calculate_suitability(terrain)
    np.testing.assert_array_equal(terrain.classes, before_classes)
    np.testing.assert_array_equal(terrain.valid_mask, before_mask)
    for name in ("free_mask", "road_distance_m", "barrier_distance_m", "scores"):
        array = getattr(first, name)
        np.testing.assert_array_equal(array, getattr(second, name))
        assert array.shape == terrain.classes.shape
        assert not np.shares_memory(array, terrain.classes)
        assert not np.shares_memory(array, terrain.valid_mask)
        with pytest.raises(ValueError, match="read-only"):
            array[0, 0] = 0
    assert first.free_mask.sum() == 8668


def test_seed_42_roads_match_hand_drawn_grid() -> None:
    classes = np.zeros((5, 7), dtype=np.uint8)
    classes[3, 2] = ClassCode.EXISTING_ROAD
    result = generate_roads(terrain_from(classes), Scenario(street_spacing_cells=5))
    assert result.axis_offsets == (0, 3)
    expected = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=bool)
    np.testing.assert_array_equal(result.new_roads, expected)
    expected[3, 2] = True
    np.testing.assert_array_equal(result.road_mask, expected)


@pytest.mark.parametrize("seed", [0, 42, 43, 2**32 - 1])
@pytest.mark.parametrize("spacing", [5, 8, 20])
def test_river_keeps_new_roads_on_the_original_bank(seed: int, spacing: int) -> None:
    classes = np.zeros((20, 24), dtype=np.uint8)
    classes[:, 12:14] = ClassCode.BLOCKED
    classes[:, 4] = ClassCode.EXISTING_ROAD
    terrain = terrain_from(classes)
    scenario = Scenario(seed=seed, street_spacing_cells=spacing)
    first = generate_roads(terrain, scenario)
    second = generate_roads(terrain, scenario)
    assert first.new_roads[:, :12].any()
    assert not first.new_roads[:, 12:14].any()
    assert not first.new_roads[:, 14:].any()
    assert not first.new_roads[:, 4].any()
    np.testing.assert_array_equal(first.new_roads, second.new_roads)
    np.testing.assert_array_equal(first.road_mask, second.road_mask)


def test_each_original_component_seeds_its_own_bank() -> None:
    classes = np.zeros((20, 24), dtype=np.uint8)
    classes[:, 12:14] = ClassCode.BLOCKED
    classes[:, 4] = ClassCode.EXISTING_ROAD
    classes[:, 18] = ClassCode.EXISTING_ROAD
    result = generate_roads(terrain_from(classes), Scenario())
    assert result.new_roads[:, :12].any()
    assert result.new_roads[:, 14:].any()
    assert not result.road_mask[:, 12:14].any()
    assert result.road_mask[:, 4].all()
    assert result.road_mask[:, 18].all()


@pytest.mark.parametrize("barrier", [ClassCode.BLOCKED, ClassCode.NODATA, ClassCode.EXISTING_BUILDING])
def test_barriers_stop_growth_along_a_single_axis(barrier: ClassCode) -> None:
    result = generate_roads(terrain_from([[2, 0, barrier, 0, 0, 0, 0]]), Scenario(street_spacing_cells=5))
    np.testing.assert_array_equal(result.new_roads, [[False, True, False, False, False, False, False]])


def test_mask_hole_stops_road_growth() -> None:
    terrain = terrain_from([[2, 0, 0, 0, 0, 0, 0]], mask=np.array([[True, True, False, True, True, True, True]]))
    result = generate_roads(terrain, Scenario(street_spacing_cells=5))
    np.testing.assert_array_equal(result.new_roads, [[False, True, False, False, False, False, False]])


def test_diagonal_contact_does_not_connect_a_road() -> None:
    scenario = Scenario(street_spacing_cells=5)
    isolated = generate_roads(terrain_from([[0, 1], [1, 2]]), scenario)
    assert not isolated.new_roads.any()
    connected = generate_roads(terrain_from([[0, 0], [1, 2]]), scenario)
    np.testing.assert_array_equal(connected.new_roads, [[True, True], [False, False]])


def test_free_land_outside_axes_cannot_bridge_to_an_isolated_grid() -> None:
    classes = np.zeros((5, 7), dtype=np.uint8)
    classes[3, 1] = ClassCode.EXISTING_ROAD
    result = generate_roads(terrain_from(classes), Scenario(street_spacing_cells=5))
    assert not result.new_roads.any()
    np.testing.assert_array_equal(result.road_mask, classes == ClassCode.EXISTING_ROAD)


@pytest.mark.parametrize("classes", [[[2]], [[2, 3], [1, 255]]])
def test_occupied_terrain_keeps_only_its_existing_roads(classes) -> None:
    terrain = terrain_from(classes)
    result = generate_roads(terrain, Scenario())
    assert not result.new_roads.any()
    np.testing.assert_array_equal(result.road_mask, terrain.classes == ClassCode.EXISTING_ROAD)


def test_roads_match_independent_breadth_first_search() -> None:
    fixture_rng = np.random.Generator(np.random.PCG64(2026))
    classes = fixture_rng.choice([0, 1, 3, 255], size=(11, 13), p=[0.75, 0.1, 0.1, 0.05]).astype(np.uint8)
    classes[2, 2] = classes[8, 10] = ClassCode.EXISTING_ROAD
    terrain = terrain_from(classes)
    scenario = Scenario(street_spacing_cells=5, seed=43)
    result = generate_roads(terrain, scenario)
    offsets = np.random.Generator(np.random.PCG64(scenario.seed)).integers(0, 5, size=2)
    reached = classes == ClassCode.EXISTING_ROAD
    queue = deque(map(tuple, np.argwhere(reached)))
    while queue:
        row, column = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = row + dr, column + dc
            if not (0 <= r < 11 and 0 <= c < 13) or reached[r, c]:
                continue
            on_axis = r % 5 == offsets[0] or c % 5 == offsets[1]
            if classes[r, c] == ClassCode.FREE and on_axis:
                reached[r, c] = True
                queue.append((r, c))
    np.testing.assert_array_equal(result.road_mask, reached)
    np.testing.assert_array_equal(result.new_roads, reached & (classes == ClassCode.FREE))


def test_demo_road_components_have_an_original_road_and_source_is_unchanged() -> None:
    terrain = read_terrain(DEMO_PATH.read_bytes())
    before_classes, before_mask = terrain.classes.copy(), terrain.valid_mask.copy()
    result = generate_roads(terrain, Scenario())
    assert result.new_roads[:, :56].any()
    assert not result.new_roads[:, 56:].any()
    assert not result.new_roads[~result.suitability.free_mask].any()
    components, count = label(result.road_mask, structure=generate_binary_structure(2, 1))
    old_roads = terrain.classes == ClassCode.EXISTING_ROAD
    for component in range(1, count + 1):
        assert np.any((components == component) & old_roads)
    np.testing.assert_array_equal(terrain.classes, before_classes)
    np.testing.assert_array_equal(terrain.valid_mask, before_mask)
    for array in (result.new_roads, result.road_mask):
        assert not np.shares_memory(array, terrain.classes)
        assert not np.shares_memory(array, terrain.valid_mask)
        with pytest.raises(ValueError, match="read-only"):
            array[0, 0] = False


def test_seed_and_spacing_change_the_demo_grid_but_other_scenario_fields_do_not() -> None:
    terrain = read_terrain(DEMO_PATH.read_bytes())
    scenario = Scenario()
    baseline = generate_roads(terrain, scenario)
    other_seed = generate_roads(terrain, replace(scenario, seed=43))
    other_spacing = generate_roads(terrain, replace(scenario, street_spacing_cells=5))
    assert not np.array_equal(baseline.new_roads, other_seed.new_roads)
    assert not np.array_equal(baseline.new_roads, other_spacing.new_roads)
    other_parameters = replace(
        scenario, target_population=20_000, floors=9,
        school_budget=0, school_capacity=500, max_school_distance_m=2_000,
    )
    np.testing.assert_array_equal(baseline.new_roads, generate_roads(terrain, other_parameters).new_roads)


def test_road_generation_does_not_touch_global_numpy_random_state() -> None:
    before = np.random.get_state()
    generate_roads(terrain_from([[2, 0, 0], [0, 0, 0]]), Scenario())
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
