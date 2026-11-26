"""Пригодность: ручные расстояния, границы и сохранность исходных слоёв."""

from hashlib import sha256
from math import dist, sqrt

import numpy as np
import pytest

from examples.create_demo import DEMO_PATH
from urban.generate import calculate_suitability
from urban.io import read_terrain
from urban.model import ClassCode, CoordinateMode, Terrain


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
