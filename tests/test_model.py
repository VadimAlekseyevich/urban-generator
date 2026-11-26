"""Контракт моделей территории и сценария (§2–4 DESIGN.md)."""

from dataclasses import FrozenInstanceError, asdict, fields
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import Affine

from urban.model import (
    AREA_PER_RESIDENT_M2,
    HOUSE_SIDE_CELLS,
    INPUT_CLASS_CODES,
    SCHOOLCHILD_FRACTION,
    USEFUL_AREA_FRACTION,
    ClassCode,
    CoordinateMode,
    Scenario,
    Terrain,
)


def test_demo_defaults_and_units() -> None:
    assert asdict(Scenario()) == {
        "target_population": 600,
        "floors": 3,
        "street_spacing_cells": 8,
        "school_budget": 1,
        "school_capacity": 80,
        "max_school_distance_m": 400,
        "seed": 42,
    }
    assert {f.name: f.metadata["unit"] for f in fields(Scenario)} == {
        "target_population": "чел.", "floors": "этажей",
        "street_spacing_cells": "ячеек", "school_budget": "шт.",
        "school_capacity": "мест", "max_school_distance_m": "м", "seed": "",
    }
    assert HOUSE_SIDE_CELLS == 2
    assert USEFUL_AREA_FRACTION == 0.8
    assert AREA_PER_RESIDENT_M2 == 30
    assert SCHOOLCHILD_FRACTION == 0.12


SCENARIO_BOUNDS = [
    ("target_population", 1, 20_000),
    ("floors", 1, 9),
    ("street_spacing_cells", 5, 20),
    ("school_budget", 0, 3),
    ("school_capacity", 1, 500),
    ("max_school_distance_m", 50, 2_000),
    ("seed", 0, 2**32 - 1),
]


@pytest.mark.parametrize("name, lower, upper", SCENARIO_BOUNDS)
def test_scenario_bounds(name: str, lower: int, upper: int) -> None:
    for boundary in (lower, upper):
        assert getattr(Scenario(**{name: boundary}), name) == boundary
    for invalid in (lower - 1, upper + 1):
        with pytest.raises(ValueError, match=name):
            Scenario(**{name: invalid})


@pytest.mark.parametrize("parameter", [row[0] for row in SCENARIO_BOUNDS])
@pytest.mark.parametrize("value", [True, False, "42", None, float("nan"), float("inf")])
def test_scenario_rejects_invalid_types_and_nonfinite_values(parameter: str, value: object) -> None:
    with pytest.raises(ValueError, match=parameter):
        Scenario(**{parameter: value})


@pytest.mark.parametrize("parameter", [row[0] for row in SCENARIO_BOUNDS if row[0] != "max_school_distance_m"])
def test_counters_and_seed_must_be_integers(parameter: str) -> None:
    with pytest.raises(ValueError, match=parameter):
        Scenario(**{parameter: 1.5})


def test_scenario_is_fixed_and_numpy_scalars_can_be_exported() -> None:
    scenario = Scenario(seed=np.int64(42), max_school_distance_m=np.float64(400.5))
    assert json.loads(json.dumps(asdict(scenario)))["seed"] == 42
    assert scenario.max_school_distance_m == 400.5
    with pytest.raises(FrozenInstanceError):
        scenario.seed = 43


def make_terrain(**changes: object) -> Terrain:
    arguments = {
        "classes": np.array([[2, 0, 1], [3, 255, 0]], dtype=np.uint8),
        "valid_mask": np.array([[True, True, True], [True, True, False]]),
        "pixel_size_m": 10,
        "coordinate_mode": CoordinateMode.LOCAL,
        "input_sha256": sha256(b"source TIFF bytes").hexdigest(),
    }
    arguments.update(changes)
    return Terrain(**arguments)


def test_class_codes_cover_input_and_output() -> None:
    assert {code.name: code.value for code in ClassCode} == {
        "FREE": 0, "BLOCKED": 1, "EXISTING_ROAD": 2, "EXISTING_BUILDING": 3,
        "NEW_ROAD": 4, "NEW_BUILDING": 5, "NEW_SCHOOL": 6, "NODATA": 255,
    }
    assert INPUT_CLASS_CODES == {0, 1, 2, 3, 255}


def test_terrain_copies_source_and_combines_masks() -> None:
    classes = np.array([[0, 1, 2], [3, 255, 9]], dtype=np.int16)
    mask = np.array([[True, True, True], [True, True, False]])
    terrain = make_terrain(classes=classes, valid_mask=mask)
    np.testing.assert_array_equal(terrain.classes, [[0, 1, 2], [3, 255, 255]])
    np.testing.assert_array_equal(terrain.valid_mask, [[True, True, True], [True, False, False]])
    assert classes[1, 2] == 9
    assert mask[1, 1]
    classes[:] = 0
    mask[:] = False
    assert terrain.classes[0, 2] == ClassCode.EXISTING_ROAD
    assert terrain.valid_mask[0, 0]
    assert terrain.classes.dtype == np.uint8
    with pytest.raises(ValueError, match="read-only"):
        terrain.classes[0, 0] = 6
    with pytest.raises(ValueError, match="read-only"):
        terrain.valid_mask[0, 0] = False


@pytest.mark.parametrize("scale", [5, 7.5, 10])
def test_local_coordinates_and_scale(scale: float) -> None:
    terrain = make_terrain(pixel_size_m=scale)
    assert terrain.coordinate_mode == CoordinateMode.LOCAL
    assert terrain.crs is None
    assert terrain.cell_area_m2 == scale**2
    assert terrain.transform @ (0, 2) == (0, 0)
    assert terrain.transform @ (3, 0) == (3 * scale, 2 * scale)
    assert terrain.input_sha256 == sha256(b"source TIFF bytes").hexdigest()


def test_georeference_is_preserved() -> None:
    crs = CRS.from_epsg(32637)
    transform = Affine(10, 0, 500_000, 0, -10, 6_000_000)
    terrain = make_terrain(coordinate_mode="georeferenced", crs=crs, transform=transform)
    assert terrain.crs == crs
    assert terrain.transform == transform
    assert terrain.coordinate_mode == CoordinateMode.GEOREFERENCED


@pytest.mark.parametrize("changes, message", [
    ({"pixel_size_m": 4.9}, "Шаг"),
    ({"pixel_size_m": 10.1}, "Шаг"),
    ({"pixel_size_m": True}, "Шаг"),
    ({"pixel_size_m": float("nan")}, "Шаг"),
    ({"input_sha256": "not-a-hash"}, "SHA-256"),
    ({"coordinate_mode": "unknown"}, "режим"),
    ({"valid_mask": np.ones((2, 3), dtype=np.uint8)}, "Маска"),
    ({"valid_mask": np.ones((3, 3), dtype=bool)}, "Маска"),
    ({"valid_mask": np.zeros((2, 3), dtype=bool)}, "пуста"),
    ({"classes": np.zeros((2, 3), dtype=float)}, "целочисленными"),
    ({"classes": np.array([[2, 9, 0], [0, 0, 0]])}, "легенды"),
    ({"classes": np.array([[2, 4, 0], [0, 0, 0]])}, "легенды"),
    ({"classes": np.zeros((2, 3), dtype=np.uint8)}, "дорожная"),
    ({"classes": np.zeros((257, 3), dtype=np.uint8)}, "256"),
    ({"classes": np.zeros((0, 3), dtype=np.uint8)}, "256"),
    ({"crs": CRS.from_epsg(32637)}, "сбросьте"),
    ({"transform": Affine.identity()}, "сбросьте"),
    ({"coordinate_mode": CoordinateMode.GEOREFERENCED}, "CRS"),
])
def test_invalid_terrain_is_rejected(changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_terrain(**changes)


@pytest.mark.parametrize("crs, transform", [
    (CRS.from_epsg(4326), Affine(10, 0, 0, 0, -10, 0)),
    (CRS.from_epsg(2263), Affine(10, 0, 0, 0, -10, 0)),
    (CRS.from_epsg(32637), None),
    (CRS.from_epsg(32637), Affine(10, 0.1, 0, 0, -10, 0)),
    (CRS.from_epsg(32637), Affine(10, 0, 0, 0, 10, 0)),
    (CRS.from_epsg(32637), Affine(5, 0, 0, 0, -10, 0)),
    (CRS.from_epsg(32637), Affine(10, 0, float("inf"), 0, -10, 0)),
])
def test_invalid_georeference_is_rejected(crs: CRS, transform: Affine | None) -> None:
    with pytest.raises(ValueError):
        make_terrain(coordinate_mode="georeferenced", crs=crs, transform=transform)


def test_occupied_terrain_with_road_is_valid() -> None:
    terrain = make_terrain(classes=np.full((2, 3), ClassCode.EXISTING_ROAD))
    assert not np.any(terrain.classes == ClassCode.FREE)


@pytest.mark.parametrize("size", [1, 256])
def test_terrain_size_boundaries(size: int) -> None:
    terrain = make_terrain(
        classes=np.full((size, size), ClassCode.EXISTING_ROAD),
        valid_mask=np.ones((size, size), dtype=bool),
    )
    assert terrain.classes.shape == (size, size)


def test_masked_road_does_not_provide_network_access() -> None:
    mask = np.ones((2, 3), dtype=bool)
    mask[0, 0] = False
    with pytest.raises(ValueError, match="дорожная"):
        make_terrain(valid_mask=mask)


def test_small_georeference_rounding_error_is_accepted() -> None:
    transform = Affine(10 + 1e-6, 1e-6, 500_000, 0, -10, 6_000_000)
    terrain = make_terrain(
        coordinate_mode="georeferenced", crs=CRS.from_epsg(32637), transform=transform,
    )
    assert terrain.transform == transform


def test_core_import_does_not_load_ui() -> None:
    subprocess.run(
        [sys.executable, "-c", "from urban.model import Scenario, Terrain; "
         "import sys; assert 'streamlit' not in sys.modules; "
         "assert 'app' not in sys.modules; assert Scenario().seed == 42"],
        cwd=Path(__file__).resolve().parents[1], check=True,
    )
