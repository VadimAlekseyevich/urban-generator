"""Приёмка демофайла: воспроизводимость, семантика, топология и резерв земли."""

from hashlib import sha256
from math import ceil, floor
from pathlib import Path
import subprocess
import sys

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from scipy.ndimage import binary_dilation, generate_binary_structure, label

from examples.create_demo import DEMO_PATH
from urban.model import (
    AREA_PER_RESIDENT_M2,
    HOUSE_SIDE_CELLS,
    INPUT_CLASS_CODES,
    MAX_INPUT_BYTES,
    USEFUL_AREA_FRACTION,
    ClassCode,
    CoordinateMode,
    Scenario,
    Terrain,
)


def test_bundled_demo_matches_raster_contract() -> None:
    assert DEMO_PATH.stat().st_size <= MAX_INPUT_BYTES
    with rasterio.open(DEMO_PATH) as dataset:
        assert dataset.driver == "GTiff"
        assert dataset.count == 1
        assert dataset.shape == (96, 96)
        assert dataset.dtypes == ("uint8",)
        assert dataset.crs == CRS.from_epsg(32637)
        assert dataset.crs.is_projected
        assert dataset.crs.linear_units_factor[1] == 1
        assert dataset.res == (10, 10)
        assert dataset.transform == from_origin(500_000, 6_200_000, 10, 10)
        assert dataset.nodata == 255
        classes = dataset.read(1)
        assert set(np.unique(classes)) == INPUT_CLASS_CODES
        np.testing.assert_array_equal(dataset.read_masks(1) > 0, classes != ClassCode.NODATA)
        terrain = Terrain(
            classes=classes, valid_mask=dataset.read_masks(1) > 0,
            pixel_size_m=10, coordinate_mode=CoordinateMode.GEOREFERENCED,
            input_sha256=sha256(DEMO_PATH.read_bytes()).hexdigest(),
            crs=dataset.crs, transform=dataset.transform,
        )
        np.testing.assert_array_equal(terrain.classes, classes)
        assert terrain.cell_area_m2 == 100


def test_rebuilt_demo_matches_bundled_file(tmp_path: Path) -> None:
    output = tmp_path / "rebuilt.tif"
    subprocess.run(
        [sys.executable, "-m", "examples.create_demo", str(output)],
        cwd=DEMO_PATH.parent.parent, check=True,
    )
    with rasterio.open(DEMO_PATH) as bundled, rasterio.open(output) as rebuilt:
        np.testing.assert_array_equal(bundled.read(), rebuilt.read())
        np.testing.assert_array_equal(bundled.read_masks(), rebuilt.read_masks())
        assert bundled.profile == rebuilt.profile
        assert bundled.crs == rebuilt.crs
        assert bundled.transform == rebuilt.transform


def test_river_separates_banks_and_initial_roads_are_connected() -> None:
    with rasterio.open(DEMO_PATH) as dataset:
        classes = dataset.read(1)
    assert np.all(classes[:, 56:60] == ClassCode.BLOCKED)
    assert np.all(classes[:, 60:] == ClassCode.FREE)
    assert np.any(classes[:, :56] == ClassCode.EXISTING_BUILDING)

    structure = generate_binary_structure(2, 1)
    roads = classes == ClassCode.EXISTING_ROAD
    assert roads.any()
    assert not roads[:, 56:].any()
    _, components = label(roads, structure=structure)
    assert components == 1
    _, houses = label(classes == ClassCode.EXISTING_BUILDING, structure=structure)
    assert houses == 5

    land = np.isin(classes, [ClassCode.FREE, ClassCode.EXISTING_ROAD, ClassCode.EXISTING_BUILDING])
    regions, _ = label(land, structure=structure)
    assert regions[48, 20] != 0
    assert regions[48, 95] != 0
    assert regions[48, 20] != regions[48, 95]


def test_seed_42_leaves_house_plots_and_school_site_by_initial_roads() -> None:
    """Консервативная проверка места без реализации будущего генератора.

    Исключаем все оси seed 42, даже недостигнутые, и используем только
    исходную улицу. Проверяем непересекающиеся пятна 2×2 и одну площадку.
    Выбор резерва по S и реальная генерация будут проверяться в следующих issue.
    """
    with rasterio.open(DEMO_PATH) as dataset:
        classes = dataset.read(1)
        h = dataset.res[0]
    scenario = Scenario()
    rng = np.random.Generator(np.random.PCG64(scenario.seed))
    offset_row, offset_column = rng.integers(0, scenario.street_spacing_cells, size=2)
    rows, columns = np.indices(classes.shape)
    axes = ((rows % scenario.street_spacing_cells) == offset_row) | (
        (columns % scenario.street_spacing_cells) == offset_column
    )
    free = (classes == ClassCode.FREE) & ~axes
    near_road = binary_dilation(
        classes == ClassCode.EXISTING_ROAD, structure=generate_binary_structure(2, 1),
    )
    school_sites = np.argwhere(free & near_road)
    assert len(school_sites) > 0
    school_row, school_column = school_sites[0]
    free[school_row, school_column] = False

    side = HOUSE_SIDE_CELLS
    house_count = 0
    for row in range(classes.shape[0] - side + 1):
        for column in range(classes.shape[1] - side + 1):
            plot = np.s_[row:row + side, column:column + side]
            if free[plot].all() and near_road[plot].any():
                free[plot] = False
                house_count += 1
    residents_per_house = floor(side**2 * h**2 * scenario.floors * USEFUL_AREA_FRACTION / AREA_PER_RESIDENT_M2 + 0.5)
    assert house_count >= ceil(scenario.target_population / residents_per_house)
