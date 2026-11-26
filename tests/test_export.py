"""Контракт выгрузки одного проверенного запуска (§5 и §8)."""

import csv
from dataclasses import replace
from io import BytesIO, StringIO
import json
import warnings
from zipfile import ZipFile

import numpy as np
import pytest
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from examples.create_demo import DEMO_PATH
from urban.generate import generate_buildings
from urban.io import read_terrain
from urban.model import CoordinateMode, Scenario
from urban.report import BUILDINGS_CSV_FIELDS, METRIC_UNITS, ReportValidationError, export_result_zip, validate_result
from urban.services import calculate_school_services


def make_report(terrain, scenario=Scenario()):
    plan = generate_buildings(terrain, scenario)
    return validate_result(plan, calculate_school_services(plan))


@pytest.fixture
def demo():
    return make_report(read_terrain(DEMO_PATH.read_bytes()))


def unpack(report):
    with ZipFile(BytesIO(export_result_zip(report))) as archive:
        assert set(archive.namelist()) == {"result.tif", "metrics.json", "buildings.csv"}
        assert archive.testzip() is None
        return (
            archive.read("result.tif"),
            json.loads(archive.read("metrics.json")),
            archive.read("buildings.csv").decode("utf-8"),
        )


@pytest.mark.parametrize("local", [False, True])
def test_tiff_roundtrip_and_metadata(demo, local):
    terrain = demo.plan.terrain
    if local:
        terrain = read_terrain(DEMO_PATH.read_bytes(), local_scale=10, discard_georeferencing=True)
    report = make_report(terrain)
    tiff, metadata, table = unpack(report)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with MemoryFile(tiff) as memory, memory.open() as dataset:
            assert dataset.count == 1 and dataset.dtypes == ("uint8",)
            assert dataset.shape == terrain.classes.shape
            assert dataset.nodata == 255
            np.testing.assert_array_equal(dataset.read(1), report.classes)
            np.testing.assert_array_equal(dataset.read_masks(1) > 0, terrain.valid_mask)
            assert dataset.tags()["coordinate_mode"] == terrain.coordinate_mode.value
            assert json.loads(dataset.tags()["legend"]) == metadata["legend"]
            assert dataset.colormap(1)[6][:3] == (147, 51, 234)
            if local:
                assert dataset.crs is None
                assert dataset.transform == Affine.identity()
            else:
                assert dataset.crs == terrain.crs
                assert dataset.transform == terrain.transform
    assert metadata["coordinate_mode"] == terrain.coordinate_mode.value
    assert metadata["pixel_size_m"] == terrain.pixel_size_m
    assert metadata["input_sha256"] == terrain.input_sha256
    assert metadata["seed"] == metadata["parameters"]["seed"] == 42
    assert metadata["schema_version"] == metadata["algorithm_version"] == 1
    assert metadata["metric_units"] == METRIC_UNITS
    assert set(metadata["legend"]) == {"0", "1", "2", "3", "4", "5", "6", "255"}
    if local:
        assert metadata["crs"] is None and metadata["transform"] is None
        assert metadata["local_origin_m"] == [0, 0]
        assert metadata["local_origin_corner"] == "lower_left"
    else:
        assert metadata["crs"] == terrain.crs.to_string()
        assert metadata["transform"] == list(terrain.transform)[:6]
        assert metadata["local_origin_m"] is None
    reader = csv.DictReader(StringIO(table))
    assert tuple(reader.fieldnames) == BUILDINGS_CSV_FIELDS
    rows = list(reader)
    assert len(rows) == len({r["id"] for r in rows}) == metadata["metrics"]["new_buildings"] == 19
    for column, metric in (("population", "population"), ("school_demand", "school_demand"),
                           ("school_served", "school_served")):
        assert sum(int(r[column]) for r in rows) == metadata["metrics"][metric]
    for row in rows:
        bid = int(row["id"])
        r, c = int(row["row"]), int(row["column"])
        assert np.all(report.plan.building_ids[r:r+2, c:c+2] == bid)
        assert float(row["footprint_area_m2"]) == 400
    assert sum(s["served"] for s in metadata["schools"]) == metadata["metrics"]["school_served"]
    assert sum(metadata["school_deficit_reasons"].values()) == metadata["metrics"]["school_deficit"]
    np.testing.assert_array_equal(report.classes, demo.classes)
    assert metadata["metrics"]["population"] == 608


def test_no_demand_and_empty_csv():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", height=2, width=2, count=1, dtype="uint8", nodata=255) as ds:
                ds.write(np.array([[2, 3], [1, 255]], dtype=np.uint8), 1)
            terrain = read_terrain(memory.read(), local_scale=5)
    report = make_report(terrain)
    _, metadata, table = unpack(report)
    assert metadata["metrics"]["school_coverage_percent"] is None
    assert metadata["metrics"]["developed_percent"] is None
    assert metadata["schools"] == []
    reader = csv.DictReader(StringIO(table))
    assert tuple(reader.fieldnames) == BUILDINGS_CSV_FIELDS
    assert list(reader) == []


def test_corrupt_raster_cannot_export(demo):
    classes = demo.classes.copy()
    classes[tuple(np.argwhere(demo.plan.terrain.classes == 1)[0])] = 5
    with pytest.raises(ReportValidationError, match="запрещённые"):
        export_result_zip(replace(demo, classes=classes))


def test_corrupt_assignment_cannot_export(demo):
    school, = demo.allocation.schools
    bad = replace(school.assignments[0], places=school.capacity + 1)
    allocation = replace(demo.allocation, schools=(replace(school, assignments=(bad,)),))
    with pytest.raises(ReportValidationError, match="вместимость"):
        export_result_zip(replace(demo, allocation=allocation))


def test_corrupt_metrics_cannot_export(demo):
    with pytest.raises(ReportValidationError, match="Метрики"):
        export_result_zip(replace(demo, metrics=replace(demo.metrics, population=0)))


def test_repeat_export_is_identical(demo):
    assert export_result_zip(demo) == export_result_zip(demo)
