"""Приёмка #49: фиксированный GeoJSON и независимо подготовленный GeoTIFF."""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from rasterio.io import MemoryFile
from streamlit.testing.v1 import AppTest

from urban.generate import generate_buildings
from urban.io import TerrainReadError, read_terrain
from urban.model import MAX_INPUT_BYTES, Scenario
from urban.report import export_result_zip, validate_result
from urban.services import calculate_school_services
from urban.vector_import import read_geojson


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "examples/site.geojson").read_bytes()
EQUIVALENT = (ROOT / "tests/data/site-equivalent.tif").read_bytes()


def document():
    return json.loads(SOURCE)


def encode(value):
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def report(terrain, seed=42):
    plan = generate_buildings(terrain, Scenario(seed=seed))
    return validate_result(plan, calculate_school_services(plan))


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_fixed_vector_and_prepared_geotiff_give_identical_results(seed):
    vector = read_geojson(SOURCE, 10, source_name="site.geojson")
    raster = read_terrain(EQUIVALENT)
    assert vector.crs == raster.crs
    assert vector.transform == raster.transform
    assert vector.classes.shape == (64, 80)
    np.testing.assert_array_equal(vector.classes, raster.classes)
    np.testing.assert_array_equal(vector.valid_mask, raster.valid_mask)
    # Независимые ожидаемые ячейки, включая отверстие и приоритет запрета над дорогой.
    assert np.all(vector.classes[28:38, 32:44] == 255)
    assert np.count_nonzero(~vector.valid_mask) == 120
    assert np.all(vector.classes[:, 6] == 2)
    assert np.all(vector.classes[:, 60:63] == 1)
    assert np.all(vector.classes[10:13, 10:13] == 3)
    a, b = report(vector, seed), report(raster, seed)
    assert a.plan.buildings == b.plan.buildings
    assert a.plan.buildings
    np.testing.assert_array_equal(a.plan.building_ids, b.plan.building_ids)
    np.testing.assert_array_equal(a.classes, b.classes)
    assert asdict(a.metrics) == asdict(b.metrics)
    assert np.all(a.classes[~vector.valid_mask] == 255)
    np.testing.assert_array_equal(vector.classes, raster.classes)  # исходник сохранён


def test_original_geometries_ids_sources_and_hash_survive_deterministic_export():
    terrain = read_geojson(SOURCE, 10, source_name="my-site.geojson")
    assert terrain.input_sha256 == sha256(SOURCE).hexdigest()
    assert terrain.vector_source.data == SOURCE
    completed = report(terrain)
    output = export_result_zip(completed)
    assert output == export_result_zip(report(read_geojson(SOURCE, 10, source_name="my-site.geojson")))
    with ZipFile(BytesIO(output)) as archive:
        assert set(archive.namelist()) == {"result.tif", "metrics.json", "buildings.csv", "source.geojson"}
        assert archive.read("source.geojson") == SOURCE
        metadata = json.loads(archive.read("metrics.json"))
        source = metadata["source"]
        assert source["source_name"] == "my-site.geojson"
        assert source["source_sha256"] == metadata["input_sha256"] == sha256(SOURCE).hexdigest()
        assert source["source_crs"] == "EPSG:4326"
        assert [(f["id"], f["role"], f["source"]) for f in source["features"]] == [
            (f["id"], f["properties"]["role"], f["properties"]["source"]) for f in document()["features"]
        ]
        with MemoryFile(archive.read("result.tif")) as memory, memory.open() as ds:
            assert ds.crs == terrain.crs and ds.transform == terrain.transform
            np.testing.assert_array_equal(ds.read(1), completed.classes)


def test_missing_ids_are_stable_and_original_is_unchanged():
    doc = document()
    del doc["features"][1]["id"]
    del doc["features"][1]["properties"]["source"]
    original = deepcopy(doc)
    terrain = read_geojson(encode(doc), 10, source_name="custom.geojson")
    record = json.loads(terrain.vector_source.manifest_json)["features"][1]
    assert record == {"id": "feature-2", "role": "road", "source": "custom.geojson"}
    assert json.loads(terrain.vector_source.data) == original == doc


def test_grid_step_changes_grid_but_keeps_original_source_hash():
    fine, coarse = read_geojson(SOURCE, 5), read_geojson(SOURCE, 10)
    assert fine.classes.shape == (128, 160)
    assert coarse.classes.shape == (64, 80)
    assert fine.input_sha256 == coarse.input_sha256
    assert fine.pixel_size_m == 5 and coarse.pixel_size_m == 10


def test_all_class_priorities_are_shared_with_osm():
    doc = document()
    water = doc["features"][-1]
    house = deepcopy(water)
    house["id"] = "priority-house"
    house["properties"]["role"] = "building"
    doc["features"].append(house)
    terrain = read_geojson(encode(doc), 10)
    assert np.all(terrain.classes[:, 60:63] == 3)


@pytest.mark.parametrize("change, message", [
    (lambda d: d.update(type="Polygon"), "FeatureCollection"),
    (lambda d: d.update(crs={"name": "EPSG:3857"}), "crs"),
    (lambda d: d["features"][1]["properties"].update(role="roads"), "неизвестная роль"),
    (lambda d: d["features"][1]["properties"].update(role="building"), "Роль building"),
    (lambda d: d["features"][0]["geometry"]["coordinates"][0].pop(), "замкнуты"),
    (lambda d: d["features"][0].update(geometry=None), "Роль boundary"),
    (lambda d: d["features"][1].update(id="site"), "Повторный id"),
    (lambda d: d["features"][1].update(id=True), "id должен"),
    (lambda d: d["features"].pop(0), "Нет границы"),
    (lambda d: d["features"].append(deepcopy(d["features"][0]) | {"id": "boundary2"}), "ровно одна"),
    (lambda d: d.update(features=[f for f in d["features"] if f["properties"]["role"] != "road"]), "Нет исходной дороги"),
    (lambda d: d["features"][1]["geometry"]["coordinates"][0].__setitem__(0, 500000), "WGS84"),
    (lambda d: d["features"][1]["geometry"]["coordinates"][0].append(100), "двумерные"),
])
def test_invalid_inputs_have_actionable_errors(change, message):
    doc = document()
    change(doc)
    with pytest.raises(TerrainReadError, match=message):
        read_geojson(encode(doc), 10)


@pytest.mark.parametrize("data", [b"not JSON", b"[]", b"", b"\xff", b'{"type":"a","type":"b"}',
                                    b'{"type":NaN}'])
def test_bad_json(data):
    with pytest.raises(TerrainReadError):
        read_geojson(data, 10)


def test_file_and_spatial_limits_are_checked_before_rasterization(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Растеризация не должна начинаться для слишком большого входа")
    monkeypatch.setattr("urban.vector_import.rasterize", unexpected)
    with pytest.raises(TerrainReadError, match="8 МиБ"):
        read_geojson(b" " * (MAX_INPUT_BYTES + 1), 10)
    doc = document()
    doc["features"][0]["geometry"]["coordinates"] = [
        [[39, 51], [39.1, 51], [39.1, 51.1], [39, 51.1], [39, 51]]]
    with pytest.raises(TerrainReadError, match="предел 256×256"):
        read_geojson(encode(doc), 10)


def test_roads_outside_boundary_or_in_hole_do_not_satisfy_contract():
    doc = document()
    roads = [f for f in doc["features"] if f["properties"]["role"] == "road"]
    for road in roads:
        road["geometry"]["coordinates"] = [[39.1, 51], [39.1, 51.0001]]
    with pytest.raises(TerrainReadError, match="Нет исходной дороги внутри boundary"):
        read_geojson(encode(doc), 10)
    hole = doc["features"][0]["geometry"]["coordinates"][1]
    mid = np.mean(np.asarray(hole[:-1]), axis=0)
    for road in roads:
        road["geometry"]["coordinates"] = [mid.tolist(), (mid + [0.00001, 0]).tolist()]
    with pytest.raises(TerrainReadError, match="Нет исходной дороги внутри boundary"):
        read_geojson(encode(doc), 10)


def test_invalid_polygon_topology_is_rejected():
    for ring in (
        [[39, 51], [39.01, 51.01], [39.01, 51], [39, 51.01], [39, 51]],
        [[39, 51], [39.01, 51], [39.02, 51], [39, 51]],
    ):
        doc = document()
        doc["features"][0]["geometry"]["coordinates"] = [ring]
        with pytest.raises(TerrainReadError, match="площадь|[Сс]амопересечение"):
            read_geojson(encode(doc), 10)
    doc = document()
    doc["features"][0]["geometry"]["coordinates"][1] = [
        [39.1, 51], [39.11, 51], [39.11, 51.01], [39.1, 51.01], [39.1, 51]]
    with pytest.raises(TerrainReadError, match="Отверстие"):
        read_geojson(encode(doc), 10)


def test_multipart_roles():
    doc = document()
    for f in doc["features"]:
        geometry = f["geometry"]
        geometry["type"] = "Multi" + geometry["type"]
        geometry["coordinates"] = [geometry["coordinates"]]
    np.testing.assert_array_equal(read_geojson(encode(doc), 10).classes, read_terrain(EQUIVALENT).classes)


def test_geojson_through_main_form_to_result_and_download():
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=15)
    app.radio[0].set_value("GeoJSON").run()
    assert not app.exception
    assert not app.session_state.terrain_ready
    app.file_uploader[0].set_value(("site.geojson", SOURCE, "application/geo+json")).run()
    assert not app.exception and not app.error
    assert not app.session_state.terrain_ready
    app.checkbox(key="legend_confirmed").check().run()
    assert not app.session_state.terrain_ready
    app.checkbox(key="scale_confirmed").check().run()
    assert app.session_state.terrain_ready
    app.button[0].click().run(timeout=15)
    assert not app.exception and not app.error
    completed = app.session_state.verified_report
    assert completed.metrics == report(read_terrain(EQUIVALENT)).metrics
    assert completed.plan.buildings
    with ZipFile(BytesIO(app.session_state.result_zip)) as archive:
        assert archive.read("source.geojson") == SOURCE
    app.number_input(key="vector_scale").set_value(5).run()
    assert not app.exception
    assert app.session_state.terrain.classes.shape == (128, 160)
    assert app.session_state.verified_report is None
    assert app.session_state.result_zip is None
    assert not app.checkbox(key="legend_confirmed").value
    assert not app.checkbox(key="scale_confirmed").value
    bad = document()
    bad["features"][1]["properties"]["role"] = "wrong"
    app.file_uploader[0].set_value(("bad.geojson", encode(bad), "application/geo+json")).run()
    assert not app.exception
    assert "неизвестная роль" in app.error[0].value
    assert app.session_state.terrain is None and not app.session_state.terrain_ready


def test_plain_tiff_still_uses_same_pipeline():
    local = read_terrain(EQUIVALENT, local_scale=10, discard_georeferencing=True)
    vector = read_geojson(SOURCE, 10)
    assert local.vector_source is None and local.crs is None
    assert report(local).metrics == report(vector).metrics
