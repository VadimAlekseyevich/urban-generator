"""Тесты импорта, включая небольшой настоящий бинарный PBF."""

import json

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.warp import transform

from urban.generate import generate_buildings
from urban.io import read_terrain
from urban.model import ClassCode, Scenario
from urban.osm_import import ATTRIBUTION, classify, import_pbf, make_window
from urban.report import validate_result
from urban.services import calculate_school_services


def test_window_utm_and_size():
    window = make_window(39.72, 54.62, 256, 10)
    assert window.crs.to_epsg() == 32637
    assert window.transform.a == 10 and window.transform.e == -10
    west, south, east, north = window.wgs84_bounds
    assert west < 39.72 < east and south < 54.62 < north
    assert make_window(18, -30, 100, 5).crs.to_epsg() == 32734


@pytest.mark.parametrize("args", [(39, 90, 64, 10), (39, 54, 257, 10), (39, 54, 64, 2)])
def test_invalid_window(args):
    with pytest.raises(ValueError):
        make_window(*args)


@pytest.mark.parametrize("tags,kind,expected", [
    ({"highway":"residential"}, "way", 2),
    ({"highway":"primary", "bridge":"yes"}, "way", 1),
    ({"highway":"service", "access":"private"}, "way", 1),
    ({"highway":"footway"}, "way", 1),
    ({"building":"yes"}, "area", 3),
    ({"natural":"water"}, "area", 1),
    ({"leisure":"park"}, "area", 1),
    ({"amenity":"school"}, "node", 1),
    ({"landuse":"residential"}, "area", None),
])
def test_classification(tags, kind, expected):
    assert classify(tags, kind) == expected


def test_real_pbf_import(tmp_path):
    osmium = pytest.importorskip("osmium")
    window = make_window(39.72, 54.62, 64, 10)
    nodes, ways, relations = [], [], []
    def node(x, y, tags=""):
        lon, lat = transform(window.crs, "EPSG:4326",
                             [window.transform.c + x * 10],
                             [window.transform.f - y * 10])
        n = len(nodes) + 1
        nodes.append(f'<node id="{n}" lon="{lon[0]}" lat="{lat[0]}" version="1">{tags}</node>')
        return n
    def way(points, tags="", closed=False):
        refs = [node(x,y) for x,y in points]
        if closed:
            refs.append(refs[0])
        n = len(ways) + 1
        ways.append(f'<way id="{n}" version="1">' + "".join(f'<nd ref="{ref}"/>' for ref in refs)
                    + tags + '</way>')
        return n
    way([(2.5, 10.5), (60.5, 10.5)], '<tag k="highway" v="residential"/>')
    way([(8.2, 8.2),(12.8,8.2),(12.8,13.8),(8.2,13.8)], '<tag k="building" v="yes"/>', True)
    way([(20.2,20.2),(28.8,20.2),(28.8,28.8),(20.2,28.8)], '<tag k="leisure" v="park"/>', True)
    outer = way([(40.2,4.2),(52.8,4.2),(52.8,24.8),(40.2,24.8)], closed=True)
    inner = way([(44.2,14.2),(48.8,14.2),(48.8,19.8),(44.2,19.8)], closed=True)
    relations.append(f'<relation id="1" version="1"><member type="way" ref="{outer}" role="outer"/>'
                     f'<member type="way" ref="{inner}" role="inner"/>'
                     '<tag k="type" v="multipolygon"/><tag k="natural" v="water"/></relation>')
    node(33.5, 33.5, '<tag k="amenity" v="school"/>')
    way([(100.2,100.2),(104.8,100.2),(104.8,104.8),(100.2,104.8)], '<tag k="building" v="yes"/>', True)
    xml = tmp_path / "fixture.osm"
    xml.write_text('<osm version="0.6" generator="pytest">' + "".join(nodes + ways + relations) + '</osm>', encoding="utf-8")
    pbf = tmp_path / "fixture.osm.pbf"
    with osmium.SimpleWriter(str(pbf)) as writer:
        for obj in osmium.FileProcessor(str(xml)):
            writer.add(obj)
    data, meta = import_pbf(pbf, window)
    terrain = read_terrain(data)
    assert terrain.crs.to_epsg() == 32637 and terrain.transform == window.transform
    assert terrain.classes.shape == (64,64)
    assert terrain.classes[10, 5] == ClassCode.EXISTING_ROAD
    assert terrain.classes[10, 10] == ClassCode.EXISTING_BUILDING
    assert terrain.classes[24, 24] == ClassCode.BLOCKED
    assert terrain.classes[10, 43] == ClassCode.BLOCKED  # вода важнее дороги
    assert terrain.classes[16, 46] == ClassCode.FREE  # отверстие multipolygon
    assert terrain.classes[33, 33] == ClassCode.BLOCKED
    assert terrain.classes[40, 10] == ClassCode.FREE
    assert meta["features_in_window"]["3"] == 1
    assert meta["invalid_geometries_in_scan"] == 0
    assert meta["attribution"] == ATTRIBUTION
    with MemoryFile(data) as mem, mem.open() as ds:
        assert ds.nodata == 255
        assert json.loads(ds.tags()["OSM_IMPORT"]) == meta
    plan = generate_buildings(terrain, Scenario())
    report = validate_result(plan, calculate_school_services(plan))
    assert report.metrics.new_buildings > 0
    np.testing.assert_array_equal(report.classes[terrain.classes != 0], terrain.classes[terrain.classes != 0])
