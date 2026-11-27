"""Малый бинарный PBF и управляемый импортёр для проверок хранения и UI."""

from hashlib import sha256
import json

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.warp import transform

from urban import osm_cache
from urban.osm_import import ATTRIBUTION, make_window


@pytest.fixture
def fake_osm_importer(monkeypatch):
    calls = []
    def importer(path, window, *, progress, index, threads):
        calls.append((path, window, index, threads))
        progress("[2/3] Контрольное сканирование PBF")
        classes = np.zeros((window.size, window.size), dtype=np.uint8)
        classes[min(3, window.size - 1), :] = 2
        with path.open("rb") as source:
            digest = sha256(source.read()).hexdigest()
        metadata = {
            "import_version": osm_cache.IMPORT_VERSION, "source_name": path.name,
            "source_sha256": digest, "attribution": ATTRIBUTION,
            "bounds_wgs84": list(window.wgs84_bounds), "crs": window.crs.to_string(),
            "transform": list(window.transform)[:6], "size": window.size,
            "pixel_size_m": window.pixel_size_m,
        }
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", width=window.size, height=window.size, count=1,
                             dtype="uint8", crs=window.crs, transform=window.transform, nodata=255) as ds:
                ds.write(classes, 1)
                ds.update_tags(OSM_IMPORT=json.dumps(metadata), ATTRIBUTION=ATTRIBUTION)
            data = memory.read()
        return data, metadata
    monkeypatch.setattr(osm_cache, "import_pbf", importer)
    return calls


@pytest.fixture
def small_pbf(tmp_path):
    osmium = pytest.importorskip("osmium")
    window = make_window(39.72, 54.62, 64, 10)
    nodes = []
    for identifier, (column, row) in enumerate([(1.5, 10.5), (62.5, 10.5)], 1):
        lon, lat = transform(window.crs, "EPSG:4326",
                             [window.transform.c + column * 10], [window.transform.f - row * 10])
        nodes.append(f'<node id="{identifier}" lon="{lon[0]}" lat="{lat[0]}" version="1"/>')
    xml = tmp_path / "city.osm"
    xml.write_text('<osm version="0.6">' + ''.join(nodes) +
                   '<way id="1" version="1"><nd ref="1"/><nd ref="2"/>'
                   '<tag k="highway" v="residential"/></way></osm>', encoding="utf-8")
    pbf = tmp_path / "city.osm.pbf"
    with osmium.SimpleWriter(str(pbf)) as writer:
        for obj in osmium.FileProcessor(str(xml)):
            writer.add(obj)
    return pbf
