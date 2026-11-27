"""Идентичность источника, версии/окна, целостность и атомарность кэша #52."""

from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from rasterio.io import MemoryFile

from urban import osm_cache
from urban.generate import generate_buildings
from urban.model import Scenario
from urban.osm_cache import prepare_pbf
from urban.osm_import import ATTRIBUTION, make_window
from urban.report import export_result_zip, validate_result
from urban.services import calculate_school_services


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "city.osm.pbf"
    path.write_bytes(b"source-1")
    return path


def test_cache_hit_keeps_classes_metadata_and_needs_no_optional_dependency(source, tmp_path, fake_osm_importer, monkeypatch):
    cache = tmp_path / "cache"
    window = make_window(39.72, 54.62, 64, 10)
    first = prepare_pbf(source, window, cache_dir=cache)
    assert not first.reused and len(fake_osm_importer) == 1
    assert len(list(cache.glob("*.zip"))) == 1
    monkeypatch.setattr(osm_cache, "import_pbf", lambda *a, **kw: pytest.fail("Повторного сканирования быть не должно"))
    progress = []
    # Нельзя случайно загрузить большой PBF целиком через Path.read_bytes.
    original = Path.read_bytes
    def read_bytes(path):
        assert path != source
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    second = prepare_pbf(source, window, cache_dir=cache, progress=progress.append, threads=4)
    assert second.reused and first.data == second.data
    assert first.metadata_json == second.metadata_json
    np.testing.assert_array_equal(first.terrain.classes, second.terrain.classes)
    assert any("повторное сканирование" in p for p in progress)
    assert json.loads(second.metadata_json)["source_sha256"] == sha256(b"source-1").hexdigest()


def test_different_window_scale_size_file_path_content_and_version_do_not_reuse(source, tmp_path, fake_osm_importer, monkeypatch):
    cache = tmp_path / "cache"
    windows = [make_window(39.72, 54.62, 64, 10), make_window(39.721, 54.62, 64, 10),
               make_window(39.72, 54.62, 64, 5), make_window(39.72, 54.62, 32, 10)]
    results = [prepare_pbf(source, w, cache_dir=cache) for w in windows]
    assert len({r.cache_key for r in results}) == 4 and len(fake_osm_importer) == 4
    assert all(not r.reused for r in results)
    other = tmp_path / "other" / source.name
    other.parent.mkdir()
    other.write_bytes(source.read_bytes())
    different_file = prepare_pbf(other, windows[0], cache_dir=cache)
    assert different_file.cache_key != results[0].cache_key
    # Даже одинаковые размер/mtime/имя не скрывают изменённые байты.
    before = source.stat()
    source.write_bytes(b"source-2")
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    changed = prepare_pbf(source, windows[0], cache_dir=cache)
    assert changed.cache_key != results[0].cache_key
    monkeypatch.setattr(osm_cache, "IMPORT_VERSION", osm_cache.IMPORT_VERSION + 1)
    upgraded = prepare_pbf(source, windows[0], cache_dir=cache)
    assert upgraded.cache_key != changed.cache_key and not upgraded.reused


@pytest.mark.parametrize("damage", ["truncated", "tiff_hash", "metadata", "manifest_type"])
def test_corrupt_cache_is_rebuilt(source, tmp_path, fake_osm_importer, damage):
    cache = tmp_path / "cache"
    window = make_window(39.72, 54.62, 64, 10)
    first = prepare_pbf(source, window, cache_dir=cache)
    target = cache / f"{first.cache_key}.zip"
    with ZipFile(target) as archive:
        data, manifest = archive.read("terrain.tif"), json.loads(archive.read("manifest.json"))
    if damage == "truncated":
        target.write_bytes(b"not a complete ZIP")
    else:
        if damage == "tiff_hash":
            data = b"corrupt TIFF"
        elif damage == "metadata":
            manifest["osm_import"] = []
        else:
            manifest = []
        with ZipFile(target, "w") as archive:
            archive.writestr("terrain.tif", data)
            archive.writestr("manifest.json", json.dumps(manifest))
    rebuilt = prepare_pbf(source, window, cache_dir=cache)
    assert not rebuilt.reused and len(fake_osm_importer) == 2
    assert rebuilt.data == first.data


def test_partial_record_and_interruption_never_publish_or_destroy_valid_window(source, tmp_path, fake_osm_importer, monkeypatch):
    cache = tmp_path / "cache"
    window = make_window(39.72, 54.62, 64, 10)
    first = prepare_pbf(source, window, cache_dir=cache)
    (cache / "pending-crash.tmp").write_bytes(b"interrupted process")
    def interrupted_replace(*args):
        raise KeyboardInterrupt()
    with monkeypatch.context() as patch:
        patch.setattr(osm_cache.os, "replace", interrupted_replace)
        with pytest.raises(KeyboardInterrupt):
            prepare_pbf(source, make_window(39.721, 54.62, 64, 10), cache_dir=cache)
    assert [p.name for p in cache.glob("*.tmp")] == ["pending-crash.tmp"]
    assert len(list(cache.glob("*.zip"))) == 1
    restored = prepare_pbf(source, window, cache_dir=cache)
    assert restored.reused and restored.data == first.data


def test_file_changed_during_scan_is_not_cached(source, tmp_path, fake_osm_importer, monkeypatch):
    importer = osm_cache.import_pbf
    def change(path, *args, **kwargs):
        path.write_bytes(b"source-2")
        return importer(path, *args, **kwargs)
    monkeypatch.setattr(osm_cache, "import_pbf", change)
    with pytest.raises(ValueError, match="изменился"):
        prepare_pbf(source, make_window(39.72, 54.62, 64, 10), cache_dir=tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("*.zip"))


def test_osm_attribution_and_source_hash_survive_report_and_result_tiff(source, tmp_path, fake_osm_importer):
    prepared = prepare_pbf(source, make_window(39.72, 54.62, 64, 10), cache_dir=tmp_path / "cache")
    plan = generate_buildings(prepared.terrain, Scenario())
    report = validate_result(plan, calculate_school_services(plan))
    with ZipFile(BytesIO(export_result_zip(report))) as archive:
        metadata = json.loads(archive.read("metrics.json"))
        assert metadata["source"] == json.loads(prepared.metadata_json)
        assert metadata["source"]["attribution"] == ATTRIBUTION
        assert metadata["source"]["source_sha256"] == sha256(source.read_bytes()).hexdigest()
        assert metadata["input_sha256"] == sha256(prepared.data).hexdigest()
        with MemoryFile(archive.read("result.tif")) as mem, mem.open() as ds:
            assert ds.tags()["ATTRIBUTION"] == ATTRIBUTION
            assert json.loads(ds.tags()["OSM_IMPORT"]) == metadata["source"]


def test_real_binary_pbf_import_and_disk_reuse(small_pbf, tmp_path, monkeypatch):
    window = make_window(39.72, 54.62, 64, 10)
    first = prepare_pbf(small_pbf, window, cache_dir=tmp_path / "cache")
    monkeypatch.setattr(osm_cache, "import_pbf", lambda *a, **kw: pytest.fail("Кэш должен работать без импортёра"))
    second = prepare_pbf(small_pbf, window, cache_dir=tmp_path / "cache")
    assert first.data == second.data and second.reused
    assert first.terrain.classes[10, 5] == 2


def test_real_scan_interruption_can_be_followed_by_successful_retry(small_pbf, tmp_path):
    cache = tmp_path / "cache"
    window = make_window(39.72, 54.62, 64, 10)
    def interrupt(message):
        if "Кандидатов проверено" in message:
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        prepare_pbf(small_pbf, window, cache_dir=cache, progress=interrupt)
    assert not list(cache.glob("*.zip"))
    completed = prepare_pbf(small_pbf, window, cache_dir=cache)
    assert completed.terrain.classes[10, 5] == 2
