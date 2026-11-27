"""Основная форма PBF, стадии, кэш и сохранение проекта при неудаче #52."""

from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from urban import osm_cache


APP = Path(__file__).resolve().parents[1] / "app.py"


def demo_report():
    app = AppTest.from_file(str(APP)).run(timeout=15)
    app.checkbox(key="legend_confirmed").check().run()
    generate(app)
    app.button(key="pin_variant_a").click().run()
    assert not app.exception
    return app


def generate(app):
    next(b for b in app.button if b.label == "Сгенерировать").click().run(timeout=15)


def osm_form(app, path):
    app.radio[0].set_value("OSM PBF").run()
    assert not app.file_uploader  # регион остаётся на диске
    app.text_input(key="osm_path").set_value(str(path))
    app.number_input(key="osm_size").set_value(64)
    return app


@pytest.mark.parametrize("exception,message", [
    (ModuleNotFoundError("osmium"), "Pyosmium"),
    (ValueError("нет дороги"), "нет дороги"),
    (KeyboardInterrupt(), "Импорт прерван"),
])
def test_failed_import_preserves_previous_report_zip_comparison_and_confirmations(tmp_path, monkeypatch, exception, message):
    app = demo_report()
    previous, previous_zip = app.session_state.verified_report, app.session_state.result_zip
    previous_terrain = app.session_state.terrain
    identity = app.session_state.input_identity
    source = tmp_path / "city.osm.pbf"
    source.write_bytes(b"a source")
    def fail(*args, **kwargs):
        kwargs["progress"]("[2/3] Проверка сохранения предыдущего проекта")
        raise exception
    monkeypatch.setattr(osm_cache, "import_pbf", fail)
    monkeypatch.setattr(osm_cache, "DEFAULT_CACHE_DIR", tmp_path / "cache")
    osm_form(app, source).button(key="open_osm").click().run()
    assert not app.exception
    assert any(message in el.value for el in [*app.error, *app.warning])
    assert app.session_state.verified_report is previous
    assert app.session_state.variant_a is previous
    assert app.session_state.result_zip == previous_zip
    assert app.session_state.input_identity == identity
    assert app.session_state.terrain is previous_terrain
    assert app.session_state.terrain_ready
    assert any(b.key == "download_result_zip" for b in app.download_button)


def test_streamlit_stop_during_import_keeps_previous_state(tmp_path, monkeypatch):
    app = demo_report()
    previous, data = app.session_state.verified_report, app.session_state.result_zip
    source = tmp_path / "city.osm.pbf"
    source.write_bytes(b"a source")
    def stop(*args, **kwargs):
        st.stop()
    monkeypatch.setattr(osm_cache, "import_pbf", stop)
    monkeypatch.setattr(osm_cache, "DEFAULT_CACHE_DIR", tmp_path / "cache")
    osm_form(app, source).button(key="open_osm").click().run()
    assert not app.exception
    assert app.session_state.verified_report is previous
    assert app.session_state.result_zip == data
    app.run()  # прежний отчёт снова доступен после остановленного прогона
    assert not app.exception
    assert any(b.key == "download_result_zip" for b in app.download_button)
    assert not list((tmp_path / "cache").glob("*.zip"))


def test_success_cache_hit_and_different_window_reset_project(tmp_path, monkeypatch, fake_osm_importer):
    monkeypatch.setattr(osm_cache, "DEFAULT_CACHE_DIR", tmp_path / "cache")
    source = tmp_path / "city.osm.pbf"
    source.write_bytes(b"a source")
    app = osm_form(demo_report(), source)
    app.button(key="open_osm").click().run()
    assert not app.exception and not app.error
    assert app.session_state.verified_report is None
    assert not app.session_state.terrain_ready
    assert app.session_state.terrain.classes.shape == (64, 64)
    app.checkbox(key="legend_confirmed").check().run()
    app.checkbox(key="scale_confirmed").check().run()
    generate(app)
    assert not app.exception and not app.error
    previous, data = app.session_state.verified_report, app.session_state.result_zip
    app.button(key="open_osm").click().run()
    assert len(fake_osm_importer) == 1
    assert app.session_state.verified_report is previous and app.session_state.result_zip == data
    app.number_input(key="osm_lon").set_value(39.721)
    app.button(key="open_osm").click().run()
    assert not app.exception and not app.error
    assert len(fake_osm_importer) == 2
    assert app.session_state.verified_report is None
    assert app.session_state.terrain.transform != previous.plan.terrain.transform
    assert not app.session_state.terrain_ready
    identity = app.session_state.input_identity
    other = tmp_path / "another.osm.pbf"
    other.write_bytes(b"another source")
    app.text_input(key="osm_path").set_value(str(other))
    app.button(key="open_osm").click().run()
    assert not app.exception and not app.error
    assert len(fake_osm_importer) == 3
    assert app.session_state.input_identity != identity
    assert app.session_state.active_source_name == other.name


def test_real_binary_pbf_through_application_to_export(small_pbf, tmp_path, monkeypatch):
    monkeypatch.setattr(osm_cache, "DEFAULT_CACHE_DIR", tmp_path / "cache")
    app = osm_form(AppTest.from_file(str(APP)).run(timeout=15), small_pbf)
    app.button(key="open_osm").click().run(timeout=30)
    assert not app.exception and not app.error
    assert app.session_state.terrain.classes[10, 5] == 2
    app.checkbox(key="legend_confirmed").check().run()
    app.checkbox(key="scale_confirmed").check().run()
    generate(app)
    assert not app.exception and not app.error
    assert app.session_state.verified_report.metrics.new_buildings > 0
    with ZipFile(BytesIO(app.session_state.result_zip)) as archive:
        source = json.loads(archive.read("metrics.json"))["source"]
        assert source["source_name"] == small_pbf.name
        assert "OpenStreetMap" in source["attribution"]
    # Новый сеанс приложения переиспользует запись с диска.
    monkeypatch.setattr(osm_cache, "import_pbf", lambda *a, **kw: pytest.fail("Повторный импорт не нужен"))
    fresh = osm_form(AppTest.from_file(str(APP)).run(timeout=15), small_pbf)
    fresh.button(key="open_osm").click().run()
    assert not fresh.exception and not fresh.error
    assert fresh.session_state.terrain.input_sha256 == app.session_state.terrain.input_sha256
