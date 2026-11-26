"""Приёмка страницы: демо, скачивание, загрузка, ошибки и подтверждения."""

from io import BytesIO
from pathlib import Path
import warnings

import numpy as np
from PIL import Image
import pytest
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
import streamlit as st
from streamlit.testing.v1 import AppTest

from examples.create_demo import DEMO_PATH
from urban.report import terrain_areas, terrain_image


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def upload_bytes(classes=None, *, crs=None, transform=None) -> bytes:
    classes = np.array([[2, 0, 1], [3, 255, 0]], dtype=np.uint8) if classes is None else classes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with MemoryFile() as memory:
            with memory.open(
                driver="GTiff", width=classes.shape[1], height=classes.shape[0],
                count=1, dtype=classes.dtype, nodata=255, crs=crs, transform=transform,
            ) as dataset:
                dataset.write(classes, 1)
            return memory.read()


def own_file_app(data: bytes) -> AppTest:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
    app.radio[0].set_value("Свой TIFF").run()
    app.file_uploader[0].set_value(("uploaded.tif", data, "image/tiff")).run()
    assert not app.exception
    return app


def confirm_local(app: AppTest, scale: float = 10) -> AppTest:
    app.radio(key="coordinate_selection").set_value("Локальные координаты").run()
    app.number_input(key="local_scale").set_value(scale).run()
    app.checkbox(key="legend_confirmed").check().run()
    app.checkbox(key="scale_confirmed").check().run()
    assert app.session_state.terrain_ready
    return app


def test_demo_page_has_map_legend_units_and_confirmation() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
    assert not app.exception
    assert app.title[0].value == "Urban Generator"
    assert len(app.get("image")) == 1
    assert any("CRS: EPSG:32637" in item.value for item in app.markdown)
    assert any("96×96 ячеек" in item.value and "960×960 м" in item.value for item in app.markdown)
    assert app.metric[0].value == "918400"
    rows = app.dataframe[0].value.set_index("Код")
    assert rows.loc[0, "Площадь, м²"] == 866800
    assert rows.loc[1, "Площадь, м²"] == 38400
    assert rows.loc[2, "Ячеек"] == 112
    assert rows.loc[3, "Ячеек"] == 20
    assert rows.loc[255, "Ячеек"] == 32
    assert np.isnan(rows.loc[255, "Площадь, м²"])
    assert not app.session_state.terrain_ready
    app.checkbox(key="legend_confirmed").check().run()
    assert not app.exception
    assert app.session_state.terrain_ready
    assert app.success[0].value == "Легенда и масштаб подтверждены."


def test_downloaded_demo_uploaded_through_form_matches_classes_and_areas(monkeypatch) -> None:
    downloaded = []
    original = st.download_button

    def capture_download(label, data, **kwargs):
        downloaded.append(data)
        return original(label, data, **kwargs)

    monkeypatch.setattr(st, "download_button", capture_download)
    app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
    demo = app.session_state.terrain
    areas = app.dataframe[0].value.copy()
    app.checkbox(key="legend_confirmed").check().run()
    app.download_button[0].click().run()
    source_bytes = downloaded[-1]
    assert source_bytes == DEMO_PATH.read_bytes()
    app.radio[0].set_value("Свой TIFF").run()
    assert app.session_state.terrain is None
    assert not app.session_state.terrain_ready
    app.file_uploader[0].set_value(("demo.tif", source_bytes, "image/tiff")).run()
    assert not app.exception
    np.testing.assert_array_equal(app.session_state.terrain.classes, demo.classes)
    np.testing.assert_array_equal(app.session_state.terrain.valid_mask, demo.valid_mask)
    assert app.session_state.terrain.input_sha256 == demo.input_sha256
    assert app.dataframe[0].value.equals(areas)
    assert not app.checkbox(key="legend_confirmed").value
    assert not app.session_state.terrain_ready


def test_local_tiff_requires_scale_and_both_confirmations() -> None:
    app = own_file_app(upload_bytes())
    assert "масштаб" in app.error[0].value
    assert app.session_state.terrain is None
    app.radio(key="coordinate_selection").set_value("Локальные координаты").run()
    assert app.number_input(key="local_scale").value is None
    assert not app.dataframe
    app.number_input(key="local_scale").set_value(7.5).run()
    assert not app.exception
    assert not app.error
    assert any("Локальные координаты; начало" in item.value for item in app.markdown)
    assert app.metric[0].value == "281.25"
    app.checkbox(key="legend_confirmed").check().run()
    assert not app.session_state.terrain_ready
    app.checkbox(key="scale_confirmed").check().run()
    assert app.session_state.terrain_ready


def test_changing_scale_resets_confirmations_and_recalculates_areas() -> None:
    app = confirm_local(own_file_app(upload_bytes()))
    assert app.metric[0].value == "500"
    app.number_input(key="local_scale").set_value(5).run()
    assert not app.exception
    assert app.metric[0].value == "125"
    assert not app.checkbox(key="legend_confirmed").value
    assert not app.checkbox(key="scale_confirmed").value
    assert not app.session_state.terrain_ready


def test_new_file_does_not_inherit_local_mode_or_confirmations() -> None:
    app = confirm_local(own_file_app(upload_bytes()))
    app.file_uploader[0].set_value(("demo.tif", DEMO_PATH.read_bytes(), "image/tiff")).run()
    assert not app.exception
    assert app.radio(key="coordinate_selection").value == "Геопривязка GeoTIFF"
    assert app.session_state.terrain.crs.to_epsg() == 32637
    assert not app.checkbox(key="legend_confirmed").value
    assert not app.session_state.terrain_ready


def test_partial_georeference_requires_reset_checkbox() -> None:
    data = upload_bytes(transform=from_origin(500_000, 6_200_000, 10, 10))
    app = own_file_app(data)
    assert "неполная" in app.error[0].value
    app.radio(key="coordinate_selection").set_value("Локальные координаты").run()
    app.number_input(key="local_scale").set_value(10).run()
    assert "неполная" in app.error[0].value
    app.checkbox(key="discard_georeferencing").check().run()
    assert not app.exception
    assert not app.error
    assert app.session_state.terrain.crs is None
    assert not app.session_state.terrain_ready


@pytest.mark.parametrize("data, message", [
    (b"not a TIFF", "Неизвестный формат"),
    (upload_bytes(np.array([[2, 9]], dtype=np.uint8), crs="EPSG:32637", transform=from_origin(0, 100, 10, 10)), "Коды вне легенды"),
])
def test_invalid_upload_shows_russian_error_without_traceback(data: bytes, message: str) -> None:
    app = own_file_app(data)
    assert not app.exception
    assert message in app.error[0].value
    assert not app.get("image")
    assert not app.dataframe
    assert app.session_state.terrain is None
    assert not app.session_state.terrain_ready


def test_invalid_replacement_and_removed_upload_cannot_leave_terrain_ready() -> None:
    app = own_file_app(DEMO_PATH.read_bytes())
    app.checkbox(key="legend_confirmed").check().run()
    assert app.session_state.terrain_ready
    app.file_uploader[0].set_value(("bad.tif", b"bad file", "image/tiff")).run()
    assert not app.exception
    assert app.error
    assert app.session_state.terrain is None
    assert not app.session_state.terrain_ready
    app.file_uploader[0].clear().run()
    assert not app.exception
    assert not app.error
    assert app.session_state.terrain is None
    assert not app.session_state.terrain_ready


def test_preview_has_exact_class_colors_and_does_not_modify_source() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
    terrain = app.session_state.terrain
    before = terrain.classes.copy()
    image = terrain_image(terrain)
    assert image.size == (768, 768)
    assert image.getpixel((4, 4)) == (17, 24, 39)
    assert image.getpixel((56 * 8 + 4, 4)) == (37, 99, 235)
    assert len(image.getcolors()) == 5
    output = BytesIO()
    image.save(output, format="PNG")
    with Image.open(BytesIO(output.getvalue())) as reopened:
        assert reopened.size == image.size
    np.testing.assert_array_equal(terrain.classes, before)
    assert terrain_areas(terrain)[-1]["Площадь, м²"] is None
