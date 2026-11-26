"""Smoke-проверка стартовой страницы без отдельного веб-сервера."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_start_page_renders() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=15)

    assert not app.exception
    assert app.title[0].value == "Urban Generator"
    assert "Приложение запущено" in app.info[0].value
