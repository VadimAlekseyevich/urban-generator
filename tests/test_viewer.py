from pathlib import Path

import pytest

from urban.generate import generate_buildings
from urban.io import read_terrain
from urban.model import Scenario
from urban.report import validate_result
from urban.services import calculate_school_services
from urban.viewer import object_at, viewer_payload


@pytest.fixture
def report():
    terrain = read_terrain((Path(__file__).parents[1] / "examples/demo.tif").read_bytes())
    plan = generate_buildings(terrain, Scenario())
    return validate_result(plan, calculate_school_services(plan))


def test_four_cells_select_one_house_and_authoritative_services(report):
    for house in report.plan.buildings:
        selected = [object_at(report, house.row+dr, house.column+dc) for dr,dc in ((0,0),(0,1),(1,0),(1,1))]
        assert all(s == selected[0] for s in selected)
        assert selected[0]["id"] == house.id
        assert f"Жителей: {house.population}" in selected[0]["details"]
        assert f"Спрос: {house.school_demand}" in selected[0]["details"]


def test_school_selection_contains_capacity_from_report(report):
    for school in report.allocation.schools:
        selection = object_at(report, *school.site.coordinate)
        assert selection["kind"] == "school" and selection["id"] == school.id
        assert f"Вместимость: {school.capacity}" in selection["details"]
        assert f"Занято: {school.served}" in selection["details"]


@pytest.mark.parametrize("point", [(-1,0),(0,96),(96,0),(True,1),(None,0),(1.5,2)])
def test_invalid_cell_never_selects_object(report, point):
    assert object_at(report, *point) is None


def test_payload_uses_actual_width_and_ids(report):
    payload = viewer_payload(report.plan.terrain, report)
    assert payload["width"] == payload["height"] == 96
    for house in report.plan.buildings:
        assert payload["ids"][house.row*96+house.column] == f"house-{house.id}"
    assert payload["project"] == report.classes.ravel().tolist()
    assert viewer_payload(report.plan.terrain)["objects"] == {}


def test_map_resources_are_utf8_even_with_cp1251_default(monkeypatch):
    """Reproduce a Windows CP1251 default independently of the host locale."""
    import runpy
    import streamlit.components.v2 as components

    root = Path(__file__).parents[1] / "urban" / "map_frontend"
    original = Path.read_text
    loaded = {}

    def read_text(path, encoding=None, *args, **kwargs):
        if path.parent == root:
            loaded[path.name] = encoding
            encoding = encoding or "cp1251"
        return original(path, encoding, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    captured = {}

    def component(name, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(components, "component", component)
    runpy.run_path(str(root.parent / "map_ui.py"))
    assert loaded == {"map.html": "utf-8", "map.css": "utf-8", "map.js": "utf-8"}
    for argument, filename in (("html", "map.html"), ("css", "map.css"), ("js", "map.js")):
        assert captured[argument] == original(root / filename, encoding="utf-8")
