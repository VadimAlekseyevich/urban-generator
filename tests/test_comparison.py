"""Приёмка сравнения двух завершённых сценариев (§5)."""

import csv
from dataclasses import replace
from io import StringIO

import pytest

from examples.create_demo import DEMO_PATH
from urban.generate import generate_buildings
from urban.io import read_terrain
from urban.model import Scenario
from urban.report import METRIC_UNITS, ReportValidationError, comparison_rows, export_comparison_csv, validate_result
from urban.services import calculate_school_services


def run(terrain, scenario):
    plan = generate_buildings(terrain, scenario)
    return validate_result(plan, calculate_school_services(plan))


def test_school_budget_comparison():
    terrain = read_terrain(DEMO_PATH.read_bytes())
    a = run(terrain, Scenario(school_budget=1))
    b = run(terrain, Scenario(school_budget=0))
    rows = {row["metric"]: row for row in comparison_rows(a, b)}
    assert set(rows) == set(METRIC_UNITS)
    assert rows["population"]["A"] == rows["population"]["B"] == 608
    assert rows["population"]["delta"] == rows["new_road_cells"]["delta"] == 0
    assert rows["school_served"]["A"] > 0 and rows["school_served"]["B"] == 0
    assert rows["school_served"]["delta"] == -a.metrics.school_served
    assert rows["school_deficit"]["delta"] == a.metrics.school_served
    reader = csv.DictReader(StringIO(export_comparison_csv(a, b).decode("utf-8")))
    assert reader.fieldnames == ["metric", "unit", "A", "B", "delta"]
    table = {row["metric"]: row for row in reader}
    assert int(table["school_served"]["delta"]) == -a.metrics.school_served
    assert table["school_coverage_percent"]["unit"] == "%"


def test_undefined_metrics_are_blank_and_delta_is_undefined():
    terrain = read_terrain(DEMO_PATH.read_bytes())
    a = run(terrain, Scenario())
    # Достаточно использовать отсутствующее значение одной стороны, чтобы
    # проверить представление None в общем формате сравнения.
    b = replace(a, metrics=replace(a.metrics, school_coverage_percent=None))
    row = next(r for r in comparison_rows(a, b) if r["metric"] == "school_coverage_percent")
    assert row["A"] is not None and row["B"] is None and row["delta"] is None
    table = {r["metric"]: r for r in csv.DictReader(StringIO(export_comparison_csv(a, b).decode()))}
    assert table["school_coverage_percent"]["B"] == table["school_coverage_percent"]["delta"] == ""


@pytest.mark.parametrize("mode", ["scale", "coordinate_mode", "hash"])
def test_incompatible_reports_rejected(mode):
    data = DEMO_PATH.read_bytes()
    terrain = read_terrain(data, local_scale=10, discard_georeferencing=True)
    a = run(terrain, Scenario())
    if mode == "scale":
        other = read_terrain(data, local_scale=5, discard_georeferencing=True)
    elif mode == "coordinate_mode":
        other = read_terrain(data)
    else:
        # Идентичная геометрия не означает идентичный входной файл.
        other = replace(terrain, input_sha256="0" * 64, transform=None)
    b = run(other, Scenario())
    with pytest.raises(ReportValidationError, match="одна исходная карта"):
        comparison_rows(a, b)
    with pytest.raises(ReportValidationError):
        export_comparison_csv(a, b)
