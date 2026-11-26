"""Проверки методики замера, без хрупких требований к скорости CI."""

import numpy as np
import pytest

from examples.create_demo import DEMO_PATH
from examples.run_benchmark import STAGES, TARGETS, benchmark, large_demo_bytes, measure
from urban.generate import generate_buildings
from urban.io import read_terrain
from urban.model import ClassCode, Scenario
from urban.report import validate_result
from urban.services import calculate_school_services


def test_large_benchmark_terrain():
    data = large_demo_bytes()
    terrain = read_terrain(data)
    assert terrain.classes.shape == (256, 256)
    assert terrain.pixel_size_m == 10
    assert terrain.crs.to_epsg() == 32637
    assert np.all(terrain.classes[:, 149:160] == ClassCode.BLOCKED)
    assert np.all(terrain.classes[:, 160:] == ClassCode.FREE)
    assert np.count_nonzero(terrain.classes == ClassCode.EXISTING_BUILDING) == 20
    assert read_terrain(large_demo_bytes()).input_sha256 == terrain.input_sha256


def test_measured_stages_match_regular_generation():
    data = DEMO_PATH.read_bytes()
    scenario = Scenario()
    times, facts = measure(data, scenario)
    assert set(times) == set(STAGES)
    assert all(value >= 0 for value in times.values())
    assert times["generation_s"] == pytest.approx(sum(
        times[k] for k in ("roads_s", "reserve_s", "buildings_s", "schools_s", "validation_s")
    ))
    from dataclasses import asdict
    plan = generate_buildings(read_terrain(data), scenario)
    report = validate_result(plan, calculate_school_services(plan))
    assert facts["metrics"] == asdict(report.metrics)
    assert facts["zip_bytes"] > 0


def test_benchmark_warmup_repeats_and_medians(monkeypatch):
    import examples.run_benchmark as module
    from statistics import median

    calls = []
    def fake_measure(data, scenario):
        calls.append(scenario)
        times = dict.fromkeys(STAGES, float(len(calls)))
        return times, {"metrics": {"population": scenario.target_population}}
    monkeypatch.setattr(module, "measure", fake_measure)
    result = benchmark()
    assert len(calls) == 12
    for index, case in enumerate(result["cases"]):
        assert case["warmups"] == 1
        assert [row["repeat"] for row in case["repeats"]] == [1, 2, 3, 4, 5]
        assert case["medians_s"]["generation_s"] == median(range(2+index*6, 7+index*6))
        assert case["target_s"] == TARGETS[case["size"]]
        assert case["within_target"] == (case["medians_s"]["generation_s"] <= case["target_s"])
    assert calls[6].street_spacing_cells == 5 and calls[6].school_budget == 3
    assert calls[6].target_population == 20000
