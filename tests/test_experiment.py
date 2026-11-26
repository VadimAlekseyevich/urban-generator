"""Контроль воспроизводимого вычислительного эксперимента."""

import csv
import json
import sys

import pytest

from examples.create_demo import DEMO_PATH
from examples.run_experiment import FIELDS, main, paired_experiment, u_experiment


def test_six_paired_scenarios_are_reproducible():
    data = DEMO_PATH.read_bytes()
    first = paired_experiment(data)
    repeat = paired_experiment(data)
    assert len(first) == 6
    assert {(r["seed"], r["school_budget"]) for r in first} == {
        (seed, budget) for seed in (42, 43, 44) for budget in (0, 1)
    }
    assert len({r["input_sha256"] for r in first}) == 1
    for a, b in zip(first, repeat, strict=True):
        assert a["elapsed_s"] >= 0 and b["elapsed_s"] >= 0
        assert {k: v for k, v in a.items() if k != "elapsed_s"} == {
            k: v for k, v in b.items() if k != "elapsed_s"
        }
        assert a["school_deficit"] == a["school_demand"] - a["school_served"]
    for zero, one in zip(first[::2], first[1::2], strict=True):
        assert zero["population"] == one["population"]
        assert zero["school_demand"] == one["school_demand"]
        assert zero["school_served"] == 0
        assert zero["school_coverage_percent"] <= one["school_coverage_percent"]
    assert any(r["school_served"] > 0 for r in first)


def test_u_path_compares_same_endpoints_and_actual_assignment():
    u = u_experiment()
    assert u["school_cell"] == [0, 1]
    assert u["house_entry"] == [1, 5]
    assert u["house_road_entry"] == [1, 4]
    assert u["school_road_entry"] == [1, 1]
    assert u["direct_distance_m"] == pytest.approx(10 * (17 ** 0.5))
    assert u["network_distance_m"] == 130
    assert u["direct_within_threshold"] and not u["network_within_threshold"]
    assert u["school_demand"] == 4
    assert u["served_at_threshold"] == 0
    assert u["served_at_network_boundary"] == 4


def test_cli_writes_csv_and_experiment_metadata(tmp_path, monkeypatch):
    directory = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["run_experiment", "--output-dir", str(directory)])
    main()
    with (directory / "paired.csv").open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert tuple(reader.fieldnames) == FIELDS
        assert len(list(reader)) == 6
    meta = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    assert meta["parameters_except_seed_and_budget"]["floors"] == 3
    assert meta["u_path"]["network_distance_m"] == 130
