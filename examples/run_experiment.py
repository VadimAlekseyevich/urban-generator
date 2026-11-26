"""Воспроизводимый парный эксперимент по §8 DESIGN.md.

Запуск: uv run python -m examples.run_experiment --output-dir experiment-results
"""

import argparse
import csv
from dataclasses import asdict, replace
from hashlib import sha256
import json
from math import dist
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from examples.create_demo import DEMO_PATH
from urban.generate import calculate_suitability, generate_buildings, place_buildings
from urban.io import read_terrain
from urban.model import CoordinateMode, RoadNetwork, Scenario, SchoolReserve, Terrain
from urban.report import validate_result
from urban.services import calculate_school_services


SEEDS = (42, 43, 44)
BUDGETS = (0, 1)
FIELDS = (
    "input_sha256", "seed", "school_budget", "population", "school_demand",
    "school_served", "school_deficit", "school_coverage_percent", "elapsed_s",
)


def paired_experiment(data: bytes) -> list[dict]:
    """Шесть независимых запусков; меняем только seed и бюджет.

    Время включает генерацию, школьные расчёты и независимую проверку,
    но не чтение входа, экспорт или UI. Один замер на сценарий.
    """
    terrain = read_terrain(data)
    rows = []
    for seed in SEEDS:
        baseline = None
        for budget in BUDGETS:
            scenario = Scenario(seed=seed, school_budget=budget)
            start = perf_counter()
            plan = generate_buildings(terrain, scenario)
            report = validate_result(plan, calculate_school_services(plan))
            elapsed = perf_counter() - start
            m = report.metrics
            if baseline is None:
                baseline = report
                if m.school_served != 0:
                    raise RuntimeError("Нулевой бюджет не должен обслуживать спрос")
            else:
                if (m.population != baseline.metrics.population
                        or m.school_demand != baseline.metrics.school_demand
                        or not np.array_equal(plan.building_ids, baseline.plan.building_ids)
                        or not np.array_equal(plan.roads.road_mask, baseline.plan.roads.road_mask)
                        or m.school_served < baseline.metrics.school_served):
                    raise RuntimeError(f"Нарушена сопоставимость пары seed={seed}")
            rows.append({
                "input_sha256": terrain.input_sha256, "seed": seed,
                "school_budget": budget, "population": m.population,
                "school_demand": m.school_demand, "school_served": m.school_served,
                "school_deficit": m.school_deficit,
                "school_coverage_percent": m.school_coverage_percent,
                "elapsed_s": round(elapsed, 9),
            })
    return rows


def u_experiment() -> dict:
    """Контрольная геометрия 7×7, тот же подход к входам, что в ядре."""
    classes = np.ones((7, 7), dtype=np.uint8)
    classes[1:6, 1] = classes[1:6, 4] = classes[5, 1:5] = 2
    classes[0, 1] = 0
    classes[0:2, 5:7] = 0
    terrain = Terrain(classes, np.ones_like(classes, dtype=bool), 10,
                      CoordinateMode.LOCAL, sha256(classes.tobytes()).hexdigest())
    road_mask = classes == 2
    roads = RoadNetwork(calculate_suitability(terrain), np.zeros_like(road_mask),
                        road_mask, (0, 0))
    reserve = np.zeros_like(road_mask)
    reserve[0, 1] = True
    scenario = Scenario(target_population=1, max_school_distance_m=100)
    plan = place_buildings(terrain, scenario, roads, SchoolReserve(((0, 1),), reserve))
    house, = plan.buildings
    allocation = calculate_school_services(plan)
    report = validate_result(plan, allocation)
    site, = allocation.paths.sites
    direct = dist(site.coordinate, house.house_entry) * terrain.pixel_size_m
    network = float(allocation.paths.distances_m[0, 0])
    threshold = scenario.max_school_distance_m
    boundary_plan = replace(plan, scenario=replace(scenario, max_school_distance_m=network))
    boundary = validate_result(boundary_plan, calculate_school_services(boundary_plan))
    return {
        "pixel_size_m": terrain.pixel_size_m,
        "house_id": house.id,
        "house_entry": list(house.house_entry),
        "house_road_entry": list(house.road_entry),
        "school_cell": list(site.coordinate),
        "school_road_entry": list(site.road_entry),
        "direct_distance_m": direct,
        "network_distance_m": network,
        "threshold_m": threshold,
        "direct_within_threshold": direct <= threshold,
        "network_within_threshold": bool(allocation.paths.reachable(threshold)[0, 0]),
        "school_demand": report.metrics.school_demand,
        "served_at_threshold": report.metrics.school_served,
        "served_at_network_boundary": boundary.metrics.school_served,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("experiment-results"))
    args = parser.parse_args()
    rows = paired_experiment(DEMO_PATH.read_bytes())
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "parameters_except_seed_and_budget": {
            k: v for k, v in asdict(Scenario()).items() if k not in ("seed", "school_budget")
        },
        "timing_scope": "generation + allocation + validation; excludes reading, export and UI",
        "u_path": u_experiment(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "paired.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output_dir / "experiment.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Вывод позволяет сохранить фактические данные непосредственно из CI.
    print(csv_path.read_text(encoding="utf-8"), end="")
    print(json_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
