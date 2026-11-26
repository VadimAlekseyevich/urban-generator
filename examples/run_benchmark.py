"""Замер стадий ядра на двух размерах (§8 DESIGN.md)."""

import argparse
import csv
from dataclasses import asdict
from importlib.metadata import version
from io import StringIO
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
from time import perf_counter

import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from examples.create_demo import DEMO_PATH
from urban.generate import generate_roads, place_buildings, reserve_school_sites
from urban.io import read_terrain
from urban.model import ClassCode, Scenario
from urban.report import export_result_zip, validate_result
from urban.services import calculate_school_services


STAGES = ("read_s", "roads_s", "reserve_s", "buildings_s", "schools_s", "validation_s", "generation_s", "export_s")
REPEATS = 5
TARGETS = {96: 3.0, 256: 10.0}


def large_demo_bytes() -> bytes:
    """Схема демо на 256×256, шаг 10 м, без внешних данных.

    Координаты реки, исходных дорог и домов масштабированы относительно
    исходного демо; пятна домов остаются 2×2, река пересекает всю карту.
    """
    size = 256
    def at(n):
        return round(n * size / 96)
    classes = np.full((size, size), ClassCode.FREE, dtype=np.uint8)
    classes[:, at(56):at(60)] = ClassCode.BLOCKED
    classes[:4, :4] = classes[-4:, :4] = ClassCode.NODATA
    classes[at(12):at(84), at(20)] = ClassCode.EXISTING_ROAD
    classes[at(48), at(8):at(49)] = ClassCode.EXISTING_ROAD
    for r, c in ((at(18), at(20)-2), (at(30), at(20)+1),
                 (at(42), at(20)-2), (at(48)+1, at(30)), (at(68), at(20)+1)):
        # Пятна 2×2 по-прежнему соседствуют с исходной дорогой.
        classes[r:r+2, c:c+2] = ClassCode.EXISTING_BUILDING
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", height=size, width=size, count=1, dtype="uint8",
                      crs="EPSG:32637", transform=from_origin(500000, 6200000, 10, 10),
                      nodata=255, compress="deflate") as ds:
            ds.write(classes, 1)
        return mem.read()


def measure(data: bytes, scenario: Scenario) -> tuple[dict, dict]:
    """Один полный проход. Чтение и экспорт исключены из generation_s."""
    start = perf_counter()
    terrain = read_terrain(data)
    t0 = perf_counter()
    roads = generate_roads(terrain, scenario)
    t1 = perf_counter()
    reserve = reserve_school_sites(roads)
    t2 = perf_counter()
    plan = place_buildings(terrain, scenario, roads, reserve)
    t3 = perf_counter()
    allocation = calculate_school_services(plan)
    t4 = perf_counter()
    report = validate_result(plan, allocation)
    t5 = perf_counter()
    archive = export_result_zip(report)
    t6 = perf_counter()
    timings = dict(zip(STAGES, (t0-start, t1-t0, t2-t1, t3-t2, t4-t3,
                                t5-t4, t5-t0, t6-t5), strict=True))
    facts = {"input_sha256": terrain.input_sha256, "shape": list(terrain.classes.shape),
             "pixel_size_m": terrain.pixel_size_m, "parameters": asdict(scenario),
             "metrics": asdict(report.metrics), "zip_bytes": len(archive)}
    return timings, facts


def benchmark() -> dict:
    cases = [
        (96, DEMO_PATH.read_bytes(), Scenario()),
        (256, large_demo_bytes(), Scenario(target_population=20000, street_spacing_cells=5,
                                           school_budget=3)),
    ]
    results = []
    for size, data, scenario in cases:
        _, baseline = measure(data, scenario)  # Прогрев, не входит в медиану.
        rows = []
        for repeat in range(1, REPEATS + 1):
            times, facts = measure(data, scenario)
            if facts != baseline:
                raise RuntimeError(f"Невоспроизводимые метрики на {size}×{size}")
            rows.append({"repeat": repeat, **times})
        medians = {key: statistics.median(row[key] for row in rows) for key in STAGES}
        results.append({"size": size, "warmups": 1, "repeats": rows, "medians_s": medians,
                        "target_s": TARGETS[size], "within_target": medians["generation_s"] <= TARGETS[size],
                        **baseline})
    return {"environment": environment(), "cases": results}


def environment() -> dict:
    cpu = platform.processor() or platform.machine()
    ram = None
    try:
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
            ram = int(Path("/proc/meminfo").read_text().splitlines()[0].split()[1]) * 1024
        elif platform.system() == "Windows":
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in
                    ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual",
                     "avail_virtual", "avail_extended")
                ]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                ram = status.total_phys
            cpu = os.environ.get("PROCESSOR_IDENTIFIER", cpu)
        elif platform.system() == "Darwin":
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "ram_bytes": ram,
            "os": platform.platform(), "python": platform.python_version(),
            "packages": {p: version(p) for p in ("numpy", "scipy", "rasterio", "pillow", "streamlit", "pytest")},
            "github_actions": os.environ.get("GITHUB_ACTIONS") == "true"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results"))
    args = parser.parse_args()
    result = benchmark()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = StringIO(newline="")
    writer = csv.DictWriter(raw, fieldnames=("size", "repeat", *STAGES), lineterminator="\n")
    writer.writeheader()
    for case in result["cases"]:
        for row in case["repeats"]:
            writer.writerow({"size": case["size"], **row})
    (args.output_dir / "timings.csv").write_text(raw.getvalue(), encoding="utf-8")
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (args.output_dir / "benchmark.json").write_text(payload, encoding="utf-8")
    print("BENCHMARK_JSON_BEGIN")
    print(payload, end="")
    print("BENCHMARK_JSON_END")
    print("BENCHMARK_CSV_BEGIN")
    print(raw.getvalue(), end="")
    print("BENCHMARK_CSV_END")


if __name__ == "__main__":
    main()
