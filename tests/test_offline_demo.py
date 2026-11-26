"""Сквозная репетиция по README; AppTest, а не настоящий браузер.

В CI URBAN_OFFLINE_CHECK=1 запускается в Linux network namespace с одним lo.
"""

import csv
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from time import perf_counter, sleep
from urllib.request import build_opener, ProxyHandler
from urllib.error import URLError
from zipfile import ZipFile

import numpy as np
from rasterio.io import MemoryFile
import streamlit as st
from streamlit.testing.v1 import AppTest

from examples.create_demo import DEMO_PATH


ROOT = Path(__file__).resolve().parents[1]


def test_readme_demo_and_export(tmp_path, monkeypatch):
    offline = os.environ.get("URBAN_OFFLINE_CHECK") == "1"
    if offline:
        assert {name for _, name in socket.if_nameindex()} == {"lo"}

    start = perf_counter()
    # Проверяем запуск настоящего локального HTTP-сервера независимо от AppTest.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log = tmp_path / "server.log"
    with log.open("w", encoding="utf-8") as out:
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
             "--server.headless=true", "--server.address=127.0.0.1",
             f"--server.port={port}", "--browser.gatherUsageStats=false"],
            cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
        )
        try:
            opener = build_opener(ProxyHandler({}))
            deadline = perf_counter() + 30
            while True:
                assert process.poll() is None, log.read_text(encoding="utf-8")
                try:
                    with opener.open(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                        assert response.status == 200
                        break
                except (OSError, URLError):
                    assert perf_counter() < deadline, log.read_text(encoding="utf-8")
                    sleep(0.1)
            with opener.open(f"http://127.0.0.1:{port}/", timeout=5) as response:
                assert response.status == 200
                assert b"streamlit" in response.read().lower()
            server_s = perf_counter() - start
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    # Получаем настоящие байты кнопок через AppTest и сохраняем на диск.
    captured = {}
    original = st.download_button
    def capture(label, data, **kwargs):
        captured[kwargs["file_name"]] = data
        return original(label, data, **kwargs)
    monkeypatch.setattr(st, "download_button", capture)

    app_start = perf_counter()
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception and not app.error
    demo = captured["demo.tif"]
    assert demo == DEMO_PATH.read_bytes()
    (tmp_path / "demo.tif").write_bytes(demo)
    app.radio[0].set_value("Свой TIFF").run()
    app.file_uploader[0].set_value(("demo.tif", demo, "image/tiff")).run()
    app.checkbox(key="legend_confirmed").check().run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    a = app.session_state.verified_report
    assert a.metrics.population == 608 and a.metrics.new_buildings == 19
    assert a.metrics.school_demand == 76 and a.metrics.school_served == 72
    assert a.metrics.school_deficit == 4
    app.button(key="pin_variant_a").click().run()
    app.number_input(key="school_budget").set_value(0)
    app.button[0].click().run()
    assert not app.exception and not app.error
    b = app.session_state.verified_report
    assert b.metrics.population == 608 and b.metrics.school_served == 0
    assert b.metrics.school_deficit == 76
    np.testing.assert_array_equal(a.plan.building_ids, b.plan.building_ids)
    np.testing.assert_array_equal(a.plan.roads.road_mask, b.plan.roads.road_mask)
    for key in ("download_result_zip", "download_variant_a", "download_comparison"):
        app.download_button(key=key).click().run()
    assert not app.exception and not app.error
    for name in ("result.zip", "result_A.zip", "comparison.csv"):
        (tmp_path / name).write_bytes(captured[name])

    for name, report in (("result_A.zip", a), ("result.zip", b)):
        with ZipFile(tmp_path / name) as z:
            assert z.testzip() is None
            assert set(z.namelist()) == {"result.tif", "metrics.json", "buildings.csv"}
            meta = json.loads(z.read("metrics.json"))
            assert meta["input_sha256"] == sha256(demo).hexdigest()
            assert meta["parameters"]["school_budget"] == report.plan.scenario.school_budget
            assert meta["metrics"]["school_served"] == report.metrics.school_served
            rows = list(csv.DictReader(StringIO(z.read("buildings.csv").decode("utf-8"))))
            assert len({row["id"] for row in rows}) == len(rows) == 19
            assert sum(int(row["population"]) for row in rows) == 608
            with MemoryFile(z.read("result.tif")) as mem, mem.open() as ds:
                assert ds.crs == report.plan.terrain.crs
                assert ds.transform == report.plan.terrain.transform
                assert ds.nodata == 255
                np.testing.assert_array_equal(ds.read(1), report.classes)

    with (tmp_path / "comparison.csv").open(encoding="utf-8", newline="") as file:
        rows = {row["metric"]: row for row in csv.DictReader(file)}
    assert float(rows["population"]["delta"]) == 0
    assert float(rows["school_served"]["delta"]) == -72
    # §1: показать исчерпание мест после сравнения и экспорта.
    app.number_input(key="school_budget").set_value(1)
    app.number_input(key="target_population").set_value(20000)
    app.button[0].click().run()
    assert not app.exception and not app.error
    final = app.session_state.verified_report
    assert final.plan.stop_reason == "candidates_exhausted"
    assert final.metrics.population_shortfall == 192
    print("DEMO_PROTOCOL " + json.dumps({
        "network_isolated": offline, "interfaces": [name for _, name in socket.if_nameindex()],
        "server_http_s": round(server_s, 3), "apptest_and_files_s": round(perf_counter()-app_start, 3),
        "total_s": round(perf_counter()-start, 3), "browser": False,
        "A_served": 72, "B_served": 0, "population": 608, "shortfall": 192,
        "opened_files": ["demo.tif", "result_A.zip", "result.zip", "comparison.csv"],
        "both_tiffs_and_reports_verified": True,
    }, ensure_ascii=False))
