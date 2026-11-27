"""Настоящий Chromium: геометрия кликов, изменение размера и работа без внешней сети."""
import os
from pathlib import Path
import socket
import subprocess
import sys
from time import monotonic, sleep
from urllib.request import ProxyHandler, build_opener

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from urban.generate import generate_buildings
from urban.io import read_terrain
from urban.model import Scenario


pytestmark = pytest.mark.skipif(os.environ.get("URBAN_BROWSER_CHECK") != "1", reason="Chromium runs in browser CI job")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def browser_app(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    count = tmp_path / "generations.log"
    wrapper = tmp_path / "browser_app.py"
    wrapper.write_text(f'''import sys,runpy
sys.path.insert(0, {str(ROOT)!r})
import urban.generate as core
if not getattr(core.generate_buildings, '_counted', False):
    original = core.generate_buildings
    def counted(*args, **kwargs):
        with open({str(count)!r}, 'a') as log: log.write('generation\\n')
        return original(*args, **kwargs)
    counted._counted = True
    core.generate_buildings = counted
runpy.run_path({str(ROOT / 'app.py')!r}, run_name='__main__')
''')
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (tmp_path / "server.log").open("w") as output:
        process = subprocess.Popen([sys.executable,"-m","streamlit","run",str(wrapper),
                                    "--server.address=127.0.0.1",f"--server.port={port}",
                                    "--server.headless=true","--browser.gatherUsageStats=false"],
                                   stdout=output,stderr=subprocess.STDOUT,cwd=ROOT)
        try:
            deadline = monotonic()+30
            opener = build_opener(ProxyHandler({}))
            while True:
                assert process.poll() is None
                try:
                    opener.open(f"http://127.0.0.1:{port}/_stcore/health", timeout=1).close()
                    break
                except OSError:
                    assert monotonic() < deadline
                    sleep(.1)
            with playwright.sync_playwright() as p:
                browser = p.chromium.launch()
                context = browser.new_context(viewport={"width":1280,"height":1000},device_scale_factor=1.5)
                context.route("**/*",lambda route: route.continue_() if route.request.url.startswith(f"http://127.0.0.1:{port}/") else route.abort())
                page = context.new_page()
                page.goto(f"http://127.0.0.1:{port}")
                try:
                    yield page, count
                finally:
                    artifact=Path(os.environ.get("URBAN_BROWSER_ARTIFACTS","/tmp/urban-browser"))
                    artifact.mkdir(parents=True,exist_ok=True)
                    page.screenshot(path=str(artifact/(tmp_path.name+".png")),full_page=True)
                    (artifact/(tmp_path.name+".txt")).write_text(page.locator("body").inner_text())
                    (artifact/(tmp_path.name+"-server.log")).write_text((tmp_path/"server.log").read_text())
                    browser.close()
        finally:
            process.terminate()
            process.wait(timeout=10)


def click_cell(page, row, column):
    canvas=page.locator(".viewer canvas")
    canvas.scroll_into_view_if_needed()
    dims=canvas.evaluate("e=>({s:+e.dataset.scale,x:+e.dataset.offsetX,y:+e.dataset.offsetY})")
    canvas.click(position={"x":dims["x"]+(column+.5)*dims["s"],"y":dims["y"]+(row+.5)*dims["s"]})


def download_zip(page):
    with page.expect_download() as event:
        page.get_by_role("button",name="Скачать ZIP текущего проекта",exact=True).click()
    return Path(event.value.path()).read_bytes()


def test_selection_zoom_resize_layers_and_zip_are_stable(browser_app):
    page,count=browser_app
    page.get_by_text("Подтверждаю: карта соответствует легенде классов 0–3 и NoData=255",exact=True).click()
    page.get_by_role("button",name="Сгенерировать",exact=True).click()
    page.get_by_role("button",name="Скачать ZIP текущего проекта",exact=True).wait_for()
    original_zip=download_zip(page)
    plan=generate_buildings(read_terrain((ROOT/"examples/demo.tif").read_bytes()),Scenario())
    house=plan.buildings[0]
    page.get_by_role("button",name="Общий вид",exact=True).click()
    # Масштабируем вокруг конкретного дома, затем проверяем все четыре ячейки.
    click_cell(page,house.row,house.column)
    for viewport in ({"width":1280,"height":1000},{"width":900,"height":850}):
        page.set_viewport_size(viewport)
        page.get_by_role("button",name="Общий вид",exact=True).click()
        canvas=page.locator(".viewer canvas");canvas.scroll_into_view_if_needed()
        dims=canvas.evaluate("e=>({s:+e.dataset.scale,x:+e.dataset.offsetX,y:+e.dataset.offsetY})")
        box=canvas.bounding_box()
        page.mouse.move(box["x"]+dims["x"]+(house.column+1)*dims["s"],
                        box["y"]+dims["y"]+(house.row+1)*dims["s"])
        page.mouse.wheel(0,-300)
        assert canvas.evaluate("e=>+e.dataset.scale") > dims["s"]
        for dr,dc in ((0,0),(0,1),(1,0),(1,1)):
            click_cell(page,house.row+dr,house.column+dc)
            page.get_by_text(f"Дом №{house.id}",exact=True).wait_for()
            assert f"Жителей: {house.population}" in page.locator(".object-details").inner_text()
    page.get_by_label("Новая улица",exact=True).uncheck()
    page.get_by_label("Новая улица",exact=True).check()
    page.get_by_role("button",name="Линейка",exact=True).click()
    click_cell(page,20,20);click_cell(page,20,30)
    assert "100.0 м" in page.locator(".position").inner_text()
    page.get_by_role("button",name="Линейка",exact=True).click()
    page.get_by_text("Подтверждаю: карта соответствует легенде классов 0–3 и NoData=255",exact=True).click()
    page.get_by_text(f"Дом №{house.id}",exact=True).wait_for()
    assert download_zip(page)==original_zip
    assert count.read_text().splitlines()==["generation"]
    artifact=Path(os.environ.get("URBAN_BROWSER_ARTIFACTS","/tmp/urban-browser"))
    artifact.mkdir(parents=True,exist_ok=True)
    page.locator(".viewer").screenshot(path=str(artifact/"viewer.png"))
    page.get_by_text("Свой TIFF",exact=True).click()
    assert page.locator(".viewer canvas").count()==0


@pytest.mark.parametrize("local",[False,True])
def test_rectangular_raster_last_cell_and_offline_loading(browser_app,tmp_path,local):
    page,_=browser_app
    classes=np.zeros((32,64),dtype=np.uint8);classes[-1,-1]=2
    with MemoryFile() as memory:
        with memory.open(driver="GTiff",width=64,height=32,count=1,dtype="uint8",nodata=255,
                         crs="EPSG:32637",transform=from_origin(500000,6200000,10,10)) as ds:
            ds.write(classes,1)
        data=memory.read()
    path=tmp_path/"rectangle.tif";path.write_bytes(data)
    page.get_by_text("Свой TIFF",exact=True).click()
    page.locator('input[type="file"]').set_input_files(str(path))
    if local:
        page.get_by_text("Локальные координаты",exact=True).click()
        page.get_by_label("Шаг растра, м/пиксель",exact=True).fill("10")
        page.get_by_label("Шаг растра, м/пиксель",exact=True).press("Enter")
        page.get_by_text("Сбросить геопривязку файла",exact=True).click()
    page.locator('.viewer canvas[data-columns="64"][data-rows="32"]').wait_for()
    click_cell(page,31,63)
    assert "Строка 31, столбец 63" in page.locator(".position").inner_text()
