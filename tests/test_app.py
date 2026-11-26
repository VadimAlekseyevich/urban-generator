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
from urban.report import (
    building_project_classes, building_project_image, road_project_classes,
    road_project_image, school_project_classes, school_project_image,
    school_deficit_reasons, terrain_areas, terrain_image,
)


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


def generated_demo_app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
    app.checkbox(key="legend_confirmed").check().run()
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state.building_plan is not None
    return app


def test_generation_requires_confirmed_terrain() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
    assert app.button[0].label == "Сгенерировать"
    assert app.button[0].disabled
    assert app.session_state.building_plan is None
    app.checkbox(key="legend_confirmed").check().run()
    assert not app.button[0].disabled
    assert app.session_state.building_plan is None


def test_demo_generation_shows_two_maps_and_keeps_scenario() -> None:
    from scipy.ndimage import generate_binary_structure, label

    app = generated_demo_app()
    run = app.session_state.building_plan
    terrain, roads, scenario = run.terrain, run.roads, run.scenario
    assert scenario.seed == 42
    assert scenario.street_spacing_cells == 8
    assert len(app.get("image")) == 2
    assert all(item.proto.imgs[0].url.endswith(".png") for item in app.get("image"))
    assert any(item.value == "Проект" for item in app.subheader)
    assert any("4 — Новая улица" in item.value for item in app.markdown)
    assert roads.new_roads.any()
    assert not roads.new_roads[:, 56:].any()
    components, count = label(roads.road_mask, structure=generate_binary_structure(2, 1))
    for component in range(1, count + 1):
        assert np.any((components == component) & (terrain.classes == 2))
    metrics = {item.label: item.value for item in app.metric}
    assert int(metrics["Новые улицы, ячеек"]) == int(roads.new_roads.sum())
    assert float(metrics["Площадь новых улиц, м²"]) == roads.new_roads.sum() * 100


def test_repeat_seed_42_and_change_spacing_through_form() -> None:
    app = generated_demo_app()
    first = app.session_state.building_plan.roads.new_roads.copy()
    first_buildings = app.session_state.building_plan.buildings
    first_ids = app.session_state.building_plan.building_ids.copy()
    app.button[0].click().run()
    assert not app.exception
    np.testing.assert_array_equal(app.session_state.building_plan.roads.new_roads, first)
    assert app.session_state.building_plan.buildings == first_buildings
    np.testing.assert_array_equal(app.session_state.building_plan.building_ids, first_ids)
    app.number_input(key="street_spacing_cells").set_value(5)
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state.building_plan.scenario.seed == 42
    assert app.session_state.building_plan.scenario.street_spacing_cells == 5
    assert not np.array_equal(app.session_state.building_plan.roads.new_roads, first)


def test_form_edit_without_submit_preserves_completed_run_and_its_parameters() -> None:
    app = generated_demo_app()
    first = app.session_state.building_plan.roads.new_roads.copy()
    # Правка внутри st.form остаётся черновиком до нажатия кнопки.
    app.number_input(key="seed").set_value(43)
    assert not app.exception
    assert app.session_state.building_plan.scenario.seed == 42
    np.testing.assert_array_equal(app.session_state.building_plan.roads.new_roads, first)
    assert any("seed 42" in item.value for item in app.caption)
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state.building_plan.scenario.seed == 43
    assert not np.array_equal(app.session_state.building_plan.roads.new_roads, first)


def test_new_source_clears_the_previous_road_project() -> None:
    app = generated_demo_app()
    app.radio[0].set_value("Свой TIFF").run()
    assert app.session_state.building_plan is None
    assert not app.get("image")
    app.file_uploader[0].set_value(("demo.tif", DEMO_PATH.read_bytes(), "image/tiff")).run()
    assert not app.exception
    assert app.session_state.building_plan is None
    assert app.button[0].disabled
    assert len(app.get("image")) == 1


def test_local_scale_change_clears_project_and_blocks_new_run() -> None:
    app = confirm_local(own_file_app(upload_bytes()))
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state.building_plan.terrain.pixel_size_m == 10
    app.number_input(key="local_scale").set_value(5).run()
    assert not app.exception
    assert app.session_state.building_plan is None
    assert app.button[0].disabled
    assert len(app.get("image")) == 1


def test_project_colors_and_scale_match_the_source_and_preserve_original_objects() -> None:
    app = generated_demo_app()
    run = app.session_state.building_plan
    terrain, roads = run.terrain, run.roads
    before = terrain.classes.copy()
    project = road_project_classes(terrain, roads)
    np.testing.assert_array_equal(project[~roads.new_roads], before[~roads.new_roads])
    assert np.all(project[roads.new_roads] == 4)
    assert not project.flags.writeable
    assert not np.shares_memory(project, terrain.classes)
    original_image, project_image = terrain_image(terrain), road_project_image(terrain, roads)
    assert original_image.size == project_image.size == (768, 768)
    original_pixels, project_pixels = np.asarray(original_image), np.asarray(project_image)
    assert len(project_image.getcolors()) == 6
    enlarged_new_roads = np.repeat(np.repeat(roads.new_roads, 8, axis=0), 8, axis=1)
    np.testing.assert_array_equal(project_pixels[~enlarged_new_roads], original_pixels[~enlarged_new_roads])
    assert np.all(project_pixels[enlarged_new_roads] == (239, 68, 68))
    np.testing.assert_array_equal(terrain.classes, before)


def test_failed_regeneration_keeps_last_successful_scenario(monkeypatch) -> None:
    import urban.generate

    app = generated_demo_app()
    before = app.session_state.building_plan.roads.new_roads.copy()
    building_ids_before = app.session_state.building_plan.building_ids.copy()

    def fail(*args, **kwargs):
        raise ValueError("Контрольный сбой генерации")

    monkeypatch.setattr(urban.generate, "generate_roads", fail)
    app.number_input(key="seed").set_value(43)
    app.button[0].click().run()
    assert not app.exception
    assert "Не удалось сгенерировать проект" in app.error[0].value
    assert app.session_state.building_plan.scenario.seed == 42
    np.testing.assert_array_equal(app.session_state.building_plan.roads.new_roads, before)
    np.testing.assert_array_equal(app.session_state.building_plan.building_ids, building_ids_before)
    assert len(app.get("image")) == 2


def test_occupied_map_displays_zero_new_roads_without_error() -> None:
    data = upload_bytes(np.array([[2, 3], [1, 255]], dtype=np.uint8))
    app = confirm_local(own_file_app(data))
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert not app.session_state.building_plan.roads.new_roads.any()
    assert len(app.get("image")) == 2
    assert any("Нет достижимых свободных ячеек" in item.value for item in app.info)
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Новые дома, шт."] == "0"
    assert metrics["Фактический прирост, чел."] == "0"
    assert metrics["Недобор, чел."] == "600"
    assert "исчерпаны" in app.warning[0].value


def test_form_accepts_maximum_seed_and_street_spacing() -> None:
    app = generated_demo_app()
    app.number_input(key="seed").set_value(2**32 - 1)
    app.number_input(key="street_spacing_cells").set_value(20)
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state.building_plan.scenario.seed == 2**32 - 1
    assert app.session_state.building_plan.scenario.street_spacing_cells == 20


def test_demo_houses_population_and_deviation_match_building_table() -> None:
    app = generated_demo_app()
    plan = app.session_state.building_plan
    metrics = {item.label: item.value for item in app.metric}
    assert plan.scenario.target_population == 600
    assert plan.scenario.floors == 3
    assert metrics["Новые дома, шт."] == "19"
    assert metrics["Целевой прирост, чел."] == "600"
    assert metrics["Фактический прирост, чел."] == "608"
    assert metrics["Отклонение от цели, чел."] == "+8"
    assert metrics["Недобор, чел."] == "0"
    table = app.dataframe[1].value
    assert len(table) == len(plan.buildings)
    assert table["Жителей, чел."].sum() == plan.population == 608
    assert table["ID"].nunique() == 19
    assert set(table["Этажей"]) == {3}
    assert any("Причина остановки: целевой прирост достигнут" in item.value for item in app.success)
    assert not app.warning
    assert any("5 — Новый дом" in item.value for item in app.markdown)


def test_unattainable_population_shows_shortfall_and_stop_reason() -> None:
    app = generated_demo_app()
    app.number_input(key="target_population").set_value(20_000)
    app.button[0].click().run()
    assert not app.exception
    plan = app.session_state.building_plan
    metrics = {item.label: item.value for item in app.metric}
    assert plan.stop_reason == "candidates_exhausted"
    assert plan.population == sum(building.population for building in plan.buildings)
    assert plan.population < 20_000
    shortage = 20_000 - plan.population
    assert int(metrics["Недобор, чел."]) == shortage
    assert int(metrics["Отклонение от цели, чел."]) == -shortage
    assert int(metrics["Фактический прирост, чел."]) == app.dataframe[1].value["Жителей, чел."].sum()
    assert str(shortage) in app.warning[0].value
    assert "площадки 2×2 у дорог исчерпаны" in app.warning[0].value


def test_draft_population_and_floors_do_not_replace_completed_result() -> None:
    app = generated_demo_app()
    app.number_input(key="target_population").set_value(20_000)
    app.number_input(key="floors").set_value(1)
    assert app.session_state.building_plan.scenario.target_population == 600
    assert app.session_state.building_plan.scenario.floors == 3
    assert {item.label: item.value for item in app.metric}["Фактический прирост, чел."] == "608"
    assert any("цель 600 чел., этажность 3" in item.value for item in app.caption)
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state.building_plan.scenario.target_population == 20_000
    assert app.session_state.building_plan.scenario.floors == 1
    assert any("цель 20000 чел., этажность 1" in item.value for item in app.caption)


def test_floor_change_recalculates_population_and_preserves_roads() -> None:
    app = generated_demo_app()
    roads_before = app.session_state.building_plan.roads.new_roads.copy()
    app.number_input(key="floors").set_value(9)
    app.button[0].click().run()
    assert not app.exception
    plan = app.session_state.building_plan
    assert plan.population == 672
    assert len(plan.buildings) == 7
    assert all(building.floors == 9 for building in plan.buildings)
    assert set(app.dataframe[1].value["Этажей"]) == {9}
    np.testing.assert_array_equal(plan.roads.new_roads, roads_before)


def test_building_project_preserves_source_and_uses_distinct_class_color() -> None:
    app = generated_demo_app()
    plan = app.session_state.building_plan
    before = plan.terrain.classes.copy()
    project = building_project_classes(plan)
    houses = plan.building_ids > 0
    assert np.all(project[houses] == 5)
    assert np.all(project[plan.roads.new_roads] == 4)
    unchanged = ~houses & ~plan.roads.new_roads
    np.testing.assert_array_equal(project[unchanged], before[unchanged])
    assert not project.flags.writeable
    assert not np.shares_memory(project, plan.terrain.classes)
    image = building_project_image(plan)
    assert image.size == terrain_image(plan.terrain).size == (768, 768)
    assert len(image.getcolors()) == 7
    pixels = np.asarray(image)
    enlarged_houses = np.repeat(np.repeat(houses, 8, axis=0), 8, axis=1)
    assert np.all(pixels[enlarged_houses] == (22, 163, 74))
    np.testing.assert_array_equal(plan.terrain.classes, before)


def test_school_budget_comparison_through_form():
    app = generated_demo_app()
    app.number_input(key="school_budget").set_value(0)
    app.button[0].click().run()
    assert not app.exception
    zero_plan = app.session_state.building_plan
    zero = app.session_state.school_allocation
    assert zero.served == 0 and zero.deficit == zero.demand == 76
    assert not zero.schools
    assert school_deficit_reasons(zero) == (76, 0)
    app.number_input(key="school_budget").set_value(1)
    app.button[0].click().run()
    assert not app.exception
    plan = app.session_state.building_plan
    result = app.session_state.school_allocation
    assert result.served > 0 and len(result.schools) == 1
    assert plan.population == zero_plan.population == 608
    assert plan.buildings == zero_plan.buildings
    np.testing.assert_array_equal(plan.building_ids, zero_plan.building_ids)
    np.testing.assert_array_equal(plan.roads.road_mask, zero_plan.roads.road_mask)
    metrics = {item.label: item.value for item in app.metric}
    assert int(metrics["Спрос, мест"]) == result.demand
    assert int(metrics["Обслужено, мест"]) == result.served
    assert int(metrics["Дефицит, мест"]) == result.deficit
    assert metrics["Покрытие спроса"] == f"{100 * result.served / result.demand:.2f}%"
    assert int(metrics["Нет школы в пределах пути, мест"]) + int(metrics["Не хватило мест в достижимой школе"]) == result.deficit
    assert any("6 — Новая школа" in item.value for item in app.markdown)
    row = app.dataframe[-1].value.iloc[0]
    assert row["Занято, мест"] == result.schools[0].served
    assert row["Вместимость, мест"] == 80
    assert row["Макс. назначенный путь, м"] <= 400


def test_school_capacity_and_distance_and_draft_parameters():
    app = generated_demo_app()
    previous = app.session_state.school_allocation
    app.number_input(key="school_capacity").set_value(1)
    app.number_input(key="max_school_distance_m").set_value(200.0)
    assert app.session_state.school_allocation.schools == previous.schools
    assert app.session_state.building_plan.scenario.school_capacity == 80
    assert any("вместимость 80 мест" in item.value for item in app.caption)
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state.school_allocation
    assert app.session_state.building_plan.scenario.max_school_distance_m == 200
    assert result.served == 1
    no_route, no_capacity = school_deficit_reasons(result)
    assert no_capacity > 0
    assert no_route + no_capacity == result.deficit
    assert all(s.max_assigned_distance_m <= 200 for s in result.schools)
    assert any("вместимость 1 мест" in item.value for item in app.caption)


def test_zero_school_demand_is_not_a_percentage():
    app = confirm_local(own_file_app(upload_bytes(np.array([[2, 3], [1, 255]], dtype=np.uint8))))
    app.button[0].click().run()
    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Покрытие спроса"] == "нет спроса"
    assert metrics["Дефицит, мест"] == "0"
    assert metrics["Открыто школ, шт."] == "0"


def test_school_map_preserves_source_and_marks_only_open_sites():
    app = generated_demo_app()
    plan = app.session_state.building_plan
    allocation = app.session_state.school_allocation
    original = plan.terrain.classes.copy()
    before = building_project_classes(plan)
    after = school_project_classes(plan, allocation)
    mask = after == 6
    assert mask.sum() == len(allocation.schools)
    for school in allocation.schools:
        coordinate = school.site.coordinate
        assert plan.reserve.mask[coordinate]
        assert before[coordinate] == 0
        assert mask[coordinate]
    np.testing.assert_array_equal(after[~mask], before[~mask])
    np.testing.assert_array_equal(plan.terrain.classes, original)
    assert not after.flags.writeable
    image = school_project_image(plan, allocation)
    assert image.size == terrain_image(plan.terrain).size
    assert (147, 51, 234) in {color for _, color in image.getcolors(maxcolors=1000000)}


def test_failed_school_calculation_preserves_pair_and_source_change_clears_it(monkeypatch):
    import urban.services

    app = generated_demo_app()
    plan = app.session_state.building_plan
    allocation = app.session_state.school_allocation

    def fail(*args, **kwargs):
        raise ValueError("Контрольная ошибка школ")

    monkeypatch.setattr(urban.services, "calculate_school_services", fail)
    app.number_input(key="school_budget").set_value(0)
    app.button[0].click().run()
    assert not app.exception
    assert app.error
    assert app.session_state.building_plan.scenario == plan.scenario
    assert app.session_state.school_allocation.schools == allocation.schools
    app.radio[0].set_value("Свой TIFF").run()
    assert app.session_state.building_plan is None
    assert app.session_state.school_allocation is None


def test_invalid_result_cannot_replace_verified_report(monkeypatch):
    import urban.report

    app = generated_demo_app()
    before = app.session_state.verified_report
    assert before.metrics.violations_m2 == before.metrics.disconnected_road_cells == 0
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Нарушения запретов, м²"] == "0"
    assert metrics["Несвязанные новые дороги, ячеек"] == "0"

    def fail(*args, **kwargs):
        raise urban.report.ReportValidationError("Контрольное нарушение")

    monkeypatch.setattr(urban.report, "validate_result", fail)
    app.number_input(key="seed").set_value(43)
    app.button[0].click().run()
    assert not app.exception
    assert any("Внутренняя проверка проекта не пройдена" in item.value for item in app.error)
    assert app.session_state.verified_report is before
    assert app.session_state.building_plan.scenario.seed == 42
    app.radio[0].set_value("Свой TIFF").run()
    assert app.session_state.verified_report is None


def test_compare_budgets_and_download_completed_reports(monkeypatch):
    import csv
    import json
    from io import StringIO
    from zipfile import ZipFile

    downloads = {}
    original = st.download_button

    def capture(label, data, **kwargs):
        downloads[kwargs["file_name"]] = data
        return original(label, data, **kwargs)

    monkeypatch.setattr(st, "download_button", capture)
    app = generated_demo_app()
    a = app.session_state.verified_report
    app.button(key="pin_variant_a").click().run()
    assert app.session_state.variant_a is a
    assert app.session_state.variant_b is None
    assert "comparison.csv" not in downloads
    a_zip = app.session_state.variant_a_zip
    app.number_input(key="school_budget").set_value(0)
    assert app.session_state.variant_b is None
    app.button[0].click().run()
    assert not app.exception
    b = app.session_state.variant_b
    assert b is app.session_state.verified_report
    assert a.plan.scenario.school_budget == 1 and b.plan.scenario.school_budget == 0
    assert a.metrics.population == b.metrics.population == 608
    assert a.metrics.school_served > b.metrics.school_served == 0
    assert len(app.get("image")) == 4
    np.testing.assert_array_equal(a.plan.building_ids, b.plan.building_ids)
    np.testing.assert_array_equal(a.plan.roads.new_roads, b.plan.roads.new_roads)
    frame = app.dataframe[-1].value.set_index("Показатель")
    assert frame.loc["Фактический прирост", "Δ (B − A)"] == 0
    assert frame.loc["Обслужено", "A"] == a.metrics.school_served
    assert frame.loc["Обслужено", "B"] == 0
    assert frame.loc["Обслужено", "Δ (B − A)"] == -a.metrics.school_served

    # Проверяем байты, переданные именно кнопкам загрузки.
    with ZipFile(BytesIO(downloads["result.zip"])) as archive:
        meta = json.loads(archive.read("metrics.json"))
        assert meta["parameters"]["school_budget"] == 0
        assert meta["metrics"]["population"] == b.metrics.population
        assert meta["metrics"]["school_served"] == 0
    with ZipFile(BytesIO(downloads["result_A.zip"])) as archive:
        meta = json.loads(archive.read("metrics.json"))
        assert meta["parameters"]["school_budget"] == 1
        assert meta["metrics"]["school_served"] == a.metrics.school_served
    rows = {r["metric"]: r for r in csv.DictReader(StringIO(downloads["comparison.csv"].decode()))}
    assert int(rows["school_served"]["delta"]) == -a.metrics.school_served
    assert downloads["result_A.zip"] == a_zip
    previous_zip = app.session_state.result_zip
    app.number_input(key="school_budget").set_value(2)
    app.download_button(key="download_result_zip").click().run()
    assert not app.exception
    assert app.session_state.result_zip == previous_zip
    assert app.session_state.variant_b is b
    assert app.session_state.variant_a is a
    app.button(key="clear_comparison").click().run()
    assert app.session_state.variant_a is None and app.session_state.variant_b is None
    assert app.session_state.verified_report is b


def test_comparison_reset_on_scale_change():
    app = own_file_app(DEMO_PATH.read_bytes())
    app.radio(key="coordinate_selection").set_value("Локальные координаты").run()
    app.number_input(key="local_scale").set_value(10).run()
    # Для GeoTIFF требуется явный сброс привязки.
    app.checkbox(key="discard_georeferencing").check().run()
    app.checkbox(key="legend_confirmed").check().run()
    app.checkbox(key="scale_confirmed").check().run()
    app.button[0].click().run()
    assert not app.exception
    app.button(key="pin_variant_a").click().run()
    app.number_input(key="school_budget").set_value(0)
    app.button[0].click().run()
    assert app.session_state.variant_b is not None
    app.number_input(key="local_scale").set_value(5).run()
    assert app.session_state.variant_a is None
    assert app.session_state.variant_b is None
    assert app.session_state.verified_report is None
    assert app.session_state.result_zip is None
    assert app.session_state.variant_a_zip is None


def test_failed_regeneration_preserves_comparison_and_downloads(monkeypatch):
    import urban.services

    app = generated_demo_app()
    app.button(key="pin_variant_a").click().run()
    a = app.session_state.variant_a
    app.number_input(key="school_budget").set_value(0)
    app.button[0].click().run()
    b = app.session_state.variant_b
    previous_zip = app.session_state.result_zip

    def fail(*args, **kwargs):
        raise ValueError("Контрольный сбой")

    monkeypatch.setattr(urban.services, "calculate_school_services", fail)
    app.number_input(key="school_budget").set_value(2)
    app.button[0].click().run()
    assert not app.exception and app.error
    assert app.session_state.variant_a is a
    assert app.session_state.variant_b is b
    assert app.session_state.verified_report is b
    assert app.session_state.result_zip == previous_zip
    app.radio[0].set_value("Свой TIFF").run()
    assert app.session_state.variant_a is None and app.session_state.variant_b is None
    assert app.session_state.result_zip is None
