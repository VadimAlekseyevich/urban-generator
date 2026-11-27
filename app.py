"""Точка входа локального приложения Streamlit."""

from uuid import uuid4
from hashlib import sha256
import json
from pathlib import Path
from dataclasses import fields

import streamlit as st
from rasterio.errors import RasterioError

from urban.generate import generate_buildings
from urban.map_ui import show_map
from urban.development import apply_development_boundary, clear_development_boundary
from urban.io import TerrainReadError, read_terrain
from urban.vector_import import read_geojson
from urban.osm_cache import prepare_pbf
from urban.osm_import import make_window
from urban.model import CoordinateMode, MAX_INPUT_BYTES, PIXEL_SIZE_M_RANGE, Scenario
from urban.services import calculate_school_services
from urban.report import (
    SCHOOL_LEGEND, INPUT_LEGEND, building_rows, school_project_image,
    ReportValidationError, VerifiedReport, school_rows, terrain_areas, terrain_image, validate_result,
    comparison_rows, export_comparison_csv, export_result_zip,
)


DEMO_PATH = Path(__file__).resolve().parent / "examples" / "demo.tif"
GEOJSON_PATH = DEMO_PATH.with_name("site.geojson")


def show_legend(legend: dict) -> None:
    st.subheader("Легенда")
    for code, (label, color) in sorted(legend.items()):
        st.markdown(
            f'<span style="display:inline-block;width:1em;height:1em;'
            f'background:{color};border:1px solid #888;margin-right:0.5em"></span>'
            f"{int(code)} — {label}", unsafe_allow_html=True,
        )


def show_school_results(report: VerifiedReport) -> None:
    allocation = report.allocation
    metrics = report.metrics
    st.subheader("Школы и доступность")
    st.caption("Спрос и обслуживание относятся только к новым жителям. Свободные места существующих школ не учитываются.")
    columns = st.columns(3)
    columns[0].metric("Спрос, мест", metrics.school_demand)
    columns[1].metric("Обслужено, мест", metrics.school_served)
    columns[2].metric("Дефицит, мест", metrics.school_deficit)
    coverage = f"{metrics.school_coverage_percent:.2f}%" if metrics.school_coverage_percent is not None else "нет спроса"
    columns = st.columns(2)
    columns[0].metric("Покрытие спроса", coverage)
    columns[1].metric("Открыто школ, шт.", metrics.opened_schools)
    no_route, no_capacity = metrics.no_route_places, metrics.no_capacity_places
    columns = st.columns(2)
    columns[0].metric("Нет школы в пределах пути, мест", no_route)
    columns[1].metric("Не хватило мест в достижимой школе", no_capacity)
    if metrics.school_deficit:
        st.info("Дефицит: нет доступной открытой школы в пределах сетевого порога "
                f"— {no_route} мест; не хватило вместимости достижимых школ — {no_capacity} мест.")
    if allocation.schools:
        st.dataframe(school_rows(allocation), hide_index=True)
    else:
        st.caption("Новые школы не открыты.")


METRIC_LABELS = {
    "violations_m2": "Нарушения запретов",
    "disconnected_road_cells": "Несвязанные новые дороги",
    "new_road_cells": "Новые улицы",
    "new_road_area_m2": "Площадь новых улиц",
    "new_buildings": "Новые дома",
    "building_area_m2": "Площадь пятен домов",
    "developed_area_m2": "Освоенная площадь",
    "developed_percent": "Освоение свободной земли",
    "target_population": "Целевой прирост",
    "population": "Фактический прирост",
    "population_delta": "Отклонение от цели",
    "population_shortfall": "Недобор населения",
    "school_demand": "Спрос на школьные места",
    "school_served": "Обслужено",
    "school_deficit": "Дефицит",
    "school_coverage_percent": "Покрытие спроса",
    "opened_schools": "Открыто школ",
    "no_route_places": "Нет достижимой школы",
    "no_capacity_places": "Не хватило мест",
}


def scenario_summary(report: VerifiedReport) -> str:
    s = report.plan.scenario
    return (
        f"Цель {s.target_population} чел. · этажи {s.floors} · шаг улиц {s.street_spacing_cells} "
        f"ячеек · бюджет школ {s.school_budget} · вместимость {s.school_capacity} мест · "
        f"путь {s.max_school_distance_m:g} м · seed {s.seed}"
    )


def show_comparison(report: VerifiedReport) -> None:
    st.subheader("Экспорт и сравнение вариантов")
    st.download_button(
        "Скачать ZIP текущего проекта", st.session_state.result_zip,
        file_name="result.zip", mime="application/zip", key="download_result_zip", on_click="ignore",
    )
    if st.button("Зафиксировать вариант A", key="pin_variant_a"):
        st.session_state.variant_a = report
        st.session_state.variant_a_zip = st.session_state.result_zip
        st.session_state.variant_b = None

    a = st.session_state.get("variant_a")
    b = st.session_state.get("variant_b")
    if a is None:
        st.caption("Зафиксируйте текущий проект как A, измените параметры и сгенерируйте вариант B.")
        return
    if st.button("Сбросить сравнение", key="clear_comparison"):
        st.session_state.variant_a = None
        st.session_state.variant_a_zip = None
        st.session_state.variant_b = None
        return
    st.download_button(
        "Скачать ZIP варианта A", st.session_state.variant_a_zip,
        file_name="result_A.zip", mime="application/zip", key="download_variant_a", on_click="ignore",
    )
    if b is None:
        st.info("Вариант A зафиксирован. Следующий успешный расчёт станет B.")
        st.caption(scenario_summary(a))
        return
    rows = comparison_rows(a, b)
    left, right = st.columns(2)
    for column, label, variant in ((left, "A", a), (right, "B", b)):
        with column:
            st.markdown(f"**Вариант {label}**")
            st.caption(scenario_summary(variant))
            st.image(school_project_image(variant.plan, variant.allocation),
                     alt=f"Карта варианта {label}", width="stretch", output_format="PNG")
    st.caption("Разность = B − A. Для процентов разность в процентных пунктах. Пустое значение означает, что показатель не определён.")
    st.dataframe([
        {"Показатель": METRIC_LABELS[row["metric"]], "Единица": row["unit"],
         "A": row["A"], "B": row["B"], "Δ (B − A)": row["delta"]}
        for row in rows
    ], hide_index=True)
    st.download_button(
        "Скачать comparison.csv", export_comparison_csv(a, b),
        file_name="comparison.csv", mime="text/csv", key="download_comparison", on_click="ignore",
    )


def reset_project(input_identity) -> None:
    if st.session_state.get("input_identity") != input_identity:
        st.session_state.input_identity = input_identity
        st.session_state.viewer_token = uuid4().hex
        st.session_state.map_selection = None
        st.session_state.legend_confirmed = False
        st.session_state.scale_confirmed = False
        st.session_state.building_plan = None
        st.session_state.school_allocation = None
        st.session_state.verified_report = None
        st.session_state.result_zip = None
        st.session_state.variant_a = None
        st.session_state.variant_a_zip = None
        st.session_state.variant_b = None


def reset_source(identity):
    if st.session_state.get("base_input_identity") != identity:
        st.session_state.base_input_identity = identity
        st.session_state.pop("development_boundary", None)
        reset_project((identity, None))


def read_file_source(source: str, demo_bytes: bytes):
    st.session_state.terrain = None
    st.session_state.terrain_ready = False
    if source == "Демо-территория":
        data = demo_bytes
        source_name = "demo.tif"
    elif source == "Свой TIFF":
        uploaded = st.file_uploader(
            "Карта классов TIFF/GeoTIFF", type=["tif", "tiff"],
            max_upload_size=MAX_INPUT_BYTES // (1024 * 1024),
        )
        data = uploaded.getvalue() if uploaded is not None else None
        source_name = uploaded.name if uploaded is not None else ""
    else:
        st.download_button("Скачать пример GeoJSON", GEOJSON_PATH.read_bytes(),
                           file_name="site.geojson", mime="application/geo+json")
        st.caption("WGS84 [долгота, широта]. properties.role: boundary — граница, road — ось дороги, "
                   "building — существующее здание, exclusion — запрет. Одна граница Polygon/MultiPolygon, "
                   "дороги LineString/MultiLineString, здания и запреты Polygon/MultiPolygon.")
        uploaded = st.file_uploader(
            "Участок GeoJSON", type=["geojson", "json"],
            max_upload_size=MAX_INPUT_BYTES // (1024 * 1024),
        )
        data = uploaded.getvalue() if uploaded is not None else None
        source_name = uploaded.name if uploaded is not None else ""

    source_identity = (source, source_name, sha256(data).hexdigest() if data is not None else None)
    if st.session_state.get("source_identity") != source_identity:
        st.session_state.source_identity = source_identity
        # Выбор локального режима и сброса привязки относится к конкретному файлу.
        for key in ("coordinate_selection", "local_scale", "discard_georeferencing", "vector_scale"):
            st.session_state.pop(key, None)

    local = False
    local_scale = None
    discard = False
    vector_scale = None
    if source == "GeoJSON":
        vector_scale = st.number_input(
            "Шаг расчётной сетки, м", min_value=PIXEL_SIZE_M_RANGE[0],
            max_value=PIXEL_SIZE_M_RANGE[1], value=10.0, step=0.5, key="vector_scale",
            help="Местная UTM выбирается автоматически. Предел — 256×256 ячеек; проверьте охват на предпросмотре.",
        )
    if source == "Свой TIFF" and data is not None:
        mode = st.radio(
            "Режим координат", ["Геопривязка GeoTIFF", "Локальные координаты"],
            key="coordinate_selection", horizontal=True,
        )
        local = mode == "Локальные координаты"
        if local:
            local_scale = st.number_input(
                "Шаг растра, м/пиксель", min_value=PIXEL_SIZE_M_RANGE[0],
                max_value=PIXEL_SIZE_M_RANGE[1], value=None, step=0.5, key="local_scale",
            )
            discard = st.checkbox("Сбросить геопривязку файла", key="discard_georeferencing")
            st.caption("Локальное начало координат — левый нижний угол карты. Сброс удаляет исходную привязку.")

    input_identity = (source_identity, local, local_scale, discard, vector_scale)
    reset_source(input_identity)

    if data is None:
        st.info("Загрузите GeoJSON FeatureCollection с ролями объектов, до 8 МиБ." if source == "GeoJSON"
                else "Загрузите один TIFF с кодами классов 0–3 и NoData=255, до 256×256 ячеек и 8 МиБ.")
        st.stop()
    if local and local_scale is None:
        st.info("Укажите масштаб от 5 до 10 м/пиксель для локальной карты.")
        st.stop()

    try:
        terrain = (read_geojson(data, vector_scale, source_name=source_name) if source == "GeoJSON" else
                   read_terrain(data, local_scale=local_scale, discard_georeferencing=discard))
    except TerrainReadError as error:
        st.error(str(error))
        st.stop()

    return terrain, source_name, local or source == "GeoJSON"


def show_osm_source():
    st.caption("PBF читается с диска на компьютере, где запущено приложение. "
               "Скопируйте полный путь к файлу; загружать регион в браузер не требуется.")
    with st.form("osm_input"):
        path = st.text_input("Путь к локальному .osm.pbf", key="osm_path",
                             placeholder=r"C:\Projects\data\city.osm.pbf")
        left, right = st.columns(2)
        lon = left.number_input("Долгота центра", min_value=-179.999, max_value=179.999,
                                value=39.72, format="%.6f", key="osm_lon")
        lat = right.number_input("Широта центра", min_value=-80.0, max_value=84.0,
                                 value=54.62, format="%.6f", key="osm_lat")
        size = left.number_input("Сторона окна, ячеек", min_value=1, max_value=256,
                                 value=256, step=1, key="osm_size")
        scale = right.number_input("Шаг окна, м", min_value=5.0, max_value=10.0,
                                    value=10.0, step=0.5, key="osm_scale")
        threads = left.number_input("Потоков чтения PBF", min_value=1, max_value=16,
                                     value=2, step=1, key="osm_threads")
        index = right.selectbox("Индекс координат узлов", ["flex_mem", "sparse_mem_array"],
                                 key="osm_index")
        st.caption("Региональный PBF сканируется целиком; индекс может занять несколько ГБ памяти. "
                   "Число потоков не ограничивает память индекса. Стоп в меню Streamlit прервёт импорт "
                   "при ближайшем обновлении стадии; на подготовке отношений обновления могут задерживаться.")
        submitted = st.form_submit_button("Открыть участок OSM", key="open_osm")
    if not submitted:
        return None
    if not path.strip():
        st.error("Укажите полный путь к локальному файлу .osm.pbf.")
        return None
    try:
        window = make_window(lon, lat, size, scale)
        with st.status("Подготовка окна OSM…", expanded=True) as status:
            activity = st.empty()
            def progress(message):
                activity.write(message)
                status.update(label=message)
            prepared = prepare_pbf(Path(path.strip().strip('"')), window,
                                   index=index, threads=threads, progress=progress)
            status.update(label="Готовое окно переиспользовано" if prepared.reused else "Окно OSM подготовлено",
                          state="complete", expanded=False)
        return prepared
    except ImportError:
        st.error("Для первого импорта нужен Pyosmium. Перезапустите приложение командой: "
                 "uv run --frozen --with osmium==4.3.2 streamlit run app.py. "
                 "Готовое окно из кэша можно открыть без Pyosmium.")
    except FileNotFoundError:
        st.error("Файл PBF не найден. Проверьте полный путь на компьютере, где запущено приложение.")
    except KeyboardInterrupt:
        st.warning("Импорт прерван. Предыдущая успешная территория и её результаты сохранены.")
    except (ValueError, OSError, RuntimeError, RasterioError) as error:
        st.error(f"Не удалось импортировать участок OSM: {error}")
    return None


def main() -> None:
    st.set_page_config(page_title="Urban Generator", page_icon="🏙️", layout="wide")
    st.title("Urban Generator")
    st.write("Генерация городской застройки и оценка доступности школ.")
    demo_bytes = DEMO_PATH.read_bytes()
    source = st.radio("Источник территории", ["Демо-территория", "Свой TIFF", "GeoJSON", "OSM PBF"], horizontal=True)
    st.download_button("Скачать демо GeoTIFF", demo_bytes, file_name="demo.tif", mime="image/tiff")
    if source == "OSM PBF":
        prepared = show_osm_source()
        if prepared is not None:
            reset_source(("OSM PBF", prepared.cache_key))
            st.session_state.terrain = prepared.terrain
            st.session_state.active_source_name = json.loads(prepared.metadata_json)["source_name"]
            st.session_state.active_requires_scale = True
            st.session_state.active_osm_tiff = prepared.data
        terrain = st.session_state.get("terrain")
        if terrain is None:
            st.info("Выберите локальный PBF и нажмите «Открыть участок OSM».")
            st.stop()
        source_name = st.session_state.active_source_name
        requires_scale = st.session_state.active_requires_scale
        st.caption("Ниже открыта текущая успешная территория. Поля PBF применяются только кнопкой «Открыть участок OSM».")
    else:
        terrain, source_name, requires_scale = read_file_source(source, demo_bytes)
        st.session_state.active_source_name = source_name
        st.session_state.active_requires_scale = requires_scale
        st.session_state.active_osm_tiff = None

    if terrain.development_source is not None:
        terrain = clear_development_boundary(terrain)
    with st.expander("Граница освоения (необязательно)"):
        st.caption("Ограничивает только новые объекты. Существующие дороги снаружи остаются доступными для прохода. "
                   "Для геопривязанной карты — GeoJSON WGS84; для локального TIFF — JSON в локальных метрах.")
        boundary = st.file_uploader("Граница освоения", type=["geojson", "json"],
                                    key="development_boundary", max_upload_size=8)
    boundary_hash = None
    if boundary is not None:
        try:
            terrain = apply_development_boundary(terrain, boundary.getvalue(), source_name=boundary.name)
        except (ValueError, RasterioError) as error:
            st.session_state.terrain = None
            st.session_state.terrain_ready = False
            st.error(f"Не удалось применить границу освоения: {error}")
            st.stop()
        boundary_hash = (boundary.name, sha256(boundary.getvalue()).hexdigest())
    reset_project((st.session_state.base_input_identity, boundary_hash))
    st.session_state.terrain = terrain
    st.caption(f"Источник: {source_name}")
    if terrain.development_source is not None:
        restriction = json.loads(terrain.development_source.manifest_json)
        percent = restriction["eligible_free_percent"]
        share = f"{percent:.2f}%" if percent is not None else "нет свободной земли"
        st.caption(f"Граница освоения — жёлтый контур. Источник: {restriction['source']}. "
                   f"Пригодно {restriction['eligible_free_area_m2']:g} м² свободной земли ({share}). "
                   "Ячейки, которых касается контур, исключены целиком.")
        if restriction["eligible_free_area_m2"] == 0:
            st.info("В границе освоения нет пригодных свободных ячеек: новые объекты не появятся, цель даст недобор.")
    if terrain.vector_source is not None:
        st.caption(f"SHA-256 GeoJSON: {terrain.input_sha256}. Исходные геометрии и роли сохраняются в ZIP. "
                   "Граница учитывается по центрам ячеек, точность контуров ограничена шагом сетки.")
    if terrain.osm_import_json is not None:
        metadata = json.loads(terrain.osm_import_json)
        st.caption(metadata.get("attribution", ""))
        st.caption(f"Источник PBF: {metadata.get('source_name')} · SHA-256: {metadata.get('source_sha256')} · "
                   f"Классификатор: {metadata.get('import_version')}. "
                   f"Охват WGS84: {metadata.get('bounds_wgs84')}.")
        st.info("Код 0 — неизвестные или незанятые по OSM ячейки; доступность строительства нужно проверить.")
        if st.session_state.get("active_osm_tiff") is not None:
            st.download_button("Скачать подготовленный GeoTIFF", st.session_state.active_osm_tiff,
                               file_name="osm-window.tif", mime="image/tiff", key="download_osm_tiff",
                               on_click="ignore")
    if terrain.coordinate_mode == CoordinateMode.GEOREFERENCED:
        st.write(f"CRS: {terrain.crs.to_string()}")
    else:
        st.write("Локальные координаты; начало (0, 0) в левом нижнем углу.")
    rows, columns = terrain.classes.shape
    st.write(
        f"Шаг: {terrain.pixel_size_m:g} м/пиксель · Размер: {columns}×{rows} ячеек · "
        f"Габариты: {columns * terrain.pixel_size_m:g}×{rows * terrain.pixel_size_m:g} м"
    )
    st.metric("Площадь исследования, м²", f"{terrain.valid_mask.sum() * terrain.cell_area_m2:g}")

    # Место карт остаётся над подтверждениями, результат рисуется после формы.
    map_container = st.container()

    st.dataframe(terrain_areas(terrain), hide_index=True)
    st.caption("NoData исключён из площади исследования. Площади рассчитаны по подтверждаемой легенде.")
    legend_confirmed = st.checkbox(
        "Подтверждаю: карта соответствует легенде классов 0–3 и NoData=255", key="legend_confirmed",
    )
    scale_confirmed = True
    if requires_scale:
        scale_confirmed = st.checkbox(
            f"Подтверждаю масштаб {terrain.pixel_size_m:g} м/пиксель", key="scale_confirmed",
        )
    st.session_state.terrain_ready = legend_confirmed and scale_confirmed
    if st.session_state.terrain_ready:
        st.success("Легенда и масштаб подтверждены.")
    else:
        st.info("Проверьте карту и подтвердите легенду; для локального TIFF, GeoJSON и PBF подтвердите также масштаб.")

    defaults = Scenario()
    specifications = {parameter.name: parameter for parameter in fields(Scenario)}
    st.subheader("Параметры застройки")
    with st.form("scenario_parameters"):
        population_spec = specifications["target_population"].metadata
        floors_spec = specifications["floors"].metadata
        target_population = st.number_input(
            "Целевой прирост, чел.", min_value=population_spec["min"], max_value=population_spec["max"],
            value=defaults.target_population, step=1, key="target_population",
        )
        floors = st.number_input(
            "Этажей в новом доме", min_value=floors_spec["min"], max_value=floors_spec["max"],
            value=defaults.floors, step=1, key="floors",
        )
        spacing_spec = specifications["street_spacing_cells"].metadata
        seed_spec = specifications["seed"].metadata
        spacing = st.number_input(
            "Шаг улиц, ячеек", min_value=spacing_spec["min"], max_value=spacing_spec["max"],
            value=defaults.street_spacing_cells, step=1, key="street_spacing_cells",
        )
        budget_spec = specifications["school_budget"].metadata
        capacity_spec = specifications["school_capacity"].metadata
        distance_spec = specifications["max_school_distance_m"].metadata
        school_budget = st.number_input(
            "Бюджет новых школ, шт.", min_value=budget_spec["min"], max_value=budget_spec["max"],
            value=defaults.school_budget, step=1, key="school_budget",
        )
        school_capacity = st.number_input(
            "Вместимость школы, мест", min_value=capacity_spec["min"], max_value=capacity_spec["max"],
            value=defaults.school_capacity, step=1, key="school_capacity",
        )
        max_distance = st.number_input(
            "Предельный путь до школы, м", min_value=float(distance_spec["min"]),
            max_value=float(distance_spec["max"]), value=defaults.max_school_distance_m,
            step=10.0, key="max_school_distance_m",
            help="Расстояние по дорожной сети с учётом подходов к дому и школе.",
        )
        seed = st.number_input(
            "Seed", min_value=seed_spec["min"], max_value=seed_spec["max"],
            value=defaults.seed, step=1, key="seed",
            help="Одинаковые файл, масштаб, параметры и seed дают одинаковый проект.",
        )
        submitted = st.form_submit_button("Сгенерировать", disabled=not st.session_state.terrain_ready)
    if submitted and st.session_state.terrain_ready:
        try:
            scenario = Scenario(
                target_population=target_population, floors=floors,
                street_spacing_cells=spacing, seed=seed,
                school_budget=school_budget, school_capacity=school_capacity,
                max_school_distance_m=max_distance,
            )
            plan = generate_buildings(terrain, scenario)
            allocation = calculate_school_services(plan)
            report = validate_result(plan, allocation)
            result_zip = export_result_zip(report)
        except ReportValidationError as error:
            st.error(f"Внутренняя проверка проекта не пройдена: {error}.")
        except (ValueError, RasterioError, OSError):
            st.error("Не удалось сгенерировать проект. Проверьте параметры и повторите расчёт.")
        else:
            st.session_state.viewer_token = uuid4().hex
            st.session_state.map_selection = None
            st.session_state.building_plan = plan
            st.session_state.school_allocation = allocation
            st.session_state.verified_report = report
            st.session_state.result_zip = result_zip
            if st.session_state.get("variant_a") is not None:
                st.session_state.variant_b = report

    report = st.session_state.get("verified_report")
    plan = report.plan if report is not None else None
    allocation = report.allocation if report is not None else None
    with map_container:
        st.subheader("Карта территории")
        show_map(terrain, report, st.session_state.viewer_token)
        show_legend(SCHOOL_LEGEND if report is not None else INPUT_LEGEND)

    if plan is not None:
        used = plan.scenario
        metrics = report.metrics
        count = metrics.new_road_cells
        st.caption(
            f"Параметры выполненного расчёта: цель {used.target_population} чел., "
            f"этажность {used.floors}, шаг улиц {used.street_spacing_cells} ячеек "
            f"({used.street_spacing_cells * plan.terrain.pixel_size_m:g} м), seed {used.seed}. "
            f"Бюджет школ {used.school_budget}, вместимость {used.school_capacity} мест, "
            f"предельный путь {used.max_school_distance_m:g} м."
        )
        st.metric("Новые улицы, ячеек", count)
        st.metric("Площадь новых улиц, м²", f"{metrics.new_road_area_m2:g}")
        if count == 0:
            st.info("Нет достижимых свободных ячеек на осях новых улиц.")
        st.subheader("Прирост населения")
        columns = st.columns(3)
        columns[0].metric("Новые дома, шт.", metrics.new_buildings)
        columns[1].metric("Целевой прирост, чел.", used.target_population)
        columns[2].metric("Фактический прирост, чел.", metrics.population)
        delta = metrics.population_delta
        columns = st.columns(2)
        columns[0].metric("Отклонение от цели, чел.", f"{delta:+d}")
        columns[1].metric("Недобор, чел.", metrics.population_shortfall)
        if plan.stop_reason == "target_reached":
            st.success("Причина остановки: целевой прирост достигнут.")
        else:
            st.warning(
                f"Недобор — {-delta} чел. Причина остановки: подходящие свободные "
                "площадки 2×2 у дорог исчерпаны с учётом школьного резерва."
            )
        if plan.buildings:
            with st.expander("Новые дома: параметры"):
                st.dataframe(building_rows(plan), hide_index=True)
        show_school_results(report)
        st.subheader("Проверка ограничений")
        columns = st.columns(2)
        columns[0].metric("Нарушения запретов, м²", f"{metrics.violations_m2:g}")
        columns[1].metric("Несвязанные новые дороги, ячеек", metrics.disconnected_road_cells)
        columns = st.columns(2)
        columns[0].metric("Площадь пятен домов, м²", f"{metrics.building_area_m2:g}")
        developed = f"{metrics.developed_percent:.2f}%" if metrics.developed_percent is not None else "нет данных"
        columns[1].metric("Освоение свободной земли", developed)
        st.success("Жёсткие ограничения проекта проверены.")
        show_comparison(report)


if __name__ == "__main__":
    main()
