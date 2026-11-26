"""Точка входа локального приложения Streamlit."""

from hashlib import sha256
from pathlib import Path
from dataclasses import fields

import streamlit as st

from urban.generate import generate_buildings
from urban.io import TerrainReadError, read_terrain
from urban.model import CoordinateMode, MAX_INPUT_BYTES, PIXEL_SIZE_M_RANGE, Scenario
from urban.report import (
    BUILDING_LEGEND, INPUT_LEGEND, building_project_image, building_rows,
    terrain_areas, terrain_image,
)


DEMO_PATH = Path(__file__).resolve().parent / "examples" / "demo.tif"


def show_legend(legend: dict) -> None:
    st.subheader("Легенда")
    for code, (label, color) in sorted(legend.items()):
        st.markdown(
            f'<span style="display:inline-block;width:1em;height:1em;'
            f'background:{color};border:1px solid #888;margin-right:0.5em"></span>'
            f"{int(code)} — {label}", unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="Urban Generator", page_icon="🏙️", layout="wide")
    st.title("Urban Generator")
    st.write("Генерация городской застройки и оценка доступности школ.")
    st.session_state.terrain = None
    st.session_state.terrain_ready = False

    demo_bytes = DEMO_PATH.read_bytes()
    source = st.radio("Источник территории", ["Демо-территория", "Свой TIFF"], horizontal=True)
    st.download_button("Скачать демо GeoTIFF", demo_bytes, file_name="demo.tif", mime="image/tiff")
    if source == "Демо-территория":
        data = demo_bytes
        source_name = "demo.tif"
    else:
        uploaded = st.file_uploader(
            "Карта классов TIFF/GeoTIFF", type=["tif", "tiff"],
            max_upload_size=MAX_INPUT_BYTES // (1024 * 1024),
        )
        data = uploaded.getvalue() if uploaded is not None else None
        source_name = uploaded.name if uploaded is not None else ""

    source_identity = (source, sha256(data).hexdigest() if data is not None else None)
    if st.session_state.get("source_identity") != source_identity:
        st.session_state.source_identity = source_identity
        # Выбор локального режима и сброса привязки относится к конкретному файлу.
        for key in ("coordinate_selection", "local_scale", "discard_georeferencing"):
            st.session_state.pop(key, None)

    local = False
    local_scale = None
    discard = False
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

    input_identity = (source_identity, local, local_scale, discard)
    if st.session_state.get("input_identity") != input_identity:
        st.session_state.input_identity = input_identity
        st.session_state.legend_confirmed = False
        st.session_state.scale_confirmed = False
        st.session_state.building_plan = None

    if data is None:
        st.info("Загрузите один TIFF с кодами классов 0–3 и NoData=255, до 256×256 ячеек и 8 МиБ.")
        st.stop()
    if local and local_scale is None:
        st.info("Укажите масштаб от 5 до 10 м/пиксель для локальной карты.")
        st.stop()

    try:
        terrain = read_terrain(data, local_scale=local_scale, discard_georeferencing=discard)
    except TerrainReadError as error:
        st.error(str(error))
        st.stop()

    st.session_state.terrain = terrain
    st.caption(f"Источник: {source_name}")
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
    if local:
        scale_confirmed = st.checkbox(
            f"Подтверждаю масштаб {terrain.pixel_size_m:g} м/пиксель", key="scale_confirmed",
        )
    st.session_state.terrain_ready = legend_confirmed and scale_confirmed
    if st.session_state.terrain_ready:
        st.success("Легенда и масштаб подтверждены.")
    else:
        st.info("Проверьте карту и подтвердите легенду; в локальном режиме подтвердите также масштаб.")

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
            )
            plan = generate_buildings(terrain, scenario)
        except ValueError:
            st.error("Не удалось сгенерировать проект. Проверьте параметры и повторите расчёт.")
        else:
            st.session_state.building_plan = plan

    plan = st.session_state.get("building_plan")
    with map_container:
        left, right = st.columns(2)
        with left:
            st.subheader("Исходная территория")
            st.image(
                terrain_image(terrain), alt="Карта классов исходной территории",
                width="stretch", output_format="PNG",
            )
        with right:
            if plan is None:
                show_legend(INPUT_LEGEND)
            else:
                st.subheader("Проект")
                st.image(
                    building_project_image(plan),
                    alt="Проект с новыми улицами и домами", width="stretch", output_format="PNG",
                )
        if plan is not None:
            show_legend(BUILDING_LEGEND)

    if plan is not None:
        used = plan.scenario
        count = int(plan.roads.new_roads.sum())
        st.caption(
            f"Параметры выполненного расчёта: цель {used.target_population} чел., "
            f"этажность {used.floors}, шаг улиц {used.street_spacing_cells} ячеек "
            f"({used.street_spacing_cells * plan.terrain.pixel_size_m:g} м), seed {used.seed}."
        )
        st.metric("Новые улицы, ячеек", count)
        st.metric("Площадь новых улиц, м²", f"{count * plan.terrain.cell_area_m2:g}")
        if count == 0:
            st.info("Нет достижимых свободных ячеек на осях новых улиц.")
        st.subheader("Прирост населения")
        columns = st.columns(3)
        columns[0].metric("Новые дома, шт.", len(plan.buildings))
        columns[1].metric("Целевой прирост, чел.", used.target_population)
        columns[2].metric("Фактический прирост, чел.", plan.population)
        delta = plan.population - used.target_population
        columns = st.columns(2)
        columns[0].metric("Отклонение от цели, чел.", f"{delta:+d}")
        columns[1].metric("Недобор, чел.", max(0, -delta))
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
    st.caption("Школы и распределение школьных мест пока недоступны.")


if __name__ == "__main__":
    main()
