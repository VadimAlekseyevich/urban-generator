"""Точка входа локального приложения Streamlit."""

from hashlib import sha256
from pathlib import Path
from dataclasses import fields

import streamlit as st

from urban.generate import generate_roads
from urban.io import TerrainReadError, read_terrain
from urban.model import CoordinateMode, MAX_INPUT_BYTES, PIXEL_SIZE_M_RANGE, Scenario
from urban.report import INPUT_LEGEND, ROAD_LEGEND, road_project_image, terrain_areas, terrain_image


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
        st.session_state.road_run = None

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
    st.subheader("Параметры улиц")
    with st.form("road_parameters"):
        spacing_spec = specifications["street_spacing_cells"].metadata
        seed_spec = specifications["seed"].metadata
        spacing = st.number_input(
            "Шаг улиц, ячеек", min_value=spacing_spec["min"], max_value=spacing_spec["max"],
            value=defaults.street_spacing_cells, step=1, key="street_spacing_cells",
        )
        seed = st.number_input(
            "Seed", min_value=seed_spec["min"], max_value=seed_spec["max"],
            value=defaults.seed, step=1, key="seed",
            help="Одинаковые файл, масштаб, шаг улиц и seed дают одинаковый проект.",
        )
        submitted = st.form_submit_button("Сгенерировать", disabled=not st.session_state.terrain_ready)
    if submitted and st.session_state.terrain_ready:
        try:
            scenario = Scenario(street_spacing_cells=spacing, seed=seed)
            roads = generate_roads(terrain, scenario)
        except ValueError:
            st.error("Не удалось построить улицы. Проверьте параметры и повторите расчёт.")
        else:
            st.session_state.road_run = {"terrain": terrain, "scenario": scenario, "roads": roads}

    run = st.session_state.get("road_run")
    with map_container:
        left, right = st.columns(2)
        with left:
            st.subheader("Исходная территория")
            st.image(
                terrain_image(terrain), alt="Карта классов исходной территории",
                width="stretch", output_format="PNG",
            )
        with right:
            if run is None:
                show_legend(INPUT_LEGEND)
            else:
                st.subheader("Проект")
                st.image(
                    road_project_image(run["terrain"], run["roads"]),
                    alt="Проект с новыми улицами", width="stretch", output_format="PNG",
                )
        if run is not None:
            show_legend(ROAD_LEGEND)

    if run is not None:
        used = run["scenario"]
        count = int(run["roads"].new_roads.sum())
        st.caption(
            f"Параметры выполненного расчёта: шаг улиц {used.street_spacing_cells} ячеек "
            f"({used.street_spacing_cells * run['terrain'].pixel_size_m:g} м), seed {used.seed}."
        )
        st.metric("Новые улицы, ячеек", count)
        st.metric("Площадь новых улиц, м²", f"{count * run['terrain'].cell_area_m2:g}")
        if count == 0:
            st.info("Нет достижимых свободных ячеек на осях новых улиц.")
    st.caption("В этой версии рассчитываются улицы. Дома и школы пока недоступны.")


if __name__ == "__main__":
    main()
