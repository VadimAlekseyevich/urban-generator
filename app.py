"""Точка входа локального приложения Streamlit."""

from hashlib import sha256
from pathlib import Path

import streamlit as st

from urban.io import TerrainReadError, read_terrain
from urban.model import CoordinateMode, MAX_INPUT_BYTES, PIXEL_SIZE_M_RANGE
from urban.report import INPUT_LEGEND, terrain_areas, terrain_image


DEMO_PATH = Path(__file__).resolve().parent / "examples" / "demo.tif"


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

    map_column, legend_column = st.columns([2, 1])
    with map_column:
        st.subheader("Исходная территория")
        st.image(terrain_image(terrain), alt="Карта классов исходной территории")
    with legend_column:
        st.subheader("Легенда")
        for code, (label, color) in INPUT_LEGEND.items():
            st.markdown(
                f'<span style="display:inline-block;width:1em;height:1em;'
                f'background:{color};border:1px solid #888;margin-right:0.5em"></span>'
                f"{int(code)} — {label}", unsafe_allow_html=True,
            )

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
    st.caption("Расчёт застройки пока недоступен.")


if __name__ == "__main__":
    main()
