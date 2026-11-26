"""Точка входа локального приложения Streamlit."""

import streamlit as st


st.set_page_config(page_title="Urban Generator", page_icon="🏙️", layout="wide")

st.title("Urban Generator")
st.write("Генерация городской застройки и оценка доступности школ.")
st.info(
    "Приложение запущено. Загрузка TIFF/GeoTIFF, демотерритория и расчёт "
    "сценариев появятся в следующих задачах."
)
