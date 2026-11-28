"""Встроенный локальный canvas без сборки и сторонних картографических сервисов."""

from pathlib import Path

import streamlit as st
import streamlit.components.v2 as components

from urban.viewer import object_at, viewer_payload


ROOT = Path(__file__).with_name("map_frontend")
MAP = components.component("urban_map", html=(ROOT / "map.html").read_text(encoding="utf-8"),
                           css=(ROOT / "map.css").read_text(encoding="utf-8"), js=(ROOT / "map.js").read_text(encoding="utf-8"))


def show_map(terrain, report, identity):
    payload = viewer_payload(terrain, report)
    payload["identity"] = identity
    result = MAP(data=payload, key=f"map-{identity}", default={"selection": None},
                 on_selection_change=lambda: None)
    selection = result.selection
    selected = (object_at(report, selection.get("row"), selection.get("column"))
                if isinstance(selection, dict) else None)
    st.session_state.map_selection = selected
    if selected:
        st.caption(f"Выбран {selected['title']}. {selected['details']}")
