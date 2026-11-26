"""Изображения исходной территории, проекта улиц и площади по классам."""

import numpy as np
from PIL import Image

from urban.model import ClassCode, RoadNetwork, Terrain


# Подписи и цвета общие для карты и легенды интерфейса.
INPUT_LEGEND = {
    ClassCode.FREE: ("Свободная территория", "#e5e7eb"),
    ClassCode.BLOCKED: ("Запрет, в том числе вода", "#2563eb"),
    ClassCode.EXISTING_ROAD: ("Существующая дорога", "#475569"),
    ClassCode.EXISTING_BUILDING: ("Существующая застройка", "#b45309"),
    ClassCode.NODATA: ("Вне исследования / NoData", "#111827"),
}

ROAD_LEGEND = {**INPUT_LEGEND, ClassCode.NEW_ROAD: ("Новая улица", "#ef4444")}


def terrain_image(terrain: Terrain) -> Image.Image:
    """Предпросмотр с целочисленным увеличением без смешивания классов."""
    return _classes_image(terrain.classes, INPUT_LEGEND)


def road_project_classes(terrain: Terrain, roads: RoadNetwork) -> np.ndarray:
    """Наложить новые улицы на копию исходного растра."""
    project = terrain.classes.copy()
    project[roads.new_roads] = ClassCode.NEW_ROAD
    project.setflags(write=False)
    return project


def road_project_image(terrain: Terrain, roads: RoadNetwork) -> Image.Image:
    return _classes_image(road_project_classes(terrain, roads), ROAD_LEGEND)


def _classes_image(classes: np.ndarray, legend: dict) -> Image.Image:
    rgb = np.zeros((*classes.shape, 3), dtype=np.uint8)
    for code, (_, color) in legend.items():
        rgb[classes == code] = tuple(bytes.fromhex(color[1:]))
    image = Image.fromarray(rgb)
    factor = max(1, 768 // max(image.size))
    return image.resize((image.width * factor, image.height * factor), Image.Resampling.NEAREST)


def terrain_areas(terrain: Terrain) -> list[dict]:
    """NoData имеет число ячеек, но исключён из площади исследования."""
    rows = []
    for code, (label, _) in INPUT_LEGEND.items():
        count = int(np.count_nonzero(terrain.classes == code))
        rows.append({
            "Код": int(code), "Класс": label, "Ячеек": count,
            "Площадь, м²": count * terrain.cell_area_m2 if code != ClassCode.NODATA else None,
        })
    return rows
