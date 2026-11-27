"""Данные просмотра и выбор объектов только из завершённого отчёта."""

from numbers import Integral
import json

from urban.model import Terrain
from urban.report import SCHOOL_LEGEND, VerifiedReport


def object_at(report: VerifiedReport | None, row: int, column: int) -> dict | None:
    if (report is None or isinstance(row, bool) or isinstance(column, bool)
            or not isinstance(row, Integral) or not isinstance(column, Integral)
            or not 0 <= row < report.classes.shape[0]
            or not 0 <= column < report.classes.shape[1]):
        return None
    bid = int(report.plan.building_ids[row, column])
    if bid:
        building = next(b for b in report.plan.buildings if b.id == bid)
        service = next(b for b in report.allocation.buildings if b.building_id == bid)
        return {"kind": "house", "id": bid, "row": building.row, "column": building.column,
                "side": 2, "title": f"Дом №{bid}", "details":
                f"Жителей: {building.population} · Спрос: {building.school_demand} мест · "
                f"Обслужено: {service.served} · Дефицит: {service.remaining} · Этажей: {building.floors}"}
    for school in report.allocation.schools:
        if school.site.coordinate == (row, column):
            return {"kind": "school", "id": school.id, "row": row, "column": column,
                    "side": 1, "title": f"Школа №{school.id}", "details":
                    f"Вместимость: {school.capacity} мест · Занято: {school.served} · "
                    f"Свободно: {school.capacity - school.served}"}
    return None


def viewer_payload(terrain: Terrain, report: VerifiedReport | None = None) -> dict:
    height, width = terrain.classes.shape
    objects = {}
    ids = [0] * (height * width)
    if report is not None:
        for building in report.plan.buildings:
            key = f"house-{building.id}"
            objects[key] = object_at(report, building.row, building.column)
            for row in range(building.row, building.row + 2):
                for col in range(building.column, building.column + 2):
                    ids[row * width + col] = key
        for school in report.allocation.schools:
            row, col = school.site.coordinate
            key = f"school-{school.id}"
            objects[key] = object_at(report, row, col)
            ids[row * width + col] = key
    contours = []
    if terrain.development_source is not None:
        geometry = json.loads(terrain.development_source.manifest_json)["geometry_projected"]
        polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        inverse = ~terrain.transform
        contours = [[[inverse.a*x+inverse.b*y+inverse.c, inverse.d*x+inverse.e*y+inverse.f] for x,y in ring]
                    for polygon in polygons for ring in polygon]
    return {"width": width, "height": height, "step": terrain.pixel_size_m, "contours": contours,
            "transform": list(terrain.transform)[:6], "original": terrain.classes.ravel().tolist(),
            "project": report.classes.ravel().tolist() if report is not None else None,
            "objects": objects, "ids": ids,
            "palette": {int(code): {"label": label, "color": color}
                        for code, (label, color) in SCHOOL_LEGEND.items()}}
