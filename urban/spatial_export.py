"""Геометрия проверенного проекта: WGS84 GeoJSON или локальный метрический WKT."""

import csv
from io import StringIO
import json

import numpy as np
from rasterio.features import shapes
from rasterio.warp import transform_geom

from urban.model import CoordinateMode


OBJECT_FIELDS = ("object_id", "role", "building_id", "school_id", "area_m2",
                 "population", "school_demand", "school_served", "capacity", "representation")


def _square(terrain, row, column, side=1):
    corners = [(column, row), (column, row+side), (column+side, row+side),
               (column+side, row), (column, row)]
    t = terrain.transform
    return {"type": "Polygon", "coordinates": [[[t.a*x+t.b*y+t.c, t.d*x+t.e*y+t.f] for x,y in corners]]}


def _area(geometry):
    total = 0.0
    for index, ring in enumerate(geometry["coordinates"]):
        points = np.asarray(ring, dtype=float)
        points -= points[0].copy()
        area = abs(np.sum(points[:-1,0]*points[1:,1] - points[1:,0]*points[:-1,1])) / 2
        total += area if index == 0 else -area
    return total


def project_features(report):
    """Новые объекты в метрической сетке Terrain, без исходных зданий и улиц."""
    terrain = report.plan.terrain
    features = []
    def add(object_id, role, geometry, **properties):
        features.append({"type": "Feature", "id": object_id, "geometry": geometry,
                         "properties": {"object_id": object_id, "role": role, **properties}})
    roads = report.plan.roads.new_roads.astype("uint8")
    for index, (geometry, _) in enumerate(shapes(roads, mask=roads.astype(bool),
                                                transform=terrain.transform, connectivity=4), 1):
        cells = round(_area(geometry) / terrain.cell_area_m2)
        add(f"road-{index}", "new_road", geometry, area_m2=cells*terrain.cell_area_m2,
            cell_count=cells, representation="raster_cells")
    services = {b.building_id: b for b in report.allocation.buildings}
    for building in report.plan.buildings:
        add(f"house-{building.id}", "new_building", _square(terrain, building.row, building.column, 2),
            building_id=building.id, area_m2=building.footprint_area_m2, population=building.population,
            floors=building.floors, school_demand=building.school_demand,
            school_served=services[building.id].served, representation="raster_footprint")
    for school in report.allocation.schools:
        add(f"school-{school.id}", "new_school", _square(terrain, *school.site.coordinate),
            school_id=school.id, area_m2=terrain.cell_area_m2, capacity=school.capacity,
            school_served=school.served, representation="raster_school_site")
    return features


def _table(features, *, wkt=False):
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=OBJECT_FIELDS + (("WKT",) if wkt else ()),
                            lineterminator="\n")
    writer.writeheader()
    for feature in features:
        row = {key: feature["properties"].get(key) for key in OBJECT_FIELDS}
        if wkt:
            rings = ["(" + ", ".join(f"{x:.12g} {y:.12g}" for x,y in ring) + ")"
                     for ring in feature["geometry"]["coordinates"]]
            row["WKT"] = "POLYGON (" + ", ".join(rings) + ")"
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def export_spatial(report, metadata):
    """Получает уже независимо проверенный отчёт от экспортёра ZIP."""
    terrain = report.plan.terrain
    features = project_features(report)
    counts = dict.fromkeys(("new_road", "new_building", "new_school"), 0)
    for feature in features:
        role = feature["properties"]["role"]
        counts[role] = counts.get(role, 0) + 1
    descriptor = {"objects_table": "objects.csv", "join_key": "object_id", "counts": counts,
                  "road_representation": "raster_cells", "area_unit": "m2"}
    members = {"objects.csv": _table(features)}
    if terrain.coordinate_mode == CoordinateMode.GEOREFERENCED:
        descriptor.update(file="project.geojson", format="GeoJSON", crs="EPSG:4326", coordinate_unit="degree")
        geographic = [{**f, "geometry": transform_geom(terrain.crs, "EPSG:4326", f["geometry"], precision=-1)}
                      for f in features]
        collection = {"type": "FeatureCollection", "features": geographic,
                      "urban_generator": {"schema_version": metadata["schema_version"],
                                          "input_sha256": terrain.input_sha256, "source": metadata["source"],
                                          "metrics": metadata["metrics"], "parameters": metadata["parameters"],
                                          "spatial_export": descriptor}}
        members["project.geojson"] = (json.dumps(collection, ensure_ascii=False, allow_nan=False)+"\n").encode("utf-8")
    else:
        descriptor.update(file="project-local-wkt.csv", format="CSV/WKT", crs=None,
                          coordinate_unit="m", origin=[0,0], origin_corner="lower_left")
        members["project-local-wkt.csv"] = _table(features, wkt=True)
    return members, descriptor
