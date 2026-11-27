"""Отдельная пользовательская граница освоения; исходная территория не обрезается."""

from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
from rasterio.features import rasterize
from rasterio.warp import transform_geom

from urban.io import TerrainReadError
from urban.model import ClassCode, CoordinateMode, VectorSource
from urban.vector_import import _geometry, _parse


def clear_development_boundary(terrain):
    if terrain.development_source is None:
        return terrain
    return replace(terrain, development_mask=None, development_source=None,
                   transform=None if terrain.coordinate_mode == CoordinateMode.LOCAL else terrain.transform)


def apply_development_boundary(terrain, data: bytes, *, source_name="boundary.geojson"):
    document = _parse(data)
    features = document["features"]
    if (len(features) != 1 or not isinstance(features[0], dict)
            or features[0].get("type") != "Feature"
            or not isinstance(features[0].get("properties"), dict)
            or features[0]["properties"].get("role") != "boundary"):
        raise TerrainReadError("Нужен один Feature с properties.role=boundary: Polygon или MultiPolygon.")
    feature = features[0]
    geographic = terrain.coordinate_mode == CoordinateMode.GEOREFERENCED
    if not geographic and (document.get("coordinate_mode") != "local" or document.get("coordinate_unit") != "m"):
        raise TerrainReadError("Для локального TIFF задайте JSON границы в метрах: coordinate_mode=local, coordinate_unit=m.")
    if geographic and document.get("coordinate_mode") == "local":
        raise TerrainReadError("Для геопривязанной территории нужна граница WGS84, не локальные метры.")
    geometry = feature.get("geometry")
    _geometry(geometry, "boundary", geographic=geographic)
    projected = transform_geom("EPSG:4326", terrain.crs, geometry) if geographic else geometry
    shape = terrain.classes.shape
    inside = rasterize([projected], out_shape=shape, transform=terrain.transform,
                       all_touched=False, default_value=1, dtype="uint8").astype(bool)
    polygons = [projected["coordinates"]] if projected["type"] == "Polygon" else projected["coordinates"]
    edges = [{"type": "LineString", "coordinates": ring} for polygon in polygons for ring in polygon]
    boundary_cells = rasterize(edges, out_shape=shape, transform=terrain.transform,
                               all_touched=True, default_value=1, dtype="uint8").astype(bool)
    # Ячейки, которых касается контур (включая очень маленькие отверстия),
    # исключаются целиком: пятно не пересечёт векторную границу между центрами.
    allowed = inside & ~boundary_cells & terrain.valid_mask
    free = terrain.valid_mask & (terrain.classes == ClassCode.FREE)
    eligible = int(np.count_nonzero(allowed & free))
    metadata = {"source_kind": "user_supplied", "source_name": source_name,
                "source_sha256": sha256(data).hexdigest(), "rules_version": 1,
                "source": feature["properties"].get("source", "Автор сценария"),
                "source_coordinate_mode": "wgs84" if geographic else "local", "geometry_projected": projected,
                "projected_crs": terrain.crs.to_string() if geographic else None, "coordinate_unit": "m",
                "boundary_rule": "exclude_all_boundary_touched_cells",
                "eligible_free_area_m2": eligible * terrain.cell_area_m2,
                "eligible_free_percent": 100 * eligible / int(free.sum()) if free.any() else None}
    return replace(terrain, development_mask=allowed,
                   transform=None if terrain.coordinate_mode == CoordinateMode.LOCAL else terrain.transform,
                   development_source=VectorSource(source_name, data, json.dumps(metadata, ensure_ascii=False)))
