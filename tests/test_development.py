from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from rasterio.warp import transform_geom
from rasterio.crs import CRS
from rasterio.transform import from_origin
from streamlit.testing.v1 import AppTest

from urban.development import apply_development_boundary, clear_development_boundary
from urban.generate import generate_buildings
from urban.model import CoordinateMode, Scenario, Terrain
from urban.report import ReportValidationError, export_result_zip, validate_result
from urban.services import calculate_school_services
from urban.viewer import viewer_payload


ROOT = Path(__file__).resolve().parents[1]


def ring(x1,y1,x2,y2):
    return [[x1,y1],[x2,y1],[x2,y2],[x1,y2],[x1,y1]]


def boundary(geometry=None):
    geometry = geometry or {"type":"Polygon", "coordinates":[
        [[102,52],[348,52],[348,348],[252,348],[252,252],[102,252],[102,52]],
        ring(182,122,218,158),ring(133,183,137,187)]}
    return {"type":"FeatureCollection", "coordinate_mode":"local", "coordinate_unit":"m",
            "features":[{"type":"Feature","properties":{"role":"boundary","source":"Контрольный пользовательский полигон"},"geometry":geometry}]}


def encode(document):
    return json.dumps(document).encode()


def terrain():
    classes = np.zeros((40,40), dtype="uint8")
    classes[20,:]=2
    classes[21:31,26]=1
    classes[25:27,12:14]=3
    return Terrain(classes, np.ones_like(classes,dtype=bool), 10, CoordinateMode.LOCAL, "a"*64)


def completed(ground):
    plan=generate_buildings(ground,Scenario(target_population=20000))
    return validate_result(plan,calculate_school_services(plan))


def test_concavity_holes_barrier_and_existing_external_road():
    original=terrain();data=encode(boundary())
    limited=apply_development_boundary(original,data)
    report=completed(limited)
    new=(report.classes>=4)&(report.classes<=6)
    assert new.any() and report.metrics.new_buildings > 0
    assert not np.any(new & ~limited.development_mask)
    assert not limited.development_mask[21,13]  # маленькое отверстие внутри одной ячейки
    # Независимая геометрическая проверка всех новых ячеек на известном полигоне.
    for row,col in np.argwhere(new):
        x1,x2=col*10,(col+1)*10;y1,y2=400-(row+1)*10,400-row*10
        assert 102<=x1<x2<=348 and 52<=y1<y2<=348
        assert not (x1<252 and y2>252)
        for a,b,c,d in ((182,122,218,158),(133,183,137,187)):
            assert x2<=a or x1>=c or y2<=b or y1>=d
    assert np.all(report.classes[20,:]==2)  # коридор за границей сохранён целиком
    assert report.metrics.disconnected_road_cells==0
    np.testing.assert_array_equal(limited.classes,original.classes)
    assert not limited.development_mask.flags.writeable
    assert np.all(report.classes[original.classes==1]==1)
    assert report.metrics.population_shortfall > 0
    manifest=json.loads(limited.development_source.manifest_json)
    eligible=int(np.count_nonzero(limited.development_mask&(limited.classes==0)))
    assert manifest["eligible_free_area_m2"]==eligible*100
    assert len(viewer_payload(limited)["contours"])==3
    with ZipFile(BytesIO(export_result_zip(report))) as archive:
        assert archive.read("development-boundary.json")==data
        metadata=json.loads(archive.read("metrics.json"))["development_area"]
        assert metadata["source_sha256"]==sha256(data).hexdigest()
        assert metadata["source_kind"]=="user_supplied"


def test_empty_allowed_zone_explains_shortfall_and_default_is_unchanged():
    original=terrain()
    empty=apply_development_boundary(original,encode(boundary({"type":"Polygon","coordinates":[ring(501,501,601,601)]})))
    report=completed(empty)
    assert report.metrics.population==0 and report.metrics.population_shortfall==20000
    assert report.metrics.opened_schools==0 and report.metrics.new_road_cells==0
    assert json.loads(empty.development_source.manifest_json)["eligible_free_area_m2"]==0
    cleared=clear_development_boundary(empty)
    np.testing.assert_array_equal(completed(original).classes,completed(cleared).classes)


def test_independent_validation_rejects_an_unrestricted_plan_on_restricted_input():
    old=completed(terrain())
    limited=apply_development_boundary(old.plan.terrain,encode(boundary()))
    assert np.any((old.classes>=4)&(old.classes<=6)&~limited.development_mask)
    with pytest.raises(ReportValidationError,match="границы освоения"):
        validate_result(replace(old.plan,terrain=limited),old.allocation)


def test_local_coordinates_require_explicit_units():
    document=boundary();document.pop("coordinate_mode")
    with pytest.raises(ValueError,match="локального TIFF"):
        apply_development_boundary(terrain(),encode(document))


def test_geographic_boundary_matches_local_mask_and_is_in_gis_provenance():
    local=terrain()
    projected=replace(local,coordinate_mode=CoordinateMode.GEOREFERENCED,
                      crs=CRS.from_epsg(32637),transform=from_origin(500000,6200000,10,10))
    document=boundary()
    geometry=document["features"][0]["geometry"]
    shifted={"type":"Polygon","coordinates":[[[500000+x,6199600+y] for x,y in ring] for ring in geometry["coordinates"]]}
    document["features"][0]["geometry"]=transform_geom(projected.crs,"EPSG:4326",shifted)
    document.pop("coordinate_mode");document.pop("coordinate_unit")
    limited=apply_development_boundary(projected,encode(document))
    np.testing.assert_array_equal(limited.development_mask,apply_development_boundary(local,encode(boundary())).development_mask)
    with ZipFile(BytesIO(export_result_zip(completed(limited)))) as archive:
        geojson=json.loads(archive.read("project.geojson"))
        assert geojson["urban_generator"]["development_area"]["source_sha256"]==sha256(encode(document)).hexdigest()


def test_boundary_upload_resets_confirmation_and_clear_restores_base_input():
    app=AppTest.from_file(str(ROOT/"app.py")).run(timeout=15)
    app.checkbox(key="legend_confirmed").check().run()
    app.button[0].click().run()
    before=app.session_state.verified_report
    original=app.session_state.terrain
    geometry={"type":"Polygon","coordinates":[ring(500020,6199100,500500,6199980)]}
    document=boundary(transform_geom(original.crs,"EPSG:4326",geometry))
    document.pop("coordinate_mode");document.pop("coordinate_unit")
    app.file_uploader(key="development_boundary").set_value(("boundary.geojson",encode(document),"application/geo+json")).run()
    assert not app.exception and not app.error
    assert app.session_state.terrain.development_mask is not None
    assert app.session_state.verified_report is None
    assert not app.session_state.terrain_ready
    app.checkbox(key="legend_confirmed").check().run();app.button[0].click().run()
    assert not app.exception and not app.error
    limited=app.session_state.verified_report
    assert limited is not before
    app.file_uploader(key="development_boundary").clear().run()
    assert not app.exception and not app.error
    assert app.session_state.terrain.development_mask is None
    assert app.session_state.verified_report is None
    assert not app.session_state.terrain_ready
