"""GIS-выгрузка и независимое чтение/перепроекция через GDAL/OGR CLI."""
import csv
from io import BytesIO, StringIO
import json
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile

import numpy as np
import pytest
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from urban.generate import generate_buildings
from urban.io import read_terrain
from urban.model import Scenario
from urban.osm_cache import prepare_pbf
from urban.osm_import import ATTRIBUTION, make_window
from urban.report import export_result_zip, validate_result


from urban.services import calculate_school_services

ROOT = Path(__file__).resolve().parents[1]


def complete(terrain):
    plan = generate_buildings(terrain, Scenario())
    return validate_result(plan, calculate_school_services(plan))


def demo(local=False):
    return complete(read_terrain((ROOT/"examples/demo.tif").read_bytes(),
                                local_scale=10 if local else None, discard_georeferencing=local))


def unpack(report):
    archive = ZipFile(BytesIO(export_result_zip(report)))
    return archive, json.loads(archive.read("metrics.json"))


@pytest.mark.parametrize("local", [False, True])
def test_object_keys_population_areas_and_schools_match_all_reports(local):
    report = demo(local)
    archive,metadata=unpack(report)
    rows=list(csv.DictReader(StringIO(archive.read("objects.csv").decode())))
    by_id={row["object_id"]:row for row in rows}
    assert len(by_id)==len(rows)
    houses=[row for row in rows if row["role"]=="new_building"]
    schools=[row for row in rows if row["role"]=="new_school"]
    roads=[row for row in rows if row["role"]=="new_road"]
    assert len(houses)==report.metrics.new_buildings
    assert sum(int(row["population"]) for row in houses)==report.metrics.population
    assert sum(float(row["area_m2"]) for row in roads)==report.metrics.new_road_area_m2
    assert sum(float(row["area_m2"]) for row in houses)==report.metrics.building_area_m2
    assert len(schools)==report.metrics.opened_schools
    buildings=list(csv.DictReader(StringIO(archive.read("buildings.csv").decode())))
    assert {r["object_id"] for r in buildings}=={r["object_id"] for r in houses}
    for row in buildings:
        assert row["id"]==by_id[row["object_id"]]["building_id"]
        assert row["population"]==by_id[row["object_id"]]["population"]
    for school in metadata["schools"]:
        assert int(by_id[school["object_id"]]["school_served"])==school["served"]
    if local:
        assert "project.geojson" not in archive.namelist()
        assert metadata["spatial_export"]["crs"] is None
        assert metadata["spatial_export"]["coordinate_unit"]=="m"
        geometries=list(csv.DictReader(StringIO(archive.read("project-local-wkt.csv").decode())))
        assert {r["object_id"] for r in geometries}==set(by_id)
    else:
        collection=json.loads(archive.read("project.geojson"))
        assert collection["type"]=="FeatureCollection" and "crs" not in collection
        assert {f["id"] for f in collection["features"]}==set(by_id)
        assert collection["urban_generator"]["input_sha256"]==metadata["input_sha256"]
        assert all(f["properties"]["representation"]=="raster_cells"
                   for f in collection["features"] if f["properties"]["role"]=="new_road")


@pytest.mark.parametrize("local", [False, True])
def test_independent_ogr_reader_reprojects_exact_cell_geometry(tmp_path,local):
    if not shutil.which("ogrinfo") or not shutil.which("ogr2ogr"):
        pytest.skip("Отдельный GIS CI устанавливает GDAL CLI")
    report=demo(local)
    archive,metadata=unpack(report)
    name=metadata["spatial_export"]["file"]
    path=tmp_path/name;path.write_bytes(archive.read(name))
    info=json.loads(subprocess.check_output(["ogrinfo","-json","-al","-features",str(path)],text=True))
    layer=info["layers"][0]
    assert layer["featureCount"]==sum(metadata["spatial_export"]["counts"].values())
    if local:
        assert layer["geometryFields"][0].get("coordinateSystem") is None
        features=layer["features"]
    else:
        converted=json.loads(subprocess.check_output(["ogr2ogr","-f","GeoJSON","/vsistdout/",str(path),
                                                     "-t_srs",report.plan.terrain.crs.to_string()],text=True))
        features=converted["features"]
    houses=[f for f in features if f["properties"]["role"]=="new_building"]
    first=next(f for f in houses if f["properties"]["object_id"]=="house-1")
    building=report.plan.buildings[0]
    t=report.plan.terrain.transform
    np.testing.assert_allclose(first["geometry"]["coordinates"][0][0],
                               [t.c+building.column*t.a,t.f+building.row*t.e],atol=1e-5,rtol=0)
    # Независимый ридер → растр: геометрия улиц должна точно вернуть исходную маску новых улиц.
    road_shapes=[f["geometry"] for f in features if f["properties"]["role"]=="new_road"]
    rebuilt=rasterize(road_shapes,out_shape=report.classes.shape,transform=t,default_value=1)
    np.testing.assert_array_equal(rebuilt.astype(bool),report.plan.roads.new_roads)


@pytest.mark.parametrize("local", [False, True])
def test_empty_result_is_a_valid_empty_spatial_export(local):
    with MemoryFile() as mem:
        with mem.open(driver="GTiff",width=2,height=2,count=1,dtype="uint8",nodata=255,
                      crs="EPSG:32637",transform=from_origin(500000,6200000,10,10)) as ds:
            ds.write(np.array([[2,3],[1,255]],dtype="uint8"),1)
        terrain=read_terrain(mem.read(),local_scale=10 if local else None,discard_georeferencing=local)
    archive,metadata=unpack(complete(terrain))
    assert all(count==0 for count in metadata["spatial_export"]["counts"].values())
    assert list(csv.DictReader(StringIO(archive.read("objects.csv").decode())))==[]
    if not local:
        assert json.loads(archive.read("project.geojson"))["features"]==[]
    else:
        assert list(csv.DictReader(StringIO(archive.read("project-local-wkt.csv").decode())))==[]


def test_geojson_keeps_osm_attribution_and_input_identity(tmp_path,fake_osm_importer):
    pbf=tmp_path/"city.osm.pbf";pbf.write_bytes(b"source")
    prepared=prepare_pbf(pbf,make_window(39.72,54.62,64,10),cache_dir=tmp_path/"cache")
    archive,metadata=unpack(complete(prepared.terrain))
    source=json.loads(archive.read("project.geojson"))["urban_generator"]["source"]
    assert source==metadata["source"]==json.loads(prepared.metadata_json)
    assert source["attribution"]==ATTRIBUTION
