"""Проверки загрузчика на реальных TIFF в памяти и контрольных ошибках."""

from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
import warnings

import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.transform import Affine, from_origin

from examples.create_demo import DEMO_PATH
from urban import io
from urban.io import TerrainReadError, read_terrain
from urban.model import MAX_INPUT_BYTES, ClassCode, CoordinateMode


def tiff_bytes(classes=None, *, mask=None, gcps=None, **changes) -> bytes:
    if classes is None:
        classes = np.array([[2, 0, 1], [3, 255, 0]], dtype=np.uint8)
    array = np.asarray(classes)
    if array.ndim == 2:
        array = array[np.newaxis]
    profile = {
        "driver": "GTiff", "height": array.shape[1], "width": array.shape[2],
        "count": array.shape[0], "dtype": array.dtype,
        "crs": "EPSG:32637", "transform": from_origin(500_000, 6_200_000, 10, 10),
        "nodata": 255, "compress": "deflate",
    }
    profile.update(changes)
    with warnings.catch_warnings(), rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with MemoryFile() as memory:
            with memory.open(**profile) as dataset:
                dataset.write(array)
                if mask is not None:
                    dataset.write_mask(np.asarray(mask, dtype=np.uint8) * 255)
                if gcps is not None:
                    dataset.gcps = (gcps, rasterio.crs.CRS.from_epsg(32637))
            return memory.read()


def test_demo_geotiff_and_equivalent_local_tiff() -> None:
    source = DEMO_PATH.read_bytes()
    geo = read_terrain(source)
    local_data = tiff_bytes(geo.classes, crs=None, transform=None)
    local = read_terrain(local_data, local_scale=10)
    np.testing.assert_array_equal(geo.classes, local.classes)
    np.testing.assert_array_equal(geo.valid_mask, local.valid_mask)
    assert geo.coordinate_mode == CoordinateMode.GEOREFERENCED
    assert geo.crs.to_epsg() == 32637
    assert geo.transform == from_origin(500_000, 6_200_000, 10, 10)
    assert local.coordinate_mode == CoordinateMode.LOCAL
    assert local.crs is None
    assert local.transform @ (0, 96) == (0, 0)
    assert local.pixel_size_m == geo.pixel_size_m == 10
    assert geo.input_sha256 == sha256(source).hexdigest()
    assert local.input_sha256 == sha256(local_data).hexdigest()
    assert geo.input_sha256 != local.input_sha256
    assert not geo.classes.flags.writeable


@pytest.mark.parametrize("scale", [5, 7.5, 10])
def test_plain_tiff_accepts_confirmed_local_scale(scale: float) -> None:
    terrain = read_terrain(tiff_bytes(crs=None, transform=None), local_scale=scale)
    assert terrain.pixel_size_m == scale
    assert terrain.transform @ (0, 2) == (0, 0)


@pytest.mark.parametrize("scale", [4.9, 10.1, True, "10", float("nan"), float("inf")])
def test_invalid_local_scale(scale: object) -> None:
    with pytest.raises(TerrainReadError, match="5 до 10"):
        read_terrain(tiff_bytes(crs=None, transform=None), local_scale=scale)


def test_plain_tiff_requires_scale() -> None:
    with pytest.raises(TerrainReadError, match="масштаб"):
        read_terrain(tiff_bytes(crs=None, transform=None))


@pytest.mark.parametrize("metadata", [
    {"crs": None}, {"transform": None}, {"crs": None, "transform": Affine.identity()},
])
def test_partial_georeference_requires_explicit_reset(metadata: dict) -> None:
    data = tiff_bytes(**metadata)
    for scale in (None, 10):
        with pytest.raises(TerrainReadError, match="неполная"):
            read_terrain(data, local_scale=scale)
    terrain = read_terrain(data, local_scale=10, discard_georeferencing=True)
    assert terrain.coordinate_mode == CoordinateMode.LOCAL
    assert terrain.crs is None
    assert terrain.transform == from_origin(0, 20, 10, 10)
    assert terrain.input_sha256 == sha256(data).hexdigest()


def test_full_georeference_also_needs_explicit_reset_to_go_local() -> None:
    data = tiff_bytes()
    with pytest.raises(TerrainReadError, match="явно сбросьте"):
        read_terrain(data, local_scale=10)
    with pytest.raises(TerrainReadError, match="укажите локальный масштаб"):
        read_terrain(data, discard_georeferencing=True)
    local = read_terrain(data, local_scale=5, discard_georeferencing=True)
    assert local.coordinate_mode == CoordinateMode.LOCAL
    assert local.transform == from_origin(0, 10, 5, 5)


def test_gcp_georeference_is_not_silently_treated_as_local() -> None:
    gcps = [GroundControlPoint(row=0, col=0, x=500_000, y=6_200_000)]
    data = tiff_bytes(crs=None, transform=None, gcps=gcps)
    with pytest.raises(TerrainReadError, match="GCP/RPC"):
        read_terrain(data, local_scale=10)
    assert read_terrain(data, local_scale=10, discard_georeferencing=True).crs is None


@pytest.mark.parametrize("metadata", [
    {"crs": "EPSG:4326"},
    {"crs": "EPSG:2263"},
    {"transform": Affine(10, 0.1, 0, 0, -10, 100)},
    {"transform": Affine(10, 0, 0, 0.1, -10, 100)},
    {"transform": Affine(10, 0, 0, 0, 10, 100)},
    {"transform": Affine(-10, 0, 0, 0, -10, 100)},
    {"transform": Affine(10, 0, 0, 0, -5, 100)},
    {"transform": from_origin(500_000, 6_200_000, 4, 4)},
    {"transform": from_origin(500_000, 6_200_000, 11, 11)},
])
def test_invalid_georeference_has_reprojection_hint(metadata: dict) -> None:
    with pytest.raises(TerrainReadError, match="ближайшего соседа"):
        read_terrain(tiff_bytes(**metadata))


def test_geographic_crs_is_reported_even_with_degree_pixel_size() -> None:
    data = tiff_bytes(crs="EPSG:4326", transform=from_origin(37, 56, 0.001, 0.001))
    with pytest.raises(TerrainReadError, match="проецированная CRS"):
        read_terrain(data)


@pytest.mark.parametrize("bigtiff", ["YES", "NO"])
@pytest.mark.parametrize("endianness", ["BIG", "LITTLE"])
def test_tiff_header_variants(bigtiff: str, endianness: str) -> None:
    data = tiff_bytes(BIGTIFF=bigtiff, ENDIANNESS=endianness)
    assert read_terrain(data).classes[0, 0] == ClassCode.EXISTING_ROAD


@pytest.mark.parametrize("size", [1, 256])
def test_raster_size_boundaries(size: int) -> None:
    data = tiff_bytes(np.full((size, size), ClassCode.EXISTING_ROAD, dtype=np.uint8))
    assert read_terrain(data).classes.shape == (size, size)


@pytest.mark.parametrize("classes, metadata, message", [
    (np.full((3, 2, 3), 2, dtype=np.uint8), {}, "один канал"),
    (np.array([[2, 9]], dtype=np.uint8), {}, "Коды вне легенды: 9"),
    (np.array([[2, 4]], dtype=np.uint8), {}, "Коды вне легенды: 4"),
    (np.array([[2, 0]], dtype=np.float32), {}, "целочисленный"),
    (np.zeros((2, 257), dtype=np.uint8), {}, "256"),
    (np.zeros((257, 2), dtype=np.uint8), {}, "256"),
    (np.zeros((2, 3), dtype=np.uint8), {}, "Нет исходной дороги"),
    (np.full((2, 3), 255, dtype=np.uint8), {}, "покрытия пуста"),
])
def test_invalid_raster(classes, metadata: dict, message: str) -> None:
    with pytest.raises(TerrainReadError, match=message):
        read_terrain(tiff_bytes(classes, **metadata))


@pytest.mark.parametrize("nodata", [0, 1, 2, 3])
def test_nodata_cannot_hide_a_semantic_class(nodata: int) -> None:
    with pytest.raises(TerrainReadError, match="Перенесите NoData в 255"):
        read_terrain(tiff_bytes(nodata=nodata))


def test_internal_mask_declared_nodata_and_reserved_255_are_combined() -> None:
    classes = np.array([[2, 0, 255], [9, 999, 3]], dtype=np.uint16)
    mask = np.array([[True, True, True], [False, True, True]])
    terrain = read_terrain(tiff_bytes(classes, mask=mask, nodata=999))
    np.testing.assert_array_equal(terrain.valid_mask, [[True, True, False], [False, False, True]])
    np.testing.assert_array_equal(terrain.classes, [[2, 0, 255], [255, 255, 3]])
    assert terrain.classes.dtype == np.uint8


def test_reserved_255_is_masked_without_nodata_metadata() -> None:
    terrain = read_terrain(tiff_bytes(nodata=None))
    assert not terrain.valid_mask[1, 1]
    assert terrain.valid_mask[0, 1]  # Свободная земля не пропадает.


def test_masked_road_and_empty_internal_mask_are_rejected() -> None:
    with pytest.raises(TerrainReadError, match="Нет исходной дороги"):
        read_terrain(tiff_bytes(mask=[[False, True, True], [True, True, True]]))
    with pytest.raises(TerrainReadError, match="покрытия пуста"):
        read_terrain(tiff_bytes(mask=np.zeros((2, 3), dtype=bool)))


def test_occupied_terrain_with_disconnected_roads_is_valid() -> None:
    terrain = read_terrain(tiff_bytes(np.array([[2, 3, 2], [3, 1, 3]], dtype=np.uint8)))
    assert not np.any(terrain.classes == ClassCode.FREE)
    assert np.count_nonzero(terrain.classes == ClassCode.EXISTING_ROAD) == 2


@pytest.mark.parametrize("payload", [b"", b"not a TIFF", b"II*\x00broken header"])
def test_invalid_file_has_russian_error(payload: bytes) -> None:
    with pytest.raises(TerrainReadError, match="TIFF"):
        read_terrain(payload)


def test_non_tiff_raster_is_rejected() -> None:
    output = BytesIO()
    Image.fromarray(np.array([[2, 0]], dtype=np.uint8)).save(output, format="PNG")
    with pytest.raises(TerrainReadError, match="Неизвестный формат"):
        read_terrain(output.getvalue())


def test_oversize_file_is_rejected_before_opening(monkeypatch) -> None:
    def unexpected_open(*args, **kwargs):
        pytest.fail("Нельзя открывать файл сверх лимита")
    monkeypatch.setattr(io, "MemoryFile", unexpected_open)
    with pytest.raises(TerrainReadError, match="8 МиБ"):
        read_terrain(b"II*\x00" + bytes(MAX_INPUT_BYTES - 3))


@pytest.mark.parametrize("changes, message", [
    ({"classes": np.zeros((2, 257), dtype=np.uint8)}, "256"),
    ({"classes": np.full((3, 2, 3), 2, dtype=np.uint8)}, "один канал"),
    ({"classes": np.array([[2, 0]], dtype=np.float32)}, "целочисленный"),
    ({"nodata": 0}, "NoData"),
    ({"crs": "EPSG:4326"}, "ближайшего соседа"),
    ({"crs": None}, "неполная"),
    ({"crs": None, "transform": None}, "масштаб"),
])
def test_invalid_header_is_rejected_before_pixel_or_mask_read(monkeypatch, changes: dict, message: str) -> None:
    data = tiff_bytes(**changes)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", NotGeoreferencedWarning)
        with MemoryFile(data) as memory, memory.open() as dataset:
            missing_georeference = any(
                issubclass(item.category, NotGeoreferencedWarning) for item in caught
            )
            class HeaderOnly:
                def __getattr__(self, name):
                    if name in ("read", "read_masks"):
                        pytest.fail("Данные нельзя читать до проверки заголовка")
                    return getattr(dataset, name)

            @contextmanager
            def guarded_dataset():
                if missing_georeference:
                    warnings.warn("Dataset has no geotransform", NotGeoreferencedWarning)
                yield HeaderOnly()

            @contextmanager
            def guarded_memory(payload):
                yield SimpleNamespace(open=guarded_dataset)

            monkeypatch.setattr(io, "MemoryFile", guarded_memory)
            with pytest.raises(TerrainReadError, match=message):
                read_terrain(data)


def test_truncated_pixel_data_has_readable_error() -> None:
    data = tiff_bytes()
    with Image.open(BytesIO(data)) as image:
        first_strip_offset = image.tag_v2[273][0]
    with pytest.raises(TerrainReadError, match="декодировать TIFF"):
        read_terrain(data[:first_strip_offset + 1])


def test_full_result_matches_between_geotiff_and_plain_tiff():
    from urban.generate import generate_buildings
    from urban.model import Scenario
    from urban.report import validate_result
    from urban.services import calculate_school_services

    geo = read_terrain(DEMO_PATH.read_bytes())
    plain = read_terrain(tiff_bytes(geo.classes, crs=None, transform=None), local_scale=10)
    assert geo.input_sha256 != plain.input_sha256
    assert geo.crs is not None and plain.crs is None
    reports = []
    for terrain in (geo, plain):
        plan = generate_buildings(terrain, Scenario())
        reports.append(validate_result(plan, calculate_school_services(plan)))
    a, b = reports
    np.testing.assert_array_equal(a.classes, b.classes)
    np.testing.assert_array_equal(a.plan.building_ids, b.plan.building_ids)
    np.testing.assert_array_equal(a.plan.roads.road_mask, b.plan.roads.road_mask)
    np.testing.assert_array_equal(a.plan.reserve.mask, b.plan.reserve.mask)
    np.testing.assert_array_equal(a.allocation.paths.distances_m, b.allocation.paths.distances_m)
    assert a.plan.buildings == b.plan.buildings
    assert a.allocation.schools == b.allocation.schools
    assert a.allocation.buildings == b.allocation.buildings
    assert a.metrics == b.metrics
