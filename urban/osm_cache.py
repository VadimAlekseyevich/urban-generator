"""Проверенный дисковый кэш окон PBF; исходный регион не копируется в память."""

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile
from zlib import error as ZlibError

from urban.io import read_terrain
from urban.model import MAX_INPUT_BYTES, Terrain
from urban.osm_import import (
    ATTRIBUTION, IMPORT_VERSION, Progress, Window, _hash_file, _silent, import_pbf,
)


CACHE_VERSION = 1
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".urban-cache" / "osm"


@dataclass(frozen=True, slots=True)
class PreparedOSM:
    terrain: Terrain
    data: bytes
    metadata_json: str
    cache_key: str
    reused: bool


def _fingerprint(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _validate(data: bytes, metadata: dict, request: dict, window: Window) -> Terrain:
    if not isinstance(metadata, dict):
        raise ValueError("Неверный формат метаданных OSM.")
    terrain = read_terrain(data)
    if (terrain.crs != window.crs or terrain.transform != window.transform
            or terrain.classes.shape != (window.size, window.size)
            or metadata.get("import_version") != IMPORT_VERSION
            or metadata.get("source_sha256") != request["source_sha256"]
            or metadata.get("source_name") != Path(request["source_path"]).name
            or metadata.get("attribution") != ATTRIBUTION
            or metadata.get("size") != window.size
            or metadata.get("pixel_size_m") != window.pixel_size_m
            or metadata.get("crs") != window.crs.to_string()
            or metadata.get("transform") != list(window.transform)[:6]
            or metadata.get("bounds_wgs84") != list(window.wgs84_bounds)
            or terrain.osm_import_json is None
            or json.loads(terrain.osm_import_json) != metadata):
        raise ValueError("Окно или происхождение кэша не соответствует запросу. Повторите импорт.")
    return terrain


def _read_cache(path: Path, request: dict, window: Window):
    try:
        with ZipFile(path) as archive:
            if (set(archive.namelist()) != {"terrain.tif", "manifest.json"}
                    or len(archive.namelist()) != 2
                    or archive.getinfo("terrain.tif").file_size > MAX_INPUT_BYTES
                    or archive.getinfo("manifest.json").file_size > 64 * 1024):
                return None
            manifest = json.loads(archive.read("manifest.json"))
            data = archive.read("terrain.tif")
        if (manifest["request"] != request or manifest["cache_version"] != CACHE_VERSION
                or manifest["tiff_sha256"] != sha256(data).hexdigest()):
            return None
        metadata = manifest["osm_import"]
        terrain = _validate(data, metadata, request, window)
        return terrain, data, metadata
    except (OSError, BadZipFile, ValueError, KeyError, TypeError, RecursionError, RuntimeError, ZlibError):
        # Повреждённый/неполный кэш не подменяет успешную территорию.
        return None


def prepare_pbf(path: Path, window: Window, *, cache_dir: Path | None = None,
                index: str = "flex_mem", threads: int = 2,
                progress: Progress = _silent) -> PreparedOSM:
    """Хешируется потоково даже на попадании в кэш (без повторного скана OSM).

    Ключ учитывает абсолютный путь, содержимое, версию классификатора и сетку.
    Потоки и тип индекса влияют на ресурсы импорта, а не на классы. Готовая
    запись — один ZIP; публикация os.replace атомарна. Любое исключение,
    включая KeyboardInterrupt/остановку Streamlit, удаляет временную запись.
    """
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file() or not path.name.lower().endswith(".osm.pbf"):
        raise ValueError("Укажите путь к существующему локальному файлу .osm.pbf.")
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 16:
        raise ValueError("Число потоков должно быть от 1 до 16.")
    before = _fingerprint(path)
    source_hash = _hash_file(path, progress)
    if before != _fingerprint(path):
        raise ValueError("PBF изменился во время чтения. Дождитесь окончания загрузки файла и повторите импорт.")
    request = {"source_path": str(path), "source_sha256": source_hash,
               "import_version": IMPORT_VERSION, "crs": window.crs.to_string(),
               "transform": list(window.transform)[:6], "size": window.size,
               "pixel_size_m": window.pixel_size_m}
    key = sha256(json.dumps({"cache_version": CACHE_VERSION, **request},
                            sort_keys=True, allow_nan=False).encode()).hexdigest()
    directory = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    target = directory / f"{key}.zip"
    cached = _read_cache(target, request, window)
    if cached is not None:
        if before != _fingerprint(path):
            raise ValueError("PBF изменился во время открытия кэша. Повторите импорт.")
        terrain, data, metadata = cached
        progress("Готовое окно найдено в кэше; повторное сканирование PBF не требуется.")
        return PreparedOSM(terrain, data, json.dumps(metadata, ensure_ascii=False), key, True)

    # Импортёр независимо хеширует исходник; сопоставление хешей ниже
    # исключает подмену файла между поиском кэша и сканированием.
    progress("Готового окна нет. Импортируем локальный PBF; процент сканирования неизвестен.")
    data, metadata = import_pbf(path, window, index=index, threads=threads, progress=progress)
    if before != _fingerprint(path):
        raise ValueError("PBF изменился во время импорта. Результат не сохранён; повторите импорт.")
    terrain = _validate(data, metadata, request, window)
    manifest = {"cache_version": CACHE_VERSION, "request": request,
                "tiff_sha256": sha256(data).hexdigest(), "osm_import": metadata}
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=directory, prefix="pending-", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            with ZipFile(file, "w", compression=ZIP_DEFLATED) as archive:
                archive.writestr("terrain.tif", data)
                archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, allow_nan=False))
            file.flush()
            os.fsync(file.fileno())
        # Закрытый файл важен для os.replace на Windows.
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    progress("Окно проверено и сохранено в кэше.")
    return PreparedOSM(terrain, data, json.dumps(metadata, ensure_ascii=False), key, False)
