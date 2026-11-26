"""Подготовить фрагмент города из локального .osm.pbf."""

import argparse
import json
from pathlib import Path

from urban.osm_import import import_pbf, make_window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pbf", type=Path, help="Локальный исходный .osm.pbf")
    parser.add_argument("--lon", required=True, type=float, help="Долгота центра WGS84")
    parser.add_argument("--lat", required=True, type=float, help="Широта центра WGS84")
    parser.add_argument("--size", type=int, default=256, help="Сторона в пикселях, 1–256")
    parser.add_argument("--pixel-size", type=float, default=10, help="Метры на пиксель, 5–10")
    parser.add_argument("--output", required=True, type=Path, help="Новый файл .tif")
    parser.add_argument("--index", default="flex_mem", help="Тип индекса Pyosmium (по умолчанию flex_mem)")
    args = parser.parse_args()
    if args.output.suffix.lower() not in (".tif", ".tiff"):
        parser.error("Выход должен иметь расширение .tif или .tiff.")
    sidecar = args.output.with_suffix(".osm.json")
    if args.output.exists() or sidecar.exists():
        parser.error("Выходной TIFF или JSON уже существует. Выберите новое имя.")
    try:
        window = make_window(args.lon, args.lat, args.size, args.pixel_size)
        print(f"Читаем локальный PBF: {args.pbf.name}; окно {window.crs}, {window.size}×{window.size}.",
              flush=True)
        print("Региональный файл сканируется целиком; дождитесь завершения процесса.", flush=True)
        data, meta = import_pbf(args.pbf, window, index=args.index)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as f:
            f.write(data)
        with sidecar.open("x", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except ImportError:
        parser.exit(2, "Нужен osmium: uv run --frozen --with osmium==4.3.2 python -m examples.import_osm ...\n")
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"Импорт не выполнен: {error}\n")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Готово: {args.output}. Загрузите TIFF через «Свой TIFF».")
    print("Проверьте соответствие карты территории; код 0 не означает разрешение на строительство.")


if __name__ == "__main__":
    main()
