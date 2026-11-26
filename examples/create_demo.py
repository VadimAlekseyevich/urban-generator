"""Создание синтетической демотерритории: python -m examples.create_demo."""

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import rasterio
from rasterio.transform import from_origin

from urban.model import ClassCode


DEMO_PATH = Path(__file__).with_name("demo.tif")


def build_demo_classes() -> NDArray[np.uint8]:
    """Фиксированная карта; seed относится к будущему сценарию, не к входу."""
    classes = np.full((96, 96), ClassCode.FREE, dtype=np.uint8)
    classes[:, 56:60] = ClassCode.BLOCKED  # Река без мостов, от края до края.
    classes[:4, :4] = ClassCode.NODATA
    classes[-4:, :4] = ClassCode.NODATA

    # Одна четырёхсвязная улица с перекрёстком на западном берегу.
    classes[12:84, 20] = ClassCode.EXISTING_ROAD
    classes[48, 8:49] = ClassCode.EXISTING_ROAD
    for row, column in ((18, 18), (30, 21), (42, 18), (49, 30), (68, 21)):
        classes[row:row + 2, column:column + 2] = ClassCode.EXISTING_BUILDING
    return classes


def write_demo(path: Path = DEMO_PATH) -> Path:
    """Записать одноканальный GeoTIFF, не требующий внешних данных."""
    classes = build_demo_classes()
    with rasterio.open(
        path, "w", driver="GTiff", height=96, width=96, count=1,
        dtype="uint8", crs="EPSG:32637",
        transform=from_origin(500_000, 6_200_000, 10, 10),
        nodata=int(ClassCode.NODATA), compress="deflate",
    ) as dataset:
        dataset.write(classes, 1)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Создать демо GeoTIFF 96×96, 10 м/пиксель")
    parser.add_argument("output", nargs="?", type=Path, default=DEMO_PATH)
    arguments = parser.parse_args()
    print(write_demo(arguments.output))


if __name__ == "__main__":
    main()
