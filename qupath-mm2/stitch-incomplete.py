import argparse
import re
from pathlib import Path

DEFAULT_SCAN_DIRECTORY = Path(r"C:\QPSC\Data\JOVE_PPM\shg_10x_4\890-445nm_2")
ROW_COLUMN_PATTERN = re.compile(r"_R(?P<row>\d+)_C(?P<column>\d+)$")


def find_tiles(scan_directory):
    if not scan_directory.is_dir():
        raise FileNotFoundError(f"Scan directory does not exist: {scan_directory}")

    tiles = []
    for folder in scan_directory.iterdir():
        if not folder.is_dir():
            continue

        match = ROW_COLUMN_PATTERN.search(folder.name)
        if match is None:
            raise ValueError(
                f"Could not extract row and column from folder: {folder.name}"
            )

        tiles.append((int(match["row"]), int(match["column"]), folder))

    return sorted(tiles, key=lambda tile: (tile[0], tile[1]))


def main():
    parser = argparse.ArgumentParser(
        description="List scan folders and their row/column positions."
    )
    parser.add_argument(
        "scan_directory",
        nargs="?",
        type=Path,
        default=DEFAULT_SCAN_DIRECTORY,
        help=f"directory containing scan folders (default: {DEFAULT_SCAN_DIRECTORY})",
    )
    args = parser.parse_args()

    tiles = find_tiles(args.scan_directory)
    if not tiles:
        print(f"No scan folders found in {args.scan_directory}")
        return

    for row, column, folder in tiles:
        print(f"Row {row}, column {column}: {folder.name}")


if __name__ == "__main__":
    main()
