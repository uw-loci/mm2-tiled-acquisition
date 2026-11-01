import argparse
import re
from pathlib import Path

import pandas as pd

DEFAULT_SCAN_DIRECTORY = Path(r"C:\QPSC\Data\JOVE_PPM\shg_10x_4\890-445nm_2")
ROW_COLUMN_PATTERN = re.compile(r"_R(?P<row>\d+)_C(?P<column>\d+)$")
IMAGE_PATTERN = re.compile(
    r"_time(?P<time>\d+)_z(?P<z>\d+)\.tiff?$",
    re.IGNORECASE,
)


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


def build_dataframes(tiles):
    images = []
    for row, column, folder in tiles:
        for image in folder.iterdir():
            if not image.is_file() or image.suffix.lower() not in {".tif", ".tiff"}:
                continue

            match = IMAGE_PATTERN.search(image.name)
            if match is None:
                raise ValueError(f"Could not extract time and Z from image: {image}")

            images.append(
                {
                    "row": row,
                    "column": column,
                    "time": int(match["time"]),
                    "z": int(match["z"]),
                    "folder": folder.name,
                    "image": image.name,
                }
            )

    if not images:
        raise ValueError("No TIFF images found in the scan folders")

    image_dataframe = pd.DataFrame(images)
    timepoint_dataframe = (
        image_dataframe.groupby(["row", "column", "time"], as_index=False)
        .agg(
            image_count=("image", "count"),
            z_count=("z", "nunique"),
            z_positions=("z", lambda values: ", ".join(map(str, sorted(values.unique())))),
        )
        .sort_values(["row", "column", "time"])
        .reset_index(drop=True)
    )
    return image_dataframe, timepoint_dataframe


def main():
    parser = argparse.ArgumentParser(
        description="Summarize acquired TIFF images by row, column, time, and Z."
    )
    parser.add_argument(
        "scan_directory",
        nargs="?",
        type=Path,
        default=DEFAULT_SCAN_DIRECTORY,
        help=f"directory containing scan folders (default: {DEFAULT_SCAN_DIRECTORY})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="save the per-row/column/time summary dataframe to this CSV file",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="print every row of the per-row/column/time summary dataframe",
    )
    args = parser.parse_args()

    tiles = find_tiles(args.scan_directory)
    if not tiles:
        print(f"No scan folders found in {args.scan_directory}")
        return

    image_dataframe, timepoint_dataframe = build_dataframes(tiles)
    print(f"Scan folders: {len(tiles)}")
    print(f"Rows: {image_dataframe['row'].nunique()}")
    print(f"Columns: {image_dataframe['column'].nunique()}")
    print(f"Time points: {image_dataframe['time'].nunique()}")
    print(f"Z positions: {image_dataframe['z'].nunique()}")
    print(f"TIFF images: {len(image_dataframe)}")
    print("\nImages per row, column, and time point (including acquired Z positions):")
    if args.all_rows:
        print(timepoint_dataframe.to_string(index=False))
    else:
        print(timepoint_dataframe.head(30).to_string(index=False))
        if len(timepoint_dataframe) > 30:
            print(f"... {len(timepoint_dataframe) - 30} more rows; use --all-rows to display them.")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        timepoint_dataframe.to_csv(args.csv, index=False)
        print(f"\nSummary CSV written to: {args.csv}")


if __name__ == "__main__":
    main()
