"""Stitch time points of an incomplete scan into Z-summed, downsampled mosaics.

Tile XY positions come from the Micro-Manager .pos file (labels match folder names).
Each channel of each tile is summed over Z, block-averaged by the downsample factor,
then placed on the mosaic; overlapping pixels are averaged. One multichannel (CYX)
TIFF is written per time point.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import tifffile

DEFAULT_SCAN_DIRECTORY = Path(r"C:\QPSC\Data\JOVE_PPM\shg_10x_4\890-445nm_4")
DEFAULT_POS_FILE = Path(
    r"C:\QPSC\Data\JOVE_PPM\Brightfield_10x_4\46286_8529\TileConfiguration_Stage.pos"
)
PIXEL_SIZE_UM = 0.52
DOWNSAMPLE = 1
X_INVERT = True
Y_INVERT = False


def load_positions(pos_file):
    document = json.loads(pos_file.read_text(encoding="utf-8"))
    positions = {}
    for entry in document["map"]["StagePositions"]["array"]:
        xy_stage = next(
            device
            for device in entry["DevicePositions"]["array"]
            if device["Device"]["scalar"] == "XYStage"
        )
        x, y = xy_stage["Position_um"]["array"]
        positions[entry["Label"]["scalar"]] = (x, y)
    return positions


def find_time_points(scan_directory):
    pattern = re.compile(r"_time(\d+)_z\d+\.tiff?$", re.IGNORECASE)
    times = set()
    for folder in scan_directory.iterdir():
        if folder.is_dir():
            times.update(
                int(match[1])
                for path in folder.iterdir()
                if (match := pattern.search(path.name))
            )
    return sorted(times)


def find_channels(scan_directory):
    pattern = re.compile(r"channel(\d+)_", re.IGNORECASE)
    channels = set()
    for folder in scan_directory.iterdir():
        if folder.is_dir():
            channels.update(
                int(match[1])
                for path in folder.iterdir()
                if (match := pattern.search(path.name))
            )
    return sorted(channels)


def load_tile(folder, time_index, channels, downsample, x_invert=False, y_invert=False):
    """Sum each channel's Z planes, block-average and optionally flip.

    Returns (tile with shape (channels, y, x), {channel: number of Z planes}).
    Channels without images in this folder are left as NaN.
    """
    z_planes = {}
    tile = None
    for index, channel in enumerate(channels):
        pattern = re.compile(
            rf"channel{channel:03d}_.*_time{time_index:09d}_z\d+\.tiff?$", re.IGNORECASE
        )
        planes = [path for path in folder.iterdir() if pattern.search(path.name)]
        if not planes:
            continue

        total = None
        for path in planes:
            plane = tifffile.imread(path).astype(np.float32)
            total = plane if total is None else total + plane

        height, width = total.shape
        height -= height % downsample
        width -= width % downsample
        blocks = total[:height, :width].reshape(
            height // downsample, downsample, width // downsample, downsample
        )
        reduced = blocks.mean(axis=(1, 3))
        if x_invert:
            reduced = reduced[:, ::-1]
        if y_invert:
            reduced = reduced[::-1, :]

        if tile is None:
            tile = np.full((len(channels), *reduced.shape), np.nan, dtype=np.float32)
        tile[index] = reduced
        z_planes[channel] = len(planes)
    return tile, z_planes


def stitch(
    scan_directory,
    positions,
    time_index,
    channels,
    pixel_size_um,
    downsample,
    x_invert=False,
    y_invert=False,
):
    pixel_um = pixel_size_um * downsample
    folders = {p.name: p for p in scan_directory.iterdir() if p.is_dir()}

    tiles = []
    z_counts = {}
    for label, (x_um, y_um) in positions.items():
        folder = folders.get(label)
        if folder is None:
            continue
        tile, z_planes = load_tile(
            folder, time_index, channels, downsample, x_invert, y_invert
        )
        if tile is None:
            continue
        tiles.append((x_um, y_um, tile))
        for channel, count in z_planes.items():
            counts = z_counts.setdefault(channel, {})
            counts[count] = counts.get(count, 0) + 1

    if not tiles:
        raise ValueError(f"No tiles found for time point {time_index}")

    min_x = min(x for x, _, _ in tiles)
    min_y = min(y for _, y, _ in tiles)
    offsets = [
        (round((y - min_y) / pixel_um), round((x - min_x) / pixel_um), tile)
        for x, y, tile in tiles
    ]
    height = max(row + tile.shape[1] for row, _, tile in offsets)
    width = max(col + tile.shape[2] for _, col, tile in offsets)

    shape = (len(channels), height, width)
    accumulated = np.zeros(shape, dtype=np.float32)
    weight = np.zeros(shape, dtype=np.float32)
    for row, col, tile in offsets:
        window = (
            slice(None),
            slice(row, row + tile.shape[1]),
            slice(col, col + tile.shape[2]),
        )
        valid = ~np.isnan(tile)
        accumulated[window] += np.where(valid, tile, 0)
        weight[window] += valid

    mosaic = np.divide(
        accumulated, weight, out=np.zeros_like(accumulated), where=weight > 0
    )
    return mosaic, len(tiles), z_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-directory", type=Path, default=DEFAULT_SCAN_DIRECTORY)
    parser.add_argument("--pos-file", type=Path, default=DEFAULT_POS_FILE)
    parser.add_argument(
        "--time", type=int, default=0, help="time point to stitch (default: 0)"
    )
    parser.add_argument(
        "--all-times",
        action="store_true",
        help="stitch every time point found in the scan folders",
    )
    parser.add_argument(
        "--pixel-size", type=float, default=PIXEL_SIZE_UM, help="um per pixel"
    )
    parser.add_argument("--downsample", type=int, default=DOWNSAMPLE)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_SCAN_DIRECTORY)
    parser.add_argument(
        "--x-invert",
        action="store_true",
        default=X_INVERT,
        help="flip each tile horizontally before stitching",
    )
    parser.add_argument(
        "--y-invert",
        action="store_true",
        default=Y_INVERT,
        help="flip each tile vertically before stitching",
    )
    args = parser.parse_args()

    positions = load_positions(args.pos_file)
    time_points = (
        find_time_points(args.scan_directory) if args.all_times else [args.time]
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    flips = ("_xinv" if args.x_invert else "") + ("_yinv" if args.y_invert else "")
    pixel_um = args.pixel_size * args.downsample
    channels = find_channels(args.scan_directory)
    print(f"Channels found: {channels}")

    for time_index in time_points:
        mosaic, tile_count, z_counts = stitch(
            args.scan_directory,
            positions,
            time_index,
            channels,
            args.pixel_size,
            args.downsample,
            args.x_invert,
            args.y_invert,
        )
        output = args.output_directory / (
            f"stitched_time{time_index:03d}_sumZ_ds{args.downsample}{flips}.tif"
        )
        tifffile.imwrite(
            output,
            mosaic,
            imagej=True,
            resolution=(1.0 / pixel_um, 1.0 / pixel_um),
            metadata={"unit": "um", "axes": "CYX"},
        )
        print(f"Time point {time_index}: stitched {tile_count}/{len(positions)} tiles")
        for channel, counts in z_counts.items():
            print(f"  channel {channel} Z planes per tile (planes: tiles): {counts}")
        print(
            f"Mosaic {mosaic.shape[2]} x {mosaic.shape[1]} px, "
            f"{mosaic.shape[0]} channels at {pixel_um:.2f} um/px"
        )
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
