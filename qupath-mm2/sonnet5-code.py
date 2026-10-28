import json
import re
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

# Outputs go next to this script, named after it, regardless of the working directory
SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_POS = SCRIPT_PATH.with_suffix(".pos")
OUTPUT_Z_PLOT = SCRIPT_PATH.with_name(SCRIPT_PATH.stem + "_z_map.png")
OUTPUT_GRID_PLOT = SCRIPT_PATH.with_name(SCRIPT_PATH.stem + "_grid_map.png")

# --- 1. CONFIGURATION ---
# Define the 4 text file datasets with their catalog prefixes
dataset_files = {"L002_RL_Pr_HE_110": """
0.tif; ; (2673.400, -2679.700, 1757.980)
1.tif; ; (1469.800, -2679.700, 1758.000)
2.tif; ; (266.200, -2679.700, 1758.000)
3.tif; ; (-937.400, -2679.700, 1757.960)
4.tif; ; (-2141.000, -2679.700, 1757.960)
5.tif; ; (-3344.600, -2679.700, 1758.060)
6.tif; ; (-4548.300, -2679.700, 1758.060)
7.tif; ; (-9362.700, -1476.100, 1758.040)
8.tif; ; (-8159.100, -1476.100, 1758.020)
9.tif; ; (-6955.500, -1476.100, 1760.800)
10.tif; ; (-5751.900, -1476.100, 1760.740)
11.tif; ; (-4548.300, -1476.100, 1760.740)
12.tif; ; (-3344.600, -1476.100, 1760.740)
13.tif; ; (-2141.000, -1476.100, 1758.060)
14.tif; ; (-937.400, -1476.100, 1747.020)
15.tif; ; (266.200, -1476.100, 1746.960)
16.tif; ; (1469.800, -1476.100, 1758.060)
17.tif; ; (2673.400, -1476.100, 1758.060)
18.tif; ; (-3344.600, -272.500, 1746.980)
19.tif; ; (-4548.300, -272.500, 1760.800)
20.tif; ; (-5751.900, -272.500, 1758.300)
21.tif; ; (-6955.500, -272.500, 1760.820)
22.tif; ; (-8159.100, -272.500, 1760.780)
23.tif; ; (-9362.700, -272.500, 1765.380)
24.tif; ; (-10566.300, -272.500, 1765.320)
25.tif; ; (-11769.900, -272.500, 1765.300)
26.tif; ; (-12973.500, -272.500, 1765.380)
27.tif; ; (-14177.100, -272.500, 1765.400)
28.tif; ; (-15380.700, -272.500, 1765.360)
29.tif; ; (-15380.700, 931.100, 1765.360)
30.tif; ; (-14177.100, 931.100, 1765.360)
31.tif; ; (-12973.500, 931.100, 1765.320)
32.tif; ; (-11769.900, 931.100, 1765.400)
33.tif; ; (-10566.300, 931.100, 1765.360)
34.tif; ; (-9362.700, 931.100, 1765.380)
"""}

# Physical FOV size of the original tiles in micrometers
# Adjust if your standard full field width/height differs
ORIGINAL_FOV_WIDTH_UM = 1203.6
ORIGINAL_FOV_HEIGHT_UM = 1203.6

SCAN_SIZE_UM = 355.0  # new, smaller scan field of view
OVERLAP = 0.02  # 2% overlap between neighbouring scans
EDGE_TOLERANCE_UM = 0.5  # rounding slack when deciding which lattice points to keep

# --- 2. PARSE ALL COORDINATES ---
pattern = re.compile(r"(\d+)\.tif;\s*;\s*\(([-?\d.]+),\s*([-?\d.]+),\s*([-?\d.]+)\)")

all_entries = []
all_points_for_fit = []

for catalog, data_str in dataset_files.items():
    for line in data_str.strip().splitlines():
        match = pattern.search(line)
        if match:
            tile_id, x, y, z = match.groups()
            x_val, y_val, z_val = float(x), float(y), float(z)
            all_entries.append(
                {
                    "catalog": catalog,
                    "tile_id": int(tile_id),
                    "x": x_val,
                    "y": y_val,
                    "z": z_val,
                }
            )
            all_points_for_fit.append([x_val, y_val, z_val])

# --- 3. Z MODEL ---
# "plane": one global plane Z = a*X + b*Y + c fitted to all tiles
# "idw":   inverse-distance weighting between the nearest tile centers (passes
#          through the measured tile Z values, varies smoothly in between)
Z_MODE = "idw"
Z_OFFSET_UM = -15.0  # fixed offset added to every interpolated Z
PLOT_Z = True  # save z_map.png comparing plane vs. IDW Z
IDW_NEIGHBORS = 4
IDW_POWER = 2

pts = np.array(all_points_for_fit)
A = np.c_[pts[:, 0], pts[:, 1], np.ones(pts.shape[0])]
plane_coeffs, _, _, _ = np.linalg.lstsq(A, pts[:, 2], rcond=None)


def z_plane(x, y):
    return float(plane_coeffs[0] * x + plane_coeffs[1] * y + plane_coeffs[2])


def z_idw(x, y):
    d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
    idx = np.argsort(d)[:IDW_NEIGHBORS]
    if d[idx[0]] < 1e-6:
        return float(pts[idx[0], 2])
    w = 1.0 / d[idx] ** IDW_POWER
    return float(np.sum(w * pts[idx, 2]) / np.sum(w))


def interpolate_z(x, y, mode=None):
    base = z_idw(x, y) if (mode or Z_MODE) == "idw" else z_plane(x, y)
    return base + Z_OFFSET_UM


# --- 4. GENERATE 5x5 SUB-GRID POSITIONS WITH CATALOG LABELS ---
pitch_x = SCAN_SIZE_UM * (1 - OVERLAP)
pitch_y = SCAN_SIZE_UM * (1 - OVERLAP)

# One global lattice anchored at the top-left corner of the whole region, so
# neighbouring original tiles share lattice points instead of duplicating them.
origin_x = (
    min(e["x"] for e in all_entries) - ORIGINAL_FOV_WIDTH_UM / 2 + SCAN_SIZE_UM / 2
)
origin_y = (
    min(e["y"] for e in all_entries) - ORIGINAL_FOV_HEIGHT_UM / 2 + SCAN_SIZE_UM / 2
)

# A lattice point is kept if its scan footprint touches an original tile
# (center within the tile expanded by half a scan). The first tile claiming it owns it.
reach_x = ORIGINAL_FOV_WIDTH_UM / 2 + SCAN_SIZE_UM / 2 - EDGE_TOLERANCE_UM
reach_y = ORIGINAL_FOV_HEIGHT_UM / 2 + SCAN_SIZE_UM / 2 - EDGE_TOLERANCE_UM

lattice = {}  # (row, col) -> (entry, x, y)
for entry in all_entries:
    c_lo = int(np.floor((entry["x"] - reach_x - origin_x) / pitch_x))
    c_hi = int(np.ceil((entry["x"] + reach_x - origin_x) / pitch_x))
    r_lo = int(np.floor((entry["y"] - reach_y - origin_y) / pitch_y))
    r_hi = int(np.ceil((entry["y"] + reach_y - origin_y) / pitch_y))
    for r in range(r_lo, r_hi + 1):
        for c in range(c_lo, c_hi + 1):
            px, py = origin_x + c * pitch_x, origin_y + r * pitch_y
            if (
                abs(px - entry["x"]) <= reach_x
                and abs(py - entry["y"]) <= reach_y
                and (r, c) not in lattice
            ):
                lattice[(r, c)] = (entry, px, py)

# Serpentine order (row by row, alternating direction) to minimize stage travel
min_r = min(r for r, _ in lattice)
min_c = min(c for _, c in lattice)
ordered = sorted(
    lattice,
    key=lambda rc: (rc[0], rc[1] if (rc[0] - min_r) % 2 == 0 else -rc[1]),
)

stage_positions = []
total_pos_counter = 0

for gr, gc in ordered:
    entry, px, py = lattice[(gr, gc)]
    cat = entry["catalog"]
    tid = entry["tile_id"]
    r, c = gr - min_r, gc - min_c

    sub_x = round(px, 3)
    sub_y = round(py, 3)
    sub_z = round(interpolate_z(sub_x, sub_y), 3)

    # Label includes Catalog ID, owning Tile ID, and global Row/Column
    label_name = f"Cat{cat}_Tile{tid}_R{r}_C{c}"

    pos_entry = {
        "DefaultXYStage": {"type": "STRING", "scalar": "XYStage"},
        "DefaultZStage": {"type": "STRING", "scalar": "ZDrive"},
        "DevicePositions": {
            "type": "PROPERTY_MAP",
            "array": [
                {
                    "Device": {"type": "STRING", "scalar": "ZDrive"},
                    "Position_um": {"type": "DOUBLE", "array": [sub_z]},
                },
                {
                    "Device": {"type": "STRING", "scalar": "XYStage"},
                    "Position_um": {"type": "DOUBLE", "array": [sub_x, sub_y]},
                },
            ],
        },
        "GridCol": {"type": "INTEGER", "scalar": c},
        "GridRow": {"type": "INTEGER", "scalar": r},
        "Label": {"type": "STRING", "scalar": label_name},
        "Properties": {"type": "PROPERTY_MAP", "scalar": {}},
    }
    stage_positions.append(pos_entry)
    total_pos_counter += 1

# --- 5. EXPORT COMBINED JSON ---
combined_pos_structure = {
    "encoding": "UTF-8",
    "format": "Micro-Manager Property Map",
    "major_version": 2,
    "minor_version": 0,
    "map": {"StagePositions": {"type": "PROPERTY_MAP", "array": stage_positions}},
}

with open(OUTPUT_POS, "w", encoding="utf-8") as f:
    json.dump(combined_pos_structure, f, indent=2)

print(f"Successfully exported {total_pos_counter} positions")

# --- 6. VISUALIZE Z ---
if PLOT_Z:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    sx = np.array(
        [
            p["DevicePositions"]["array"][1]["Position_um"]["array"][0]
            for p in stage_positions
        ]
    )
    sy = np.array(
        [
            p["DevicePositions"]["array"][1]["Position_um"]["array"][1]
            for p in stage_positions
        ]
    )
    sz = np.array(
        [
            p["DevicePositions"]["array"][0]["Position_um"]["array"][0]
            for p in stage_positions
        ]
    )
    zp = np.array([interpolate_z(x, y, "plane") for x, y in zip(sx, sy)])
    zi = np.array([interpolate_z(x, y, "idw") for x, y in zip(sx, sy)])

    vmin = min(sz.min(), pts[:, 2].min() + Z_OFFSET_UM)
    vmax = max(sz.max(), pts[:, 2].max() + Z_OFFSET_UM)
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True, sharey=True)

    ax = axes[0]
    for t in pts:
        ax.add_patch(
            Rectangle(
                (t[0] - ORIGINAL_FOV_WIDTH_UM / 2, t[1] - ORIGINAL_FOV_HEIGHT_UM / 2),
                ORIGINAL_FOV_WIDTH_UM,
                ORIGINAL_FOV_HEIGHT_UM,
                fc=plt.cm.viridis((t[2] + Z_OFFSET_UM - vmin) / (vmax - vmin)),
                ec="k",
            )
        )
        ax.text(
            t[0],
            t[1],
            f"{t[2] + Z_OFFSET_UM:.1f}",
            ha="center",
            va="center",
            fontsize=7,
            color="w",
        )
    ax.set_title(f"Original tile Z (with {Z_OFFSET_UM:+.0f} um offset)")

    for ax, z, title in (
        (axes[1], zp, "Scan Z: global plane"),
        (axes[2], zi, "Scan Z: IDW between tiles"),
    ):
        sc = ax.scatter(
            sx, sy, c=z, s=18, marker="s", cmap="viridis", vmin=vmin, vmax=vmax
        )
        ax.set_title(f"{title}  (range {z.min():.1f}..{z.max():.1f} um)")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_ylabel("Stage Y (um)")
    axes[2].set_xlabel("Stage X (um)")
    fig.colorbar(sc, ax=axes, label="Z (um)", shrink=0.6)
    fig.savefig(OUTPUT_Z_PLOT, dpi=130)
    print(
        f"Saved {OUTPUT_Z_PLOT.name} (Z_MODE={Z_MODE}); exported Z range {sz.min():.1f}..{sz.max():.1f} um"
    )

    # --- Row/column map ---
    gr = np.array([p["GridRow"]["scalar"] for p in stage_positions])
    gc = np.array([p["GridCol"]["scalar"] for p in stage_positions])
    fig2, ax2 = plt.subplots(figsize=(30, 10))
    for t in pts:
        ax2.add_patch(
            Rectangle(
                (t[0] - ORIGINAL_FOV_WIDTH_UM / 2, t[1] - ORIGINAL_FOV_HEIGHT_UM / 2),
                ORIGINAL_FOV_WIDTH_UM,
                ORIGINAL_FOV_HEIGHT_UM,
                fill=False,
                ec="tab:blue",
                lw=1.0,
                ls="--",
                alpha=0.6,
            )
        )
    for x, y in zip(sx, sy):
        ax2.add_patch(
            Rectangle(
                (x - SCAN_SIZE_UM / 2, y - SCAN_SIZE_UM / 2),
                SCAN_SIZE_UM,
                SCAN_SIZE_UM,
                fill=False,
                ec="0.75",
                lw=0.5,
            )
        )
    ax2.plot(sx, sy, "-", color="tab:red", lw=0.5, alpha=0.5, zorder=2)  # acquisition order
    for x, y, r, c in zip(sx, sy, gr, gc):
        ax2.text(x, y, f"{r}\n{c}", ha="center", va="center", fontsize=6, zorder=4)

    # Row/column index ticks on the axes
    col_x = {c: x for c, x in zip(gc, sx)}
    row_y = {r: y for r, y in zip(gr, sy)}
    cs = sorted(col_x)
    rs = sorted(row_y)
    ax2.set_xticks([col_x[c] for c in cs])
    ax2.set_xticklabels([f"C{c}" for c in cs], fontsize=6, rotation=90)
    ax2.set_yticks([row_y[r] for r in rs])
    ax2.set_yticklabels([f"R{r}" for r in rs], fontsize=7)
    ax2.set_aspect("equal")
    ax2.autoscale()
    ax2.set_title(
        f"Grid map: {gr.max() + 1} rows x {gc.max() + 1} cols, {len(sx)} scans "
        f"(labels = row,col; dashed = original tiles; line = acquisition order)"
    )
    ax2.set_xlabel("Stage X (um)")
    ax2.set_ylabel("Stage Y (um)")
    fig2.tight_layout()
    fig2.savefig(OUTPUT_GRID_PLOT, dpi=150)
    print(f"Saved {OUTPUT_GRID_PLOT.name}")
