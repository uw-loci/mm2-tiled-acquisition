import json
import re
import numpy as np

# --- 1. CONFIGURATION ---
# Define the 4 text file datasets with their catalog prefixes
dataset_files = {
    "462": """
0.tif; ; (-5839.500, -683.800, 2094.380)
1.tif; ; (-5839.500, 519.800, 2094.340)
2.tif; ; (-7043.100, 519.800, 2094.320)
3.tif; ; (-8246.700, 519.800, 2094.320)
4.tif; ; (-9450.300, 519.800, 2094.320)
5.tif; ; (-10653.900, 519.800, 2094.360)
6.tif; ; (-13061.100, 1723.400, 2094.240)
7.tif; ; (-11857.500, 1723.400, 2094.360)
8.tif; ; (-10653.900, 1723.400, 2088.980)
9.tif; ; (-9450.300, 1723.400, 2088.900)
10.tif; ; (-8246.700, 1723.400, 2088.920)
11.tif; ; (-7043.100, 1723.400, 2092.220)
12.tif; ; (-11857.500, 2927.000, 2089.000)
13.tif; ; (-13061.100, 2927.000, 2088.920)
""",
    "570": """
0.tif; ; (-11057.000, -1735.700, 2090.040)
1.tif; ; (-12260.600, -1735.700, 2089.960)
2.tif; ; (-13464.200, -1735.700, 2089.940)
3.tif; ; (-14667.800, -532.100, 2089.920)
4.tif; ; (-13464.200, -532.100, 2089.920)
5.tif; ; (-12260.600, -532.100, 2089.920)
6.tif; ; (-11057.000, -532.100, 2089.920)
""",
    "702": """
0.tif; ; (-16789.500, -1372.700, 2082.300)
1.tif; ; (-15585.900, -1372.700, 2082.240)
2.tif; ; (-14382.300, -1372.700, 2082.220)
3.tif; ; (-14382.300, -169.100, 2082.220)
4.tif; ; (-15585.900, -169.100, 2082.200)
5.tif; ; (-16789.500, -169.100, 2082.160)
""",
    "278": """
0.tif; ; (448.400, -3398.100, 2104.820)
1.tif; ; (-755.200, -3398.100, 2104.760)
2.tif; ; (-1958.800, -3398.100, 2104.780)
3.tif; ; (-3162.500, -3398.100, 2104.760)
4.tif; ; (-4366.100, -3398.100, 2104.760)
5.tif; ; (-5569.700, -3398.100, 2104.760)
6.tif; ; (-6773.300, -3398.100, 2104.800)
7.tif; ; (-10384.100, -2194.500, 2089.180)
8.tif; ; (-9180.500, -2194.500, 2089.140)
9.tif; ; (-7976.900, -2194.500, 2089.120)
10.tif; ; (-6773.300, -2194.500, 2089.100)
11.tif; ; (-5569.700, -2194.500, 2089.120)
12.tif; ; (-4366.100, -2194.500, 2095.540)
13.tif; ; (-3162.500, -2194.500, 2095.480)
14.tif; ; (-1958.800, -2194.500, 2095.480)
15.tif; ; (-755.200, -2194.500, 2101.700)
16.tif; ; (448.400, -2194.500, 2104.800)
17.tif; ; (-6773.300, -990.900, 2095.520)
18.tif; ; (-7976.900, -990.900, 2092.400)
19.tif; ; (-9180.500, -990.900, 2092.340)
20.tif; ; (-10384.100, -990.900, 2089.160)
"""
}

# Physical FOV size of the original tiles in micrometers
# Adjust if your standard full field width/height differs
ORIGINAL_FOV_WIDTH_UM = 1203.6
ORIGINAL_FOV_HEIGHT_UM = 1203.6

GRID_DIM = 5  # 5x5 sub-positions per original FOV tile

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
            all_entries.append({
                "catalog": catalog,
                "tile_id": int(tile_id),
                "x": x_val,
                "y": y_val,
                "z": z_val
            })
            all_points_for_fit.append([x_val, y_val, z_val])

# --- 3. GLOBAL Z-INTERPOLATION PLANAR FIT ---
# Fits Z = a*X + b*Y + c across all combined tile coordinates
pts = np.array(all_points_for_fit)
A = np.c_[pts[:, 0], pts[:, 1], np.ones(pts.shape[0])]
plane_coeffs, _, _, _ = np.linalg.lstsq(A, pts[:, 2], rcond=None)

def interpolate_z(x, y):
    return float(plane_coeffs[0] * x + plane_coeffs[1] * y + plane_coeffs[2])

# --- 4. GENERATE 5x5 SUB-GRID POSITIONS WITH CATALOG LABELS ---
step_x = ORIGINAL_FOV_WIDTH_UM / GRID_DIM
step_y = ORIGINAL_FOV_HEIGHT_UM / GRID_DIM

x_offsets = np.linspace(-ORIGINAL_FOV_WIDTH_UM/2 + step_x/2, ORIGINAL_FOV_WIDTH_UM/2 - step_x/2, GRID_DIM)
y_offsets = np.linspace(-ORIGINAL_FOV_HEIGHT_UM/2 + step_y/2, ORIGINAL_FOV_HEIGHT_UM/2 - step_y/2, GRID_DIM)

stage_positions = []
total_pos_counter = 0

for entry in all_entries:
    cat = entry["catalog"]
    tid = entry["tile_id"]
    cx, cy = entry["x"], entry["y"]

    for r, dy in enumerate(y_offsets):
        for c, dx in enumerate(x_offsets):
            sub_x = round(cx + dx, 3)
            sub_y = round(cy + dy, 3)
            sub_z = round(interpolate_z(sub_x, sub_y), 3)

            # Label includes Catalog ID, Tile ID, Row, and Column
            label_name = f"Cat{cat}_Tile{tid}_R{r}_C{c}"

            pos_entry = {
                "DefaultXYStage": {"type": "STRING", "scalar": "XYStage"},
                "DefaultZStage": {"type": "STRING", "scalar": "ZDrive"},
                "DevicePositions": {
                    "type": "PROPERTY_MAP",
                    "array": [
                        {
                            "Device": {"type": "STRING", "scalar": "ZDrive"},
                            "Position_um": {"type": "DOUBLE", "array": [sub_z]}
                        },
                        {
                            "Device": {"type": "STRING", "scalar": "XYStage"},
                            "Position_um": {"type": "DOUBLE", "array": [sub_x, sub_y]}
                        }
                    ]
                },
                "GridCol": {"type": "INTEGER", "scalar": c},
                "GridRow": {"type": "INTEGER", "scalar": r},
                "Label": {"type": "STRING", "scalar": label_name},
                "Properties": {"type": "PROPERTY_MAP", "scalar": {}}
            }
            stage_positions.append(pos_entry)
            total_pos_counter += 1

# --- 5. EXPORT COMBINED JSON ---
combined_pos_structure = {
    "encoding": "UTF-8",
    "format": "Micro-Manager Property Map",
    "major_version": 2,
    "minor_version": 0,
    "map": {
        "StagePositions": {
            "type": "PROPERTY_MAP",
            "array": stage_positions
        }
    }
}

with open("combined_positions.pos", "w", encoding="utf-8") as f:
    json.dump(combined_pos_structure, f, indent=2)

print(f"Successfully exported {total_pos_counter} positions across catalogs 462, 570, 702, and 278.")