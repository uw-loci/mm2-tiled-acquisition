"""Visualize a TileConfiguration_Stage.txt file and compute the tile grid size.

Usage: python visualize_tile_config.py [path] [--save out.png]
Requires: numpy, matplotlib
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

LINE_RE = re.compile(
    r"^(?P<name>[^;]+);\s*;\s*\(\s*(?P<x>[-\d.]+)\s*,\s*(?P<y>[-\d.]+)(?:\s*,\s*(?P<z>[-\d.]+))?\s*\)"
)


def parse(path):
    names, pts = [], []
    for line in Path(path).read_text().splitlines():
        m = LINE_RE.match(line.strip())
        if m:
            names.append(m["name"].strip())
            pts.append((float(m["x"]), float(m["y"]), float(m["z"] or 0)))
    return names, np.array(pts)


def cluster(values, tol):
    """Return sorted cluster centers of 1-D values within tol of each other."""
    vals = np.sort(values)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return np.array([np.mean(g) for g in groups])


def step_size(centers):
    d = np.diff(centers)
    return float(np.median(d)) if len(d) else 0.0


def analyze(pts):
    x, y = pts[:, 0], pts[:, 1]
    # Estimate step from the smallest typical nonzero gap between sorted unique coords
    def raw_step(v):
        d = np.diff(np.sort(np.unique(np.round(v, 1))))
        d = d[d > 1]
        return float(np.median(d)) if len(d) else 0.0

    sx, sy = raw_step(x), raw_step(y)
    xs = cluster(x, sx * 0.25 if sx else 1)
    ys = cluster(y, sy * 0.25 if sy else 1)
    step_x, step_y = step_size(xs), step_size(ys)
    # Grid indices relative to the min coordinate
    ix = np.round((x - x.min()) / step_x).astype(int) if step_x else np.zeros(len(x), int)
    iy = np.round((y - y.min()) / step_y).astype(int) if step_y else np.zeros(len(y), int)
    return dict(
        step_x=step_x, step_y=step_y,
        cols=int(ix.max()) + 1, rows=int(iy.max()) + 1,
        ix=ix, iy=iy,
        unique_cols=len(xs), unique_rows=len(ys),
    )


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=here / "TileConfiguration_Stage.txt")
    ap.add_argument("--save", help="save figure to this file instead of showing")
    args = ap.parse_args()

    names, pts = parse(args.path)
    r = analyze(pts)
    n = len(pts)
    cols, rows = r["cols"], r["rows"]
    print(f"Tiles:            {n}")
    print(f"Step (um):        X={r['step_x']:.1f}  Y={r['step_y']:.1f}")
    print(f"Grid size:        {cols} columns x {rows} rows (bounding grid, {cols * rows} cells)")
    print(f"Unique X / Y:     {r['unique_cols']} / {r['unique_rows']}")
    print(f"Empty cells:      {cols * rows - n}")
    print(f"Extent (um):      X {pts[:,0].min():.1f}..{pts[:,0].max():.1f}  "
          f"Y {pts[:,1].min():.1f}..{pts[:,1].max():.1f}")
    print(f"Z range (um):     {pts[:,2].min():.2f}..{pts[:,2].max():.2f}")
    for j in range(rows):
        print(f"  row {j}: {int((r['iy'] == j).sum())} tiles")

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 1]})
    for p in pts:
        ax.add_patch(Rectangle((p[0] - r["step_x"] / 2, p[1] - r["step_y"] / 2),
                               r["step_x"], r["step_y"], fill=False, ec="0.7", lw=0.8))
    ax.plot(pts[:, 0], pts[:, 1], "-", color="tab:blue", lw=0.8, alpha=0.6, label="acquisition order")
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=np.arange(n), cmap="viridis", zorder=3)
    for i, p in enumerate(pts):
        ax.annotate(str(i), (p[0], p[1]), ha="center", va="center", fontsize=7,
                    color="white", zorder=4)
    fig.colorbar(sc, ax=ax, label="tile index")
    ax.set_aspect("equal")
    ax.set_xlabel("Stage X (um)")
    ax.set_ylabel("Stage Y (um)")
    ax.set_title(f"{n} tiles, grid {cols} x {rows} (cols x rows)")
    ax.legend(loc="upper right")

    axz.plot(np.arange(n), pts[:, 2], ".-")
    axz.set_xlabel("tile index")
    axz.set_ylabel("Stage Z (um)")
    axz.set_title("Z per tile")
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
