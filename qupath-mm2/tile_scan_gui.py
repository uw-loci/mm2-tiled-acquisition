"""Minimal GUI: TileConfiguration_Stage.txt -> Micro-Manager .pos + Z-map / grid-map PNGs.
Requires: numpy, matplotlib (tkinter ships with Python)."""
import json
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

PAT = re.compile(r"(\d+)\.tif;\s*;\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)")


def load_tiles(path):
    rows = [(int(i), float(x), float(y), float(z)) for i, x, y, z in PAT.findall(Path(path).read_text())]
    if not rows:
        raise ValueError("No tile lines found in file")
    return rows


def auto_fov(tiles):
    def step(v):
        d = np.diff(np.unique(np.round(v, 1)))
        d = d[d > 1]
        return float(d.min()) if len(d) else 1000.0
    a = np.array(tiles)
    return step(a[:, 1]), step(a[:, 2])


def run(p):
    tiles = load_tiles(p["file"])
    pts = np.array([t[1:] for t in tiles])
    fw, fh = p["fov_w"], p["fov_h"]
    if fw is None or fh is None:
        aw, ah = auto_fov(tiles)
        fw, fh = fw or aw, fh or ah
    scan, pitch = p["scan"], p["scan"] * (1 - p["overlap"])
    off, k, pw = p["z_offset"], p["idw_k"], p["idw_power"]

    A = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
    plane = np.linalg.lstsq(A, pts[:, 2], rcond=None)[0]

    def z_at(x, y, mode):
        if mode == "plane":
            return float(plane[0] * x + plane[1] * y + plane[2]) + off
        d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        idx = np.argsort(d)[:k]
        if d[idx[0]] < 1e-6:
            return float(pts[idx[0], 2]) + off
        w = 1.0 / d[idx] ** pw
        return float(np.sum(w * pts[idx, 2]) / w.sum()) + off

    ox = pts[:, 0].min() - fw / 2 + scan / 2
    oy = pts[:, 1].min() - fh / 2 + scan / 2
    rx, ry = fw / 2 + scan / 2 - p["tol"], fh / 2 + scan / 2 - p["tol"]
    lattice = {}
    for tid, tx, ty, _ in tiles:
        for r in range(int(np.floor((ty - ry - oy) / pitch)), int(np.ceil((ty + ry - oy) / pitch)) + 1):
            for c in range(int(np.floor((tx - rx - ox) / pitch)), int(np.ceil((tx + rx - ox) / pitch)) + 1):
                px, py = ox + c * pitch, oy + r * pitch
                if abs(px - tx) <= rx and abs(py - ty) <= ry and (r, c) not in lattice:
                    lattice[(r, c)] = (tid, px, py)
    min_r = min(r for r, _ in lattice)
    min_c = min(c for _, c in lattice)
    order = sorted(lattice, key=lambda rc: (rc[0], rc[1] if (rc[0] - min_r) % 2 == 0 else -rc[1]))

    cat = p["catalog"]
    out = Path(p["out_dir"]) / p["base"]
    sx, sy, sz, gr, gc, positions = [], [], [], [], [], []
    for r0, c0 in order:
        tid, px, py = lattice[(r0, c0)]
        x, y = round(px, 3), round(py, 3)
        z = round(z_at(x, y, p["z_mode"]), 3)
        r, c = r0 - min_r, c0 - min_c
        sx.append(x); sy.append(y); sz.append(z); gr.append(r); gc.append(c)
        positions.append({
            "DefaultXYStage": {"type": "STRING", "scalar": "XYStage"},
            "DefaultZStage": {"type": "STRING", "scalar": "ZDrive"},
            "DevicePositions": {"type": "PROPERTY_MAP", "array": [
                {"Device": {"type": "STRING", "scalar": "ZDrive"}, "Position_um": {"type": "DOUBLE", "array": [z]}},
                {"Device": {"type": "STRING", "scalar": "XYStage"}, "Position_um": {"type": "DOUBLE", "array": [x, y]}},
            ]},
            "GridCol": {"type": "INTEGER", "scalar": c},
            "GridRow": {"type": "INTEGER", "scalar": r},
            "Label": {"type": "STRING", "scalar": f"Cat{cat}_Tile{tid}_R{r}_C{c}"},
            "Properties": {"type": "PROPERTY_MAP", "scalar": {}},
        })
    sx, sy, sz, gr, gc = map(np.array, (sx, sy, sz, gr, gc))
    msgs = [f"{len(positions)} positions (FOV {fw:.1f} x {fh:.1f} um)"]

    if p["save_pos"]:
        doc = {"encoding": "UTF-8", "format": "Micro-Manager Property Map", "major_version": 2,
               "minor_version": 0, "map": {"StagePositions": {"type": "PROPERTY_MAP", "array": positions}}}
        f = out.with_suffix(".pos")
        f.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        msgs.append(f.name)

    if p["save_z"]:
        zp = [z_at(x, y, "plane") for x, y in zip(sx, sy)]
        zi = [z_at(x, y, "idw") for x, y in zip(sx, sy)]
        vmin = min(sz.min(), pts[:, 2].min() + off)
        vmax = max(sz.max(), pts[:, 2].max() + off)
        fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True, sharey=True)
        for t in pts:
            axes[0].add_patch(Rectangle((t[0] - fw / 2, t[1] - fh / 2), fw, fh, ec="k",
                                        fc=plt.cm.viridis((t[2] + off - vmin) / ((vmax - vmin) or 1))))
            axes[0].text(t[0], t[1], f"{t[2] + off:.1f}", ha="center", va="center", fontsize=7, color="w")
        axes[0].set_title(f"Original tile Z ({off:+.0f} um offset)")
        for ax, z, title in ((axes[1], zp, "Scan Z: global plane"), (axes[2], zi, "Scan Z: IDW")):
            sc = ax.scatter(sx, sy, c=z, s=18, marker="s", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"{title} (range {min(z):.1f}..{max(z):.1f} um)")
        for ax in axes:
            ax.set_aspect("equal"); ax.set_ylabel("Stage Y (um)")
        axes[2].set_xlabel("Stage X (um)")
        fig.colorbar(sc, ax=axes, label="Z (um)", shrink=0.6)
        f = out.with_name(out.name + "_z_map.png")
        fig.savefig(f, dpi=130); plt.close(fig)
        msgs.append(f.name)

    if p["save_grid"]:
        fig, ax = plt.subplots(figsize=(30, 10))
        for t in pts:
            ax.add_patch(Rectangle((t[0] - fw / 2, t[1] - fh / 2), fw, fh, fill=False,
                                   ec="tab:blue", lw=1, ls="--", alpha=0.6))
        for x, y in zip(sx, sy):
            ax.add_patch(Rectangle((x - scan / 2, y - scan / 2), scan, scan, fill=False, ec="0.75", lw=0.5))
        ax.plot(sx, sy, "-", color="tab:red", lw=0.5, alpha=0.5)
        for x, y, r, c in zip(sx, sy, gr, gc):
            ax.text(x, y, f"{r}\n{c}", ha="center", va="center", fontsize=6)
        ax.autoscale(); ax.set_aspect("equal")
        ax.set_title(f"Grid map: {gr.max() + 1} rows x {gc.max() + 1} cols, {len(sx)} scans")
        ax.set_xlabel("Stage X (um)"); ax.set_ylabel("Stage Y (um)")
        fig.tight_layout()
        f = out.with_name(out.name + "_grid_map.png")
        fig.savefig(f, dpi=150); plt.close(fig)
        msgs.append(f.name)
    return msgs


def main():
    root = tk.Tk()
    root.title("Tile scan generator")
    frm = ttk.Frame(root, padding=10)
    frm.pack()
    v = {k: tk.StringVar(value=d) for k, d in dict(
        file="", out_dir="", base="scan", catalog="", fov_w="", fov_h="", scan="355.0", overlap="0.02",
        tol="0.5", z_mode="idw", z_offset="-15.0", idw_k="4", idw_power="2").items()}
    b = {k: tk.BooleanVar(value=True) for k in ("save_pos", "save_z", "save_grid")}

    def browse_file():
        f = filedialog.askopenfilename(filetypes=[("Tile config", "*.txt"), ("All", "*.*")])
        if f:
            v["file"].set(f)
            v["out_dir"].set(v["out_dir"].get() or str(Path(f).parent))
            v["catalog"].set(v["catalog"].get() or Path(f).stem)
            v["base"].set(Path(f).stem)

    def browse_dir():
        d = filedialog.askdirectory()
        if d:
            v["out_dir"].set(d)

    rows = [("Tile config file", "file", browse_file), ("Output folder", "out_dir", browse_dir),
            ("Output base name", "base", None), ("Catalog label", "catalog", None),
            ("Tile FOV width um (blank=auto)", "fov_w", None), ("Tile FOV height um (blank=auto)", "fov_h", None),
            ("Scan size um", "scan", None), ("Overlap (0-1)", "overlap", None),
            ("Edge tolerance um", "tol", None), ("Z offset um", "z_offset", None),
            ("IDW neighbors", "idw_k", None), ("IDW power", "idw_power", None)]
    for i, (lbl, key, cmd) in enumerate(rows):
        ttk.Label(frm, text=lbl).grid(row=i, column=0, sticky="w", pady=2)
        ttk.Entry(frm, textvariable=v[key], width=50 if cmd else 14).grid(row=i, column=1, sticky="w")
        if cmd:
            ttk.Button(frm, text="...", width=3, command=cmd).grid(row=i, column=2)
    n = len(rows)
    ttk.Label(frm, text="Z mode").grid(row=n, column=0, sticky="w")
    ttk.Combobox(frm, textvariable=v["z_mode"], values=["idw", "plane"], state="readonly", width=11).grid(row=n, column=1, sticky="w")
    for j, (k, t) in enumerate((("save_pos", "Save .pos"), ("save_z", "Save Z-map PNG"), ("save_grid", "Save grid-map PNG"))):
        ttk.Checkbutton(frm, text=t, variable=b[k]).grid(row=n + 1 + j, column=1, sticky="w")

    def go():
        try:
            opt = lambda k: float(v[k].get()) if v[k].get().strip() else None
            p = dict(file=v["file"].get(), out_dir=v["out_dir"].get(), base=v["base"].get() or "scan",
                     catalog=v["catalog"].get() or "Tile", fov_w=opt("fov_w"), fov_h=opt("fov_h"),
                     scan=float(v["scan"].get()), overlap=float(v["overlap"].get()), tol=float(v["tol"].get()),
                     z_mode=v["z_mode"].get(), z_offset=float(v["z_offset"].get()),
                     idw_k=int(v["idw_k"].get()), idw_power=float(v["idw_power"].get()),
                     **{k: x.get() for k, x in b.items()})
            if not Path(p["file"]).is_file() or not Path(p["out_dir"]).is_dir():
                raise ValueError("Choose a valid tile config file and output folder")
            messagebox.showinfo("Done", "\n".join(run(p)))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    ttk.Button(frm, text="Generate", command=go).grid(row=n + 4, column=1, sticky="w", pady=8)
    root.mainloop()


if __name__ == "__main__":
    main()
