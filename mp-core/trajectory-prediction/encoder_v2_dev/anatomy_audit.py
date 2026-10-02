"""
anatomy_audit.py  (Encoder V2 — Phase 1, READ-ONLY audit helper)

Reconstructs the production encoder's intermediate structures IN MEMORY by
importing the frozen encode_spatial_auto_v2.py (v2.1C constants) and running its
functions on already-encoded recordings. NOTHING is written into the repo or the
datasets; figures go ONLY to the Desktop phase1 folder. The frozen encoder file
is not modified or executed via main().

Purpose: render anatomy figures and prove which structures exist for Phase 2
directional features (incl. nearest-obstacle coords via distance_transform_edt
return_indices — a feasibility probe, not a new feature).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import distance_transform_edt

PRED = Path(__file__).resolve().parent.parent           # trajectory-prediction
sys.path.insert(0, str(PRED))
import encode_spatial_auto_v2 as v2                      # frozen encoder (read-only import)

# v2.1C production constants (same overrides as the Barcelona batch runner)
v2.ENVELOPE_DILATE_M = 4.0
v2.WALKABLE_CLOSE_M = 1.5
v2.DBSCAN_EPS_M = 2.75
v2.DBSCAN_MIN_SAMPLES = 12
RES = v2.GRID_RES_M

ENC = Path(r"C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded")
OUTFIG = Path(r"C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\06_encoder_v2_dev\phase1_encoder_anatomy\figures")
RECS = ["red_bridge_combined_01", "placa_catalunya_01"]


def reconstruct(rec):
    """Run the frozen encoder functions in-memory; return all intermediates."""
    df = pd.read_csv(ENC / rec / "spatial_v21C" / "trajectories_encoded.csv")
    ali = {"track_id": "track_id", "frame": "frame", "world_x": "world_x", "world_y": "world_y"}
    grid = v2.build_grid(df, ali)
    masks = v2.build_walkable_and_envelope(grid["occupancy"])
    obstacle = v2.derive_obstacles(masks["walkable"], masks["envelope"])
    boundary = v2.derive_boundary(masks["walkable"])
    d_obs = v2.distance_map_metres(obstacle)
    d_bnd = v2.distance_map_metres(boundary)
    entries = v2.derive_entry_exit(df, ali)
    extent = (grid["x_min"], grid["x_min"] + grid["W"] * RES,
              grid["y_min"], grid["y_min"] + grid["H"] * RES)
    return dict(df=df, grid=grid, walk=masks["walkable"], occ=masks["occupancy_mask"],
                obstacle=obstacle, boundary=boundary, d_obs=d_obs, d_bnd=d_bnd,
                entries=entries, extent=extent)


def save(fig, name):
    OUTFIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTFIG / name, dpi=150, bbox_inches="tight"); plt.close(fig)


def figures_for(rec, R):
    ext = R["extent"]; df = R["df"]
    def base(ax, mask, cmap, title):
        ax.imshow(mask, origin="lower", extent=ext, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=10); ax.set_xlabel("world_x (m)"); ax.set_ylabel("world_y (m)")
        ax.set_aspect("equal")

    f, a = plt.subplots(figsize=(7, 7)); base(a, R["walk"], "Greens", f"{rec} — walkable mask"); save(f, f"{rec}_walkable_mask_anatomy.png")
    f, a = plt.subplots(figsize=(7, 7)); base(a, R["obstacle"], "Reds", f"{rec} — obstacle mask"); save(f, f"{rec}_obstacle_mask_anatomy.png")
    f, a = plt.subplots(figsize=(7, 7)); base(a, R["boundary"], "Greys", f"{rec} — boundary mask"); save(f, f"{rec}_boundary_mask_anatomy.png")

    f, a = plt.subplots(figsize=(8, 7))
    im = a.imshow(R["d_obs"], origin="lower", extent=ext, cmap="viridis")
    f.colorbar(im, ax=a, label="metres"); a.set_title(f"{rec} — distance_to_obstacle (m)")
    a.set_xlabel("world_x (m)"); a.set_ylabel("world_y (m)"); a.set_aspect("equal")
    # feasibility probe: nearest-obstacle vectors via return_indices for sample points
    if R["obstacle"].any():
        inds = distance_transform_edt(~R["obstacle"], return_indices=True)[1]  # (2,H,W): [row_idx,col_idx]
        g = R["grid"]; wx = df.world_x.to_numpy(); wy = df.world_y.to_numpy()
        col = np.clip(((wx - g["x_min"]) / RES).astype(int), 0, g["W"] - 1)
        row = np.clip(((wy - g["y_min"]) / RES).astype(int), 0, g["H"] - 1)
        sel = np.random.default_rng(0).choice(len(wx), size=min(40, len(wx)), replace=False)
        for i in sel:
            nr = inds[0, row[i], col[i]]; nc = inds[1, row[i], col[i]]
            nwx = g["x_min"] + nc * RES; nwy = g["y_min"] + nr * RES
            a.annotate("", xy=(nwx, nwy), xytext=(wx[i], wy[i]),
                       arrowprops=dict(arrowstyle="->", color="white", lw=0.6, alpha=0.8))
    save(f, f"{rec}_distance_to_obstacle_anatomy.png")

    f, a = plt.subplots(figsize=(8, 7))
    im = a.imshow(R["d_bnd"], origin="lower", extent=ext, cmap="magma")
    f.colorbar(im, ax=a, label="metres"); a.set_title(f"{rec} — distance_to_boundary (m)")
    a.set_xlabel("world_x (m)"); a.set_ylabel("world_y (m)"); a.set_aspect("equal")
    save(f, f"{rec}_distance_to_boundary_anatomy.png")

    f, a = plt.subplots(figsize=(8, 7))
    a.scatter(df.world_x, df.world_y, s=0.3, c="#bbbbbb", alpha=0.3)
    if not R["entries"].empty:
        a.scatter(R["entries"].center_x, R["entries"].center_y, s=160, c="#e67e22",
                  marker="*", edgecolor="black", linewidth=0.8, zorder=3)
        for _, r in R["entries"].iterrows():
            a.annotate(f"#{int(r.cluster_id)}", (r.center_x, r.center_y), fontsize=8,
                       xytext=(4, 4), textcoords="offset points")
    a.set_title(f"{rec} — entry/exit clusters ({len(R['entries'])})"); a.set_aspect("equal")
    a.set_xlabel("world_x (m)"); a.set_ylabel("world_y (m)"); save(f, f"{rec}_entry_exit_points_anatomy.png")

    f, a = plt.subplots(figsize=(8, 7))
    a.imshow(R["walk"], origin="lower", extent=ext, cmap="Greens", alpha=0.5, interpolation="nearest")
    a.imshow(np.ma.masked_where(~R["obstacle"], R["obstacle"]), origin="lower", extent=ext, cmap="autumn", alpha=0.7, interpolation="nearest")
    a.scatter(df.world_x, df.world_y, s=0.3, c="#1f3a8a", alpha=0.35)
    a.set_title(f"{rec} — trajectory points over walkable(green)/obstacle(red)")
    a.set_xlabel("world_x (m)"); a.set_ylabel("world_y (m)"); a.set_aspect("equal")
    save(f, f"{rec}_trajectory_points_over_masks.png")
    print(f"  [{rec}] grid={R['grid']['W']}x{R['grid']['H']} "
          f"walk_px={int(R['walk'].sum())} obs_px={int(R['obstacle'].sum())} "
          f"bnd_px={int(R['boundary'].sum())} entries={len(R['entries'])}")


def contact_sheet():
    names = [f"{r}_{k}_anatomy.png" for r in RECS for k in
             ["walkable_mask", "obstacle_mask", "boundary_mask", "distance_to_obstacle",
              "distance_to_boundary", "entry_exit_points"]]
    names = [n.replace("entry_exit_points_anatomy", "entry_exit_points_anatomy") for n in names]
    cols = 6
    f, axes = plt.subplots(2, cols, figsize=(4 * cols, 8))
    for ax, n in zip(axes.flat, names):
        p = OUTFIG / n
        if p.exists():
            ax.imshow(mpimg.imread(str(p)))
        ax.axis("off"); ax.set_title(n.replace("_anatomy.png", ""), fontsize=7)
    f.suptitle("Encoder V2 — Phase 1 anatomy (top: red_bridge, bottom: placa_catalunya)", fontweight="bold", fontsize=13)
    f.tight_layout(rect=[0, 0, 1, 0.97])
    f.savefig(OUTFIG / "encoder_v2_phase1_anatomy_contact_sheet.png", dpi=110); plt.close(f)


def main():
    print(f"[v2.1C] RES={RES} ENVELOPE_DILATE={v2.ENVELOPE_DILATE_M} WALKABLE_CLOSE={v2.WALKABLE_CLOSE_M} "
          f"DBSCAN eps={v2.DBSCAN_EPS_M} min={v2.DBSCAN_MIN_SAMPLES}")
    for rec in RECS:
        R = reconstruct(rec)
        figures_for(rec, R)
    contact_sheet()
    print(f"[done] figures -> {OUTFIG}")


if __name__ == "__main__":
    main()
