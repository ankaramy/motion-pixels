"""
encoding_truth_audit.py  (READ-ONLY forensic audit of spatial_v21C obstacle truth)
No training, no data/encoder modification. Reconstructs frozen masks in memory,
decomposes obstacle into interior vs exterior, overlays masks on the real plan
images via calib transform, and relates turns to encoded obstacles.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import label as cc_label

DEV = Path(__file__).resolve().parent
sys.path.insert(0, str(DEV))
import directional_features as DF   # reconstruct(), frozen v2.1C
v2 = DF.v2; RES = DF.RES

NEW = Path(r"C:\Users\OWNER\Desktop\new_datasets")
ENC = NEW / "Barcelona_v1_encoded"
OUT = Path(r"C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\07_encoding_truth_audit")
FIG = OUT / "figures"; TAB = OUT / "tables"
PHASE2 = Path(r"C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\06_encoder_v2_dev\phase2_directional_features\datasets\model_C_directional_N10.csv")
RECS = ["placa_catalunya_01", "esplanade_espanya_01", "placa_espanya_01",
        "red_bridge_combined_01", "stairs_montjuic_01"]


def load_calib(rec):
    c = json.load(open(NEW / rec / "calibration" / "calib.json"))
    return dict(mpp=c["meters_per_plan_pixel"], ox=c["plan_origin_pixel"][0],
                oy=c["plan_origin_pixel"][1], invert=c["invert_plan_y"],
                plan=c["plan_image"])


def world_to_plan(wx, wy, cal):
    px = cal["ox"] + wx / cal["mpp"]
    py = cal["oy"] + (-wy if cal["invert"] else wy) / cal["mpp"]
    return px, py


def warp_mask_to_plan(mask, grid, cal, plan_shape):
    """Backward-warp a world-grid boolean mask into plan-pixel space."""
    Hp, Wp = plan_shape[:2]
    yy, xx = np.mgrid[0:Hp, 0:Wp]
    wx = (xx - cal["ox"]) * cal["mpp"]
    wy = (yy - cal["oy"]) * cal["mpp"]
    if cal["invert"]:
        wy = -wy
    col = ((wx - grid["x_min"]) / RES).astype(np.int64)
    row = ((wy - grid["y_min"]) / RES).astype(np.int64)
    ok = (col >= 0) & (col < grid["W"]) & (row >= 0) & (row < grid["H"])
    out = np.zeros((Hp, Wp), bool)
    out[ok] = mask[row[ok], col[ok]]
    return out


def decompose_obstacle(walk, obstacle):
    """Split obstacle into exterior (connected to grid border through non-walkable)
    vs interior (enclosed voids). Interior = real enclosed objects."""
    free = ~walk
    lbl, n = cc_label(free)
    border = set(np.unique(np.concatenate([lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]])))
    border.discard(0)
    exterior_free = np.isin(lbl, list(border))
    obs_ext = obstacle & exterior_free
    obs_int = obstacle & ~exterior_free
    return obs_int, obs_ext


def main():
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    ph2 = pd.read_csv(PHASE2)
    rows = []
    recon = {}
    for rec in RECS:
        R = DF.reconstruct(rec)
        recon[rec] = R
        g = R["grid"]; walk = R["walk"]; obs = R["obstacle"]; bnd = R["boundary"]
        total = g["W"] * g["H"]; cell_m2 = RES * RES
        obs_int, obs_ext = decompose_obstacle(walk, obs)
        lbl_o, n_obs = cc_label(obs)
        lbl_i, n_int = cc_label(obs_int)
        int_sizes = np.bincount(lbl_i.ravel())[1:] if n_int else np.array([])
        # trajectory dist_to_obstacle from encoded CSV
        enc = pd.read_csv(ENC / rec / "spatial_v21C" / "trajectories_encoded.csv")
        d = pd.to_numeric(enc["dist_to_obstacle"], errors="coerce").dropna().to_numpy()
        # turn-event proximity: sample d_obs at moving turn points
        sub = ph2[(ph2.recording_id == rec) & (ph2.is_moving)]
        col = np.clip(((sub.world_x - g["x_min"]) / RES).astype(int), 0, g["W"]-1)
        row = np.clip(((sub.world_y - g["y_min"]) / RES).astype(int), 0, g["H"]-1)
        dt = R["obstacle"]  # need distance field
        d_obs = v2.distance_map_metres(obs)
        turn_d = d_obs[row, col]
        turns = sub.turn_class.to_numpy()
        def within(arr, t): return float(np.mean(arr <= t)) if len(arr) else np.nan
        rows.append({
            "recording": rec, "grid_cells": total,
            "walkable_pct": 100*walk.sum()/total, "obstacle_pct": 100*obs.sum()/total,
            "obstacle_interior_pct": 100*obs_int.sum()/total,
            "obstacle_exterior_pct": 100*obs_ext.sum()/total,
            "interior_obstacle_area_m2": float(obs_int.sum()*cell_m2),
            "n_obstacle_cc": int(n_obs), "n_interior_obstacle_cc": int(n_int),
            "median_interior_cc_m2": float(np.median(int_sizes)*cell_m2) if len(int_sizes) else 0.0,
            "max_interior_cc_m2": float(int_sizes.max()*cell_m2) if len(int_sizes) else 0.0,
            "boundary_px": int(bnd.sum()),
            "dist_obs_median_m": float(np.median(d)), "dist_obs_mean_m": float(d.mean()),
            "traj_within_0.5m_pct": 100*within(d, 0.5), "traj_within_1m_pct": 100*within(d, 1.0),
            "traj_within_2m_pct": 100*within(d, 2.0),
            "turns_within_0.5m_pct": 100*within(turn_d, 0.5), "turns_within_1m_pct": 100*within(turn_d, 1.0),
            "turns_within_2m_pct": 100*within(turn_d, 2.0),
            "n_turn_events_moving": int(len(sub)),
        })
        print(f"  [{rec:24s}] walk%={rows[-1]['walkable_pct']:.1f} obs%={rows[-1]['obstacle_pct']:.1f} "
              f"INT_obs%={rows[-1]['obstacle_interior_pct']:.2f} n_int_cc={n_int} "
              f"int_area={rows[-1]['interior_obstacle_area_m2']:.1f}m2")
    tab = pd.DataFrame(rows)
    tab.to_csv(TAB / "mask_diagnostics.csv", index=False)

    # interior/exterior decomposition figure (all 5)
    f, axes = plt.subplots(1, 5, figsize=(26, 5.4))
    for ax, rec in zip(axes, RECS):
        R = recon[rec]; g = R["grid"]
        ext = (g["x_min"], g["x_min"]+g["W"]*RES, g["y_min"], g["y_min"]+g["H"]*RES)
        oi, oe = decompose_obstacle(R["walk"], R["obstacle"])
        ax.imshow(R["walk"], origin="lower", extent=ext, cmap="Greens", alpha=0.45, interpolation="nearest")
        ax.imshow(np.ma.masked_where(~oe, oe), origin="lower", extent=ext, cmap="autumn", alpha=0.5, interpolation="nearest")
        ax.imshow(np.ma.masked_where(~oi, oi), origin="lower", extent=ext, cmap="cool", alpha=0.95, interpolation="nearest")
        ai = float(oi.sum()*RES*RES)
        ax.set_title(f"{rec}\ninterior obs (cyan) = {ai:.0f} m2", fontsize=9); ax.set_aspect("equal")
    f.suptitle("Obstacle decomposition — green=walkable, orange=exterior obstacle (ring), cyan=INTERIOR obstacle (enclosed)", fontweight="bold")
    f.tight_layout(rect=[0, 0, 1, 0.94]); f.savefig(FIG / "obstacle_interior_vs_exterior.png", dpi=140); plt.close(f)

    print("\nSaved mask_diagnostics.csv + obstacle_interior_vs_exterior.png")
    return tab, recon


if __name__ == "__main__":
    main()
