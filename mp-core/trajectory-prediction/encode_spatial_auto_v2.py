"""
encode_spatial_auto_v2.py
-------------------------
Trajectory-informed automatic spatial encoder.

Why a v2?
The first auto encoder (encode_spatial_auto.py) relied on image thresholding
and produced near-zero, near-constant distance features (mean dist_to_obstacle
0.27 m, mean dist_to_boundary 0.36 m, Pearson r vs manual ≈ 0.1).

This v2 instead INFERS the space from how pedestrians actually moved.
Image segmentation is optional and only ever a weak prior.

The script:
  1. Loads the trajectory CSV (column names auto-aliased).
  2. Builds a world-space occupancy grid (resolution = GRID_RES_M).
  3. Walkable mask  = dilated + closed occupancy, largest CC(s).
  4. Envelope mask  = wide dilation around walkable (defines "site").
  5. Obstacle mask  = envelope AND NOT walkable, min-area filtered.
  6. Boundary mask  = outer contour of walkable.
  7. Distance maps  = scipy distance_transform_edt × GRID_RES_M (metres).
  8. Entry/exit    = first/last points of sufficiently-long tracks,
                     DBSCAN clustered.
  9. Samples each distance map at every trajectory point.
 10. Writes outputs and a summary that honestly compares against the manual
     encoding when present.

Read-only with respect to existing files. All v2 outputs are under
mp-data/processed/encoded/auto_v2/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_closing, binary_dilation, binary_fill_holes,
    distance_transform_edt, label as cc_label,
)
from sklearn.cluster import DBSCAN


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE     = Path(__file__).resolve().parent
MP_ROOT  = HERE.parent.parent
MP_DATA  = MP_ROOT / "mp-data"
ENCODED  = MP_DATA / "processed" / "encoded"
AUTO_V2  = ENCODED / "auto_v2"
SRC_CSV  = ENCODED / "trajectories_encoded_auto.csv"  # has world_x/world_y
MANUAL_CSV = ENCODED / "trajectories_encoded.csv"     # benchmark only

# --------------------------------------------------------------------------- #
# Tunables — every magic number is here.
# --------------------------------------------------------------------------- #
GRID_RES_M           = 0.10   # metres per pixel in the world grid
PAD_M                = 3.0    # padding around min/max trajectory extent
WALKABLE_DILATE_M    = 0.40   # walkable = dilate occupancy by this radius
WALKABLE_CLOSE_M     = 1.00   # then close gaps of up to this radius
WALKABLE_MIN_CC_FRAC = 0.05   # keep CCs that are >= 5% of biggest CC
ENVELOPE_DILATE_M    = 2.50   # envelope = walkable dilated by this much
OBSTACLE_MIN_AREA_M2 = 0.30   # discard obstacle blobs smaller than this
LONG_TRACK_MIN_LEN   = 10     # only tracks with >= this many frames count
                              # for entry/exit detection
DBSCAN_EPS_M         = 1.50   # cluster start/end points within this radius
DBSCAN_MIN_SAMPLES   = 5

# Column-name aliases.
ALIAS = {
    "track_id": ["track_id", "person_id", "pid", "id"],
    "frame":    ["frame", "frame_number", "frame_idx", "fid"],
    "world_x":  ["world_x", "wx"],
    "world_y":  ["world_y", "wy"],
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def find_alias(cols, canonical) -> Optional[str]:
    for n in ALIAS.get(canonical, [canonical]):
        if n in cols:
            return n
    return None


def m_to_px(metres: float) -> int:
    return max(1, int(round(metres / GRID_RES_M)))


def disk_kernel(radius_px: int) -> np.ndarray:
    r = radius_px
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


# --------------------------------------------------------------------------- #
# 1. Load trajectories
# --------------------------------------------------------------------------- #
def load_trajectories(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = pd.read_csv(path)
    cols = df.columns.tolist()
    aliases = {
        "track_id": find_alias(cols, "track_id"),
        "frame":    find_alias(cols, "frame"),
        "world_x":  find_alias(cols, "world_x"),
        "world_y":  find_alias(cols, "world_y"),
    }
    missing = [k for k, v in aliases.items() if v is None]
    if missing:
        raise SystemExit(f"[FATAL] trajectory CSV missing columns: {missing}")
    return df, aliases


# --------------------------------------------------------------------------- #
# 2-3. Build world-space occupancy + walkable + envelope
# --------------------------------------------------------------------------- #
def build_grid(df: pd.DataFrame, ali: Dict[str, str]) -> Dict:
    wx = df[ali["world_x"]].to_numpy(dtype=np.float64)
    wy = df[ali["world_y"]].to_numpy(dtype=np.float64)
    valid = np.isfinite(wx) & np.isfinite(wy)
    wx, wy = wx[valid], wy[valid]

    x_min, x_max = wx.min() - PAD_M, wx.max() + PAD_M
    y_min, y_max = wy.min() - PAD_M, wy.max() + PAD_M
    W = int(np.ceil((x_max - x_min) / GRID_RES_M))
    H = int(np.ceil((y_max - y_min) / GRID_RES_M))

    # Grid indices (col, row) for every trajectory point.
    col = np.clip(((wx - x_min) / GRID_RES_M).astype(np.int32), 0, W - 1)
    row = np.clip(((wy - y_min) / GRID_RES_M).astype(np.int32), 0, H - 1)

    occupancy = np.zeros((H, W), dtype=np.int32)
    np.add.at(occupancy, (row, col), 1)

    return {
        "x_min": x_min, "y_min": y_min, "W": W, "H": H,
        "occupancy": occupancy,
        "wx": wx, "wy": wy,
    }


def build_walkable_and_envelope(occupancy: np.ndarray) -> Dict[str, np.ndarray]:
    occ_mask = occupancy > 0

    # Walkable: dilate then close.
    walk = binary_dilation(occ_mask,
                           structure=disk_kernel(m_to_px(WALKABLE_DILATE_M)))
    walk = binary_closing(walk,
                          structure=disk_kernel(m_to_px(WALKABLE_CLOSE_M)))

    # Keep meaningful connected components.
    lbl, n_cc = cc_label(walk)
    if n_cc > 0:
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        biggest = sizes.max()
        keep = np.where(sizes >= WALKABLE_MIN_CC_FRAC * biggest)[0]
        walk = np.isin(lbl, keep)

    # Envelope: walkable dilated by ENVELOPE_DILATE_M and hole-filled.
    env = binary_dilation(walk,
                          structure=disk_kernel(m_to_px(ENVELOPE_DILATE_M)))
    env = binary_fill_holes(env)

    return {"walkable": walk, "envelope": env, "occupancy_mask": occ_mask}


# --------------------------------------------------------------------------- #
# 4. Obstacles
# --------------------------------------------------------------------------- #
def derive_obstacles(walkable: np.ndarray,
                     envelope: np.ndarray) -> np.ndarray:
    raw = envelope & (~walkable)
    # Connected-component filter by min area in metres².
    min_area_px = int(round(OBSTACLE_MIN_AREA_M2 / (GRID_RES_M ** 2)))
    lbl, n_cc = cc_label(raw)
    if n_cc == 0:
        return np.zeros_like(raw)
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    keep = np.where(sizes >= min_area_px)[0]
    obstacle = np.isin(lbl, keep)
    return obstacle


# --------------------------------------------------------------------------- #
# 5. Boundary + distance maps
# --------------------------------------------------------------------------- #
def derive_boundary(walkable: np.ndarray) -> np.ndarray:
    """Outer contour of the walkable mask."""
    boundary = walkable & (~binary_dilation(~walkable))  # 1-pixel inner rim
    # OpenCV gives us a slightly nicer thin outline.
    walk_u8 = walkable.astype(np.uint8) * 255
    contours, _ = cv2.findContours(walk_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    boundary = np.zeros_like(walkable, dtype=bool)
    canvas = np.zeros(walkable.shape, dtype=np.uint8)
    cv2.drawContours(canvas, contours, -1, 255, thickness=1)
    boundary |= canvas > 0
    return boundary


def distance_map_metres(mask: np.ndarray) -> np.ndarray:
    """Distance (in metres) from every pixel to the nearest True cell of mask.
    If mask is all-False, returns an array filled with NaN."""
    if not mask.any():
        return np.full(mask.shape, np.nan, dtype=np.float32)
    # distance_transform_edt computes distance to nearest 0 → invert.
    return distance_transform_edt(~mask).astype(np.float32) * GRID_RES_M


# --------------------------------------------------------------------------- #
# 6. Entry / exit detection
# --------------------------------------------------------------------------- #
def derive_entry_exit(df: pd.DataFrame, ali: Dict[str, str]) -> pd.DataFrame:
    pts = []
    grouped = df.groupby(ali["track_id"])
    for tid, g in grouped:
        if len(g) < LONG_TRACK_MIN_LEN:
            continue
        gs = g.sort_values(ali["frame"])
        first = gs.iloc[0]
        last  = gs.iloc[-1]
        pts.append((float(first[ali["world_x"]]), float(first[ali["world_y"]]),
                    "start", int(tid)))
        pts.append((float(last [ali["world_x"]]), float(last [ali["world_y"]]),
                    "end",   int(tid)))
    if not pts:
        return pd.DataFrame(columns=["cluster_id", "center_x", "center_y",
                                     "n_points"])
    pts_df = pd.DataFrame(pts, columns=["x", "y", "kind", "track_id"])
    coords = pts_df[["x", "y"]].to_numpy()
    db = DBSCAN(eps=DBSCAN_EPS_M, min_samples=DBSCAN_MIN_SAMPLES).fit(coords)
    pts_df["cluster"] = db.labels_

    clusters = []
    for cid in sorted(set(db.labels_)):
        if cid == -1:
            continue
        members = pts_df[pts_df["cluster"] == cid]
        clusters.append({
            "cluster_id": int(cid),
            "center_x": float(members["x"].mean()),
            "center_y": float(members["y"].mean()),
            "n_points": int(len(members)),
        })
    return pd.DataFrame(clusters)


# --------------------------------------------------------------------------- #
# 7. Sample distance maps at trajectory points
# --------------------------------------------------------------------------- #
def world_to_grid(wx: np.ndarray, wy: np.ndarray,
                  grid: Dict) -> Tuple[np.ndarray, np.ndarray]:
    col = np.clip(((wx - grid["x_min"]) / GRID_RES_M).astype(np.int32),
                  0, grid["W"] - 1)
    row = np.clip(((wy - grid["y_min"]) / GRID_RES_M).astype(np.int32),
                  0, grid["H"] - 1)
    return row, col


def sample_distances(df: pd.DataFrame, ali: Dict[str, str], grid: Dict,
                     d_obs: np.ndarray, d_bnd: np.ndarray,
                     d_ent: Optional[np.ndarray]) -> pd.DataFrame:
    wx = df[ali["world_x"]].to_numpy(dtype=np.float64)
    wy = df[ali["world_y"]].to_numpy(dtype=np.float64)
    row, col = world_to_grid(wx, wy, grid)
    out = df.copy()
    out["dist_to_obstacle"] = d_obs[row, col]
    out["dist_to_boundary"] = d_bnd[row, col]
    if d_ent is not None:
        out["dist_to_entrance"] = d_ent[row, col]
    else:
        out["dist_to_entrance"] = np.nan
    return out


# --------------------------------------------------------------------------- #
# 8. Validation against manual
# --------------------------------------------------------------------------- #
def compare_against_manual(auto_df: pd.DataFrame,
                           ali: Dict[str, str]) -> Dict:
    rep = {"compared": False}
    if not MANUAL_CSV.exists():
        rep["reason"] = "manual CSV not present"
        return rep
    try:
        man = pd.read_csv(MANUAL_CSV)
    except Exception as e:
        rep["reason"] = f"could not read manual: {e}"
        return rep

    tid_m   = find_alias(man.columns, "track_id")
    frame_m = find_alias(man.columns, "frame")
    if not (tid_m and frame_m):
        rep["reason"] = "manual CSV missing track/frame columns"
        return rep

    a = auto_df.rename(columns={ali["track_id"]: "_tid",
                                ali["frame"]:    "_fr"})
    m = man.rename(columns={tid_m: "_tid", frame_m: "_fr"})
    keep_a = ["_tid", "_fr", "dist_to_obstacle", "dist_to_boundary",
              "dist_to_entrance"]
    keep_m = ["_tid", "_fr"]
    for c in ("dist_to_obstacle", "dist_to_boundary", "dist_to_entrance"):
        if c in m.columns:
            keep_m.append(c)
    merged = pd.merge(a[keep_a], m[keep_m], on=["_tid", "_fr"],
                      suffixes=("_auto", "_manual"))

    per_feat = {}
    for feat in ("dist_to_obstacle", "dist_to_boundary", "dist_to_entrance"):
        a_col, m_col = f"{feat}_auto", f"{feat}_manual"
        if a_col not in merged.columns or m_col not in merged.columns:
            per_feat[feat] = {"present_in_manual": m_col in merged.columns}
            continue
        sa = pd.to_numeric(merged[a_col], errors="coerce")
        sm = pd.to_numeric(merged[m_col], errors="coerce")
        ok = sa.notna() & sm.notna()
        if ok.sum() < 3:
            per_feat[feat] = {"pairs": int(ok.sum())}
            continue
        per_feat[feat] = {
            "pairs": int(ok.sum()),
            "auto_mean": float(sa[ok].mean()),
            "auto_median": float(sa[ok].median()),
            "auto_std": float(sa[ok].std()),
            "manual_mean": float(sm[ok].mean()),
            "manual_median": float(sm[ok].median()),
            "manual_std": float(sm[ok].std()),
            "mean_abs_diff": float((sa[ok] - sm[ok]).abs().mean()),
            "pearson_r": float(sa[ok].corr(sm[ok])),
        }
    rep["compared"] = True
    rep["joined_rows"] = int(len(merged))
    rep["per_feature"] = per_feat
    return rep


def feature_warnings(auto_df: pd.DataFrame) -> List[str]:
    warns = []
    for feat in ("dist_to_obstacle", "dist_to_boundary", "dist_to_entrance"):
        if feat not in auto_df.columns:
            warns.append(f"{feat}: column missing")
            continue
        s = pd.to_numeric(auto_df[feat], errors="coerce")
        n_nan = int(s.isna().sum())
        if n_nan > 0.5 * len(s):
            warns.append(f"{feat}: majority NaN ({n_nan}/{len(s)})")
        if n_nan == len(s):
            warns.append(f"{feat}: all NaN")
            continue
        mean = float(s.mean())
        std  = float(s.std())
        if std < 1e-6:
            warns.append(f"{feat}: constant (std≈0)")
        if mean < 0.05:
            warns.append(f"{feat}: mean collapsed near zero ({mean:.3f} m)")
        zero_frac = float((s < 1e-3).sum()) / float(s.notna().sum())
        if zero_frac > 0.5:
            warns.append(f"{feat}: {zero_frac:.0%} of samples are ~0")
    return warns


# --------------------------------------------------------------------------- #
# 9. Output writers
# --------------------------------------------------------------------------- #
def save_mask(mask: np.ndarray, path: Path) -> None:
    img = np.zeros(mask.shape, dtype=np.uint8)
    img[mask] = 255
    # Flip vertically so y grows upward visually (world_y up).
    cv2.imwrite(str(path), img[::-1, :])


def save_distance_map(dmap: np.ndarray, path: Path, vmax: Optional[float] = None) -> None:
    """Save as a colormapped PNG so it's interpretable as 'distance'."""
    valid = np.isfinite(dmap)
    if not valid.any():
        cv2.imwrite(str(path), np.zeros(dmap.shape, dtype=np.uint8)[::-1, :])
        return
    v = dmap.copy()
    v[~valid] = 0.0
    upper = float(vmax) if vmax is not None else float(np.nanpercentile(dmap[valid], 99))
    if upper <= 0:
        upper = 1.0
    norm = np.clip(v / upper, 0.0, 1.0)
    u8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)
    # Flip vertically.
    cv2.imwrite(str(path), colored[::-1, :, :])


def plot_debug_stages(grid, masks, obstacle, boundary, entries_df,
                      out_path: Path) -> None:
    extent = (grid["x_min"],
              grid["x_min"] + grid["W"] * GRID_RES_M,
              grid["y_min"],
              grid["y_min"] + grid["H"] * GRID_RES_M)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    ax = axes[0, 0]
    ax.scatter(grid["wx"], grid["wy"], s=0.3, c="#2980b9", alpha=0.35)
    ax.set_title("(1) Trajectory point cloud", fontsize=10)

    ax = axes[0, 1]
    ax.imshow(masks["occupancy_mask"], cmap="Greys", origin="lower",
              extent=extent, interpolation="nearest")
    ax.set_title("(2) Raw occupancy (any-frame)", fontsize=10)

    ax = axes[0, 2]
    ax.imshow(masks["walkable"], cmap="Greens", origin="lower",
              extent=extent, interpolation="nearest")
    ax.set_title(f"(3) Walkable (dilate {WALKABLE_DILATE_M}m, "
                 f"close {WALKABLE_CLOSE_M}m)", fontsize=10)

    ax = axes[1, 0]
    ax.imshow(masks["walkable"], cmap="Greens", alpha=0.4, origin="lower",
              extent=extent, interpolation="nearest")
    ax.imshow(obstacle, cmap="Reds", alpha=0.7, origin="lower",
              extent=extent, interpolation="nearest")
    ax.set_title(f"(4) Obstacles (min area {OBSTACLE_MIN_AREA_M2} m²)",
                 fontsize=10)

    ax = axes[1, 1]
    ax.imshow(masks["walkable"], cmap="Greens", alpha=0.35, origin="lower",
              extent=extent, interpolation="nearest")
    # Draw boundary in solid red on top.
    by, bx = np.where(boundary)
    ax.scatter(grid["x_min"] + bx * GRID_RES_M,
               grid["y_min"] + by * GRID_RES_M,
               s=0.2, c="#c0392b")
    ax.set_title("(5) Boundary (outer walkable contour)", fontsize=10)

    ax = axes[1, 2]
    ax.scatter(grid["wx"], grid["wy"], s=0.2, c="#bdc3c7", alpha=0.35,
               label="all points")
    if not entries_df.empty:
        ax.scatter(entries_df["center_x"], entries_df["center_y"],
                   s=140, c="#e67e22", marker="*", edgecolor="black",
                   linewidth=0.8, label="entry/exit clusters")
        for _, r in entries_df.iterrows():
            ax.annotate(f"#{int(r['cluster_id'])} (n={int(r['n_points'])})",
                        xy=(r["center_x"], r["center_y"]),
                        fontsize=7, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points")
    ax.set_title(f"(6) Entry/exit (DBSCAN ε={DBSCAN_EPS_M} m, "
                 f"min_samples={DBSCAN_MIN_SAMPLES})", fontsize=10)
    ax.legend(fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel("world_x (m)", fontsize=8)
        ax.set_ylabel("world_y (m)", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, lw=0.2, alpha=0.4)

    fig.suptitle("encode_spatial_auto_v2 — debug stages",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_sampling(auto_df: pd.DataFrame, ali: Dict[str, str],
                             out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    feats = [
        ("dist_to_obstacle", "viridis"),
        ("dist_to_boundary", "magma"),
        ("dist_to_entrance", "plasma"),
    ]
    for ax, (feat, cmap) in zip(axes, feats):
        if feat not in auto_df.columns:
            ax.set_title(f"{feat} (missing)"); ax.set_axis_off(); continue
        s = pd.to_numeric(auto_df[feat], errors="coerce")
        ok = s.notna()
        sc = ax.scatter(auto_df.loc[ok, ali["world_x"]],
                        auto_df.loc[ok, ali["world_y"]],
                        c=s[ok], s=1.2, cmap=cmap,
                        vmin=float(np.nanpercentile(s[ok], 1)),
                        vmax=float(np.nanpercentile(s[ok], 99)))
        ax.set_title(f"Trajectories coloured by {feat}", fontsize=10)
        ax.set_xlabel("world_x (m)"); ax.set_ylabel("world_y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, lw=0.2, alpha=0.4)
        fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02, label="metres")
    fig.suptitle("encode_spatial_auto_v2 — trajectory feature sampling",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 10. Summary report
# --------------------------------------------------------------------------- #
def fmt(x):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    try:
        f = float(x)
        if not np.isfinite(f):
            return "—"
        return f"{f:.3f}"
    except Exception:
        return str(x)


def write_summary(out_path: Path, auto_df: pd.DataFrame, grid: Dict,
                  masks: Dict, obstacle: np.ndarray, boundary: np.ndarray,
                  entries_df: pd.DataFrame, compare_rep: Dict,
                  warnings: List[str], n_obstacle_cc: int,
                  n_tracks_for_entries: int) -> None:
    lines: List[str] = []
    lines.append("# Auto v2 — Spatial Encoding Summary")
    lines.append("")
    lines.append("**Trajectory-informed automatic spatial encoder.** The space "
                 "is inferred from how pedestrians actually moved, then enriched "
                 "with conservative obstacle hints derived from holes in the "
                 "walkable envelope. No model retraining. No edits to the "
                 "original `encode_spatial_auto.py`.")
    lines.append("")
    lines.append(f"- Source trajectory CSV: `{SRC_CSV.relative_to(MP_ROOT)}`")
    lines.append(f"- Output folder: `{AUTO_V2.relative_to(MP_ROOT)}`")
    lines.append("")

    lines.append("## Method parameters")
    lines.append("")
    lines.append(f"| parameter | value |")
    lines.append(f"|---|---|")
    lines.append(f"| grid resolution | {GRID_RES_M} m/px |")
    lines.append(f"| padding around extent | {PAD_M} m |")
    lines.append(f"| walkable dilation radius | {WALKABLE_DILATE_M} m |")
    lines.append(f"| walkable closing radius  | {WALKABLE_CLOSE_M} m |")
    lines.append(f"| walkable min-CC fraction  | {WALKABLE_MIN_CC_FRAC:.0%} of biggest |")
    lines.append(f"| envelope dilation radius | {ENVELOPE_DILATE_M} m |")
    lines.append(f"| obstacle min area | {OBSTACLE_MIN_AREA_M2} m² |")
    lines.append(f"| long-track min length | {LONG_TRACK_MIN_LEN} frames |")
    lines.append(f"| DBSCAN eps | {DBSCAN_EPS_M} m |")
    lines.append(f"| DBSCAN min_samples | {DBSCAN_MIN_SAMPLES} |")
    lines.append("")

    lines.append("## Grid")
    lines.append("")
    lines.append(f"- World extent: "
                 f"x ∈ [{grid['x_min']:.2f}, "
                 f"{grid['x_min'] + grid['W']*GRID_RES_M:.2f}] m, "
                 f"y ∈ [{grid['y_min']:.2f}, "
                 f"{grid['y_min'] + grid['H']*GRID_RES_M:.2f}] m")
    lines.append(f"- Grid size: **{grid['W']} × {grid['H']} pixels** "
                 f"({grid['W']*grid['H']:,} cells)")
    lines.append("")

    lines.append("## Derived masks")
    lines.append("")
    n_occ  = int(masks["occupancy_mask"].sum())
    n_walk = int(masks["walkable"].sum())
    n_env  = int(masks["envelope"].sum())
    n_obs  = int(obstacle.sum())
    n_bnd  = int(boundary.sum())
    total  = grid["W"] * grid["H"]
    lines.append(f"| mask | true pixels | area (m²) | % of grid |")
    lines.append(f"|---|---|---|---|")
    lines.append(f"| occupancy (raw) | {n_occ:,} | "
                 f"{n_occ*GRID_RES_M*GRID_RES_M:.1f} | "
                 f"{100*n_occ/total:.1f}% |")
    lines.append(f"| walkable        | {n_walk:,} | "
                 f"{n_walk*GRID_RES_M*GRID_RES_M:.1f} | "
                 f"{100*n_walk/total:.1f}% |")
    lines.append(f"| envelope        | {n_env:,} | "
                 f"{n_env*GRID_RES_M*GRID_RES_M:.1f} | "
                 f"{100*n_env/total:.1f}% |")
    lines.append(f"| obstacle        | {n_obs:,} | "
                 f"{n_obs*GRID_RES_M*GRID_RES_M:.1f} | "
                 f"{100*n_obs/total:.1f}% |")
    lines.append(f"| boundary        | {n_bnd:,} | — | "
                 f"{100*n_bnd/total:.2f}% |")
    lines.append("")
    lines.append(f"- Obstacle connected components retained: **{n_obstacle_cc}**")
    lines.append("")

    lines.append("## Entry / exit detection")
    lines.append("")
    lines.append(f"- Long tracks used (>= {LONG_TRACK_MIN_LEN} frames): "
                 f"**{n_tracks_for_entries}**")
    if entries_df.empty:
        lines.append("- **No clusters found.** dist_to_entrance is NaN.")
    else:
        lines.append(f"- Clusters: **{len(entries_df)}**")
        lines.append("")
        lines.append("| cluster | center_x (m) | center_y (m) | n_points |")
        lines.append("|---|---|---|---|")
        for _, r in entries_df.iterrows():
            lines.append(f"| {int(r['cluster_id'])} | {r['center_x']:.2f} | "
                         f"{r['center_y']:.2f} | {int(r['n_points'])} |")
    lines.append("")

    lines.append("## Feature distributions (auto v2, all rows)")
    lines.append("")
    lines.append("| feature | n_valid | mean (m) | median (m) | std (m) | "
                 "min (m) | max (m) | %≤0.05 m |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for feat in ("dist_to_obstacle", "dist_to_boundary", "dist_to_entrance"):
        if feat not in auto_df.columns:
            lines.append(f"| `{feat}` | — | — | — | — | — | — | — |")
            continue
        s = pd.to_numeric(auto_df[feat], errors="coerce")
        ok = s.notna()
        if not ok.any():
            lines.append(f"| `{feat}` | 0 | — | — | — | — | — | — |")
            continue
        nz_frac = float((s[ok] <= 0.05).sum()) / float(ok.sum())
        lines.append(f"| `{feat}` | {int(ok.sum()):,} | "
                     f"{fmt(s[ok].mean())} | {fmt(s[ok].median())} | "
                     f"{fmt(s[ok].std())} | "
                     f"{fmt(s[ok].min())} | {fmt(s[ok].max())} | "
                     f"{100*nz_frac:.1f}% |")
    lines.append("")

    lines.append("## Comparison vs manual encoding (benchmark only)")
    lines.append("")
    if not compare_rep.get("compared"):
        lines.append(f"_Not run: {compare_rep.get('reason', '—')}_")
    else:
        lines.append(f"- Joined rows: **{compare_rep['joined_rows']:,}**")
        lines.append("")
        lines.append("| feature | pairs | auto mean | manual mean | "
                     "mean |Δ| | Pearson r |")
        lines.append("|---|---|---|---|---|---|")
        for feat in ("dist_to_obstacle", "dist_to_boundary",
                     "dist_to_entrance"):
            info = compare_rep["per_feature"].get(feat, {})
            if "pearson_r" not in info:
                lines.append(f"| `{feat}` | — | — | — | — | — |")
                continue
            lines.append(f"| `{feat}` | {info['pairs']:,} | "
                         f"{fmt(info['auto_mean'])} | "
                         f"{fmt(info['manual_mean'])} | "
                         f"{fmt(info['mean_abs_diff'])} | "
                         f"{fmt(info['pearson_r'])} |")
    lines.append("")

    lines.append("## Warnings")
    lines.append("")
    if not warnings:
        lines.append("_No automatic warnings raised._")
    else:
        for w in warnings:
            lines.append(f"- {w}")
    lines.append("")

    lines.append("## Acceptance checklist")
    lines.append("")
    checks = acceptance_checks(auto_df, warnings, entries_df)
    for label, ok in checks:
        mark = "[x]" if ok else "[ ]"
        lines.append(f"- {mark} {label}")
    overall = all(ok for _, ok in checks)
    lines.append("")
    lines.append(f"**Overall: {'ACCEPT as candidate' if overall else 'NOT a candidate yet — see warnings above.'}**")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    for name in [
        "trajectories_encoded_auto_v2.csv",
        "walkable_mask.png", "obstacle_mask.png", "boundary_mask.png",
        "distance_to_obstacle_m.png", "distance_to_boundary_m.png",
        "entry_exit_points.csv", "distance_to_entrance_m.png",
        "debug_stages.png", "debug_trajectory_sampling.png",
    ]:
        lines.append(f"- `{name}`")
    lines.append("")
    lines.append("## Thesis framing")
    lines.append("")
    lines.append("This v2 encoder does not aim for perfect semantic "
                 "segmentation. It is a trajectory-informed automatic "
                 "spatial encoder: the walkable region is inferred from "
                 "observed pedestrian density, obstacles are conservatively "
                 "derived from holes inside the walkable envelope, and "
                 "entries/exits are detected from the start and end of "
                 "long tracks. The result is intentionally a function of the "
                 "data rather than of any one image-segmentation choice — "
                 "which is the property the thesis needs.")
    lines.append("")
    lines.append("_End of summary._")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def acceptance_checks(auto_df, warnings, entries_df) -> List[Tuple[str, bool]]:
    checks = []
    def col_mean(c):
        if c not in auto_df.columns: return None
        s = pd.to_numeric(auto_df[c], errors="coerce")
        if not s.notna().any(): return None
        return float(s[s.notna()].mean())

    obs_mean = col_mean("dist_to_obstacle")
    bnd_mean = col_mean("dist_to_boundary")
    ent_present = "dist_to_entrance" in auto_df.columns \
                  and pd.to_numeric(auto_df["dist_to_entrance"],
                                    errors="coerce").notna().any()

    checks.append(("dist_to_obstacle mean is not collapsed near 0 (>0.10 m)",
                   obs_mean is not None and obs_mean > 0.10))
    checks.append(("dist_to_boundary mean is not collapsed near 0 (>0.10 m)",
                   bnd_mean is not None and bnd_mean > 0.10))
    checks.append(("dist_to_entrance is present and non-empty", ent_present))
    constant_or_zero = any("constant" in w or "majority NaN" in w
                           or "mostly zero" in w or "collapsed" in w
                           for w in warnings)
    checks.append(("no spatial feature is constant / mostly zero / NaN-heavy",
                   not constant_or_zero))
    checks.append(("at least one entry/exit cluster detected",
                   not entries_df.empty))
    return checks


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    AUTO_V2.mkdir(parents=True, exist_ok=True)
    print(f"[INFO]  Output folder: {AUTO_V2}")

    if not SRC_CSV.exists():
        print(f"[FATAL] missing source CSV: {SRC_CSV}")
        sys.exit(1)

    df, ali = load_trajectories(SRC_CSV)
    print(f"[INFO]  Loaded {len(df):,} rows; "
          f"track col=`{ali['track_id']}`, frame col=`{ali['frame']}`")
    print(f"[INFO]  Unique tracks: {df[ali['track_id']].nunique():,}")

    grid = build_grid(df, ali)
    print(f"[INFO]  World extent: "
          f"x∈[{grid['x_min']:.2f},{grid['x_min']+grid['W']*GRID_RES_M:.2f}] m  "
          f"y∈[{grid['y_min']:.2f},{grid['y_min']+grid['H']*GRID_RES_M:.2f}] m  "
          f"grid={grid['W']}×{grid['H']}")

    masks = build_walkable_and_envelope(grid["occupancy"])
    print(f"[INFO]  Walkable pixels: {int(masks['walkable'].sum()):,}   "
          f"Envelope pixels: {int(masks['envelope'].sum()):,}")

    obstacle = derive_obstacles(masks["walkable"], masks["envelope"])
    lbl_o, n_obs_cc = cc_label(obstacle)
    print(f"[INFO]  Obstacle pixels: {int(obstacle.sum()):,}   "
          f"connected components: {n_obs_cc}")

    boundary = derive_boundary(masks["walkable"])
    print(f"[INFO]  Boundary pixels: {int(boundary.sum()):,}")

    d_obs = distance_map_metres(obstacle)
    d_bnd = distance_map_metres(boundary)
    print(f"[INFO]  dist_to_obstacle: mean={np.nanmean(d_obs):.3f} m   "
          f"max={np.nanmax(d_obs):.3f} m")
    print(f"[INFO]  dist_to_boundary: mean={np.nanmean(d_bnd):.3f} m   "
          f"max={np.nanmax(d_bnd):.3f} m")

    entries_df = derive_entry_exit(df, ali)
    long_tracks = (df.groupby(ali["track_id"]).size() >= LONG_TRACK_MIN_LEN)
    n_tracks_for_entries = int(long_tracks.sum())
    print(f"[INFO]  Entry/exit clusters: {len(entries_df)} "
          f"(from {n_tracks_for_entries} long tracks)")

    # Build dist_to_entrance map.
    if not entries_df.empty:
        ent_mask = np.zeros((grid["H"], grid["W"]), dtype=bool)
        ex = entries_df["center_x"].to_numpy()
        ey = entries_df["center_y"].to_numpy()
        row, col = world_to_grid(ex, ey, grid)
        ent_mask[row, col] = True
        d_ent = distance_map_metres(ent_mask)
    else:
        d_ent = None

    if d_ent is not None:
        print(f"[INFO]  dist_to_entrance: mean={np.nanmean(d_ent):.3f} m   "
              f"max={np.nanmax(d_ent):.3f} m")
    else:
        print("[WARN]  No entry/exit clusters; dist_to_entrance will be NaN.")

    # Sample at trajectory points and write CSV.
    out_df = sample_distances(df, ali, grid, d_obs, d_bnd, d_ent)
    out_csv = AUTO_V2 / "trajectories_encoded_auto_v2.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[OK]    Wrote {out_csv.name} ({len(out_df):,} rows)")

    # Save masks and distance maps as PNGs.
    save_mask(masks["walkable"], AUTO_V2 / "walkable_mask.png")
    save_mask(obstacle,          AUTO_V2 / "obstacle_mask.png")
    save_mask(boundary,          AUTO_V2 / "boundary_mask.png")
    save_distance_map(d_obs,     AUTO_V2 / "distance_to_obstacle_m.png")
    save_distance_map(d_bnd,     AUTO_V2 / "distance_to_boundary_m.png")
    if d_ent is not None:
        save_distance_map(d_ent, AUTO_V2 / "distance_to_entrance_m.png")
    else:
        # Write a placeholder black image so the artefact list is still complete.
        cv2.imwrite(str(AUTO_V2 / "distance_to_entrance_m.png"),
                    np.zeros((grid["H"], grid["W"]), dtype=np.uint8))
    entries_df.to_csv(AUTO_V2 / "entry_exit_points.csv", index=False)
    print(f"[OK]    Wrote masks, distance maps, entry_exit_points.csv")

    # Debug plots.
    plot_debug_stages(grid, masks, obstacle, boundary, entries_df,
                      AUTO_V2 / "debug_stages.png")
    plot_trajectory_sampling(out_df, ali,
                             AUTO_V2 / "debug_trajectory_sampling.png")
    print(f"[OK]    Wrote debug_stages.png, debug_trajectory_sampling.png")

    # Compare and warnings.
    compare_rep = compare_against_manual(out_df, ali)
    warnings = feature_warnings(out_df)
    write_summary(AUTO_V2 / "auto_v2_summary.md",
                  out_df, grid, masks, obstacle, boundary, entries_df,
                  compare_rep, warnings, n_obs_cc, n_tracks_for_entries)
    print(f"[OK]    Wrote auto_v2_summary.md")

    # Console summary.
    print("\n----- Headline -----")
    for feat in ("dist_to_obstacle", "dist_to_boundary", "dist_to_entrance"):
        if feat in out_df.columns:
            s = pd.to_numeric(out_df[feat], errors="coerce")
            ok = s.notna()
            if ok.any():
                print(f"  {feat:<20s}  mean={s[ok].mean():.3f} m   "
                      f"median={s[ok].median():.3f} m   "
                      f"std={s[ok].std():.3f} m   "
                      f"(n_valid={int(ok.sum()):,})")
            else:
                print(f"  {feat:<20s}  ALL NaN")
    if compare_rep.get("compared"):
        print("\n----- Pearson r vs manual (joined rows) -----")
        for feat, info in compare_rep["per_feature"].items():
            if "pearson_r" in info:
                print(f"  {feat:<20s}  r={info['pearson_r']:+.3f}   "
                      f"meanΔ={info['mean_abs_diff']:.2f} m   "
                      f"(auto mean {info['auto_mean']:.2f} m, "
                      f"manual mean {info['manual_mean']:.2f} m)")
    if warnings:
        print("\n----- Warnings -----")
        for w in warnings:
            print(f"  - {w}")

    print(f"\nSummary report : {AUTO_V2 / 'auto_v2_summary.md'}")


if __name__ == "__main__":
    main()
