"""
visualize_auto_spatial_encoding_v2.py
-------------------------------------
Visual previews of the auto_v2 spatial encoding outputs.

Read-only:
  - reads everything from mp-data/processed/encoded/auto_v2/
  - does not retrain, edit the encoder, or overwrite any v2 artefact
  - writes only into mp-data/processed/encoded/auto_v2/visual_checks/

Outputs:
  map_mask_overlay.png
  map_trajectories_on_mask.png
  map_distance_to_obstacle.png
  map_distance_to_boundary.png
  map_distance_to_entrance.png
  visual_check_contact_sheet.png
  visual_check_summary.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# --------------------------------------------------------------------------- #
# Paths (mirrors encode_spatial_auto_v2.py)
# --------------------------------------------------------------------------- #
HERE     = Path(__file__).resolve().parent
MP_ROOT  = HERE.parent.parent
MP_DATA  = MP_ROOT / "mp-data"
AUTO_V2  = MP_DATA / "processed" / "encoded" / "auto_v2"
OUT_DIR  = AUTO_V2 / "visual_checks"

TRAJ_CSV    = AUTO_V2 / "trajectories_encoded_auto_v2.csv"
WALK_PNG    = AUTO_V2 / "walkable_mask.png"
OBS_PNG     = AUTO_V2 / "obstacle_mask.png"
BND_PNG     = AUTO_V2 / "boundary_mask.png"
ENTRY_CSV   = AUTO_V2 / "entry_exit_points.csv"

# Optional: top-down image (homography unknown wrt world coords, so we don't
# warp it; we just note its existence).
TOPDOWN_IMG = MP_DATA / "raw" / "images" / "top-down.png"

# Must match the encoder constants so masks align to world coordinates.
GRID_RES_M = 0.10
PAD_M      = 3.0

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


def load_mask(path: Path) -> Optional[np.ndarray]:
    """Load a mask PNG. Encoder saved with img[::-1,:] (vertical flip), so we
    flip back here so row 0 = smallest y."""
    if not path.exists():
        return None
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    arr = arr[::-1, :]                       # undo encoder flip
    return arr > 0                            # binary mask


def compute_world_extent(df: pd.DataFrame, ali: Dict[str, str]) -> Dict:
    wx = pd.to_numeric(df[ali["world_x"]], errors="coerce").to_numpy()
    wy = pd.to_numeric(df[ali["world_y"]], errors="coerce").to_numpy()
    valid = np.isfinite(wx) & np.isfinite(wy)
    wx, wy = wx[valid], wy[valid]
    x_min, x_max = wx.min() - PAD_M, wx.max() + PAD_M
    y_min, y_max = wy.min() - PAD_M, wy.max() + PAD_M
    W = int(np.ceil((x_max - x_min) / GRID_RES_M))
    H = int(np.ceil((y_max - y_min) / GRID_RES_M))
    return {
        "x_min": x_min, "y_min": y_min,
        "x_max": x_min + W * GRID_RES_M,
        "y_max": y_min + H * GRID_RES_M,
        "W": W, "H": H,
        "extent": (x_min, x_min + W * GRID_RES_M,
                   y_min, y_min + H * GRID_RES_M),
    }


def verify_alignment(grid: Dict, masks: Dict[str, np.ndarray]) -> bool:
    """All loaded masks should be (H, W); flag mismatch."""
    expected = (grid["H"], grid["W"])
    bad = []
    for name, m in masks.items():
        if m is None:
            continue
        if m.shape != expected:
            bad.append(f"{name} {m.shape} != expected {expected}")
    if bad:
        for b in bad:
            print(f"[WARN]  Mask alignment mismatch: {b}")
        return False
    return True


# --------------------------------------------------------------------------- #
# Plotting primitives
# --------------------------------------------------------------------------- #
def draw_masks(ax, grid: Dict, walk: Optional[np.ndarray],
               obstacle: Optional[np.ndarray],
               boundary: Optional[np.ndarray],
               entries_df: Optional[pd.DataFrame],
               with_entries: bool = True) -> None:
    """Layer order: walkable (light) → obstacle (darker) → boundary outline
    → entries. Origin lower so world_y grows upward."""
    extent = grid["extent"]

    # Light walkable fill — pale green using a custom 0→1 colormap.
    if walk is not None:
        walk_cm = LinearSegmentedColormap.from_list(
            "walk", [(0, 0, 0, 0), (0.55, 0.78, 0.60, 0.55)])
        ax.imshow(walk.astype(np.uint8), cmap=walk_cm, origin="lower",
                  extent=extent, interpolation="nearest",
                  vmin=0, vmax=1)

    # Darker obstacle fill — slate red.
    if obstacle is not None:
        obs_cm = LinearSegmentedColormap.from_list(
            "obs", [(0, 0, 0, 0), (0.70, 0.20, 0.20, 0.75)])
        ax.imshow(obstacle.astype(np.uint8), cmap=obs_cm, origin="lower",
                  extent=extent, interpolation="nearest",
                  vmin=0, vmax=1)

    # Boundary as a strong outline.
    if boundary is not None and boundary.any():
        by, bx = np.where(boundary)
        ax.scatter(grid["x_min"] + (bx + 0.5) * GRID_RES_M,
                   grid["y_min"] + (by + 0.5) * GRID_RES_M,
                   s=0.6, c="#0b3d91", marker="s", alpha=0.95)

    if with_entries and entries_df is not None and not entries_df.empty:
        ax.scatter(entries_df["center_x"], entries_df["center_y"],
                   s=130, c="#f1c40f", marker="*",
                   edgecolor="black", linewidth=0.8, zorder=10,
                   label="entry/exit")


def add_mask_legend(ax) -> None:
    handles = [
        Patch(facecolor=(0.55, 0.78, 0.60, 0.55), edgecolor="none",
              label="walkable"),
        Patch(facecolor=(0.70, 0.20, 0.20, 0.75), edgecolor="none",
              label="obstacle"),
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor="#0b3d91", markersize=6,
               label="boundary"),
        Line2D([0], [0], marker="*", color="none",
               markerfacecolor="#f1c40f", markeredgecolor="black",
               markersize=11, label="entry/exit cluster"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right",
              framealpha=0.85)


def set_world_axes(ax, grid: Dict, title: str) -> None:
    ax.set_xlim(grid["x_min"], grid["x_max"])
    ax.set_ylim(grid["y_min"], grid["y_max"])
    ax.set_xlabel("world_x (metres)", fontsize=9)
    ax.set_ylabel("world_y (metres)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, lw=0.25, alpha=0.35)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_mask_overlay(grid, masks, entries_df, path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 12))
    draw_masks(ax, grid, masks["walkable"], masks["obstacle"],
               masks["boundary"], entries_df)
    add_mask_legend(ax)
    set_world_axes(ax, grid,
                   "Map mask overlay — walkable + obstacle + boundary + entry/exit")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories_on_mask(grid, masks, entries_df,
                              df, ali, path: Path,
                              n_highlight: int = 10) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 12))
    draw_masks(ax, grid, masks["walkable"], masks["obstacle"],
               masks["boundary"], entries_df)

    # All trajectories — thin grey.
    for _, g in df.groupby(ali["track_id"]):
        gs = g.sort_values(ali["frame"])
        ax.plot(gs[ali["world_x"]], gs[ali["world_y"]],
                "-", color="#2c3e50", lw=0.35, alpha=0.35)

    # Highlight the 10 longest tracks.
    lengths = df.groupby(ali["track_id"]).size().sort_values(ascending=False)
    top_ids = lengths.head(n_highlight).index.tolist()
    cmap = plt.get_cmap("tab10")
    for i, tid in enumerate(top_ids):
        g = df[df[ali["track_id"]] == tid].sort_values(ali["frame"])
        ax.plot(g[ali["world_x"]], g[ali["world_y"]],
                "-", color=cmap(i % 10), lw=1.3, alpha=0.95,
                label=f"track {int(tid)} ({len(g)})" if i < 5 else None)

    add_mask_legend(ax)
    set_world_axes(ax, grid,
                   f"Trajectories over masks (all faint, top {n_highlight} highlighted)")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories_coloured(grid, masks, entries_df, df, ali,
                               feature: str, cmap_name: str,
                               path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.4, 12))
    draw_masks(ax, grid, masks["walkable"], masks["obstacle"],
               masks["boundary"], entries_df, with_entries=True)

    if feature not in df.columns:
        ax.text(0.5, 0.5, f"{feature} not in CSV",
                transform=ax.transAxes, ha="center")
        set_world_axes(ax, grid, f"Trajectories coloured by `{feature}` (missing)")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig); return

    s = pd.to_numeric(df[feature], errors="coerce")
    ok = s.notna() & np.isfinite(df[ali["world_x"]]) \
                  & np.isfinite(df[ali["world_y"]])
    if not ok.any():
        ax.text(0.5, 0.5, f"{feature}: no valid values",
                transform=ax.transAxes, ha="center")
        set_world_axes(ax, grid, f"Trajectories coloured by `{feature}`")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig); return

    vmin = float(np.nanpercentile(s[ok], 1))
    vmax = float(np.nanpercentile(s[ok], 99))
    sc = ax.scatter(df.loc[ok, ali["world_x"]],
                    df.loc[ok, ali["world_y"]],
                    c=s[ok], s=1.6, cmap=cmap_name,
                    vmin=vmin, vmax=vmax, alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("metres", fontsize=9)
    set_world_axes(ax, grid, f"Trajectories coloured by `{feature}`")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_contact_sheet(grid, masks, entries_df, df, ali,
                       path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(22, 11))

    # 1. Masks only
    ax = axes[0]
    draw_masks(ax, grid, masks["walkable"], masks["obstacle"],
               masks["boundary"], entries_df)
    set_world_axes(ax, grid, "Masks + entries")

    # 2-4. Trajectories coloured by each distance feature.
    for ax, feat, cmap in zip(
            axes[1:],
            ["dist_to_obstacle", "dist_to_boundary", "dist_to_entrance"],
            ["viridis", "magma", "plasma"]):
        draw_masks(ax, grid, masks["walkable"], masks["obstacle"],
                   masks["boundary"], entries_df, with_entries=False)
        if feat in df.columns:
            s = pd.to_numeric(df[feat], errors="coerce")
            ok = s.notna()
            if ok.any():
                vmin = float(np.nanpercentile(s[ok], 1))
                vmax = float(np.nanpercentile(s[ok], 99))
                sc = ax.scatter(df.loc[ok, ali["world_x"]],
                                df.loc[ok, ali["world_y"]],
                                c=s[ok], s=1.0, cmap=cmap,
                                vmin=vmin, vmax=vmax, alpha=0.85)
                fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02,
                             label="metres")
        set_world_axes(ax, grid, feat)

    fig.suptitle("Auto v2 spatial encoding — visual contact sheet",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def fmt(x) -> str:
    try:
        f = float(x)
        if not np.isfinite(f):
            return "—"
        return f"{f:.3f}"
    except Exception:
        return "—"


def write_summary(path: Path, used_inputs: Dict[str, bool],
                  alignment_ok: bool, used_topdown: bool, grid: Dict,
                  df: pd.DataFrame, ali: Dict[str, str],
                  outputs: Dict[str, Path]) -> None:
    lines = []
    lines.append("# Auto v2 — Visual Check Summary")
    lines.append("")
    lines.append("_Visual previews of the auto v2 spatial encoding. "
                 "Read-only: nothing in auto_v2/ was modified._")
    lines.append("")

    lines.append("## Input files used")
    lines.append("")
    for k, ok in used_inputs.items():
        mark = "✓" if ok else "✗"
        lines.append(f"- {mark} `{k}`")
    lines.append("")
    if used_topdown:
        lines.append("Base layer: trajectories and masks drawn on **synthetic "
                     "world-space grid** (axes in metres). The top-down image "
                     "was located but not warped onto the world canvas — the "
                     "homography to world coordinates is unknown in this "
                     "context, so the masks themselves provide the visual "
                     "reference.")
    else:
        lines.append("Base layer: **synthetic world-space grid** (axes in "
                     "metres). No source plan image was used.")
    lines.append("")

    lines.append("## Alignment")
    lines.append("")
    lines.append(f"- World extent: x ∈ [{grid['x_min']:.2f}, "
                 f"{grid['x_max']:.2f}] m, y ∈ [{grid['y_min']:.2f}, "
                 f"{grid['y_max']:.2f}] m")
    lines.append(f"- Grid: {grid['W']} × {grid['H']} pixels at "
                 f"{GRID_RES_M} m/pixel")
    lines.append(f"- Mask alignment to world grid: "
                 f"{'OK' if alignment_ok else 'WARNING — mismatch detected'}")
    lines.append("")

    lines.append("## What each output shows")
    lines.append("")
    lines.append("- `map_mask_overlay.png` — pale green walkable mask, dark "
                 "red obstacle mask, blue boundary outline, gold ★ markers for "
                 "DBSCAN entry/exit cluster centres. No trajectories.")
    lines.append("- `map_trajectories_on_mask.png` — same mask layer plus "
                 "every trajectory drawn as a thin dark grey line. The 10 "
                 "longest tracks are highlighted in `tab10` colours so you "
                 "can visually trace dominant flows.")
    lines.append("- `map_distance_to_obstacle.png` — trajectory points "
                 "coloured by `dist_to_obstacle` (viridis colormap). Bright "
                 "= farther from the nearest auto-derived obstacle pixel.")
    lines.append("- `map_distance_to_boundary.png` — trajectory points "
                 "coloured by `dist_to_boundary` (magma).")
    lines.append("- `map_distance_to_entrance.png` — trajectory points "
                 "coloured by `dist_to_entrance` (plasma). Bright = farther "
                 "from any DBSCAN entry/exit cluster.")
    lines.append("- `visual_check_contact_sheet.png` — 1×4 sheet that bundles "
                 "the mask overlay plus the three distance-coloured plots "
                 "for one-glance review.")
    lines.append("")

    lines.append("## Feature stats (from `trajectories_encoded_auto_v2.csv`)")
    lines.append("")
    lines.append("| feature | n_valid | mean | median | min | max |")
    lines.append("|---|---|---|---|---|---|")
    for feat in ("dist_to_obstacle", "dist_to_boundary", "dist_to_entrance"):
        if feat not in df.columns:
            lines.append(f"| `{feat}` | — | — | — | — | — |")
            continue
        s = pd.to_numeric(df[feat], errors="coerce")
        ok = s.notna()
        if not ok.any():
            lines.append(f"| `{feat}` | 0 | — | — | — | — |")
            continue
        lines.append(f"| `{feat}` | {int(ok.sum()):,} | "
                     f"{fmt(s[ok].mean())} | {fmt(s[ok].median())} | "
                     f"{fmt(s[ok].min())} | {fmt(s[ok].max())} |")
    lines.append("")

    lines.append("## Generated files")
    lines.append("")
    for k, p in outputs.items():
        lines.append(f"- `{p.relative_to(MP_ROOT)}`")
    lines.append("")
    lines.append("_End of visual check summary._")
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO]  Output folder: {OUT_DIR}")

    used_inputs = {
        "trajectories_encoded_auto_v2.csv": TRAJ_CSV.exists(),
        "walkable_mask.png":               WALK_PNG.exists(),
        "obstacle_mask.png":               OBS_PNG.exists(),
        "boundary_mask.png":               BND_PNG.exists(),
        "entry_exit_points.csv":           ENTRY_CSV.exists(),
    }
    for k, ok in used_inputs.items():
        print(f"[INFO]  {('found ' if ok else 'MISSING')} {k}")

    if not TRAJ_CSV.exists():
        print(f"[FATAL] {TRAJ_CSV} not present.")
        sys.exit(1)

    df = pd.read_csv(TRAJ_CSV)
    ali = {k: find_alias(df.columns, k)
           for k in ("track_id", "frame", "world_x", "world_y")}
    missing = [k for k, v in ali.items() if v is None]
    if missing:
        print(f"[FATAL] CSV missing columns: {missing}")
        sys.exit(1)
    print(f"[INFO]  Trajectory CSV: {len(df):,} rows, "
          f"{df[ali['track_id']].nunique():,} unique tracks")

    grid = compute_world_extent(df, ali)
    print(f"[INFO]  World extent (m): "
          f"x∈[{grid['x_min']:.2f},{grid['x_max']:.2f}]  "
          f"y∈[{grid['y_min']:.2f},{grid['y_max']:.2f}]   "
          f"grid={grid['W']}×{grid['H']}")

    masks = {
        "walkable": load_mask(WALK_PNG),
        "obstacle": load_mask(OBS_PNG),
        "boundary": load_mask(BND_PNG),
    }
    for name, m in masks.items():
        if m is None:
            print(f"[WARN]  mask not loaded: {name}")
        else:
            print(f"[INFO]  mask {name}: shape={m.shape} "
                  f"true_px={int(m.sum()):,}")

    alignment_ok = verify_alignment(grid, masks)
    if not alignment_ok:
        print("[WARN]  At least one mask does not match the expected world "
              "grid. Plots will still be produced on the synthetic grid; the "
              "mismatched mask will simply not align — flagged in summary.")

    entries_df = (pd.read_csv(ENTRY_CSV) if ENTRY_CSV.exists()
                  else pd.DataFrame(columns=["cluster_id", "center_x",
                                             "center_y", "n_points"]))
    print(f"[INFO]  Entry/exit clusters loaded: {len(entries_df)}")

    used_topdown = TOPDOWN_IMG.exists()
    if used_topdown:
        print(f"[INFO]  top-down image present at {TOPDOWN_IMG}, but it is "
              "not warped (homography unknown wrt world coords)")

    # ------------------------- generate plots ---------------------------- #
    outputs = {
        "mask_overlay":       OUT_DIR / "map_mask_overlay.png",
        "traj_on_mask":       OUT_DIR / "map_trajectories_on_mask.png",
        "dist_obstacle":      OUT_DIR / "map_distance_to_obstacle.png",
        "dist_boundary":      OUT_DIR / "map_distance_to_boundary.png",
        "dist_entrance":      OUT_DIR / "map_distance_to_entrance.png",
        "contact_sheet":      OUT_DIR / "visual_check_contact_sheet.png",
        "summary":            OUT_DIR / "visual_check_summary.md",
    }

    plot_mask_overlay(grid, masks, entries_df, outputs["mask_overlay"])
    print(f"[OK]    {outputs['mask_overlay'].name}")

    plot_trajectories_on_mask(grid, masks, entries_df, df, ali,
                              outputs["traj_on_mask"])
    print(f"[OK]    {outputs['traj_on_mask'].name}")

    plot_trajectories_coloured(grid, masks, entries_df, df, ali,
                               "dist_to_obstacle", "viridis",
                               outputs["dist_obstacle"])
    print(f"[OK]    {outputs['dist_obstacle'].name}")

    plot_trajectories_coloured(grid, masks, entries_df, df, ali,
                               "dist_to_boundary", "magma",
                               outputs["dist_boundary"])
    print(f"[OK]    {outputs['dist_boundary'].name}")

    plot_trajectories_coloured(grid, masks, entries_df, df, ali,
                               "dist_to_entrance", "plasma",
                               outputs["dist_entrance"])
    print(f"[OK]    {outputs['dist_entrance'].name}")

    plot_contact_sheet(grid, masks, entries_df, df, ali,
                       outputs["contact_sheet"])
    print(f"[OK]    {outputs['contact_sheet'].name}")

    write_summary(outputs["summary"], used_inputs, alignment_ok,
                  used_topdown, grid, df, ali, outputs)
    print(f"[OK]    {outputs['summary'].name}")

    print("\nGenerated files:")
    for k, p in outputs.items():
        print(f"  {k:<16s}  {p}")


if __name__ == "__main__":
    main()
