"""
visualize_auto_v21_on_plan.py
-----------------------------
Overlay the selected v2.1 spatial encoding on the original top-down plan image.

Transformation:
  The plan image ↔ world coordinates relationship is a simple affine,
  recorded in mp-data/raw/calibration/calib_macba.json:

      world_x = (plan_px_x - origin_x) * mpp
      world_y = (plan_px_y - origin_y) * mpp     (negated if invert_plan_y)

  Inverse used here:
      plan_px_x = world_x / mpp + origin_x
      plan_px_y = world_y / mpp + origin_y       (sign-flipped if invert_y)

  This mapping is used everywhere — for trajectory points, mask extents,
  and entry/exit stars.

Validation:
  Before any plot is rendered, 100 random trajectory points are projected
  to plan pixels. If fewer than 85% land inside the image bounds, or the
  spread is implausible, we **stop with a warning** and only write the
  overlay_summary.md describing the failure — no misleading visuals.

Outputs (all in <variant_folder>/plan_overlays/):
  plan_overlay_masks.png
  plan_overlay_trajectories.png
  plan_overlay_distance_obstacle.png
  plan_overlay_distance_boundary.png
  plan_overlay_distance_entrance.png
  plan_overlay_contact_sheet.png
  overlay_summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE     = Path(__file__).resolve().parent
MP_ROOT  = HERE.parent.parent
MP_DATA  = MP_ROOT / "mp-data"
ENCODED  = MP_DATA / "processed" / "encoded"
CALIB    = MP_DATA / "raw" / "calibration" / "calib_macba.json"
TOPDOWN  = MP_DATA / "raw" / "images" / "top-down.png"

DEFAULT_VARIANT = "auto_v21C"

# Encoder constants (same as encode_spatial_auto_v2.py).
GRID_RES_M = 0.10
PAD_M      = 3.0

# Alignment thresholds.
MIN_INSIDE_FRAC = 0.85
N_SAMPLE_POINTS = 100


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def load_calibration(path: Path) -> Dict:
    if not path.exists():
        raise SystemExit(f"[FATAL] calibration not found: {path}")
    c = json.loads(path.read_text(encoding="utf-8"))
    needed = ("meters_per_plan_pixel", "plan_origin_pixel")
    missing = [k for k in needed if k not in c]
    if missing:
        raise SystemExit(f"[FATAL] calibration missing keys: {missing}")
    return {
        "mpp":       float(c["meters_per_plan_pixel"]),
        "origin_x":  float(c["plan_origin_pixel"][0]),
        "origin_y":  float(c["plan_origin_pixel"][1]),
        "invert_y":  bool(c.get("invert_plan_y", False)),
        "raw":       c,
    }


def world_to_plan(world_x: np.ndarray, world_y: np.ndarray,
                  calib: Dict) -> Tuple[np.ndarray, np.ndarray]:
    px = world_x / calib["mpp"] + calib["origin_x"]
    if calib["invert_y"]:
        py = -world_y / calib["mpp"] + calib["origin_y"]
    else:
        py = world_y / calib["mpp"] + calib["origin_y"]
    return px, py


def mask_extent_in_plan(grid: Dict, calib: Dict) -> Tuple[float, float, float, float]:
    """Return matplotlib `extent` (left, right, bottom, top) such that the
    mask's row 0 (smallest world_y) lands at the smaller plan_y_pixel."""
    # corners in world
    x0, x1 = grid["x_min"], grid["x_min"] + grid["W"] * GRID_RES_M
    y0, y1 = grid["y_min"], grid["y_min"] + grid["H"] * GRID_RES_M
    px0, py0 = world_to_plan(np.array([x0]), np.array([y0]), calib)
    px1, py1 = world_to_plan(np.array([x1]), np.array([y1]), calib)
    left, right = float(px0[0]), float(px1[0])
    # py0 corresponds to mask row 0; py1 corresponds to mask last row.
    # imshow with origin='upper' places row 0 at extent's `top`.
    return (left, right, float(py1[0]), float(py0[0]))


# --------------------------------------------------------------------------- #
# Data + masks
# --------------------------------------------------------------------------- #
def load_variant(folder: Path) -> Dict:
    csv = folder / "trajectories_encoded.csv"
    if not csv.exists():
        raise SystemExit(f"[FATAL] {csv} missing")
    df = pd.read_csv(csv)
    cols = df.columns.tolist()
    aliases = {
        "track_id": _alias(cols, ["track_id", "person_id", "pid"]),
        "frame":    _alias(cols, ["frame", "frame_number", "frame_idx"]),
        "world_x":  _alias(cols, ["world_x"]),
        "world_y":  _alias(cols, ["world_y"]),
    }
    missing = [k for k, v in aliases.items() if v is None]
    if missing:
        raise SystemExit(f"[FATAL] CSV missing columns: {missing}")

    walk = _load_mask(folder / "walkable_mask.png")
    obs  = _load_mask(folder / "obstacle_mask.png")
    bnd  = _load_mask(folder / "boundary_mask.png")

    entries_csv = folder / "entry_exit_points.csv"
    entries = (pd.read_csv(entries_csv) if entries_csv.exists()
               else pd.DataFrame(columns=["cluster_id", "center_x",
                                          "center_y", "n_points"]))

    return {"df": df, "ali": aliases,
            "walk": walk, "obs": obs, "bnd": bnd,
            "entries": entries, "folder": folder}


def _alias(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def _load_mask(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    arr = arr[::-1, :]                       # undo encoder vertical flip
    return arr > 0


def compute_world_grid(df: pd.DataFrame, ali: Dict[str, str]) -> Dict:
    wx = pd.to_numeric(df[ali["world_x"]], errors="coerce").to_numpy()
    wy = pd.to_numeric(df[ali["world_y"]], errors="coerce").to_numpy()
    valid = np.isfinite(wx) & np.isfinite(wy)
    wx, wy = wx[valid], wy[valid]
    x_min, x_max = wx.min() - PAD_M, wx.max() + PAD_M
    y_min, y_max = wy.min() - PAD_M, wy.max() + PAD_M
    W = int(np.ceil((x_max - x_min) / GRID_RES_M))
    H = int(np.ceil((y_max - y_min) / GRID_RES_M))
    return {"x_min": x_min, "y_min": y_min, "W": W, "H": H}


# --------------------------------------------------------------------------- #
# Alignment validation
# --------------------------------------------------------------------------- #
def validate_alignment(df, ali, calib, plan_shape) -> Dict:
    """Project N random trajectory points to plan pixels and verify."""
    H, W = plan_shape[:2]
    n = min(N_SAMPLE_POINTS, len(df))
    rng = np.random.default_rng(seed=0)
    idx = rng.choice(len(df), size=n, replace=False)
    sub = df.iloc[idx]
    wx = pd.to_numeric(sub[ali["world_x"]], errors="coerce").to_numpy()
    wy = pd.to_numeric(sub[ali["world_y"]], errors="coerce").to_numpy()
    px, py = world_to_plan(wx, wy, calib)
    inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    return {
        "n_sampled":    n,
        "n_inside":     int(inside.sum()),
        "frac_inside":  float(inside.mean()),
        "px_range":     (float(px.min()), float(px.max())),
        "py_range":     (float(py.min()), float(py.max())),
        "image_size":   (W, H),
        "sample_px":    px.tolist(),
        "sample_py":    py.tolist(),
        "sample_idx":   idx.tolist(),
    }


def alignment_passes(val: Dict) -> Tuple[bool, str]:
    if val["frac_inside"] < MIN_INSIDE_FRAC:
        return False, (f"only {val['frac_inside']:.0%} of sampled "
                       f"trajectory points project inside the plan image "
                       f"({val['n_inside']}/{val['n_sampled']}); "
                       f"threshold is {MIN_INSIDE_FRAC:.0%}")
    # Also flag absurd projected spreads.
    W, H = val["image_size"]
    span_x = val["px_range"][1] - val["px_range"][0]
    span_y = val["py_range"][1] - val["py_range"][0]
    if span_x < 5 or span_y < 5:
        return False, "projected trajectory spread is implausibly small"
    if span_x > 5 * W or span_y > 5 * H:
        return False, "projected trajectory spread vastly exceeds image"
    return True, "OK"


# --------------------------------------------------------------------------- #
# Plot primitives
# --------------------------------------------------------------------------- #
def draw_plan_base(ax, plan_rgb) -> None:
    ax.imshow(plan_rgb, interpolation="bilinear")
    ax.set_xlabel("plan_x (pixels)", fontsize=8)
    ax.set_ylabel("plan_y (pixels)", fontsize=8)


def draw_masks_on_plan(ax, walk, obs, bnd, extent,
                       walk_alpha=0.55, obs_alpha=0.65,
                       bnd_alpha=0.95) -> None:
    if walk is not None:
        cm = LinearSegmentedColormap.from_list(
            "walk", [(0, 0, 0, 0), (0.45, 0.78, 0.50, walk_alpha)])
        ax.imshow(walk.astype(np.uint8), cmap=cm, origin="upper",
                  extent=extent, interpolation="nearest", vmin=0, vmax=1)
    if obs is not None:
        cm = LinearSegmentedColormap.from_list(
            "obs", [(0, 0, 0, 0), (0.78, 0.18, 0.18, obs_alpha)])
        ax.imshow(obs.astype(np.uint8), cmap=cm, origin="upper",
                  extent=extent, interpolation="nearest", vmin=0, vmax=1)
    if bnd is not None:
        cm = LinearSegmentedColormap.from_list(
            "bnd", [(0, 0, 0, 0), (0.04, 0.24, 0.57, bnd_alpha)])
        ax.imshow(bnd.astype(np.uint8), cmap=cm, origin="upper",
                  extent=extent, interpolation="nearest", vmin=0, vmax=1)


def draw_entries_on_plan(ax, entries_df, calib, marker_size=160) -> None:
    if entries_df is None or entries_df.empty:
        return
    px, py = world_to_plan(entries_df["center_x"].to_numpy(),
                            entries_df["center_y"].to_numpy(), calib)
    ax.scatter(px, py, s=marker_size, c="#f1c40f", marker="*",
               edgecolor="black", linewidth=0.9, zorder=20)


def add_mask_legend(ax) -> None:
    handles = [
        Patch(facecolor=(0.45, 0.78, 0.50, 0.55), label="walkable"),
        Patch(facecolor=(0.78, 0.18, 0.18, 0.65), label="obstacle"),
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor="#0b3d91", markersize=8, label="boundary"),
        Line2D([0], [0], marker="*", color="none",
               markerfacecolor="#f1c40f", markeredgecolor="black",
               markersize=11, label="entry/exit"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right",
              framealpha=0.85)


def crop_to_data(ax, val: Dict, pad_px: int = 60) -> None:
    """Zoom axes to the projected trajectory bbox so the plan + masks fit
    nicely without huge whitespace."""
    x0, x1 = val["px_range"]; y0, y1 = val["py_range"]
    W, H = val["image_size"]
    left   = max(0, x0 - pad_px); right = min(W, x1 + pad_px)
    top    = max(0, y0 - pad_px); bottom = min(H, y1 + pad_px)
    ax.set_xlim(left, right); ax.set_ylim(bottom, top)


# --------------------------------------------------------------------------- #
# Plot functions
# --------------------------------------------------------------------------- #
def plot_overlay_masks(out: Path, plan, var, extent, val) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 11.5))
    draw_plan_base(ax, plan)
    draw_masks_on_plan(ax, var["walk"], var["obs"], var["bnd"], extent)
    draw_entries_on_plan(ax, var["entries"], CALIB_CACHE["calib"])
    add_mask_legend(ax)
    ax.set_title("Plan overlay — walkable + obstacle + boundary + entries",
                 fontsize=11, fontweight="bold")
    crop_to_data(ax, val)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_trajectories(out: Path, plan, var, extent, val) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 11.5))
    draw_plan_base(ax, plan)
    draw_masks_on_plan(ax, var["walk"], var["obs"], var["bnd"], extent,
                       walk_alpha=0.30, obs_alpha=0.35, bnd_alpha=0.80)
    # all trajectories — thin grey
    df, ali = var["df"], var["ali"]
    calib = CALIB_CACHE["calib"]
    for _, g in df.groupby(ali["track_id"]):
        gs = g.sort_values(ali["frame"])
        px, py = world_to_plan(gs[ali["world_x"]].to_numpy(),
                                gs[ali["world_y"]].to_numpy(), calib)
        ax.plot(px, py, "-", color="#1c2833", lw=0.35, alpha=0.45)
    # highlight top 10 longest
    top_ids = df.groupby(ali["track_id"]).size().sort_values(
        ascending=False).head(10).index.tolist()
    cmap = plt.get_cmap("tab10")
    for i, tid in enumerate(top_ids):
        g = df[df[ali["track_id"]] == tid].sort_values(ali["frame"])
        px, py = world_to_plan(g[ali["world_x"]].to_numpy(),
                                g[ali["world_y"]].to_numpy(), calib)
        ax.plot(px, py, "-", color=cmap(i % 10), lw=1.4, alpha=0.95)
    draw_entries_on_plan(ax, var["entries"], calib)
    add_mask_legend(ax)
    ax.set_title("Plan overlay — trajectories (faint) + top-10 longest "
                 "(coloured)", fontsize=11, fontweight="bold")
    crop_to_data(ax, val)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_distance(out: Path, plan, var, extent, val,
                          feat: str, cmap_name: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9.2, 11.5))
    draw_plan_base(ax, plan)
    draw_masks_on_plan(ax, var["walk"], var["obs"], var["bnd"], extent,
                       walk_alpha=0.18, obs_alpha=0.22, bnd_alpha=0.75)
    df, ali = var["df"], var["ali"]
    calib = CALIB_CACHE["calib"]
    s = pd.to_numeric(df[feat], errors="coerce") if feat in df.columns \
        else None
    if s is None or not s.notna().any():
        ax.text(0.5, 0.5, f"{feat} missing", transform=ax.transAxes,
                ha="center")
    else:
        ok = s.notna()
        vmin = float(np.nanpercentile(s[ok], 1))
        vmax = float(np.nanpercentile(s[ok], 99))
        px, py = world_to_plan(df.loc[ok, ali["world_x"]].to_numpy(),
                                df.loc[ok, ali["world_y"]].to_numpy(), calib)
        sc = ax.scatter(px, py, c=s[ok], s=1.5, cmap=cmap_name,
                        vmin=vmin, vmax=vmax, alpha=0.9)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("metres", fontsize=9)
    draw_entries_on_plan(ax, var["entries"], calib, marker_size=120)
    ax.set_title(f"Plan overlay — trajectories coloured by `{feat}`",
                 fontsize=11, fontweight="bold")
    crop_to_data(ax, val)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_contact_sheet(out: Path, plan, var, extent, val) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(22, 11))
    titles = ["Masks + entries",
              "dist_to_obstacle", "dist_to_boundary", "dist_to_entrance"]
    cmaps  = [None, "viridis", "magma", "plasma"]
    feats  = [None, "dist_to_obstacle", "dist_to_boundary", "dist_to_entrance"]
    calib = CALIB_CACHE["calib"]
    df, ali = var["df"], var["ali"]

    for ax, title, cmap_name, feat in zip(axes, titles, cmaps, feats):
        draw_plan_base(ax, plan)
        if feat is None:
            draw_masks_on_plan(ax, var["walk"], var["obs"], var["bnd"],
                               extent)
            draw_entries_on_plan(ax, var["entries"], calib)
        else:
            draw_masks_on_plan(ax, var["walk"], var["obs"], var["bnd"],
                               extent, walk_alpha=0.18, obs_alpha=0.22,
                               bnd_alpha=0.75)
            if feat in df.columns:
                s = pd.to_numeric(df[feat], errors="coerce")
                ok = s.notna()
                if ok.any():
                    vmin = float(np.nanpercentile(s[ok], 1))
                    vmax = float(np.nanpercentile(s[ok], 99))
                    px, py = world_to_plan(df.loc[ok, ali["world_x"]].to_numpy(),
                                            df.loc[ok, ali["world_y"]].to_numpy(),
                                            calib)
                    sc = ax.scatter(px, py, c=s[ok], s=0.9, cmap=cmap_name,
                                    vmin=vmin, vmax=vmax, alpha=0.9)
                    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02,
                                 label="m")
            draw_entries_on_plan(ax, var["entries"], calib, marker_size=90)
        ax.set_title(title, fontsize=10, fontweight="bold")
        crop_to_data(ax, val)

    fig.suptitle(f"Plan-overlay contact sheet — "
                 f"{var['folder'].name}",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def write_summary(path: Path, variant: Path, calib: Dict, val: Dict,
                  passed: bool, reason: str,
                  outputs: List[Path], stopped: bool) -> None:
    L = []
    L.append("# Plan-overlay Summary")
    L.append("")
    L.append(f"- Variant: `{variant.relative_to(MP_ROOT)}`")
    L.append(f"- Plan image: `{TOPDOWN.relative_to(MP_ROOT)}` "
             f"({val['image_size'][0]} × {val['image_size'][1]} pixels)")
    L.append(f"- Calibration source: `{CALIB.relative_to(MP_ROOT)}`")
    L.append("")
    L.append("## Transformation source")
    L.append("")
    L.append("The plan image ↔ world-coordinates relationship is recorded in "
             "the calibration JSON as a simple affine map.")
    L.append("")
    L.append("```")
    L.append(f"meters_per_plan_pixel = {calib['mpp']:.6f}")
    L.append(f"plan_origin_pixel     = ({calib['origin_x']:.3f}, "
             f"{calib['origin_y']:.3f})")
    L.append(f"invert_plan_y         = {calib['invert_y']}")
    L.append("```")
    L.append("")
    L.append("## Reprojection logic")
    L.append("")
    L.append("World → plan pixel (used for trajectories and entry stars):")
    L.append("")
    L.append("```")
    L.append("plan_px_x = world_x / meters_per_plan_pixel + origin_x")
    L.append("plan_px_y = world_y / meters_per_plan_pixel + origin_y"
             "   # negate world_y first if invert_plan_y")
    L.append("```")
    L.append("")
    L.append("Mask bounding box (used for matplotlib `extent`):")
    L.append("")
    L.append("```")
    L.append("(left, right) = world_x bounds projected to plan_px_x")
    L.append("(top,  bottom) = world_y bounds projected to plan_px_y "
             "(in image-y-down convention)")
    L.append("```")
    L.append("")
    L.append("## Alignment validation")
    L.append("")
    L.append(f"- {val['n_sampled']} random trajectory points projected "
             f"to plan pixels")
    L.append(f"- Inside-image fraction: **{val['frac_inside']:.1%}** "
             f"({val['n_inside']}/{val['n_sampled']})")
    L.append(f"- Projected x range: "
             f"[{val['px_range'][0]:.1f}, {val['px_range'][1]:.1f}] px "
             f"(image width {val['image_size'][0]})")
    L.append(f"- Projected y range: "
             f"[{val['py_range'][0]:.1f}, {val['py_range'][1]:.1f}] px "
             f"(image height {val['image_size'][1]})")
    L.append(f"- Calibration's recorded reprojection error (from the "
             f"original calibration step): "
             f"mean {calib['raw'].get('diagnostics', {}).get('mean_reprojection_error_m', '—')} m, "
             f"max {calib['raw'].get('diagnostics', {}).get('max_reprojection_error_m', '—')} m")
    L.append("")
    L.append(f"- Threshold for rendering: ≥ "
             f"{MIN_INSIDE_FRAC:.0%} of sampled points must land inside "
             f"the image.")
    L.append(f"- **Decision: {'PASSED' if passed else 'FAILED'}** "
             f"({reason})")
    L.append("")
    L.append("## Assumptions")
    L.append("")
    L.append("- The calibration in `calib_macba.json` matches the plan "
             "image at `mp-data/raw/images/top-down.png` (the calibration "
             "names it `top-view.png`; the size and origin coordinates "
             "match the file we have).")
    L.append("- The mapping is treated as a global affine. No per-region "
             "distortion is applied (the calibration's full homography is "
             "for the camera-to-world map, not the plan-to-world map).")
    L.append("- Mask resolution is the encoder default (GRID_RES_M = 0.10 m"
             "/pixel) and is scaled by `0.10 / mpp` ≈ "
             f"{0.10 / calib['mpp']:.3f}× during overlay.")
    L.append("")

    if stopped:
        L.append("## Outputs")
        L.append("")
        L.append("_No PNGs were rendered because alignment failed. Resolve "
                 "the calibration / plan-image mismatch before retrying._")
    else:
        L.append("## Outputs")
        L.append("")
        for p in outputs:
            L.append(f"- `{p.relative_to(MP_ROOT)}`")
    L.append("")
    L.append("_End of overlay summary._")

    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
CALIB_CACHE = {}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=DEFAULT_VARIANT,
                    help="Variant folder name under "
                         "mp-data/processed/encoded/ (default: auto_v21C)")
    args = ap.parse_args()

    variant_folder = ENCODED / args.variant
    if not variant_folder.exists():
        print(f"[FATAL] variant folder not found: {variant_folder}")
        sys.exit(1)

    out_dir = variant_folder / "plan_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO]  Variant: {variant_folder}")
    print(f"[INFO]  Output:  {out_dir}")

    calib = load_calibration(CALIB)
    CALIB_CACHE["calib"] = calib
    print(f"[INFO]  Calibration: mpp={calib['mpp']:.5f}  "
          f"origin=({calib['origin_x']:.2f},{calib['origin_y']:.2f})  "
          f"invert_y={calib['invert_y']}")

    plan = cv2.imread(str(TOPDOWN), cv2.IMREAD_COLOR)
    if plan is None:
        print(f"[FATAL] cannot read plan image: {TOPDOWN}")
        sys.exit(1)
    plan_rgb = cv2.cvtColor(plan, cv2.COLOR_BGR2RGB)
    print(f"[INFO]  Plan image: {plan_rgb.shape[1]} × {plan_rgb.shape[0]} px")

    var = load_variant(variant_folder)
    print(f"[INFO]  Loaded variant CSV: {len(var['df']):,} rows   "
          f"entries: {len(var['entries'])}")

    val = validate_alignment(var["df"], var["ali"], calib, plan_rgb.shape)
    passed, reason = alignment_passes(val)
    print(f"[INFO]  Alignment check: {val['n_inside']}/{val['n_sampled']} "
          f"sampled points inside image "
          f"({val['frac_inside']:.1%}); {'PASS' if passed else 'FAIL'} "
          f"({reason})")

    outputs: List[Path] = []
    summary_path = out_dir / "overlay_summary.md"

    if not passed:
        print(f"[STOP]  Alignment confidence too low; refusing to render "
              f"misleading visuals. Writing summary only.")
        write_summary(summary_path, variant_folder, calib, val,
                      passed, reason, outputs, stopped=True)
        print(f"[OK]    Wrote {summary_path}")
        sys.exit(2)

    grid = compute_world_grid(var["df"], var["ali"])
    extent = mask_extent_in_plan(grid, calib)
    print(f"[INFO]  Mask plan-pixel extent: x∈[{extent[0]:.1f},"
          f"{extent[1]:.1f}], y∈[{extent[3]:.1f},{extent[2]:.1f}]")

    out_masks      = out_dir / "plan_overlay_masks.png"
    out_traj       = out_dir / "plan_overlay_trajectories.png"
    out_dist_obs   = out_dir / "plan_overlay_distance_obstacle.png"
    out_dist_bnd   = out_dir / "plan_overlay_distance_boundary.png"
    out_dist_ent   = out_dir / "plan_overlay_distance_entrance.png"
    out_contact    = out_dir / "plan_overlay_contact_sheet.png"

    plot_overlay_masks(out_masks, plan_rgb, var, extent, val);    print(f"[OK]    {out_masks.name}")
    plot_overlay_trajectories(out_traj, plan_rgb, var, extent, val); print(f"[OK]    {out_traj.name}")
    plot_overlay_distance(out_dist_obs, plan_rgb, var, extent, val,
                          "dist_to_obstacle", "viridis"); print(f"[OK]    {out_dist_obs.name}")
    plot_overlay_distance(out_dist_bnd, plan_rgb, var, extent, val,
                          "dist_to_boundary", "magma");   print(f"[OK]    {out_dist_bnd.name}")
    plot_overlay_distance(out_dist_ent, plan_rgb, var, extent, val,
                          "dist_to_entrance", "plasma");  print(f"[OK]    {out_dist_ent.name}")
    plot_contact_sheet(out_contact, plan_rgb, var, extent, val)
    print(f"[OK]    {out_contact.name}")

    outputs = [out_masks, out_traj, out_dist_obs, out_dist_bnd, out_dist_ent,
               out_contact]
    write_summary(summary_path, variant_folder, calib, val,
                  passed, reason, outputs, stopped=False)
    print(f"[OK]    {summary_path.name}")

    print()
    print("Generated files:")
    for p in outputs + [summary_path]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
