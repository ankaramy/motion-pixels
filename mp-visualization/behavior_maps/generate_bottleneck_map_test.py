"""
generate_bottleneck_map_test.py
-------------------------------
Visualization-ONLY repair pass for the Plaça Catalunya bottleneck map.

This script does NOT retrack, recalibrate, or modify any source data. It reads
the existing bottleneck cell CSV and the existing plan image, then produces a
single presentation-quality figure clipped exactly to the plan image extent.

What it fixes versus the raw pipeline heatmap (compute_bottlenecks.py):
  * The raw map auto-zooms to the union of image + data bounds. Trajectory
    artifacts that fall well below the plan (down to world-y ~ -94 m, while the
    plan only covers world-y ~ -29 .. +18 m) blow the frame out, leaving the
    plan tiny inside a huge empty coordinate box.
      -> Here the plan image extent is the AUTHORITATIVE visual frame. Cells
         outside that rectangle are masked out and the axis limits are locked
         to the plan corners. (Visual clipping only — calibration untouched.)
  * The raw map draws blocky 1 m square cells.
      -> Here the cell scores are rasterized and Gaussian-smoothed into a soft
         continuous field; weak values fade to transparent.
  * The raw map carries a technical title, axes, ticks, and a large colorbar.
      -> Here axes are removed, a small corner label is added, and only a
         minimal colorbar is kept.

The geometry (plan-pixel -> world homography, plan world extent) is computed the
SAME way compute_bottlenecks._render_top_view_bg does, so this overlay aligns
with the existing pipeline output without any recalibration.

Usage (defaults point at the discovered Plaça Catalunya data):
    python generate_bottleneck_map_test.py
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patheffects import withStroke
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Default paths (discovered Plaça Catalunya data — filtered_250m is the
# cleanest / most complete bottleneck product for this site)
# ---------------------------------------------------------------------------

SITE_ROOT   = Path(r"C:\Users\OWNER\Desktop\new_datasets\placa_catalunya_01")
DEF_CSV     = SITE_ROOT / "filtered_250m" / "bottlenecks" / "bottleneck_cells.csv"
DEF_PLAN    = SITE_ROOT / "plan" / "placa-catalunya.png"
DEF_CALIB   = SITE_ROOT / "calibration" / "calib.json"

OUT_DIR      = Path(__file__).resolve().parent / "outputs"
OUT_PNG      = OUT_DIR / "placa_catalunya_bottleneck_test.png"
OUT_REPORT   = OUT_DIR / "placa_catalunya_bottleneck_test_report.md"
OUT_PNG_V2   = OUT_DIR / "placa_catalunya_bottleneck_test_v2.png"
OUT_REPORT_V2 = OUT_DIR / "placa_catalunya_bottleneck_test_v2_report.md"
OUT_PNG_V3   = OUT_DIR / "placa_catalunya_bottleneck_test_v3.png"
OUT_CMP_V3   = OUT_DIR / "placa_catalunya_bottleneck_test_v3_comparison.png"
OUT_REPORT_V3 = OUT_DIR / "placa_catalunya_bottleneck_test_v3_report.md"
OUT_PNG_V2B  = OUT_DIR / "placa_catalunya_bottleneck_test_v2b.png"
OUT_CMP_V2B  = OUT_DIR / "placa_catalunya_bottleneck_test_v2b_comparison.png"
OUT_REPORT_V2B = OUT_DIR / "placa_catalunya_bottleneck_test_v2b_report.md"
OUT_PNG_V2C  = OUT_DIR / "placa_catalunya_bottleneck_test_v2c.png"
OUT_CMP_V2C  = OUT_DIR / "placa_catalunya_bottleneck_test_v2b_v2c_comparison.png"
OUT_REPORT_V2C = OUT_DIR / "placa_catalunya_bottleneck_test_v2c_report.md"
OUT_PNG_V2D  = OUT_DIR / "placa_catalunya_bottleneck_test_v2d.png"
OUT_CMP_V2D  = OUT_DIR / "placa_catalunya_bottleneck_test_v2b_v2d_comparison.png"
OUT_REPORT_V2D = OUT_DIR / "placa_catalunya_bottleneck_test_v2d_report.md"

# --- v2d config ------------------------------------------------------------
# Grow strong cells only; never shrink weak cells below the base. Clamp the
# largest square to a fraction of grid spacing so squares never overlap.
BASE_SCALE_V2D       = 0.70
GROWTH_AMOUNT_V2D    = 0.25   # scale range 0.70 -> 0.95
SIZE_EXPONENT_V2D    = 1.6
MAX_FRAC_SPACING_V2D = 0.85

# --- v2b config ------------------------------------------------------------
# Square edge scales linearly with normalized bottleneck score.
MIN_SQUARE_SCALE = 0.45
MAX_SQUARE_SCALE = 1.00
BG_OPACITY_V2B   = 0.55     # +0.10 vs v2's 0.45

# --- v2c config ------------------------------------------------------------
# More dramatic, nonlinear size scaling. Strongest cells may exceed base size;
# weakest become noticeably small. Same palette/background/clipping as v2b.
MIN_SQUARE_SCALE_V2C = 0.25
MAX_SQUARE_SCALE_V2C = 1.20
SIZE_EXPONENT_V2C    = 2.2   # Option B (score_norm ** 2.2)

# --- v3 config -------------------------------------------------------------
# Quantile threshold: hide the bottom THRESHOLD fraction of bottleneck scores
# so weak cells disappear and strong cells dominate. (0.30 => hide bottom 30%.)
THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# Plan -> world geometry (identical method to compute_bottlenecks.py)
# ---------------------------------------------------------------------------

def warp_plan_to_world(plan_path: Path, calib_path: Path):
    """
    Warp the plan image into world space using the plan_points_px -> world_points
    homography. Returns (warped_rgb, (xmin, xmax, ymin, ymax)).

    The warped canvas is oriented exactly like the source plan image (row 0 =
    plan top edge = world ymin), so it can be drawn with
    extent=[xmin, xmax, ymax, ymin] and origin='upper'.
    """
    calib = json.loads(Path(calib_path).read_text(encoding="utf-8"))
    plan_pts  = np.array(calib.get("plan_points_px", []), dtype=np.float32)
    world_pts = np.array(calib.get("world_points",   []), dtype=np.float32)
    if len(plan_pts) < 4:
        raise ValueError("calib needs >= 4 plan_points_px / world_points pairs")

    H_pw, _ = cv2.findHomography(plan_pts, world_pts, cv2.RANSAC, 3.0)
    if H_pw is None:
        raise RuntimeError("Could not compute plan->world homography")

    img_bgr = cv2.imread(str(plan_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read plan image: {plan_path}")
    h_img, w_img = img_bgr.shape[:2]

    corners = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]],
                       dtype=np.float32)
    cw = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H_pw).reshape(-1, 2)
    xmin, xmax = float(cw[:, 0].min()), float(cw[:, 0].max())
    ymin, ymax = float(cw[:, 1].min()), float(cw[:, 1].max())

    world_diag = float(np.hypot(xmax - xmin, ymax - ymin))
    img_diag   = float(np.hypot(w_img, h_img))
    res = min(img_diag / max(world_diag, 1e-6), 400.0)  # px per world metre

    out_w = max(int((xmax - xmin) * res), 2)
    out_h = max(int((ymax - ymin) * res), 2)

    T = np.array([[res, 0,   -xmin * res],
                  [0,   res, -ymin * res],
                  [0,   0,    1          ]], dtype=np.float64)
    H_total = T @ H_pw.astype(np.float64)
    warped_bgr = cv2.warpPerspective(
        img_bgr, H_total, (out_w, out_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
    return warped_rgb, (xmin, xmax, ymin, ymax)


# ---------------------------------------------------------------------------
# Background styling — light architectural underlay
# ---------------------------------------------------------------------------

def to_architectural_underlay(rgb: np.ndarray,
                              desaturate: float = 0.85,
                              brighten: float = 0.55,
                              contrast: float = 0.65) -> np.ndarray:
    """
    Turn a plan image into a pale, low-contrast architectural underlay.

    desaturate : 0..1 fraction pulled toward grayscale
    brighten   : 0..1 fraction blended toward white
    contrast   : <1 lowers contrast around mid-grey
    """
    img = rgb.astype(np.float32) / 255.0

    # Desaturate toward luminance
    lum = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
    img = (1 - desaturate) * img + desaturate * lum[..., None]

    # Lower contrast around mid-grey
    img = 0.5 + (img - 0.5) * contrast

    # Brighten toward white
    img = (1 - brighten) * img + brighten * 1.0

    return np.clip(img, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Smooth bottleneck field
# ---------------------------------------------------------------------------

def rasterize_field(cells: pd.DataFrame, extent, res: float, sigma_m: float):
    """
    Rasterize cell bottleneck_score onto a regular world grid and Gaussian-smooth.

    Returns (field, (xmin, xmax, ymin, ymax)) where field row 0 == world ymin
    (so it displays with extent=[xmin, xmax, ymax, ymin], origin='upper').
    """
    xmin, xmax, ymin, ymax = extent
    W = max(int(round((xmax - xmin) * res)), 2)
    Hgrid = max(int(round((ymax - ymin) * res)), 2)

    field = np.zeros((Hgrid, W), dtype=np.float32)
    cols = ((cells["cell_x"].to_numpy() - xmin) / (xmax - xmin) * (W - 1))
    rows = ((cells["cell_y"].to_numpy() - ymin) / (ymax - ymin) * (Hgrid - 1))
    cols = np.clip(np.round(cols).astype(int), 0, W - 1)
    rows = np.clip(np.round(rows).astype(int), 0, Hgrid - 1)
    scores = cells["bottleneck_score"].to_numpy()

    # Accumulate max score per pixel (cells are 1 m, grid is finer)
    for r, c, s in zip(rows, cols, scores):
        if s > field[r, c]:
            field[r, c] = s

    sigma_px = sigma_m * res
    field = gaussian_filter(field, sigma=sigma_px, mode="constant")
    return field


def soft_warm_rgba(field: np.ndarray, vmax: float,
                   floor_frac: float = 0.12, gamma: float = 0.85):
    """
    Map a normalized field to a refined warm RGBA layer.

    Low values fade to fully transparent (below floor_frac of vmax); strong
    values reach near-opaque warm red. Colour comes from a custom yellow->red
    ramp; alpha follows the same intensity with a gamma curve.
    """
    norm = np.clip(field / max(vmax, 1e-9), 0.0, 1.0)

    warm = mcolors.LinearSegmentedColormap.from_list(
        "soft_warm",
        [
            (0.00, "#fff1c2"),   # pale warm sand
            (0.35, "#ffd27f"),   # amber
            (0.62, "#ff9b42"),   # orange
            (0.82, "#f5602a"),   # deep orange
            (1.00, "#c81e1e"),   # strong red
        ],
    )
    rgba = warm(norm)

    # Alpha: transparent below floor, then smooth ramp up to ~0.9
    a = (norm - floor_frac) / (1.0 - floor_frac)
    a = np.clip(a, 0.0, 1.0) ** gamma
    a = a * 0.90
    a[norm < floor_frac] = 0.0
    rgba[..., 3] = a
    return rgba


# ---------------------------------------------------------------------------
# v2 — crisp discrete cell render (architectural occupancy raster)
# ---------------------------------------------------------------------------

def muted_warm_cmap():
    """
    Muted warm yellow -> orange -> brick ramp. Slightly desaturated, no pure
    fluorescent red, no glow — reads as an architectural graphic, not a
    weather forecast.
    """
    return mcolors.LinearSegmentedColormap.from_list(
        "muted_warm",
        [
            (0.00, "#f2e3b3"),   # soft sand
            (0.30, "#e7c483"),   # muted amber
            (0.55, "#d99a55"),   # ochre orange
            (0.78, "#c2673a"),   # terracotta
            (1.00, "#9e3b2e"),   # brick red (not bright)
        ],
    )


def infer_cell_size(cells: pd.DataFrame) -> float:
    """Infer grid cell size from the minimum nonzero spacing of cell centres."""
    xs = np.sort(cells["cell_x"].unique())
    if len(xs) > 1:
        d = np.diff(xs)
        d = d[d > 1e-9]
        if len(d):
            return float(d.min())
    return 1.0


def render_v2(cells, warped_rgb, extent, top_k, out_png, dpi):
    """
    Render discrete bottleneck cells as crisp rounded squares over a pale
    architectural underlay. Keeps grid identity / occupancy resolution.
    No title, no colorbar, no labels — background + cells + rank markers only.
    """
    from matplotlib.patches import FancyBboxPatch

    xmin, xmax, ymin, ymax = extent
    cell_size = infer_cell_size(cells)
    half = cell_size / 2.0

    cmap = muted_warm_cmap()
    vmax = float(np.percentile(cells["bottleneck_score"], 99.0))
    vmax = vmax if vmax > 0 else float(cells["bottleneck_score"].max() or 1.0)
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)

    world_w, world_h = xmax - xmin, ymax - ymin
    fig_w = 11.0
    fig_h = fig_w * (world_h / world_w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    img_extent = [xmin, xmax, ymax, ymin]  # plan orientation (y down)
    ax.imshow(warped_rgb_underlay(warped_rgb), extent=img_extent, origin="upper",
              alpha=0.45, zorder=0, interpolation="bilinear")

    # Discrete cells: subtle transparency scaled with score so weak cells
    # recede but stay visible (occupancy preserved). Slightly rounded corners
    # + thin matching edge keeps them crisp, not blocky.
    rounding = cell_size * 0.18
    for _, row in cells.iterrows():
        t = norm(row["bottleneck_score"])
        color = cmap(t)
        alpha = 0.35 + 0.55 * float(np.clip(t, 0, 1))   # 0.35 .. 0.90
        patch = FancyBboxPatch(
            (row["cell_x"] - half + cell_size * 0.06,
             row["cell_y"] - half + cell_size * 0.06),
            cell_size * 0.88, cell_size * 0.88,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            facecolor=color, edgecolor=color, linewidth=0.3,
            alpha=alpha, zorder=2, mutation_aspect=1.0,
            antialiased=True,
        )
        ax.add_patch(patch)

    # Ranked markers — small, elegant, secondary to the cell layer
    top = cells.sort_values("bottleneck_score", ascending=False).head(top_k)
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        ax.scatter(row["cell_x"], row["cell_y"], s=64, facecolor="white",
                   edgecolor="#7a2018", linewidth=0.9, zorder=4, alpha=0.95)
        ax.text(row["cell_x"], row["cell_y"], str(rank),
                ha="center", va="center", fontsize=6.0, color="#7a2018",
                fontweight="bold", zorder=5)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_png, dpi=dpi, facecolor="white",
                bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return cell_size, len(cells)


def warped_rgb_underlay(warped_rgb):
    """Architectural underlay for the warped plan (cached-free helper)."""
    return to_architectural_underlay(warped_rgb)


# ---------------------------------------------------------------------------
# v3 — borderless occupancy raster with opacity hierarchy
# ---------------------------------------------------------------------------

def desaturated_warm_cmap():
    """
    Warm yellow -> orange -> muted red, a touch less saturated than v2 and
    without bright/fluorescent red. Architectural tone, not weather radar.
    """
    return mcolors.LinearSegmentedColormap.from_list(
        "warm_v3",
        [
            (0.00, "#ecd9a6"),   # soft straw
            (0.32, "#e0bd7c"),   # muted amber
            (0.58, "#cf9354"),   # ochre
            (0.80, "#b56a40"),   # clay orange
            (1.00, "#8f3a30"),   # muted brick red
        ],
    )


def render_v3(cells, warped_rgb, extent, top_k, out_png, dpi,
              threshold=THRESHOLD):
    """
    Borderless cell raster. Cells are clean squares with NO outline; raster
    structure reads through adjacency. Opacity scales with score so weak cells
    fade out and strong cells dominate. Bottom `threshold` fraction of scores
    is hidden entirely.

    Returns dict with threshold stats for the report.
    """
    from matplotlib.patches import Rectangle

    xmin, xmax, ymin, ymax = extent
    cell_size = infer_cell_size(cells)
    half = cell_size / 2.0

    scores = cells["bottleneck_score"].to_numpy()
    cutoff = float(np.quantile(scores, threshold))     # quantile cut
    keep_mask = scores >= cutoff
    kept = cells[keep_mask].copy()
    n_hidden = int((~keep_mask).sum())
    n_kept = int(keep_mask.sum())

    cmap = desaturated_warm_cmap()
    vmax = float(np.percentile(scores, 99.0))
    vmax = vmax if vmax > cutoff else float(scores.max())
    # Hue: normalize across the FULL score range for stable colour meaning
    cnorm = mcolors.Normalize(vmin=0.0, vmax=vmax)

    world_w, world_h = xmax - xmin, ymax - ymin
    fig_w = 11.0
    fig_h = fig_w * (world_h / world_w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    img_extent = [xmin, xmax, ymax, ymin]
    ax.imshow(to_architectural_underlay(warped_rgb), extent=img_extent,
              origin="upper", alpha=0.45, zorder=0, interpolation="bilinear")

    # Opacity hierarchy: map score within [cutoff, vmax] -> alpha, gamma-shaped
    # so medium cells stay light and only strong cells become fully opaque.
    span = max(vmax - cutoff, 1e-9)
    for _, row in kept.iterrows():
        s = row["bottleneck_score"]
        rgb = cmap(cnorm(s))[:3]
        t = np.clip((s - cutoff) / span, 0.0, 1.0)
        alpha = 0.15 + 0.85 * (t ** 1.25)              # 0.15 .. 1.0
        rect = Rectangle(
            (row["cell_x"] - half, row["cell_y"] - half),
            cell_size, cell_size,
            facecolor=(rgb[0], rgb[1], rgb[2], float(alpha)),
            edgecolor="none", linewidth=0.0, antialiased=False, zorder=2,
        )
        ax.add_patch(rect)

    # Ranked markers — ~20% smaller than v2 (s 64 -> 51, font 6.0 -> 4.8)
    top = cells.sort_values("bottleneck_score", ascending=False).head(top_k)
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        ax.scatter(row["cell_x"], row["cell_y"], s=51, facecolor="white",
                   edgecolor="#7a2018", linewidth=0.8, zorder=4, alpha=0.95)
        ax.text(row["cell_x"], row["cell_y"], str(rank),
                ha="center", va="center", fontsize=4.8, color="#7a2018",
                fontweight="bold", zorder=5)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_png, dpi=dpi, facecolor="white",
                bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

    return {
        "cell_size": cell_size, "threshold": threshold, "cutoff": cutoff,
        "n_total": len(cells), "n_kept": n_kept, "n_hidden": n_hidden,
        "vmax": vmax,
    }


def make_comparison(v2_png: Path, v3_png: Path, out_png: Path):
    """Side-by-side V2 | V3 panel (simple horizontal concat, matched height)."""
    from PIL import Image
    a = Image.open(v2_png).convert("RGB")
    b = Image.open(v3_png).convert("RGB")
    h = min(a.height, b.height)
    a = a.resize((int(a.width * h / a.height), h))
    b = b.resize((int(b.width * h / b.height), h))
    gap = 24
    canvas = Image.new("RGB", (a.width + gap + b.width, h), "white")
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + gap, 0))
    canvas.save(out_png)


# ---------------------------------------------------------------------------
# v2b — V2 cell look + variable square size + cleaner yellow->red palette
# ---------------------------------------------------------------------------

def clean_yellow_red_cmap():
    """Pale yellow -> orange -> red. No brown, no maroon, no muddy mid."""
    return mcolors.LinearSegmentedColormap.from_list(
        "yellow_red_v2b",
        [
            (0.00, "#FFF3B0"),   # pale yellow
            (0.40, "#FDBA3B"),   # amber
            (0.72, "#F97316"),   # orange
            (1.00, "#DC2626"),   # red
        ],
    )


def render_v2b(cells, warped_rgb, extent, top_k, out_png, dpi,
               min_scale=MIN_SQUARE_SCALE, max_scale=MAX_SQUARE_SCALE,
               bg_opacity=BG_OPACITY_V2B, size_exponent=1.0,
               max_frac_spacing=None):
    """
    V2-style discrete cells, but each square's edge scales with its bottleneck
    score (strong = larger), centred on the original cell position so location
    never shifts. Cleaner yellow->red palette; slightly more visible background.

    size_exponent > 1 makes the size response nonlinear/dramatic
    (scale = min + (score_norm ** size_exponent) * (max - min)).
    """
    from matplotlib.patches import FancyBboxPatch

    xmin, xmax, ymin, ymax = extent
    cell_size = infer_cell_size(cells)

    cmap = clean_yellow_red_cmap()
    scores = cells["bottleneck_score"].to_numpy()
    vmax = float(np.percentile(scores, 99.0))
    vmax = vmax if vmax > 0 else float(scores.max() or 1.0)
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)

    world_w, world_h = xmax - xmin, ymax - ymin
    fig_w = 11.0
    fig_h = fig_w * (world_h / world_w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    img_extent = [xmin, xmax, ymax, ymin]
    ax.imshow(to_architectural_underlay(warped_rgb), extent=img_extent,
              origin="upper", alpha=bg_opacity, zorder=0, interpolation="bilinear")

    rounding = cell_size * 0.16
    # Draw weakest first so the strongest (largest) squares sit on top.
    cells_sorted = cells.sort_values("bottleneck_score", ascending=True)
    for _, row in cells_sorted.iterrows():
        t = float(np.clip(norm(row["bottleneck_score"]), 0.0, 1.0))
        color = cmap(t)
        # Square edge scales with normalized score (centred -> no shift).
        # size_exponent shapes the response: >1 = more dramatic.
        scale = min_scale + (t ** size_exponent) * (max_scale - min_scale)
        side = cell_size * scale
        # No-overlap clamp: cap side at a fraction of the grid spacing so
        # neighbouring squares always keep visible gaps.
        if max_frac_spacing is not None:
            side = min(side, max_frac_spacing * cell_size)
        h = side / 2.0
        alpha = 0.45 + 0.50 * t                          # 0.45 .. 0.95
        patch = FancyBboxPatch(
            (row["cell_x"] - h, row["cell_y"] - h), side, side,
            boxstyle=f"round,pad=0,rounding_size={rounding * scale}",
            facecolor=color, edgecolor="none", linewidth=0.0,
            alpha=alpha, zorder=2, antialiased=True,
        )
        ax.add_patch(patch)

    # Ranked markers (kept; slightly smaller to stay secondary under big squares)
    top = cells.sort_values("bottleneck_score", ascending=False).head(top_k)
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        ax.scatter(row["cell_x"], row["cell_y"], s=54, facecolor="white",
                   edgecolor="#9e1414", linewidth=0.85, zorder=4, alpha=0.95)
        ax.text(row["cell_x"], row["cell_y"], str(rank),
                ha="center", va="center", fontsize=5.2, color="#9e1414",
                fontweight="bold", zorder=5)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_png, dpi=dpi, facecolor="white",
                bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return {"cell_size": cell_size, "n": len(cells), "vmax": vmax,
            "bg_opacity": bg_opacity, "min_scale": min_scale,
            "max_scale": max_scale, "size_exponent": size_exponent}


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Plaça Catalunya bottleneck visual repair test")
    ap.add_argument("--csv",   default=str(DEF_CSV))
    ap.add_argument("--plan",  default=str(DEF_PLAN))
    ap.add_argument("--calib", default=str(DEF_CALIB))
    ap.add_argument("--out",   default=str(OUT_PNG))
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--grid_res",  type=float, default=14.0, help="px per world metre for the smooth field")
    ap.add_argument("--sigma_m",   type=float, default=1.6,  help="Gaussian smoothing sigma in metres")
    ap.add_argument("--dpi",       type=int,   default=320)
    args = ap.parse_args()

    csv_path  = Path(args.csv)
    plan_path = Path(args.plan)
    calib_path = Path(args.calib)
    out_png   = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    cells_all = pd.read_csv(csv_path)
    n_before = len(cells_all)

    # --- Geometry: plan -> world, authoritative visual frame ---
    warped_rgb, (xmin, xmax, ymin, ymax) = warp_plan_to_world(plan_path, calib_path)

    # --- Clip cells to the plan rectangle (VISUAL clipping only) ---
    inside = (
        (cells_all["cell_x"] >= xmin) & (cells_all["cell_x"] <= xmax) &
        (cells_all["cell_y"] >= ymin) & (cells_all["cell_y"] <= ymax)
    )
    cells = cells_all[inside].copy()
    n_after = len(cells)
    n_dropped = n_before - n_after

    # --- Smooth bottleneck field over the clipped frame ---
    field = rasterize_field(cells, (xmin, xmax, ymin, ymax),
                            res=args.grid_res, sigma_m=args.sigma_m)
    vmax = float(np.percentile(field[field > 0], 99.0)) if np.any(field > 0) else 1.0
    rgba = soft_warm_rgba(field, vmax=vmax)

    # --- Background underlay ---
    underlay = to_architectural_underlay(warped_rgb)

    # --- Figure: clipped exactly to plan extent, no margins, no axes ---
    world_w = xmax - xmin
    world_h = ymax - ymin
    fig_w = 11.0
    fig_h = fig_w * (world_h / world_w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    img_extent = [xmin, xmax, ymax, ymin]  # bottom=ymax, top=ymin (plan orientation)

    ax.imshow(underlay, extent=img_extent, origin="upper",
              alpha=0.45, zorder=0, interpolation="bilinear")
    ax.imshow(rgba, extent=img_extent, origin="upper",
              zorder=2, interpolation="bilinear")

    # --- Top-k numbered labels (clean, small, legible) ---
    top = cells.sort_values("bottleneck_score", ascending=False).head(args.top_k)
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        ax.scatter(row["cell_x"], row["cell_y"], s=150, facecolor="white",
                   edgecolor="#7a1010", linewidth=1.1, zorder=4, alpha=0.92)
        ax.text(row["cell_x"], row["cell_y"], str(rank),
                ha="center", va="center", fontsize=8.5, color="#7a1010",
                fontweight="bold", zorder=5)

    # --- Lock axis limits to the plan rectangle ---
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)   # plan orientation (world-y increases downward)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # --- Small corner label ---
    ax.text(0.018, 0.965, "Plaça Catalunya — Bottleneck Intensity",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=11, color="#222222", fontweight="medium",
            path_effects=[withStroke(linewidth=3, foreground="white")], zorder=6)

    # --- Minimal, elegant colorbar ---
    sm = plt.cm.ScalarMappable(
        cmap=mcolors.LinearSegmentedColormap.from_list(
            "soft_warm",
            ["#fff1c2", "#ffd27f", "#ff9b42", "#f5602a", "#c81e1e"]),
        norm=mcolors.Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.01, aspect=28)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=6.5, length=2, color="#888888")
    cbar.set_label("low  →  high", fontsize=7.5, color="#555555")
    cbar.set_ticks([])

    fig.subplots_adjust(left=0.005, right=0.97, top=0.995, bottom=0.005)
    fig.savefig(out_png, dpi=args.dpi, facecolor="white",
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Report the achieved pixel width
    px_w = int(fig_w * args.dpi)
    print(f"[INFO] Saved {out_png}  (~{px_w}px wide target, {args.dpi} dpi)")
    print(f"[INFO] Plan world extent: X[{xmin:.2f},{xmax:.2f}] Y[{ymin:.2f},{ymax:.2f}] m")
    print(f"[INFO] Cells before clip: {n_before} | inside: {n_after} | dropped: {n_dropped}")

    # --- Report ---
    write_report(OUT_REPORT, csv_path, plan_path, calib_path, out_png,
                 (xmin, xmax, ymin, ymax), n_before, n_after, n_dropped,
                 args)


def write_report(path, csv_path, plan_path, calib_path, out_png,
                 extent, n_before, n_after, n_dropped, args):
    xmin, xmax, ymin, ymax = extent
    txt = f"""# Plaça Catalunya — Bottleneck Map Visual Repair Test

**This is a visualization-only clipping and smoothing pass. No recalibration,
retracking, or data modification was performed.**

## Sources used
- **Bottleneck CSV:** `{csv_path}`
  - Chosen over the generic `mp-data/outputs/behavior/bottlenecks/` copy because
    this `filtered_250m` product is the site-specific, distance-filtered (250 m)
    bottleneck output for Plaça Catalunya and is the most complete/clean version.
- **Background plan image:** `{plan_path}`
- **Calibration (geometry only, not modified):** `{calib_path}`

## How the image extent was determined
The plan-pixel → world homography was rebuilt from the calibration's
`plan_points_px` → `world_points` correspondences (identical method to
`compute_bottlenecks.py::_render_top_view_bg`). The four plan-image corners were
transformed into world coordinates to give the authoritative visual frame:

- World extent: **X = [{xmin:.2f}, {xmax:.2f}] m**, **Y = [{ymin:.2f}, {ymax:.2f}] m**

This rectangle — not the bottleneck data bounds — is used as the final axis
limits, so the plan fills the figure with no empty coordinate margins.

## How cells were clipped
Bottleneck cells whose centre `(cell_x, cell_y)` fell outside the plan world
rectangle were dropped from the render. The axis limits are then locked to the
plan corners so no outlier cell can expand the frame.

- **Cells before clipping:** {n_before}
- **Cells after clipping:** {n_after}
- **Cells dropped (outside plan):** {n_dropped}

## How smoothing was applied
Clipped cell `bottleneck_score` values were rasterized onto a regular world grid
at **{args.grid_res:.0f} px/m**, then Gaussian-smoothed with **σ = {args.sigma_m:.1f} m**
(`scipy.ndimage.gaussian_filter`). The smoothed field was normalized to its 99th
percentile and mapped through a custom warm ramp (sand → amber → orange → red).
Values below ~12% of the scale fade to fully transparent so weak noise disappears
and only meaningful bottleneck zones read. The result is a continuous soft field
rather than blocky 1 m cells.

## Calibration mismatch observed
**Yes — a vertical extent mismatch was observed (left untouched).** The bottleneck
cells span world-Y down to about **-94.5 m**, while the calibrated plan image only
covers world-Y **[{ymin:.2f}, {ymax:.2f}] m**. Roughly {n_dropped} cells fall below the
plan's bottom edge (trajectory/projection artifacts from the calibration). These
were the cells inflating the raw pipeline figure and shrinking the plan inside a
huge empty box.

Per task instructions, calibration was **not** corrected. Instead the plan image
extent was used as the authoritative visual boundary and out-of-frame cells were
masked out — **visual clipping only, no recalibration.**

## Output
- Figure: `{out_png}`
- DPI: {args.dpi}
- Top-{args.top_k} bottleneck zones marked with small numbered labels.
- Axes removed; minimal colorbar; small corner title.
- Clipped exactly to the plan image extent; suitable as a base for future animation.
"""
    Path(path).write_text(txt, encoding="utf-8")
    print(f"[INFO] Report saved: {path}")


def main_v2():
    """Revision: crisp discrete-cell render. Writes _v2.png (leaves v1 alone)."""
    ap = argparse.ArgumentParser(description="Plaça Catalunya bottleneck — v2 crisp-cell render")
    ap.add_argument("--csv",   default=str(DEF_CSV))
    ap.add_argument("--plan",  default=str(DEF_PLAN))
    ap.add_argument("--calib", default=str(DEF_CALIB))
    ap.add_argument("--out",   default=str(OUT_PNG_V2))
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--dpi",   type=int, default=320)
    args = ap.parse_args()

    csv_path  = Path(args.csv)
    plan_path = Path(args.plan)
    calib_path = Path(args.calib)
    out_png   = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    cells_all = pd.read_csv(csv_path)
    n_before = len(cells_all)

    warped_rgb, (xmin, xmax, ymin, ymax) = warp_plan_to_world(plan_path, calib_path)

    inside = (
        (cells_all["cell_x"] >= xmin) & (cells_all["cell_x"] <= xmax) &
        (cells_all["cell_y"] >= ymin) & (cells_all["cell_y"] <= ymax)
    )
    cells = cells_all[inside].copy()
    n_after = len(cells)
    n_dropped = n_before - n_after

    cell_size, n_drawn = render_v2(
        cells, warped_rgb, (xmin, xmax, ymin, ymax),
        top_k=args.top_k, out_png=out_png, dpi=args.dpi)

    px_w = int(11.0 * args.dpi)
    print(f"[INFO] Saved {out_png}  (~{px_w}px wide target, {args.dpi} dpi)")
    print(f"[INFO] Plan world extent: X[{xmin:.2f},{xmax:.2f}] Y[{ymin:.2f},{ymax:.2f}] m")
    print(f"[INFO] Cell size: {cell_size:.2f} m | cells before clip: {n_before} | drawn: {n_drawn} | dropped: {n_dropped}")

    txt = f"""# Plaça Catalunya — Bottleneck Map Test v2 (crisp-cell revision)

**This is a visualization-only clipping pass. No recalibration, retracking, or
data modification was performed.**

## What changed vs v1
- Reverted from the smoothed continuous heat field back to **discrete grid cells**
  so spatial occupancy / grid resolution is preserved.
- Cells drawn as crisp, lightly rounded squares with subtle score-scaled
  transparency (alpha 0.35 → 0.90) — readable and deliberate, not blocky or muddy.
- **Removed** the title text, colorbar/legend, and all explanatory labels.
  Figure now contains only: background image + cell layer + ranked markers.
- Warm palette **muted** (sand → amber → ochre → terracotta → brick); no pure
  bright red, no glow.
- Ranked bottleneck markers kept but made smaller / more elegant so the cell
  layer remains the primary information.

## Sources (unchanged from v1)
- Bottleneck CSV: `{csv_path}`
- Background plan image: `{plan_path}`
- Calibration (geometry only, not modified): `{calib_path}`

## Framing / clipping (kept from v1)
- Plan world extent (authoritative frame): X = [{xmin:.2f}, {xmax:.2f}] m, Y = [{ymin:.2f}, {ymax:.2f}] m
- Cell size: {cell_size:.2f} m
- Cells before clipping: {n_before}
- Cells drawn (inside plan): {n_drawn}
- Cells dropped (outside plan): {n_dropped}
- Image clipped exactly to plan bounds; axes locked to plan corners.

## Calibration mismatch
Same vertical mismatch noted in v1 (cells extend below the plan's bottom edge);
**not corrected** — out-of-frame cells masked out. Visual clipping only, no
recalibration.

## Output
- Figure: `{out_png}` (DPI {args.dpi})
- v1 figure left untouched.
"""
    Path(OUT_REPORT_V2).write_text(txt, encoding="utf-8")
    print(f"[INFO] Report saved: {OUT_REPORT_V2}")


def main_v3():
    """Revision v3: borderless opacity-graded raster. Writes _v3.png + comparison."""
    ap = argparse.ArgumentParser(description="Plaça Catalunya bottleneck — v3 occupancy raster")
    ap.add_argument("--csv",   default=str(DEF_CSV))
    ap.add_argument("--plan",  default=str(DEF_PLAN))
    ap.add_argument("--calib", default=str(DEF_CALIB))
    ap.add_argument("--out",   default=str(OUT_PNG_V3))
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--dpi",   type=int, default=320)
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help="Quantile of bottleneck scores below which cells are hidden")
    args = ap.parse_args()

    csv_path  = Path(args.csv)
    plan_path = Path(args.plan)
    calib_path = Path(args.calib)
    out_png   = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    cells_all = pd.read_csv(csv_path)
    n_before = len(cells_all)

    warped_rgb, (xmin, xmax, ymin, ymax) = warp_plan_to_world(plan_path, calib_path)

    inside = (
        (cells_all["cell_x"] >= xmin) & (cells_all["cell_x"] <= xmax) &
        (cells_all["cell_y"] >= ymin) & (cells_all["cell_y"] <= ymax)
    )
    cells = cells_all[inside].copy()
    n_after = len(cells)
    n_dropped = n_before - n_after

    stats = render_v3(cells, warped_rgb, (xmin, xmax, ymin, ymax),
                      top_k=args.top_k, out_png=out_png, dpi=args.dpi,
                      threshold=args.threshold)

    # Comparison panel (requires v2 to exist; generate it if missing)
    if not OUT_PNG_V2.exists():
        print("[WARN] v2 PNG missing — run main_v2 first for a full comparison.")
    else:
        make_comparison(OUT_PNG_V2, out_png, OUT_CMP_V3)
        print(f"[INFO] Saved comparison: {OUT_CMP_V3}")

    print(f"[INFO] Saved {out_png}  ({args.dpi} dpi)")
    print(f"[INFO] Plan world extent: X[{xmin:.2f},{xmax:.2f}] Y[{ymin:.2f},{ymax:.2f}] m")
    print(f"[INFO] threshold q={stats['threshold']:.2f} -> score cutoff {stats['cutoff']:.4f}")
    print(f"[INFO] in-frame cells: {n_after} | kept: {stats['n_kept']} | hidden: {stats['n_hidden']} | dropped(off-plan): {n_dropped}")

    txt = f"""# Plaça Catalunya — Bottleneck Map Test v3 (occupancy-hierarchy revision)

**Visualization-only pass. No recalibration, retracking, or data modification.**

## Goal
Make bottleneck intensity read before the grid. The raster data structure is
preserved (no smoothing, blurring, interpolation, or continuous heatmap), but
visual hierarchy is shifted onto the bottleneck values via opacity + thresholding.

## What changed vs v2
- **Cell borders removed** — clean borderless squares; raster structure now reads
  through adjacency only (no outline / grid stroke).
- **Opacity hierarchy** — alpha scales with score so weak cells fade out and
  strong cells dominate.
- **Low-value threshold** — bottom fraction of scores hidden entirely.
- **Palette** slightly desaturated, no bright red (architectural tone).
- **Ranked markers** reduced ~20% (size 64→51, font 6.0→4.8) so the raster field
  is primary and labels secondary.
- All chrome removed: no title, legend, colorbar, axes, ticks, or text other than
  the rank markers.

## Threshold
- Method: **quantile** of in-frame bottleneck scores. Configurable via the
  `THRESHOLD` constant / `--threshold` flag.
- Threshold used: **{stats['threshold']:.2f}** (hide bottom {stats['threshold']*100:.0f}% of values)
- Resulting score cutoff: **{stats['cutoff']:.4f}**
- Cells in frame: **{n_after}**
- Cells hidden (below cutoff): **{stats['n_hidden']}**
- Cells retained: **{stats['n_kept']}**

## Opacity scaling method
For each retained cell with score `s`, normalized within the retained range:
`t = clip((s - cutoff) / (vmax - cutoff), 0, 1)`, then
`alpha = 0.15 + 0.85 * t**1.25` (range 0.15 → 1.0). Hue is taken from the
desaturated warm ramp normalized over `[0, vmax]` (vmax = 99th percentile =
{stats['vmax']:.4f}) so colour meaning stays stable while opacity carries the
hierarchy. The gamma (1.25) keeps medium cells light so only the strongest cells
become fully opaque.

## Framing / clipping (kept from v1/v2)
- Plan world extent (authoritative frame): X = [{xmin:.2f}, {xmax:.2f}] m, Y = [{ymin:.2f}, {ymax:.2f}] m
- Cell size: {stats['cell_size']:.2f} m
- Cells dropped as off-plan (outside frame): {n_dropped}
- Calibration mismatch unchanged and not corrected — visual clipping only.

## Outputs
- Figure: `{out_png}`
- Comparison (V2 | V3): `{OUT_CMP_V3}`
- v1 and v2 figures left untouched.
"""
    Path(OUT_REPORT_V3).write_text(txt, encoding="utf-8")
    print(f"[INFO] Report saved: {OUT_REPORT_V3}")


def main_v2b():
    """Revision v2b: V2 cells + variable square size + yellow->red palette."""
    ap = argparse.ArgumentParser(description="Plaça Catalunya bottleneck — v2b variable-size raster")
    ap.add_argument("--csv",   default=str(DEF_CSV))
    ap.add_argument("--plan",  default=str(DEF_PLAN))
    ap.add_argument("--calib", default=str(DEF_CALIB))
    ap.add_argument("--out",   default=str(OUT_PNG_V2B))
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--dpi",   type=int, default=320)
    ap.add_argument("--min_scale", type=float, default=MIN_SQUARE_SCALE)
    ap.add_argument("--max_scale", type=float, default=MAX_SQUARE_SCALE)
    ap.add_argument("--bg_opacity", type=float, default=BG_OPACITY_V2B)
    args = ap.parse_args()

    csv_path  = Path(args.csv)
    plan_path = Path(args.plan)
    calib_path = Path(args.calib)
    out_png   = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    cells_all = pd.read_csv(csv_path)
    n_before = len(cells_all)

    warped_rgb, (xmin, xmax, ymin, ymax) = warp_plan_to_world(plan_path, calib_path)

    inside = (
        (cells_all["cell_x"] >= xmin) & (cells_all["cell_x"] <= xmax) &
        (cells_all["cell_y"] >= ymin) & (cells_all["cell_y"] <= ymax)
    )
    cells = cells_all[inside].copy()
    n_after = len(cells)
    n_dropped = n_before - n_after

    stats = render_v2b(cells, warped_rgb, (xmin, xmax, ymin, ymax),
                       top_k=args.top_k, out_png=out_png, dpi=args.dpi,
                       min_scale=args.min_scale, max_scale=args.max_scale,
                       bg_opacity=args.bg_opacity)

    if OUT_PNG_V2.exists():
        make_comparison(OUT_PNG_V2, out_png, OUT_CMP_V2B)
        print(f"[INFO] Saved comparison: {OUT_CMP_V2B}")
    else:
        print("[WARN] v2 PNG missing — run main_v2 first for a full comparison.")

    print(f"[INFO] Saved {out_png}  ({args.dpi} dpi)")
    print(f"[INFO] Plan world extent: X[{xmin:.2f},{xmax:.2f}] Y[{ymin:.2f},{ymax:.2f}] m")
    print(f"[INFO] in-frame cells rendered: {n_after} | dropped(off-plan): {n_dropped} | bg_opacity {stats['bg_opacity']:.2f}")

    txt = f"""# Plaça Catalunya — Bottleneck Map Test v2b

**Visualization-only pass. No recalibration, retracking, or data modification.**

Base direction: **V2** (discrete cell raster). Three refinements applied.

## Source files
- Bottleneck CSV: `{csv_path}`
- Background plan image: `{plan_path}`
- Calibration (geometry only, not modified): `{calib_path}`

## Kept from V2
Exact clipping to plan image bounds, discrete cell/raster representation, visible
cell structure, ranked bottleneck markers, and no title / legend / colorbar /
axes / ticks.

## Change 1 — variable square size
Each square's edge scales linearly with the cell's normalized bottleneck score,
centred on the original cell position (location never shifts):

    t     = clip(score / vmax, 0, 1)          # vmax = 99th pct = {stats['vmax']:.4f}
    scale = {args.min_scale:.2f} + ({args.max_scale:.2f} - {args.min_scale:.2f}) * t
    side  = cell_size * scale                 # cell_size = {stats['cell_size']:.2f} m

Strong bottlenecks render as larger squares so hierarchy reads from size as well
as colour. Alpha also rises with score (0.45 → 0.95).

## Change 2 — colour palette
Clean yellow → orange → red `LinearSegmentedColormap` (no brown / maroon / mud):
`#FFF3B0` (pale yellow) → `#FDBA3B` (amber) → `#F97316` (orange) → `#DC2626` (red).

## Change 3 — background visibility
Background underlay opacity raised from 0.45 (V2) to **{stats['bg_opacity']:.2f}** (+0.10),
still kept below the bottleneck layer so it doesn't compete.

## Markers
Ranked markers kept, slightly reduced (size 54, font 5.2) so they stay secondary
to the cell field under the larger squares.

## Cells rendered
- In-frame cells rendered: **{n_after}**
- Cells dropped as off-plan: {n_dropped}
- Plan world extent: X = [{xmin:.2f}, {xmax:.2f}] m, Y = [{ymin:.2f}, {ymax:.2f}] m

## Outputs
- Figure: `{out_png}` (DPI {args.dpi})
- Comparison (V2 | V2B): `{OUT_CMP_V2B}`
- V2 and V3 figures left untouched.
"""
    Path(OUT_REPORT_V2B).write_text(txt, encoding="utf-8")
    print(f"[INFO] Report saved: {OUT_REPORT_V2B}")


def main_v2c():
    """Revision v2c: v2b look with more dramatic nonlinear square-size scaling."""
    ap = argparse.ArgumentParser(description="Plaça Catalunya bottleneck — v2c dramatic size scaling")
    ap.add_argument("--csv",   default=str(DEF_CSV))
    ap.add_argument("--plan",  default=str(DEF_PLAN))
    ap.add_argument("--calib", default=str(DEF_CALIB))
    ap.add_argument("--out",   default=str(OUT_PNG_V2C))
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--dpi",   type=int, default=320)
    ap.add_argument("--min_scale", type=float, default=MIN_SQUARE_SCALE_V2C)
    ap.add_argument("--max_scale", type=float, default=MAX_SQUARE_SCALE_V2C)
    ap.add_argument("--exponent",  type=float, default=SIZE_EXPONENT_V2C)
    ap.add_argument("--bg_opacity", type=float, default=BG_OPACITY_V2B)
    args = ap.parse_args()

    csv_path  = Path(args.csv)
    plan_path = Path(args.plan)
    calib_path = Path(args.calib)
    out_png   = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    cells_all = pd.read_csv(csv_path)
    warped_rgb, (xmin, xmax, ymin, ymax) = warp_plan_to_world(plan_path, calib_path)

    inside = (
        (cells_all["cell_x"] >= xmin) & (cells_all["cell_x"] <= xmax) &
        (cells_all["cell_y"] >= ymin) & (cells_all["cell_y"] <= ymax)
    )
    cells = cells_all[inside].copy()
    n_after = len(cells)
    n_dropped = len(cells_all) - n_after

    stats = render_v2b(cells, warped_rgb, (xmin, xmax, ymin, ymax),
                       top_k=args.top_k, out_png=out_png, dpi=args.dpi,
                       min_scale=args.min_scale, max_scale=args.max_scale,
                       bg_opacity=args.bg_opacity, size_exponent=args.exponent)

    if OUT_PNG_V2B.exists():
        make_comparison(OUT_PNG_V2B, out_png, OUT_CMP_V2C)
        print(f"[INFO] Saved comparison: {OUT_CMP_V2C}")
    else:
        print("[WARN] v2b PNG missing — run main_v2b first for the comparison.")

    print(f"[INFO] Saved {out_png}  ({args.dpi} dpi)")
    print(f"[INFO] size scale {args.min_scale:.2f}..{args.max_scale:.2f}, exponent {args.exponent}")
    print(f"[INFO] in-frame cells rendered: {n_after} | dropped(off-plan): {n_dropped}")

    txt = f"""# Plaça Catalunya — Bottleneck Map Test v2c

**Visualization-only pass. No recalibration, retracking, or data modification.**

Base: **v2b**. Only the square-size scaling changed — everything else (clipping,
background treatment, yellow→red palette, no title/legend/axes, square-cell
raster language) is kept.

## Change — more dramatic square-size scaling
The size response is now nonlinear so strong bottlenecks pop by size as well as
colour, and weak cells shrink noticeably. Each square stays centred on its
original cell position — no spatial shift.

    t     = clip(score / vmax, 0, 1)          # vmax = 99th pct = {stats['vmax']:.4f}
    scale = min_scale + (t ** exponent) * (max_scale - min_scale)
    side  = cell_size * scale                 # cell_size = {stats['cell_size']:.2f} m

- **min_square_scale:** {args.min_scale:.2f}
- **max_square_scale:** {args.max_scale:.2f}  (strongest cells exceed base cell size)
- **exponent:** {args.exponent}  (Option B-style aggressive nonlinear mapping)

For reference, v2b used min 0.45 / max 1.00 / exponent 1.0 (linear).

## Kept from v2b
- Exact clipping to plan image bounds (plan world extent
  X = [{xmin:.2f}, {xmax:.2f}] m, Y = [{ymin:.2f}, {ymax:.2f}] m)
- Background underlay opacity {args.bg_opacity:.2f}
- Yellow→red palette `#FFF3B0 → #FDBA3B → #F97316 → #DC2626`
- Ranked bottleneck markers; no title / legend / colorbar / axes / ticks
- Stronger (larger) squares drawn on top of weaker ones for clean overlap

## Cells rendered
- In-frame cells: **{n_after}** | dropped off-plan: {n_dropped}

## Source files
- Bottleneck CSV: `{csv_path}`
- Background plan image: `{plan_path}`
- Calibration (geometry only, not modified): `{calib_path}`

## Outputs
- Figure: `{out_png}` (DPI {args.dpi})
- Comparison (V2B | V2C): `{OUT_CMP_V2C}`
- V2, V2B, and V3 figures left untouched.
"""
    Path(OUT_REPORT_V2C).write_text(txt, encoding="utf-8")
    print(f"[INFO] Report saved: {OUT_REPORT_V2C}")


def main_v2d():
    """Revision v2d: grow strong cells only (>= v2b base), clamp to avoid overlap."""
    ap = argparse.ArgumentParser(description="Plaça Catalunya bottleneck — v2d grow-only size scaling")
    ap.add_argument("--csv",   default=str(DEF_CSV))
    ap.add_argument("--plan",  default=str(DEF_PLAN))
    ap.add_argument("--calib", default=str(DEF_CALIB))
    ap.add_argument("--out",   default=str(OUT_PNG_V2D))
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--dpi",   type=int, default=320)
    ap.add_argument("--base_scale",    type=float, default=BASE_SCALE_V2D)
    ap.add_argument("--growth_amount", type=float, default=GROWTH_AMOUNT_V2D)
    ap.add_argument("--exponent",      type=float, default=SIZE_EXPONENT_V2D)
    ap.add_argument("--max_frac_spacing", type=float, default=MAX_FRAC_SPACING_V2D)
    ap.add_argument("--bg_opacity",    type=float, default=BG_OPACITY_V2B)
    args = ap.parse_args()

    csv_path  = Path(args.csv)
    plan_path = Path(args.plan)
    calib_path = Path(args.calib)
    out_png   = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    cells_all = pd.read_csv(csv_path)
    warped_rgb, (xmin, xmax, ymin, ymax) = warp_plan_to_world(plan_path, calib_path)

    inside = (
        (cells_all["cell_x"] >= xmin) & (cells_all["cell_x"] <= xmax) &
        (cells_all["cell_y"] >= ymin) & (cells_all["cell_y"] <= ymax)
    )
    cells = cells_all[inside].copy()
    n_after = len(cells)
    n_dropped = len(cells_all) - n_after
    spacing = infer_cell_size(cells)

    # base_scale + (t**exp)*growth  ==  min_scale + (t**exp)*(max-min)
    max_scale = args.base_scale + args.growth_amount
    stats = render_v2b(cells, warped_rgb, (xmin, xmax, ymin, ymax),
                       top_k=args.top_k, out_png=out_png, dpi=args.dpi,
                       min_scale=args.base_scale, max_scale=max_scale,
                       bg_opacity=args.bg_opacity, size_exponent=args.exponent,
                       max_frac_spacing=args.max_frac_spacing)

    if OUT_PNG_V2B.exists():
        make_comparison(OUT_PNG_V2B, out_png, OUT_CMP_V2D)
        print(f"[INFO] Saved comparison: {OUT_CMP_V2D}")
    else:
        print("[WARN] v2b PNG missing — run main_v2b first for the comparison.")

    max_side = min(max_scale, args.max_frac_spacing) * spacing
    print(f"[INFO] Saved {out_png}  ({args.dpi} dpi)")
    print(f"[INFO] grid spacing {spacing:.2f} m | scale {args.base_scale:.2f}->{max_scale:.2f} | "
          f"clamp {args.max_frac_spacing:.2f}*spacing | max side {max_side:.2f} m")
    print(f"[INFO] in-frame cells rendered: {n_after} | dropped(off-plan): {n_dropped}")

    txt = f"""# Plaça Catalunya — Bottleneck Map Test v2d

**Visualization-only pass. No recalibration, retracking, or data modification.**

Base: **v2b**. Only the square-size behaviour changed. Fixes v2c, where weak
cells shrank too far and the field lost spatial continuity.

## Square-size rule (grow strong cells only)
No cell drops below the v2b base size; size only grows with score. Each square
stays centred on its cell — no spatial shift.

    t      = clip(score / vmax, 0, 1)         # vmax = 99th pct = {stats['vmax']:.4f}
    scale  = base_scale + (t ** exponent) * growth_amount
    side   = cell_size * scale
    side   = min(side, max_frac_spacing * spacing)   # no-overlap clamp

- **base_scale:** {args.base_scale:.2f}  (= v2b base; weak/mid cells never smaller)
- **growth_amount:** {args.growth_amount:.2f}  (scale range {args.base_scale:.2f} → {max_scale:.2f})
- **exponent:** {args.exponent}
- **max_square_fraction_of_spacing:** {args.max_frac_spacing:.2f}
- **estimated grid spacing:** {spacing:.2f} m (nearest-neighbour cell-centre spacing)
- **largest rendered square side:** {max_side:.2f} m → guaranteed gap of
  {(1 - min(max_scale, args.max_frac_spacing)) * spacing:.2f} m between neighbours

## No-overlap guarantee
The largest possible square occupies at most {args.max_frac_spacing:.0%} of the grid
spacing, so neighbouring squares always keep a visible gap. Low/mid cells read as
a continuous measured raster; high cells subtly pop.

## Kept from v2b (unchanged)
- Exact clipping to plan bounds (X = [{xmin:.2f}, {xmax:.2f}] m, Y = [{ymin:.2f}, {ymax:.2f}] m)
- Background underlay opacity {args.bg_opacity:.2f}
- Yellow→red palette `#FFF3B0 → #FDBA3B → #F97316 → #DC2626`
- Ranked bottleneck markers; no title / legend / colorbar / axes / ticks

## Cells rendered
- In-frame cells: **{n_after}** | dropped off-plan: {n_dropped}

## Source files
- Bottleneck CSV: `{csv_path}`
- Background plan image: `{plan_path}`
- Calibration (geometry only, not modified): `{calib_path}`

## Outputs
- Figure: `{out_png}` (DPI {args.dpi})
- Comparison (V2B | V2D): `{OUT_CMP_V2D}`
- V2, V2B, V2C, and V3 figures left untouched.
"""
    Path(OUT_REPORT_V2D).write_text(txt, encoding="utf-8")
    print(f"[INFO] Report saved: {OUT_REPORT_V2D}")


if __name__ == "__main__":
    main_v2d()
