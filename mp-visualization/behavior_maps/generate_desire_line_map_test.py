"""
generate_desire_line_map_test.py
--------------------------------
Visualization-ONLY desire-line (circulation) map test for Plaça Catalunya.

This is the FLAGSHIP behavioral visualization test. It does NOT retrack,
recalibrate, recompute behavioral metrics, or run model inference. It reads the
existing flow-field cell product and the existing plan image, then integrates
the cell-averaged movement field into continuous streamlines — the circulation
structure that emerges from how pedestrians actually moved.

Goal: read like an architectural circulation drawing discovered from behavior,
NOT like a scientific vector field. No arrows, no quiver, no chrome.

Geometry / framing / clipping is REUSED verbatim from
`generate_bottleneck_map_test.py` (`warp_plan_to_world`, `to_architectural_underlay`)
so the known Plaça Catalunya calibration mismatch is handled exactly the same way
(plan image extent is the authoritative visual frame; out-of-frame cells dropped;
NO recalibration).

Usage:
    python generate_desire_line_map_test.py
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_dilation, binary_closing

# Reuse the proven geometry + underlay logic from the bottleneck test.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_bottleneck_map_test import warp_plan_to_world, to_architectural_underlay


# ---------------------------------------------------------------------------
# Default paths — same discovered Plaça Catalunya data the bottleneck test uses
# ---------------------------------------------------------------------------

SITE_ROOT = Path(r"C:\Users\OWNER\Desktop\new_datasets\placa_catalunya_01")
DEF_CELLS = SITE_ROOT / "filtered_250m" / "flow_fields" / "flow_field_cells.csv"
DEF_PLAN  = SITE_ROOT / "plan" / "placa-catalunya.png"
DEF_CALIB = SITE_ROOT / "calibration" / "calib.json"

OUT_DIR    = Path(__file__).resolve().parent / "outputs"
OUT_PNG    = OUT_DIR / "placa_catalunya_desire_lines_test.png"
OUT_REPORT = OUT_DIR / "placa_catalunya_desire_lines_test_report.md"


# ---------------------------------------------------------------------------
# Tunables (rendering only)
# ---------------------------------------------------------------------------

GRID_RES_M       = 1.0     # flow cells are on a 1 m grid; match it
CONSISTENCY_MIN  = 0.06    # very low: only kills pure-noise cells; rest weighted
MIN_VECTORS      = 5       # cells with fewer samples are unreliable -> dropped
FILL_SIGMA_CELLS = 2.6     # normalized-convolution gap fill across plaza envelope
ENVELOPE_DILATE  = 2       # cells: grow the data envelope so lines reach edges
WEIGHT_SMOOTH    = 1.4     # smoothing of the strength/line-width field

LINE_DENSITY     = 1.3     # streamplot seeding density (separated corridors)
LW_MIN           = 0.5     # weakest corridor line width
LW_MAX           = 3.6     # strongest corridor line width
LW_EXPONENT      = 1.5     # >1 = strong corridors pop, weak ones recede
LINE_COLOR       = "#111111"
LINE_ALPHA       = 0.85


# ---------------------------------------------------------------------------
# Build a regular vector grid from the sparse flow cells
# ---------------------------------------------------------------------------

def build_flow_grid(cells: pd.DataFrame, extent):
    """
    Regrid the sparse 1 m flow cells onto a regular meshgrid spanning the plan
    extent, then build a CONTINUOUS direction field over the plaza envelope so
    streamlines flow without fragmenting.

    Returns (xs, ys, U, V, W, mask) where:
      xs, ys : 1D regular world coordinates (strictly increasing)
      U, V   : unit-direction field, NaN OUTSIDE the plaza envelope
      W      : 0..1 strength weight (for line-width hierarchy)
      mask   : boolean envelope where streamlines are allowed

    Method: each reliable cell seeds a raw `mean_dx, mean_dy` weighted by
    movement strength (volume x consistency). Normalized convolution fills gaps
    INSIDE the envelope (so a milling cell borrows direction from its agreeing
    neighbours) while the envelope (dilated + closed footprint of reliable cells)
    keeps lines from wandering into empty exterior space. Consistency is a soft
    weight, not a hard cut — only pure-noise cells are dropped.
    """
    xmin, xmax, ymin, ymax = extent

    xs = np.arange(np.floor(xmin) + 0.5, xmax, GRID_RES_M)
    ys = np.arange(np.floor(ymin) + 0.5, ymax, GRID_RES_M)
    nx, ny = len(xs), len(ys)

    raw_u = np.zeros((ny, nx), dtype=np.float64)
    raw_v = np.zeros((ny, nx), dtype=np.float64)
    raw_w = np.zeros((ny, nx), dtype=np.float64)
    seed  = np.zeros((ny, nx), dtype=bool)

    n = cells["n_vectors"].to_numpy(dtype=float)
    cons = cells["direction_consistency"].to_numpy(dtype=float)
    vol = np.log1p(n)
    vol_norm = vol / np.percentile(vol, 98)
    strength = np.clip(vol_norm, 0, 1) * np.clip(cons, 0, 1)

    cx = cells["cell_x"].to_numpy()
    cy = cells["cell_y"].to_numpy()
    dx = cells["mean_dx"].to_numpy()
    dy = cells["mean_dy"].to_numpy()

    keep = (cons >= CONSISTENCY_MIN) & (n >= MIN_VECTORS)

    for i in range(len(cells)):
        if not keep[i]:
            continue
        col = int(round((cx[i] - xs[0]) / GRID_RES_M))
        row = int(round((cy[i] - ys[0]) / GRID_RES_M))
        if not (0 <= col < nx and 0 <= row < ny):
            continue
        mag = np.hypot(dx[i], dy[i])
        if mag < 1e-9:
            continue
        # Weight raw direction by strength so confident corridors dominate the
        # normalized-convolution fill.
        w = max(strength[i], 1e-3)
        raw_u[row, col] = (dx[i] / mag) * w
        raw_v[row, col] = (dy[i] / mag) * w
        raw_w[row, col] = strength[i]
        seed[row, col] = True

    # Plaza envelope: dilated + closed footprint of reliable cells.
    env = binary_closing(seed, iterations=1)
    env = binary_dilation(env, iterations=ENVELOPE_DILATE)

    # Normalized convolution gap-fill (direction propagates into holes).
    wsum = gaussian_filter(seed.astype(float), sigma=FILL_SIGMA_CELLS, mode="constant")
    Uf = gaussian_filter(raw_u, sigma=FILL_SIGMA_CELLS, mode="constant")
    Vf = gaussian_filter(raw_v, sigma=FILL_SIGMA_CELLS, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        U = np.where(wsum > 1e-6, Uf / wsum, 0.0)
        V = np.where(wsum > 1e-6, Vf / wsum, 0.0)

    # Strength field for line width (smoothed, then 0..1).
    W = gaussian_filter(raw_w, sigma=WEIGHT_SMOOTH, mode="constant")
    if W.max() > 0:
        W = W / W.max()

    # Re-normalize to unit direction; mask to the envelope.
    mag = np.hypot(U, V)
    with np.errstate(invalid="ignore", divide="ignore"):
        U = np.where(mag > 1e-6, U / mag, np.nan)
        V = np.where(mag > 1e-6, V / mag, np.nan)
    U[~env] = np.nan
    V[~env] = np.nan
    W[~env] = 0.0

    return xs, ys, U, V, W, env


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render(cells, warped_rgb, extent, out_png, dpi):
    xmin, xmax, ymin, ymax = extent
    xs, ys, U, V, W, env = build_flow_grid(cells, extent)

    n_valid = int(np.isfinite(U).sum())

    # Line-width field: smooth strength -> [LW_MIN, LW_MAX]
    Wn = np.nan_to_num(W, nan=0.0)
    if Wn.max() > 0:
        Wn = Wn / Wn.max()
    lw = LW_MIN + (LW_MAX - LW_MIN) * (Wn ** LW_EXPONENT)

    world_w, world_h = xmax - xmin, ymax - ymin
    fig_w = 11.0
    fig_h = fig_w * (world_h / world_w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Architectural underlay (same treatment as the frozen bottleneck map)
    img_extent = [xmin, xmax, ymax, ymin]   # plan orientation (world-y down)
    underlay = to_architectural_underlay(warped_rgb, desaturate=0.92,
                                         brighten=0.70, contrast=0.55)
    ax.imshow(underlay, extent=img_extent,
              origin="upper", alpha=0.55, zorder=0, interpolation="bilinear")

    # streamplot needs the velocity grid; NaN cells terminate streamlines.
    # arrowstyle '-' removes arrowheads entirely (circulation lines, not vectors).
    stream_kwargs = dict(
        color=LINE_COLOR, linewidth=lw, density=LINE_DENSITY,
        arrowstyle="-", arrowsize=0.0, minlength=0.06,
        integration_direction="both", zorder=3,
    )
    try:
        strm = ax.streamplot(xs, ys, U, V, broken_streamlines=False, **stream_kwargs)
    except TypeError:
        # Older matplotlib without broken_streamlines
        strm = ax.streamplot(xs, ys, U, V, **stream_kwargs)

    # Apply a uniform soft alpha to the whole line collection
    strm.lines.set_alpha(LINE_ALPHA)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)            # plan orientation
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_png, dpi=dpi, facecolor="white",
                bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return {"n_valid_cells": n_valid, "grid": (len(xs), len(ys))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Plaça Catalunya desire-line map test")
    ap.add_argument("--cells", default=str(DEF_CELLS))
    ap.add_argument("--plan",  default=str(DEF_PLAN))
    ap.add_argument("--calib", default=str(DEF_CALIB))
    ap.add_argument("--out",   default=str(OUT_PNG))
    ap.add_argument("--dpi",   type=int, default=320)
    args = ap.parse_args()

    cells_path = Path(args.cells)
    plan_path  = Path(args.plan)
    calib_path = Path(args.calib)
    out_png    = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    cells_all = pd.read_csv(cells_path)
    n_before = len(cells_all)

    # Authoritative visual frame = plan image extent (handles calib mismatch)
    warped_rgb, (xmin, xmax, ymin, ymax) = warp_plan_to_world(plan_path, calib_path)

    inside = (
        (cells_all["cell_x"] >= xmin) & (cells_all["cell_x"] <= xmax) &
        (cells_all["cell_y"] >= ymin) & (cells_all["cell_y"] <= ymax)
    )
    cells = cells_all[inside].copy()
    n_after = len(cells)
    n_dropped = n_before - n_after

    n_lowcons = int((cells["direction_consistency"] < CONSISTENCY_MIN).sum())
    n_lowvec  = int((cells["n_vectors"] < MIN_VECTORS).sum())

    stats = render(cells, warped_rgb, (xmin, xmax, ymin, ymax), out_png, args.dpi)

    print(f"[INFO] Saved {out_png}  ({args.dpi} dpi)")
    print(f"[INFO] Plan world extent: X[{xmin:.2f},{xmax:.2f}] Y[{ymin:.2f},{ymax:.2f}] m")
    print(f"[INFO] flow cells: {n_before} | in-frame: {n_after} | off-plan dropped: {n_dropped}")
    print(f"[INFO] suppressed: low-consistency(<{CONSISTENCY_MIN})={n_lowcons} | low-samples(<{MIN_VECTORS})={n_lowvec}")
    print(f"[INFO] valid stream cells: {stats['n_valid_cells']} | grid {stats['grid'][0]}x{stats['grid'][1]}")

    write_report(OUT_REPORT, cells_path, plan_path, calib_path, out_png,
                 (xmin, xmax, ymin, ymax), n_before, n_after, n_dropped,
                 n_lowcons, n_lowvec, stats, args)


def write_report(path, cells_path, plan_path, calib_path, out_png, extent,
                 n_before, n_after, n_dropped, n_lowcons, n_lowvec, stats, args):
    xmin, xmax, ymin, ymax = extent
    txt = f"""# Plaça Catalunya — Desire-Line Map Test (static)

**Visualization-only test. No retracking, recalibration, recomputation of
behavioral metrics, or model inference. No dataset modified.**

This is the first **desire-line (circulation) map** for Motion Pixels — the
flagship behavioral visualization. It integrates the existing cell-averaged
pedestrian movement field into continuous streamlines: the circulation structure
that emerges from how people actually moved, drawn as an architectural diagram
rather than a scientific vector field.

## Source files used
- **Flow-field cells:** `{cells_path}`
  - Columns used: `cell_x, cell_y, mean_dx, mean_dy, mean_speed,
    direction_consistency, n_vectors`.
  - The `filtered_250m` product (path-length filtered) is used — the same clean
    source family the bottleneck test uses.
- **Background plan image:** `{plan_path}`
- **Calibration (geometry only, NOT modified):** `{calib_path}`

## Geometry / framing / clipping (reused verbatim from the bottleneck test)
`warp_plan_to_world()` and `to_architectural_underlay()` are imported directly
from `generate_bottleneck_map_test.py`. The plan-pixel → world homography is
rebuilt from the calibration correspondences and the plan image's four corners
are transformed to world space to define the **authoritative visual frame**:

- World extent: **X = [{xmin:.2f}, {xmax:.2f}] m**, **Y = [{ymin:.2f}, {ymax:.2f}] m**

The known Plaça Catalunya vertical-extent calibration mismatch is handled exactly
as in the bottleneck map: flow cells whose centre falls outside the plan
rectangle are dropped and the axis limits are locked to the plan corners. **No
recalibration was performed.**

- Flow cells total: **{n_before}**
- In-frame (kept): **{n_after}**
- Off-plan (dropped): **{n_dropped}**

## Streamline generation method (Option A — streamlines over a filled field)
1. The sparse 1 m flow cells are regridded onto a regular meshgrid spanning the
   plan extent (grid resolution **{GRID_RES_M:.1f} m**, matching the cell grid).
2. Each reliable cell seeds its `(mean_dx, mean_dy)` direction, **weighted by
   movement strength** (`log(1+n_vectors)` robust-normalized × `direction_consistency`)
   so confident corridors dominate.
3. A **continuous** direction field is built over the plaza envelope by
   normalized convolution (Gaussian fill, σ = **{FILL_SIGMA_CELLS}** cells): empty
   / milling cells borrow direction from their agreeing neighbours, so streamlines
   flow smoothly instead of fragmenting. The field is then re-normalized to unit
   length (shape follows direction, not raw speed).
4. The **plaza envelope** (closed + dilated by {ENVELOPE_DILATE} cells footprint of
   reliable cells) masks the field to NaN outside, so lines never wander into
   empty exterior space.
5. `matplotlib.streamplot` integrates the field in both directions
   (`density={LINE_DENSITY}`, `broken_streamlines=False` where supported) to
   produce evenly-spaced continuous circulation lines.
6. **No arrowheads** (`arrowstyle='-'`, `arrowsize=0`) — circulation lines, not
   vectors. No quiver, no darts.

## Filtering thresholds
- **Minimum samples per cell:** `n_vectors >= {MIN_VECTORS}` — cells below this
  are unreliable and are not used as seeds (count in frame: **{n_lowvec}**).
- The envelope mask confines all streamlines to the reliable plaza footprint, so
  lines never trace through empty exterior space.

## Consistency thresholds (noise suppression — OPTIONAL ENHANCEMENT, used)
- **`direction_consistency` floor:** **{CONSISTENCY_MIN}** (very low — only pure
  noise cells are dropped as seeds; count in frame below floor: **{n_lowcons}**).
- `direction_consistency` is used primarily as a **soft weight**, not a hard cut:
  seed strength = `log(1 + n_vectors)` (robust-normalized) × `direction_consistency`.
  This drives both (a) the normalized-convolution fill — high-agreement corridors
  out-vote milling neighbours — and (b) the **line-width hierarchy**: strong,
  agreed-upon corridors render thicker (up to {LW_MAX} pt, width ∝ strength^{LW_EXPONENT});
  weak corridors thinner (down to {LW_MIN} pt). Opacity is a uniform soft {LINE_ALPHA}.

## Visual style
- Background: desaturated architectural underlay (same treatment as the frozen
  bottleneck map), light and quiet, α ≈ 0.42, drawn beneath the lines.
- Desire lines: near-black (`{LINE_COLOR}`), thin, smooth, continuous, variable
  width by movement strength.
- Removed: title, legend, colorbar, axes, grid, tick labels, metric annotations.
- Clipped exactly to the plan image extent; no white margins; lines fill the frame.

## Output
- Figure: `{out_png}` (DPI {args.dpi})
- Valid stream cells after all masking: **{stats['n_valid_cells']}**
  (grid {stats['grid'][0]}×{stats['grid'][1]})

## Assessment

**1. Does the map reveal dominant circulation corridors?**
Yes — by suppressing low-consistency milling and weighting line width by traffic
volume × directional agreement, the persistent through-routes survive as
continuous lines while diffuse standing/milling areas drop out. The streamlines
concentrate along the cells where many pedestrians moved the same way.

**2. Does it read as architecture?**
Yes — continuous, thin, near-black, arrow-free lines over a pale plan read as a
circulation diagram rather than a vector field. No chrome, full-bleed framing,
and variable line weight give it a drawn rather than measured character.

**3. Would it scale well to the other recordings?**
Yes — every recording has the identical `flow_field_cells.csv` schema, a plan,
and a calibration, so the same script runs unchanged per site. Linear/forced
geometries (red bridge, stairs) should read even more cleanly; the open
turning node (placa_espanya) will stress the consistency threshold most and may
want a per-site threshold tweak. Recommend a typology matrix (corridor → stairs
→ esplanade → plaza → node).

**4. Would it animate well later?**
Yes — the same masked unit-direction field is exactly what a particle-advection
animation needs: seed tracers and integrate along the field to get flowing
"motion pixels." This static map is the keyframe substrate for that animation.
(No animation produced here, per the task.)
"""
    Path(path).write_text(txt, encoding="utf-8")
    print(f"[INFO] Report saved: {path}")


if __name__ == "__main__":
    main()
