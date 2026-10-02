"""
generate_trace_dust_hero.py
---------------------------
MOTION PIXELS — Trace + Dust HERO image on black, for Plaça Catalunya.

A single cinematic hero: luminous pedestrian movement (glowing trace strings +
motion dust, density-coloured cyan->magenta) imprinted on a black architectural
plan drawn as fine white linework. Additive multi-radius bloom for the glow.

VISUALIZATION ONLY. A rendering of existing tracked pedestrian trajectories. No
retracking, recalibration, model inference, or source-data modification.

Geometry / clipping reuse `warp_plan_to_world` from
`../generate_bottleneck_map_test.py`; the plan image extent is the authoritative
frame. The pale architectural underlay is NOT used — instead the warped plan is
edge-detected into white linework over black.

Outputs (no collage, no frames, one hero per file):
  placa_catalunya_trace_dust_hero_black.png            (balanced)
  placa_catalunya_trace_dust_hero_black_more_glow.png
  placa_catalunya_trace_dust_hero_black_less_glow.png

Usage:
    python generate_trace_dust_hero.py
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from PIL import Image

BEHAVIOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BEHAVIOR_DIR))
from generate_bottleneck_map_test import warp_plan_to_world


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SITE = Path(r"C:\Users\OWNER\Desktop\new_datasets\placa_catalunya_01")
DEF_TRAJ  = SITE / "filtered_250m" / "trajectories_world_filtered_250m.csv"
DEF_PLAN  = SITE / "plan" / "placa-catalunya.png"
DEF_CALIB = SITE / "calibration" / "calib.json"

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_BASE = OUT_DIR / "placa_catalunya_trace_dust_hero_black.png"
OUT_MORE = OUT_DIR / "placa_catalunya_trace_dust_hero_black_more_glow.png"
OUT_LESS = OUT_DIR / "placa_catalunya_trace_dust_hero_black_less_glow.png"
OUT_REPORT = OUT_DIR / "placa_catalunya_trace_dust_hero_report.md"


# ---------------------------------------------------------------------------
# Tunables (rendering only)
# ---------------------------------------------------------------------------

WIDTH_PX = 4200
DPI = 200

# trajectory prep
MIN_POINTS = 4
RESAMPLE_STEP_M = 0.4
MAX_POINTS = 400
SMOOTH_SIGMA_PTS = 1.2

# traces
TRACE_W = 0.7
TRACE_ALPHA = 0.05
# dust
DUST_FRAME_STEP = 4         # sample every Nth position
DUST_SIZE = 1.1
DUST_ALPHA = 0.05

# density colouring
DENS_RES_M = 0.6
DENS_SIGMA_M = 1.2
DENS_GAMMA = 0.60

CORE_GAIN = 1.7             # brightness of the sharp movement core

# plan linework (edge detection)
CANNY_LO, CANNY_HI = 60, 150
PLAN_OPACITY = 0.16         # white linework brightness over black (secondary)

# palette low -> highest density
PAL = ["#00D5FF", "#2563EB", "#7C3AED", "#D946EF"]
PAL_CMAP = LinearSegmentedColormap.from_list("hero", PAL)

# glow presets: (radii_px, weights, glow_strength)
GLOW = {
    "base": ([5, 12, 26], [0.55, 0.40, 0.28], 1.0),
    "more": ([7, 18, 38], [0.6, 0.5, 0.4], 1.5),
    "less": ([3, 8],       [0.5, 0.32],      0.55),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def resample_smooth(x, y):
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    if d[-1] < 1e-6:
        return None
    n = int(np.clip(d[-1] / RESAMPLE_STEP_M + 1, 2, MAX_POINTS))
    t = np.linspace(0, d[-1], n)
    xi = np.interp(t, d, x); yi = np.interp(t, d, y)
    if n >= 5:
        xi = gaussian_filter1d(xi, SMOOTH_SIGMA_PTS, mode="nearest")
        yi = gaussian_filter1d(yi, SMOOTH_SIGMA_PTS, mode="nearest")
    return np.column_stack([xi, yi])


def load(traj_path, extent):
    xmin, xmax, ymin, ymax = extent
    df = pd.read_csv(traj_path, usecols=["frame", "track_id", "world_x", "world_y"])
    n_total = len(df); n_tracks = int(df.track_id.nunique())
    df = df[(df.world_x >= xmin) & (df.world_x <= xmax) &
            (df.world_y >= ymin) & (df.world_y <= ymax)].sort_values(["track_id", "frame"])

    polylines = []
    for _, g in df.groupby("track_id"):
        x = g.world_x.to_numpy(); y = g.world_y.to_numpy()
        if len(x) < MIN_POINTS:
            continue
        pl = resample_smooth(x, y)
        if pl is not None:
            polylines.append(pl)

    dust = df.iloc[::DUST_FRAME_STEP]
    dust_xy = np.column_stack([dust.world_x.to_numpy(), dust.world_y.to_numpy()])
    return polylines, dust_xy, n_total, n_tracks


def density_field(polylines, extent):
    xmin, xmax, ymin, ymax = extent
    xe = np.arange(xmin, xmax + DENS_RES_M, DENS_RES_M)
    ye = np.arange(ymin, ymax + DENS_RES_M, DENS_RES_M)
    allx = np.concatenate([p[:, 0] for p in polylines])
    ally = np.concatenate([p[:, 1] for p in polylines])
    H, _, _ = np.histogram2d(allx, ally, bins=[xe, ye])
    D = gaussian_filter(H.T, DENS_SIGMA_M / DENS_RES_M, mode="constant")
    xs = 0.5 * (xe[:-1] + xe[1:]); ys = 0.5 * (ye[:-1] + ye[1:])
    return D, xs, ys, float(np.percentile(D[D > 0], 98))


def sample_density(D, xs, ys, pts, vmax):
    fx = np.clip((pts[:, 0] - xs[0]) / DENS_RES_M, 0, len(xs) - 1.001)
    fy = np.clip((pts[:, 1] - ys[0]) / DENS_RES_M, 0, len(ys) - 1.001)
    i0 = fx.astype(int); j0 = fy.astype(int); tx = fx - i0; ty = fy - j0
    d = ((D[j0, i0] * (1 - tx) + D[j0, i0 + 1] * tx) * (1 - ty) +
         (D[j0 + 1, i0] * (1 - tx) + D[j0 + 1, i0 + 1] * tx) * ty)
    return np.clip(d / max(vmax, 1e-9), 0, 1)


# ---------------------------------------------------------------------------
# Plan linework (edges over black)
# ---------------------------------------------------------------------------

def plan_linework(warped_rgb):
    gray = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 40, 40)
    edges = cv2.Canny(gray, CANNY_LO, CANNY_HI)
    e = edges.astype(np.float32) / 255.0
    e = gaussian_filter(e, 0.6)
    if e.max() > 0:
        e /= e.max()
    return e  # 0..1 intensity, white linework


# ---------------------------------------------------------------------------
# Layer rendering (float RGB on black, exact pixel size, world-aligned)
# ---------------------------------------------------------------------------

def fig_dims(extent):
    xmin, xmax, ymin, ymax = extent
    W_in = WIDTH_PX / DPI
    H_in = W_in * (ymax - ymin) / (xmax - xmin)
    return W_in, H_in


def render_layer(extent, draw_fn):
    xmin, xmax, ymin, ymax = extent
    W_in, H_in = fig_dims(extent)
    fig = plt.figure(figsize=(W_in, H_in), dpi=DPI)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("black")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin); ax.axis("off")
    draw_fn(ax)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32) / 255.0
    plt.close(fig)
    return buf[..., :3]


def draw_movement(ax, polylines, dust_xy, D, xs, ys, vmax):
    # dust first (atmosphere), then strings on top
    dv = sample_density(D, xs, ys, dust_xy, vmax)
    drgba = PAL_CMAP(dv ** DENS_GAMMA); drgba[:, 3] = DUST_ALPHA
    ax.scatter(dust_xy[:, 0], dust_xy[:, 1], s=DUST_SIZE, c=drgba, marker=".",
               edgecolors="none", linewidths=0, zorder=2)
    segs, vals = [], []
    for p in polylines:
        if len(p) < 2:
            continue
        segs.append(np.stack([p[:-1], p[1:]], axis=1))
        vals.append(sample_density(D, xs, ys, 0.5 * (p[:-1] + p[1:]), vmax))
    segs = np.concatenate(segs, 0); vals = np.concatenate(vals, 0)
    rgba = PAL_CMAP(vals ** DENS_GAMMA); rgba[:, 3] = TRACE_ALPHA
    ax.add_collection(LineCollection(segs, colors=rgba, linewidths=TRACE_W,
                      capstyle="round", joinstyle="round", antialiased=True, zorder=3))


def bloom(M, radii, weights):
    out = np.zeros_like(M)
    for r, w in zip(radii, weights):
        for c in range(3):
            out[..., c] += w * gaussian_filter(M[..., c], r)
    return out


def compose(M, P_edges_rgb, radii, weights, glow_strength):
    b = bloom(M, radii, weights)
    # plan linework with a cool-grey tint so it sits behind the movement
    plan = PLAN_OPACITY * P_edges_rgb * np.array([0.72, 0.80, 0.95])
    out = CORE_GAIN * M + glow_strength * b + plan
    # gentle highlight rolloff keeps cores luminous without hard clipping
    out = out / (1.0 + 0.16 * out)
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=str(DEF_TRAJ))
    ap.add_argument("--plan", default=str(DEF_PLAN))
    ap.add_argument("--calib", default=str(DEF_CALIB))
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    warped_rgb, extent = warp_plan_to_world(Path(args.plan), Path(args.calib))
    xmin, xmax, ymin, ymax = extent

    polylines, dust_xy, n_total, n_tracks = load(Path(args.traj), extent)
    n_traces = len(polylines); n_dust = len(dust_xy)
    D, xs, ys, vmax = density_field(polylines, extent)

    # Movement layer (float RGB on black)
    M = render_layer(extent, lambda ax: draw_movement(ax, polylines, dust_xy, D, xs, ys, vmax))

    # Plan linework -> resize to match M pixel grid
    edges = plan_linework(warped_rgb)
    Hh, Ww = M.shape[:2]
    edges_r = cv2.resize(edges, (Ww, Hh), interpolation=cv2.INTER_AREA)
    P = np.dstack([edges_r] * 3)  # white lines

    for name, out_png in (("base", OUT_BASE), ("more", OUT_MORE), ("less", OUT_LESS)):
        radii, weights, gs = GLOW[name]
        final = compose(M, P, radii, weights, gs)
        Image.fromarray((final * 255).astype(np.uint8)).save(out_png)
        print(f"[INFO] Saved {out_png}  ({Ww}x{Hh})")

    print(f"[INFO] Plan extent X[{xmin:.2f},{xmax:.2f}] Y[{ymin:.2f},{ymax:.2f}] m")
    print(f"[INFO] traj rows {n_total} | tracks {n_tracks} | traces {n_traces} | dust {n_dust}")
    write_report(n_total, n_tracks, n_traces, n_dust, extent, (Ww, Hh))


def write_report(n_total, n_tracks, n_traces, n_dust, extent, size):
    xmin, xmax, ymin, ymax = extent
    Ww, Hh = size
    txt = f"""# Plaça Catalunya — Trace + Dust Hero (black, glowing)

**This is a visualization-only rendering of existing tracked pedestrian
trajectories. No retracking, recalibration, model inference, or source-data
modification was performed.**

A single cinematic hero image: luminous pedestrian movement (glowing trace strings
+ motion dust, density-coloured) imprinted on a black architectural plan rendered
as fine white linework. Movement is the hero; the plan is secondary.

## Source data
`{DEF_TRAJ}`
- Tracks: **{n_tracks}** · Trajectory rows: **{n_total}**
- Trace strings drawn: **{n_traces}**
- Dust points drawn: **{n_dust}** (every {DUST_FRAME_STEP}th in-frame position)

## Clipping
`warp_plan_to_world()` reused from `generate_bottleneck_map_test.py`; plan image
extent is the authoritative frame, axes locked to the plan corners (handles the
known calibration mismatch). No margins/axes/frame. **No recalibration.**
World extent X = [{xmin:.2f}, {xmax:.2f}] m, Y = [{ymin:.2f}, {ymax:.2f}] m.
Output size **{Ww} × {Hh}px** (≥ 4000 wide, {DPI} dpi-equivalent), PNG, black bg.

## Smoothing
Each track arc-length resampled (step {RESAMPLE_STEP_M:.1f} m, ≤ {MAX_POINTS} pts)
and lightly Gaussian-smoothed (σ = {SMOOTH_SIGMA_PTS} pts) — jitter removed, real
route geometry preserved. No arrows/markers/endpoints.

## Density colouring
Density grid ({DENS_RES_M:.1f} m, σ {DENS_SIGMA_M} m) from trace vertices; traces
coloured per segment and dust per point by local density (γ = {DENS_GAMMA}) on the
palette **{" → ".join(PAL)}** (cyan → blue → violet → magenta). Sparse routes read
cyan/blue; dominant corridors and decision knots saturate to violet/magenta.
Colour is applied to movement only — never to the plan.

## Trace + dust settings
- Trace strings: width {TRACE_W} px, per-trace opacity {TRACE_ALPHA} (accumulate
  by overplotting on black).
- Motion dust: marker {DUST_SIZE} px², opacity {DUST_ALPHA}, sampled every
  {DUST_FRAME_STEP}th position. Dust adds atmosphere/density beneath the strings.

## Glow (layered additive bloom, post-process)
The movement layer is rendered to a float buffer on black, then multi-radius
Gaussian blurs are added back (bloom) and a soft highlight rolloff keeps cores
luminous without hard clipping. Presets (radii px / weights / strength):
- base: {GLOW['base'][0]} / {GLOW['base'][1]} / {GLOW['base'][2]}
- more glow: {GLOW['more'][0]} / {GLOW['more'][1]} / {GLOW['more'][2]}
- less glow: {GLOW['less'][0]} / {GLOW['less'][1]} / {GLOW['less'][2]}

## Plan linework extraction
The warped plan is converted to grey, bilateral-filtered, and **Canny edge-
detected** (thresholds {CANNY_LO}/{CANNY_HI}), softened, normalized, and drawn as
white linework over black at opacity {PLAN_OPACITY}. No filled map, no pale
overlay — fine white/grey lines only.

## Output paths
- `{OUT_BASE.name}` (balanced)
- `{OUT_MORE.name}` (more glow)
- `{OUT_LESS.name}` (less glow)
"""
    OUT_REPORT.write_text(txt, encoding="utf-8")
    print(f"[INFO] Report saved: {OUT_REPORT}")


if __name__ == "__main__":
    main()
