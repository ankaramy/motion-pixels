"""
generate_trace_dust_hero_v3.py
------------------------------
MOTION PIXELS — FINAL Trace + Dust Glow HERO V3 (Plaça Catalunya).

Black background + a LAYERED architectural drawing (major geometry + faint
secondary detail) + glowing density-coloured trace strings + motion dust +
cinematic bloom. Movement is the hero; architecture is the stage.

Improvements over V2:
  - background is a layered architectural drawing, not flat Canny "noodles"
    (major rectilinear geometry via Hough long lines + a faint, speckle-cleaned
    secondary-detail layer for trees/objects/paving);
  - more visible plan, stronger strings (more traces, higher width/opacity),
    denser dust, stronger layered bloom;
  - density gamma inverted (>1) so cyan/blue spread is preserved and only the
    densest corridor saturates to violet/magenta (fixes V2's magenta collapse).

VISUALIZATION ONLY. A rendering of existing tracked pedestrian trajectories. No
retracking, recalibration, model inference, or source-data modification.

Reuses `warp_plan_to_world` from `../generate_bottleneck_map_test.py`; clipped to
the plan extent; black bg; no text/axes/legend/frame.

Usage:
    python generate_trace_dust_hero_v3.py
"""

from pathlib import Path
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
OUT_V3        = OUT_DIR / "placa_catalunya_trace_dust_hero_final_v3.png"
OUT_MORE_PLAN = OUT_DIR / "placa_catalunya_trace_dust_hero_final_v3_more_plan.png"
OUT_MORE_GLOW = OUT_DIR / "placa_catalunya_trace_dust_hero_final_v3_more_glow.png"
OUT_REPORT    = OUT_DIR / "placa_catalunya_trace_dust_hero_final_v3_report.md"


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

WIDTH_PX = 4400
DPI = 200

# trajectory prep (more strings than V2)
MIN_POINTS = 3
RESAMPLE_STEP_M = 0.4
MAX_POINTS = 420
SMOOTH_SIGMA_PTS = 1.1

# traces (stronger)
TRACE_W = 0.8
TRACE_ALPHA = 0.045
# dust (denser, more atmosphere)
DUST_FRAME_STEP = 3
DUST_SIZE = 1.1
DUST_ALPHA = 0.06

# density colouring — gamma > 1 keeps cyan/blue spread (only peaks -> magenta)
DENS_RES_M = 0.6
DENS_SIGMA_M = 1.2
DENS_GAMMA = 1.4

CORE_GAIN = 1.8
ROLLOFF = 0.13

# palette low -> highest density
PAL = ["#00D5FF", "#2563EB", "#7C3AED", "#D946EF"]
PAL_CMAP = LinearSegmentedColormap.from_list("hero_v3", PAL)

# layered architectural background
BILATERAL = (9, 80, 80)
CANNY = (50, 150)
HOUGH_THRESH = 50
HOUGH_MIN_LEN = 40
HOUGH_MAX_GAP = 7
SECONDARY_MIN_DIAG = 16        # drop speckle below this; keep trees/objects/paving
SECONDARY_MIN_AREA = 12

MAJOR_TINT = np.array([0.86, 0.90, 1.0])   # cool white
SECONDARY_TINT = np.array([0.60, 0.70, 0.86])  # cooler grey

# presets: major_op, secondary_op, (radii, weights, glow_strength)
PRESETS = {
    "v3":        (0.46, 0.18, ([4, 10, 22], [0.62, 0.46, 0.34], 1.18)),
    "more_plan": (0.64, 0.30, ([4, 10, 22], [0.62, 0.46, 0.34], 1.10)),
    "more_glow": (0.42, 0.16, ([5, 13, 28], [0.7, 0.55, 0.42], 1.65)),
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
# Layered architectural background
# ---------------------------------------------------------------------------

def architectural_layers(warped_rgb):
    gray = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.bilateralFilter(gray, *BILATERAL)
    edges = cv2.Canny(g, *CANNY)

    # Layer A — MAJOR geometry: long straight structural lines.
    major = np.zeros(gray.shape, dtype=np.float32)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=HOUGH_THRESH,
                            minLineLength=HOUGH_MIN_LEN, maxLineGap=HOUGH_MAX_GAP)
    n_lines = 0 if lines is None else len(lines)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            cv2.line(major, (x1, y1), (x2, y2), 1.0, 1, cv2.LINE_AA)
    major = gaussian_filter(major, 0.5)
    if major.max() > 0:
        major /= major.max()

    # Layer B — SECONDARY detail: speckle-cleaned edges (keep trees/objects/paving).
    num, lab, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    keep = np.zeros(edges.shape, dtype=np.float32)
    for i in range(1, num):
        w = stats[i, cv2.CC_STAT_WIDTH]; h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if np.hypot(w, h) >= SECONDARY_MIN_DIAG and area >= SECONDARY_MIN_AREA:
            keep[lab == i] = 1.0
    secondary = gaussian_filter(keep, 0.7)
    if secondary.max() > 0:
        secondary /= secondary.max()
    return major, secondary, n_lines


# ---------------------------------------------------------------------------
# Movement layer + bloom + compose
# ---------------------------------------------------------------------------

def render_movement(extent, polylines, dust_xy, D, xs, ys, vmax):
    xmin, xmax, ymin, ymax = extent
    W_in = WIDTH_PX / DPI
    H_in = W_in * (ymax - ymin) / (xmax - xmin)
    fig = plt.figure(figsize=(W_in, H_in), dpi=DPI)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("black")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin); ax.axis("off")

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
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32) / 255.0
    plt.close(fig)
    return buf[..., :3], len(segs)


def bloom(M, radii, weights):
    out = np.zeros_like(M)
    for r, w in zip(radii, weights):
        for c in range(3):
            out[..., c] += w * gaussian_filter(M[..., c], r)
    return out


def compose(M, major, secondary, major_op, secondary_op, radii, weights, glow):
    b = bloom(M, radii, weights)
    plan = (major_op * np.dstack([major] * 3) * MAJOR_TINT +
            secondary_op * np.dstack([secondary] * 3) * SECONDARY_TINT)
    out = CORE_GAIN * M + glow * b + plan
    out = out / (1.0 + ROLLOFF * out)
    return np.clip(out, 0, 1)


def resize_to(M, intensity):
    Hh, Ww = M.shape[:2]
    return cv2.resize(intensity, (Ww, Hh), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warped_rgb, extent = warp_plan_to_world(DEF_PLAN, DEF_CALIB)

    polylines, dust_xy, n_total, n_tracks = load(DEF_TRAJ, extent)
    D, xs, ys, vmax = density_field(polylines, extent)
    M, n_segs = render_movement(extent, polylines, dust_xy, D, xs, ys, vmax)
    Hh, Ww = M.shape[:2]

    major, secondary, n_lines = architectural_layers(warped_rgb)
    major_r = resize_to(M, major); secondary_r = resize_to(M, secondary)

    for name, out_png in (("v3", OUT_V3), ("more_plan", OUT_MORE_PLAN),
                          ("more_glow", OUT_MORE_GLOW)):
        mo, so, (radii, weights, glow) = PRESETS[name]
        final = compose(M, major_r, secondary_r, mo, so, radii, weights, glow)
        Image.fromarray((final * 255).astype(np.uint8)).save(out_png)
        print(f"[INFO] Saved {out_png}  ({Ww}x{Hh})")

    print(f"[INFO] traj rows {n_total} | tracks {n_tracks} | traces {len(polylines)} "
          f"| dust {len(dust_xy)} | hough lines {n_lines}")
    write_report(n_total, n_tracks, len(polylines), len(dust_xy), n_lines, extent, (Ww, Hh))


def write_report(n_total, n_tracks, n_traces, n_dust, n_lines, extent, size):
    xmin, xmax, ymin, ymax = extent
    Ww, Hh = size
    txt = f"""# Plaça Catalunya — FINAL Trace + Dust Glow Hero V3

**This is a visualization-only rendering of existing tracked pedestrian
trajectories. No retracking, recalibration, model inference, or source-data
modification was performed.**

Black background + layered architectural drawing + glowing density-coloured trace
strings + motion dust + cinematic bloom. Movement is the hero; architecture is the
stage.

## Source data
`{DEF_TRAJ}`
- Tracks: **{n_tracks}** · Trajectory rows: **{n_total}**
- Trace strings drawn: **{n_traces}** (min {MIN_POINTS} pts — more than V2)
- Dust points drawn: **{n_dust}** (every {DUST_FRAME_STEP}rd in-frame position)
- Output: **{Ww} × {Hh}px**, PNG, black background.

## Plan linework extraction (layered architectural drawing)
The warped plan is bilateral-filtered `{BILATERAL}` and Canny `{CANNY}`, then split
into two layers (no flat "noodle" Canny):
- **Layer A — major geometry:** `HoughLinesP` (threshold {HOUGH_THRESH},
  minLineLength {HOUGH_MIN_LEN}px, maxLineGap {HOUGH_MAX_GAP}) → **{n_lines}** long
  straight structural lines (buildings, roads, plaza edges). Cool white, opacity
  {PRESETS['v3'][0]} (v3).
- **Layer B — secondary detail:** Canny edges with speckle removed (connected
  components kept only if bbox diagonal ≥ {SECONDARY_MIN_DIAG}px and area ≥
  {SECONDARY_MIN_AREA}px) → trees / objects / paving hints. Cooler grey, opacity
  {PRESETS['v3'][1]} (v3).
- **Texture suppression:** bilateral pre-filter + the component-size filter remove
  tiny speckles and shadow noise so the city reads as a drawing, not a photo.

## Major geometry settings
Hough threshold {HOUGH_THRESH}, minLineLength {HOUGH_MIN_LEN}px, maxLineGap
{HOUGH_MAX_GAP}; tint {list(np.round(MAJOR_TINT,2))}.

## Secondary detail settings
Component filter diag ≥ {SECONDARY_MIN_DIAG}px / area ≥ {SECONDARY_MIN_AREA}px;
Gaussian soften σ 0.7; tint {list(np.round(SECONDARY_TINT,2))}.

## Movement strings + dust
- Strings: width {TRACE_W}px, opacity {TRACE_ALPHA} (up from V2), light smoothing
  (σ {SMOOTH_SIGMA_PTS} pts), real geometry preserved, round caps, no markers.
- Dust: marker {DUST_SIZE}px², opacity {DUST_ALPHA}, sampled every
  {DUST_FRAME_STEP}rd position — atmosphere beneath the strings.

## Density colour mapping
Palette **{" → ".join(PAL)}** (cyan → blue → violet → magenta), applied to movement
only. Density normalized to its 98th percentile; colour index = `norm^{DENS_GAMMA}`.
**γ = {DENS_GAMMA} (> 1)** spreads more of the field into cyan/blue and reserves
violet/magenta for the genuinely densest corridor — fixing V2's magenta collapse.

## Glow / bloom settings
Movement rendered to a float buffer on black; layered Gaussian bloom added back
(small + medium + wide) with `CORE_GAIN={CORE_GAIN}` and a soft rolloff
(`/(1+{ROLLOFF}·out)`):
- v3: radii {PRESETS['v3'][2][0]}, weights {PRESETS['v3'][2][1]}, strength {PRESETS['v3'][2][2]}
- more_glow: radii {PRESETS['more_glow'][2][0]}, weights {PRESETS['more_glow'][2][1]}, strength {PRESETS['more_glow'][2][2]}

## Output paths
- `{OUT_V3.name}` — **final (balanced)**
- `{OUT_MORE_PLAN.name}` — plan layers brighter (major {PRESETS['more_plan'][0]}, secondary {PRESETS['more_plan'][1]})
- `{OUT_MORE_GLOW.name}` — stronger/wider bloom (strength {PRESETS['more_glow'][2][2]})

## Recommendation
**`final_v3`** is the recommended hero: legible layered urban plan (major lines +
faint detail) on black, strong cyan→magenta movement glow, no noodles. Use
**`more_plan`** if the architecture should read more assertively (e.g. larger
prints / context-forward boards), and **`more_glow`** for a more cinematic,
movement-forward poster where the plan recedes further.

World extent X = [{xmin:.2f}, {xmax:.2f}] m, Y = [{ymin:.2f}, {ymax:.2f}] m.
No georeferenced OSM is used (no lat/lon in calibration); the plan drawing is
derived from the existing warped plan image.
"""
    OUT_REPORT.write_text(txt, encoding="utf-8")
    print(f"[INFO] Report saved: {OUT_REPORT}")


if __name__ == "__main__":
    main()
