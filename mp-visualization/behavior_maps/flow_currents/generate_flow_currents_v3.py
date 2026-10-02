"""
generate_flow_currents_v3.py
----------------------------
MOTION PIXELS — Flow Currents V3 (Plaça Catalunya).

Revises V2 on four points:
  1. LESS GLOW / readable filaments — bloom reduced ~30%, widest blur 45→30,
     core gain lowered; strings/strands read as fiber optics, not neon fog.
     Hierarchy: bright strings > dust currents > plan > glow.
  2. INFLOW — particles are oriented to flow toward the denser corridor (each
     trajectory reversed if it starts dense), so currents CONVERGE and
     accumulate into the dominant corridor (river-tributary feel).
  3. ARRIVAL FROM OFF-SCREEN — each flow path is extrapolated a few metres past
     its start, so particles fade in from beyond the network and stream inward.
  4. 16:9 CANVAS — a taller window (1920×1080) centred on the plan, no stretch;
     the city breathes above/below the movement.

Background frozen = V3 "more_plan" architectural drawing.

VISUALIZATION ONLY. Built from existing tracked trajectories. No retracking,
recalibration, model inference, or source-data modification.

Usage:
    python generate_flow_currents_v3.py
    python generate_flow_currents_v3.py --preview
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
from PIL import Image
import imageio.v2 as imageio

BEHAVIOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BEHAVIOR_DIR))
from generate_bottleneck_map_test import warp_plan_to_world
HERO_DIR = BEHAVIOR_DIR / "trace_dust_hero"
sys.path.insert(0, str(HERO_DIR))
import generate_trace_dust_hero_v3 as v3


DEF_TRAJ, DEF_PLAN, DEF_CALIB = v3.DEF_TRAJ, v3.DEF_PLAN, v3.DEF_CALIB
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_GIF = OUT_DIR / "placa_catalunya_flow_currents_v3.gif"
OUT_MP4 = OUT_DIR / "placa_catalunya_flow_currents_v3.mp4"
OUT_REPORT = OUT_DIR / "placa_catalunya_flow_currents_v3_report.md"


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

CANVAS_W, CANVAS_H = 1920, 1080      # 16:9
GIF_WIDTH = 1000
GIF_COLORS = 128
FPS = 16
SECONDS = 7.0
HOLD_S = 0.7

PAL = ["#00E5FF", "#2F6BFF", "#8B5CFF", "#FF3DF2"]
PAL_CMAP = LinearSegmentedColormap.from_list("currents_v3", PAL)
DENS_GAMMA = 1.4

# stages
PLAN_FADE  = 0.14
DRAW_START = 0.14
DRAW_END   = 0.52
DUST_START = 0.50
GLOW_FULL_T = 0.92

# traces (strings = hierarchy #1, kept bright & sharp)
TRACE_W = 0.9
TRACE_ALPHA = 0.075
TRACE_CORE_GAIN = 2.0
DRAW_DUR = 0.12

# particles / inflow
N_A = 4200; N_B = 6500
A_BLUR = 0.8; A_GAIN = 2.4           # sharp filaments
B_BLUR = 4.0; B_GAIN = 1.1
TRAIL_DECAY = 0.74                   # crisper streaks than V2
PARTICLE_CORE_GAIN = 1.8
TARGET_SPEED_MS = 3.8
K_SAMPLES = 200
MIN_LEN_M = 6.0
EXTRA_M = 5.0                        # off-screen lead-in length
FADE_IN = 0.12; FADE_OUT = 0.06      # phase envelope (hides respawn, reads as arrival)

# glow REDUCED (~30%); widest blur 45->30
BLOOM_SIGMAS = [3, 9, 20, 30]
BLOOM_WEIGHTS = [0.5, 0.38, 0.26, 0.16]
GLOW_DRAW = (0.15, 0.38)
GLOW_DUST = (0.5, 1.0)
ROLLOFF = 0.12

PLAN_MAJOR_OP, PLAN_SECONDARY_OP = 0.64, 0.30


# ---------------------------------------------------------------------------
# Geometry / window
# ---------------------------------------------------------------------------

def window_extent(plan_extent, W, H):
    xmin, xmax, ymin, ymax = plan_extent
    ww = xmax - xmin
    hh = ww * H / W                       # world height for 16:9, no stretch
    yc = 0.5 * (ymin + ymax)
    return (xmin, xmax, yc - hh / 2, yc + hh / 2)


def resample_const(pts, k):
    x, y = pts[:, 0], pts[:, 1]
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    if d[-1] < 1e-6:
        return None, 0.0
    t = np.linspace(0, d[-1], k)
    return np.column_stack([np.interp(t, d, x), np.interp(t, d, y)]), float(d[-1])


def load_geometry(plan_extent):
    xmin, xmax, ymin, ymax = plan_extent
    df = pd.read_csv(DEF_TRAJ, usecols=["frame", "track_id", "world_x", "world_y"])
    n_total = len(df); n_tracks = int(df.track_id.nunique())
    df = df[(df.world_x >= xmin) & (df.world_x <= xmax) &
            (df.world_y >= ymin) & (df.world_y <= ymax)].sort_values(["track_id", "frame"])
    polylines = []
    for _, g in df.groupby("track_id"):
        x = g.world_x.to_numpy(); y = g.world_y.to_numpy()
        if len(x) < 3:
            continue
        pl = v3.resample_smooth(x, y)
        if pl is not None and len(pl) >= 2:
            polylines.append(pl)
    D, xs, ys, vmax = v3.density_field(polylines, plan_extent)
    return polylines, (D, xs, ys, vmax), n_total, n_tracks


def build_segments(polylines, dens, rng):
    D, xs, ys, vmax = dens
    seg_xy, seg_rgb, seg_appear = [], [], []
    for pl in polylines:
        m = len(pl)
        if m < 2:
            continue
        start = DRAW_START + rng.random() * (DRAW_END - DRAW_DUR - DRAW_START)
        frac = np.linspace(0, 1, m - 1)
        mid = 0.5 * (pl[:-1] + pl[1:])
        col = PAL_CMAP(v3.sample_density(D, xs, ys, mid, vmax) ** DENS_GAMMA)[:, :3]
        seg_xy.append(np.stack([pl[:-1], pl[1:]], axis=1))
        seg_rgb.append(col); seg_appear.append(start + frac * DRAW_DUR)
    return (np.concatenate(seg_xy, 0), np.concatenate(seg_rgb, 0),
            np.concatenate(seg_appear, 0))


def build_flow(polylines, dens):
    """Flow paths oriented toward the denser corridor (convergence) and
    extrapolated a few metres past the start (off-screen arrival)."""
    D, xs, ys, vmax = dens
    samples, lengths, colors = [], [], []
    for pl in polylines:
        s, L = resample_const(pl, K_SAMPLES)
        if s is None or L < MIN_LEN_M:
            continue
        d0 = v3.sample_density(D, xs, ys, s[:1], vmax)[0]
        d1 = v3.sample_density(D, xs, ys, s[-1:], vmax)[0]
        if d0 > d1:                              # flow low-density -> high-density
            s = s[::-1].copy()
        dirv = s[0] - s[1]; nrm = np.hypot(*dirv)
        if nrm > 1e-6:                            # extrapolate start off-screen
            dirv = dirv / nrm
            lead = s[0][None, :] + dirv[None, :] * np.linspace(EXTRA_M, 0, 8)[:, None]
            s = np.vstack([lead[:-1], s])
        s, L = resample_const(s, K_SAMPLES)
        col = PAL_CMAP(v3.sample_density(D, xs, ys, s, vmax) ** DENS_GAMMA)[:, :3]
        samples.append(s); lengths.append(L); colors.append(col)
    return np.array(samples), np.array(lengths), np.array(colors)


# ---------------------------------------------------------------------------
# Plan placed into the 16:9 window
# ---------------------------------------------------------------------------

def plan_window(warped_rgb, plan_extent, win_extent, W, H):
    major, secondary, n_lines = v3.architectural_layers(warped_rgb)
    plan_small = (PLAN_MAJOR_OP * np.dstack([major] * 3) * v3.MAJOR_TINT +
                  PLAN_SECONDARY_OP * np.dstack([secondary] * 3) * v3.SECONDARY_TINT)
    plan_small = np.clip(plan_small, 0, 1).astype(np.float32)

    xmin, xmax, ymin, ymax = plan_extent
    _, _, ymin_w, ymax_w = win_extent
    row0 = int(round((ymin - ymin_w) / (ymax_w - ymin_w) * H))
    row1 = int(round((ymax - ymin_w) / (ymax_w - ymin_w) * H))
    ph = max(row1 - row0, 2)
    plan_resized = cv2.resize(plan_small, (W, ph), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((H, W, 3), dtype=np.float32)
    a, b = max(row0, 0), min(row1, H)
    canvas[a:b] = plan_resized[(a - row0):(b - row0)]
    return canvas, n_lines


def render_segment_buffer(seg_xy, seg_rgb, mask, win_extent, W, H):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("black")
    xmin, xmax, ymin_w, ymax_w = win_extent
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymax_w, ymin_w); ax.axis("off")
    if mask.any():
        rgba = np.concatenate([seg_rgb[mask], np.full((mask.sum(), 1), TRACE_ALPHA)], 1)
        ax.add_collection(LineCollection(seg_xy[mask], colors=rgba, linewidths=TRACE_W,
                          capstyle="round", joinstyle="round", antialiased=True))
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32)[..., :3] / 255.0
    plt.close(fig)
    return buf


def bloom(src, glow_strength):
    out = np.zeros_like(src)
    for s, w in zip(BLOOM_SIGMAS, BLOOM_WEIGHTS):
        for c in range(3):
            out[..., c] += w * cv2.GaussianBlur(src[..., c], (0, 0), s)
    return glow_strength * out


def splat(W, H, px, py, col, wgt):
    acc = np.zeros((H, W, 3), dtype=np.float32)
    flat = py * W + px
    for c in range(3):
        np.add.at(acc[..., c].ravel(), flat, col[:, c] * wgt)
    return acc


def lerp(a, b, t):
    return a + (b - a) * np.clip(t, 0, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    global N_A, N_B
    if args.preview:
        N_A, N_B = 1800, 2800

    warped_rgb, plan_extent = warp_plan_to_world(DEF_PLAN, DEF_CALIB)
    W = CANVAS_W - (CANVAS_W % 2); H = CANVAS_H - (CANVAS_H % 2)
    win = window_extent(plan_extent, W, H)
    xmin, xmax, ymin_w, ymax_w = win

    polylines, dens, n_total, n_tracks = load_geometry(plan_extent)
    rng = np.random.default_rng(7)
    seg_xy, seg_rgb, seg_appear = build_segments(polylines, dens, rng)
    samples, lengths, colors = build_flow(polylines, dens)
    n_flow = len(samples)

    plan, n_lines = plan_window(warped_rgb, plan_extent, win, W, H)

    pprob = lengths / lengths.sum()
    tidA = rng.choice(n_flow, size=N_A, p=pprob); ph0A = rng.random(N_A)
    tidB = rng.choice(n_flow, size=N_B, p=pprob); ph0B = rng.random(N_B)

    F = int(round(SECONDS * FPS)); F_hold = int(round(HOLD_S * FPS)); total = F + F_hold
    print(f"[INFO] flow {n_flow} | tracks {n_tracks} | rows {n_total} | {W}x{H} | F {total}")

    gif_w = GIF_WIDTH - (GIF_WIDTH % 2); gif_h = int(round(gif_w * H / W)); gif_h -= gif_h % 2
    gif_frames = []
    mp4 = imageio.get_writer(OUT_MP4, fps=FPS, codec="libx264", quality=8,
                             macro_block_size=None, output_params=["-pix_fmt", "yuv420p"])

    full_mask = np.ones(len(seg_appear), bool)
    trace_full = None
    trail = np.zeros((H, W, 3), dtype=np.float32)

    def phase_of(ph0, tid, f):
        return (ph0 + TARGET_SPEED_MS * (f / FPS) / lengths[tid]) % 1.0

    for fi in range(total):
        f = min(fi, F - 1); t = f / (F - 1)
        frame = plan * np.clip(t / PLAN_FADE, 0, 1)

        movement = np.zeros((H, W, 3), dtype=np.float32)
        if t >= DRAW_START:
            if t >= DRAW_END:
                if trace_full is None:
                    trace_full = render_segment_buffer(seg_xy, seg_rgb, full_mask, win, W, H)
                traces = trace_full
            else:
                traces = render_segment_buffer(seg_xy, seg_rgb, seg_appear <= t, win, W, H)
            movement += TRACE_CORE_GAIN * traces

        if t >= DUST_START:
            pg = np.clip((t - DUST_START) / (0.85 - DUST_START), 0, 1)
            inst = np.zeros((H, W, 3), dtype=np.float32)
            for tid, ph0, blur, gain in ((tidA, ph0A, A_BLUR, A_GAIN),
                                         (tidB, ph0B, B_BLUR, B_GAIN)):
                phase = phase_of(ph0, tid, f)
                env = np.clip(phase / FADE_IN, 0, 1) * np.clip((1 - phase) / FADE_OUT, 0, 1)
                idx = np.clip((phase * K_SAMPLES).astype(np.int32), 0, K_SAMPLES - 1)
                pos = samples[tid, idx]; col = colors[tid, idx] * env[:, None]
                px = np.clip(((pos[:, 0] - xmin) / (xmax - xmin) * (W - 1)).astype(np.int32), 0, W - 1)
                py = np.clip(((pos[:, 1] - ymin_w) / (ymax_w - ymin_w) * (H - 1)).astype(np.int32), 0, H - 1)
                acc = splat(W, H, px, py, col, gain * pg)
                for c in range(3):
                    acc[..., c] = cv2.GaussianBlur(acc[..., c], (0, 0), blur)
                inst += acc
            trail = trail * TRAIL_DECAY + inst
            movement += PARTICLE_CORE_GAIN * trail

        if t < DRAW_START:
            gs = 0.0
        elif t < DUST_START:
            gs = lerp(*GLOW_DRAW, (t - DRAW_START) / (DUST_START - DRAW_START))
        else:
            gs = lerp(*GLOW_DUST, (t - DUST_START) / (GLOW_FULL_T - DUST_START))

        out = frame + movement + bloom(movement, gs)
        out = out / (1.0 + ROLLOFF * out)
        img = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        mp4.append_data(img)
        pim = Image.fromarray(img).resize((gif_w, gif_h), Image.LANCZOS).convert("RGB")
        gif_frames.append(pim.quantize(colors=GIF_COLORS, method=Image.MEDIANCUT,
                                       dither=Image.NONE))
        if fi % 20 == 0:
            print(f"   frame {fi}/{total}")

    mp4.close()
    gif_frames[0].save(OUT_GIF, save_all=True, append_images=gif_frames[1:],
                       duration=int(round(1000 / FPS)), loop=0, optimize=True, disposal=2)
    gif_mb = OUT_GIF.stat().st_size / 1e6
    print(f"[INFO] Saved {OUT_GIF} ({gif_mb:.1f} MB) and {OUT_MP4}")
    write_report(n_total, n_tracks, n_flow, n_lines, total, (W, H), gif_mb)


def write_report(n_total, n_tracks, n_flow, n_lines, total, size, gif_mb):
    W, H = size
    txt = f"""# Plaça Catalunya — Flow Currents V3

**Visualization-only animation from existing tracked trajectories. No retracking,
recalibration, model inference, or source-data modification.**

Revises V2 to read as an **urban energy field** (not a glowing cloud): individual
flowing strands are visible, currents stream IN from the edges and converge into
the dominant corridor, and the composition breathes in 16:9.

## Format
- GIF (primary): `{OUT_GIF.name}` — **{gif_mb:.1f} MB**, {GIF_WIDTH}px, {FPS} fps, loops.
- MP4 (backup): `{OUT_MP4.name}` — H.264, **{W}×{H} (16:9)**.
- Duration ~{SECONDS:.0f}s + {HOLD_S:.1f}s hold; {total} frames.

## 1. Less glow / readable strings
Bloom reduced ~30% and widest blur **45→30 px** (sigmas {BLOOM_SIGMAS}, weights
{BLOOM_WEIGHTS}); glow strength {GLOW_DUST} (was up to 1.5). Trace strings kept
bright/sharp (width {TRACE_W}, gain {TRACE_CORE_GAIN}); particle trails crisper
(decay {TRAIL_DECAY}, Type-A blur {A_BLUR}). **Hierarchy: strings > dust > plan >
glow.** Image stays luminous (palette saturation preserved) but filaments read.

## 2. Inflow / convergence
Each flow path is oriented to run from its **lower-density end to its
higher-density end**, so currents converge and accumulate into the dominant
corridor (river-tributary behaviour) rather than circulating in place.

## 3. Arrival from off-screen
Every flow path is extrapolated **{EXTRA_M:.0f} m past its start** along its
initial heading, and particles **fade in** over the first {FADE_IN:.0%} of the
path (and out over the last {FADE_OUT:.0%}, hiding respawn). Combined with the
taller canvas, dust streams in from beyond the network — "movement is arriving."

## 4. 16:9 canvas
Window is **{W}×{H}** centred on the plan with no stretch (world window height =
width·9/16); the city now breathes above and below the movement. Plan placed at
its true world band; black margins elsewhere.

## 5. Plan preserved
Frozen V3 "more_plan" drawing: {n_lines} major Hough lines + speckle-cleaned
secondary detail (buildings, trees, plaza geometry, street edges), major op
{PLAN_MAJOR_OP} / secondary {PLAN_SECONDARY_OP} — secondary but readable.

## Staging (unchanged structure)
plan (0–{PLAN_FADE:.0%}) → strings draw in ({DRAW_START:.0%}–{DRAW_END:.0%}) →
dust flows in + glow builds ({DUST_START:.0%}–100%, hold at end).

## Data
`{DEF_TRAJ}` — tracks {n_tracks}, rows {n_total}, flow paths {n_flow}; particles
A {N_A} + B {N_B}, ~{TARGET_SPEED_MS} m/s, length-weighted onto paths.

## Output paths
`{OUT_GIF.name}` (primary), `{OUT_MP4.name}` (backup), this report.
"""
    OUT_REPORT.write_text(txt, encoding="utf-8")
    print(f"[INFO] Report saved: {OUT_REPORT}")


if __name__ == "__main__":
    main()
