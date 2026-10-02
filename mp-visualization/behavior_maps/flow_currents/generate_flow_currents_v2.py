"""
generate_flow_currents_v2.py
----------------------------
MOTION PIXELS — Flow Currents V2 (STAGED REVEAL), Plaça Catalunya.

Unlike V1 (everything present from frame 0), V2 reveals the movement layer over a
~7 s staged sequence:

    0.0–1.0s   plan only (black + white architectural linework; subtle fade-in)
    1.0–3.5s   trace strings draw in along real trajectory geometry
               (segment-by-segment, staggered — strings pulled through the city)
    3.5–7.0s   dust + glowing particles flow along/between the traced strings;
               glow blooms up to the bright Trace+Dust Hero V3 intensity

Brighter palette, much bigger layered bloom. Background frozen = V3 "more_plan".
Output: GIF (primary) + MP4 (backup).

VISUALIZATION ONLY. Built from existing tracked trajectories. No retracking,
recalibration, model inference, or source-data modification.

Usage:
    python generate_flow_currents_v2.py
    python generate_flow_currents_v2.py --preview      # fewer particles, faster
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


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEF_TRAJ, DEF_PLAN, DEF_CALIB = v3.DEF_TRAJ, v3.DEF_PLAN, v3.DEF_CALIB
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_GIF = OUT_DIR / "placa_catalunya_flow_currents_v2.gif"
OUT_MP4 = OUT_DIR / "placa_catalunya_flow_currents_v2.mp4"
OUT_REPORT = OUT_DIR / "placa_catalunya_flow_currents_v2_report.md"


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

WIDTH_PX = 1600               # MP4 (backup) render width
GIF_WIDTH = 1000              # GIF downscaled for size
GIF_COLORS = 128              # GIF palette size
FPS = 16
SECONDS = 7.0
HOLD_S = 0.7                  # extra hold on the final bright frame

# brighter palette (low -> high density)
PAL = ["#00E5FF", "#2F6BFF", "#8B5CFF", "#FF3DF2"]
PAL_CMAP = LinearSegmentedColormap.from_list("currents_v2", PAL)
DENS_GAMMA = 1.4

# stage boundaries (fraction of the 7 s sequence)
PLAN_FADE   = 0.14
DRAW_START  = 0.14
DRAW_END    = 0.52
DUST_START  = 0.50
GLOW_FULL_T = 0.92

# trace look (brighter / a touch wider than V1)
TRACE_W = 1.0
TRACE_ALPHA = 0.08
TRACE_CORE_GAIN = 2.0
DRAW_DUR = 0.12               # per-trajectory draw-in duration (timeline frac)

# particles
N_A = 3600
N_B = 6000
A_BLUR = 1.2; A_GAIN = 2.6
B_BLUR = 5.0; B_GAIN = 1.3
TRAIL_DECAY = 0.82            # afterglow trail
PARTICLE_CORE_GAIN = 2.2
TARGET_SPEED_MS = 3.6
K_SAMPLES = 180
MIN_LEN_M = 6.0

# layered bloom (BIG)
BLOOM_SIGMAS = [4, 12, 28, 45]
BLOOM_WEIGHTS = [0.7, 0.55, 0.42, 0.3]
GLOW_DRAW = (0.2, 0.5)        # glow strength ramp during trace draw
GLOW_DUST = (0.6, 1.5)        # glow strength ramp during dust phase
ROLLOFF = 0.12

# frozen background = V3 more_plan
PLAN_MAJOR_OP, PLAN_SECONDARY_OP = 0.64, 0.30


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_geometry(extent):
    xmin, xmax, ymin, ymax = extent
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
    D, xs, ys, vmax = v3.density_field(polylines, extent)
    return polylines, (D, xs, ys, vmax), n_total, n_tracks


def resample_const(pts, k):
    x, y = pts[:, 0], pts[:, 1]
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    if d[-1] < 1e-6:
        return None, 0.0
    t = np.linspace(0, d[-1], k)
    return np.column_stack([np.interp(t, d, x), np.interp(t, d, y)]), float(d[-1])


def build_segments(polylines, dens, rng):
    """Global segment arrays for the trace draw-in (with per-segment appear time)."""
    D, xs, ys, vmax = dens
    seg_xy, seg_rgb, seg_appear = [], [], []
    for pl in polylines:
        m = len(pl)
        if m < 2:
            continue
        start = DRAW_START + rng.random() * (DRAW_END - DRAW_DUR - DRAW_START)
        frac = np.linspace(0, 1, m - 1)             # along-path fraction per segment
        mid = 0.5 * (pl[:-1] + pl[1:])
        dv = v3.sample_density(D, xs, ys, mid, vmax) ** DENS_GAMMA
        col = PAL_CMAP(dv)[:, :3]
        seg_xy.append(np.stack([pl[:-1], pl[1:]], axis=1))
        seg_rgb.append(col)
        seg_appear.append(start + frac * DRAW_DUR)
    return (np.concatenate(seg_xy, 0), np.concatenate(seg_rgb, 0),
            np.concatenate(seg_appear, 0))


def build_flow(polylines, dens):
    D, xs, ys, vmax = dens
    samples, lengths, colors = [], [], []
    for pl in polylines:
        s, L = resample_const(pl, K_SAMPLES)
        if s is None or L < MIN_LEN_M:
            continue
        dv = v3.sample_density(D, xs, ys, s, vmax) ** DENS_GAMMA
        samples.append(s); lengths.append(L); colors.append(PAL_CMAP(dv)[:, :3])
    return np.array(samples), np.array(lengths), np.array(colors)


# ---------------------------------------------------------------------------
# Background + trace buffers
# ---------------------------------------------------------------------------

def plan_layer(warped_rgb, W, H):
    major, secondary, n_lines = v3.architectural_layers(warped_rgb)
    major = cv2.resize(major, (W, H), interpolation=cv2.INTER_AREA)
    secondary = cv2.resize(secondary, (W, H), interpolation=cv2.INTER_AREA)
    plan = (PLAN_MAJOR_OP * np.dstack([major] * 3) * v3.MAJOR_TINT +
            PLAN_SECONDARY_OP * np.dstack([secondary] * 3) * v3.SECONDARY_TINT)
    return np.clip(plan, 0, 1).astype(np.float32), n_lines


def render_segment_buffer(seg_xy, seg_rgb, mask, extent, W, H):
    """Render the visible trace segments to a float RGB buffer on black."""
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("black")
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin); ax.axis("off")
    if mask.any():
        rgba = np.concatenate([seg_rgb[mask], np.full((mask.sum(), 1), TRACE_ALPHA)], 1)
        ax.add_collection(LineCollection(seg_xy[mask], colors=rgba, linewidths=TRACE_W,
                          capstyle="round", joinstyle="round", antialiased=True))
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32)[..., :3] / 255.0
    plt.close(fig)
    return buf


# ---------------------------------------------------------------------------
# Glow + particles
# ---------------------------------------------------------------------------

def bloom(src, glow_strength):
    out = np.zeros_like(src)
    for s, w in zip(BLOOM_SIGMAS, BLOOM_WEIGHTS):
        for c in range(3):
            out[..., c] += w * cv2.GaussianBlur(src[..., c], (0, 0), s)
    return glow_strength * out


def splat(W, H, px, py, col, weight):
    acc = np.zeros((H, W, 3), dtype=np.float32)
    flat = py * W + px
    for c in range(3):
        np.add.at(acc[..., c].ravel(), flat, col[:, c] * weight)
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
        N_A, N_B = 1500, 2500

    warped_rgb, extent = warp_plan_to_world(DEF_PLAN, DEF_CALIB)
    xmin, xmax, ymin, ymax = extent
    W = WIDTH_PX - (WIDTH_PX % 2)
    H = int(round(W * (ymax - ymin) / (xmax - xmin))); H -= H % 2

    polylines, dens, n_total, n_tracks = load_geometry(extent)
    rng = np.random.default_rng(7)
    seg_xy, seg_rgb, seg_appear = build_segments(polylines, dens, rng)
    samples, lengths, colors = build_flow(polylines, dens)
    n_flow = len(samples)

    plan, n_lines = plan_layer(warped_rgb, W, H)

    # particle assignment (prob ∝ length); phases random (no seamless constraint)
    pprob = lengths / lengths.sum()
    tidA = rng.choice(n_flow, size=N_A, p=pprob); ph0A = rng.random(N_A)
    tidB = rng.choice(n_flow, size=N_B, p=pprob); ph0B = rng.random(N_B)
    spA = np.maximum(1, np.round(TARGET_SPEED_MS / FPS * 1000 / lengths)).astype(int)
    spB = spA  # cycle scaling reused; phase advance below uses world speed directly

    F = int(round(SECONDS * FPS)); F_hold = int(round(HOLD_S * FPS))
    print(f"[INFO] flow {n_flow} | tracks {n_tracks} | rows {n_total} | {W}x{H} | F {F}+{F_hold}")

    full_mask = np.ones(len(seg_appear), bool)
    trace_full = None
    trail = np.zeros((H, W, 3), dtype=np.float32)

    gif_w = GIF_WIDTH - (GIF_WIDTH % 2)
    gif_h = int(round(gif_w * H / W)); gif_h -= gif_h % 2
    gif_frames = []
    mp4 = imageio.get_writer(OUT_MP4, fps=FPS, codec="libx264", quality=8,
                             macro_block_size=None, output_params=["-pix_fmt", "yuv420p"])

    def advance_phase(ph0, tid, f):
        # constant world speed: distance = TARGET_SPEED_MS * (f/FPS); fraction = dist/L
        dist = TARGET_SPEED_MS * (f / FPS)
        return (ph0 + dist / lengths[tid]) % 1.0

    total = F + F_hold
    for fi in range(total):
        f = min(fi, F - 1)
        t = f / (F - 1)

        # --- plan (fades in) ---
        frame = plan * np.clip(t / PLAN_FADE, 0, 1)

        # --- traces (draw in, then full) ---
        movement = np.zeros((H, W, 3), dtype=np.float32)
        if t >= DRAW_START:
            if t >= DRAW_END:
                if trace_full is None:
                    trace_full = render_segment_buffer(seg_xy, seg_rgb, full_mask,
                                                       extent, W, H)
                traces = trace_full
            else:
                mask = seg_appear <= t
                traces = render_segment_buffer(seg_xy, seg_rgb, mask, extent, W, H)
            movement += TRACE_CORE_GAIN * traces

        # --- dust particles flowing along traces (after DUST_START) ---
        if t >= DUST_START:
            pg = np.clip((t - DUST_START) / (0.85 - DUST_START), 0, 1)
            inst = np.zeros((H, W, 3), dtype=np.float32)
            for tid, ph0, blur, gain in ((tidA, ph0A, A_BLUR, A_GAIN),
                                         (tidB, ph0B, B_BLUR, B_GAIN)):
                phase = advance_phase(ph0, tid, f)
                idx = np.clip((phase * K_SAMPLES).astype(np.int32), 0, K_SAMPLES - 1)
                pos = samples[tid, idx]; col = colors[tid, idx]
                px = np.clip(((pos[:, 0] - xmin) / (xmax - xmin) * (W - 1)).astype(np.int32), 0, W - 1)
                py = np.clip(((pos[:, 1] - ymin) / (ymax - ymin) * (H - 1)).astype(np.int32), 0, H - 1)
                acc = splat(W, H, px, py, col, gain * pg)
                for c in range(3):
                    acc[..., c] = cv2.GaussianBlur(acc[..., c], (0, 0), blur)
                inst += acc
            trail = trail * TRAIL_DECAY + inst
            movement += PARTICLE_CORE_GAIN * trail

        # --- glow strength schedule ---
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
    write_report(n_total, n_tracks, n_flow, n_lines, F, F_hold, (W, H), gif_mb)


def write_report(n_total, n_tracks, n_flow, n_lines, F, F_hold, size, gif_mb):
    W, H = size
    txt = f"""# Plaça Catalunya — Flow Currents V2 (staged reveal)

**Visualization-only animation built from existing tracked trajectories. No
retracking, recalibration, model inference, or source-data modification.**

V2 reveals the movement layer over time (V1 had everything present from frame 0):
**plan → traces drawn → dust flows → glow blooms.**

## Format / duration
- GIF (primary): `{OUT_GIF.name}` — **{gif_mb:.1f} MB**, loops forever.
- MP4 (backup): `{OUT_MP4.name}` — H.264, same frames.
- Duration: **{SECONDS:.0f} s** + {HOLD_S:.1f} s final hold; **{FPS} fps**;
  frame count **{F + F_hold}** ({F} sequence + {F_hold} hold); size **{W}×{H}**.

## Staged structure (fractions of the {SECONDS:.0f}s sequence)
- **0–{PLAN_FADE:.0%}** plan only (black + white/cool-grey linework, subtle fade-in).
- **{DRAW_START:.0%}–{DRAW_END:.0%}** trace strings **draw in** along real trajectory
  geometry: each trajectory has a staggered start and reveals segment-by-segment
  over {DRAW_DUR:.0%} of the timeline (strings pulled through the city).
- **{DUST_START:.0%}–100%** dust + glowing particles **flow along** the strings;
  glow builds to full hero intensity by {GLOW_FULL_T:.0%}, then holds.

## Data
`{DEF_TRAJ}` — tracks {n_tracks}, rows {n_total}, flow trajectories {n_flow}
(≥ {MIN_LEN_M:.0f} m). Background frozen = **V3 more_plan** ({n_lines} major Hough
lines + secondary detail; major op {PLAN_MAJOR_OP}, secondary op {PLAN_SECONDARY_OP}).

## Trace draw-in
Per-segment appearance time = staggered start + along-path fraction × {DRAW_DUR}.
Width {TRACE_W}, alpha {TRACE_ALPHA}, core gain {TRACE_CORE_GAIN}, additive, round caps,
density-coloured (γ {DENS_GAMMA}). Real geometry preserved.

## Dust flow
Begins at {DUST_START:.0%}. Type A (sharp, {N_A}, blur {A_BLUR}, gain {A_GAIN}) +
Type B (soft, {N_B}, blur {B_BLUR}, gain {B_GAIN}); both travel strictly along the
trajectory lookup (no random motion) at ~{TARGET_SPEED_MS} m/s, with an afterglow
**trail** (decay {TRAIL_DECAY}). Particle core gain {PARTICLE_CORE_GAIN}; intensity
ramps in. Currents concentrate where trajectories overlap (length-weighted).

## Glow (BIG, layered)
Bloom sigmas **{BLOOM_SIGMAS}** px, weights {BLOOM_WEIGHTS}; strength ramps
{GLOW_DRAW} (draw) → {GLOW_DUST} (dust) then holds. Soft rolloff `/(1+{ROLLOFF}·out)`
keeps the black background black while letting cores bloom.

## Palette (brighter)
`{" → ".join(PAL)}` (cyan → electric blue → violet → magenta), movement only.

## Output paths
- `{OUT_GIF.name}` (primary), `{OUT_MP4.name}` (backup), this report.
"""
    OUT_REPORT.write_text(txt, encoding="utf-8")
    print(f"[INFO] Report saved: {OUT_REPORT}")


if __name__ == "__main__":
    main()
