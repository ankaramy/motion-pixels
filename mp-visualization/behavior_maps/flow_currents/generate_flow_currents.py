"""
generate_flow_currents.py
-------------------------
MOTION PIXELS — Flow Currents animation over the static Trace + Dust Hero V3
(Plaça Catalunya).

The hero is treated as a STATIC field. Nothing about the plan, the linework, or
the trace strings is animated or revealed over time — they are fully present in
every frame. On top of that static field, glowing particles flow ALONG the real
pedestrian trajectory geometry (the trajectories are used as the flow field), so
the viewer perceives continuous urban currents, not animated trajectories.

Layers:
  1. Static architectural background (from the V3 layered drawing).
  2. Static trace strings (dim, visible throughout).
  3. Animated flow particles advecting along real trajectories:
       Type A — small / sharp / bright (active movement)
       Type B — larger / softer / low-opacity (atmospheric dust)
  4. Glow that brightens where particle density accumulates (corridors pulse
     naturally; no flashing).

Seamless loop: each trajectory does an INTEGER number of particle cycles over the
loop, so frame F == frame 0. Particle world-speed is roughly constant (longer
trajectories get fewer cycles). No random walk, no Brownian motion, no explosions.

VISUALIZATION ONLY. Built from existing tracked trajectories. No retracking,
recalibration, model inference, or source-data modification.

Usage:
    python generate_flow_currents.py            # full 24s loop
    python generate_flow_currents.py --seconds 6 --preview   # quick check
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
import imageio.v2 as imageio

BEHAVIOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BEHAVIOR_DIR))
from generate_bottleneck_map_test import warp_plan_to_world

# reuse the frozen V3 architectural background + helpers
HERO_DIR = BEHAVIOR_DIR / "trace_dust_hero"
sys.path.insert(0, str(HERO_DIR))
import generate_trace_dust_hero_v3 as v3


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SITE = v3.SITE
DEF_TRAJ, DEF_PLAN, DEF_CALIB = v3.DEF_TRAJ, v3.DEF_PLAN, v3.DEF_CALIB

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_MP4 = OUT_DIR / "placa_catalunya_flow_currents.mp4"
OUT_STILL = OUT_DIR / "placa_catalunya_flow_currents_still.png"
OUT_REPORT = OUT_DIR / "placa_catalunya_flow_currents_report.md"


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

WIDTH_PX = 1920          # animation resolution (HD; hero stills stay 4400)
FPS = 25
SECONDS = 24

K_SAMPLES = 180          # arc-length samples per trajectory (flow lookup table)
MIN_LEN_M = 6.0          # ignore very short tracks as flow paths
TARGET_SPEED_MS = 3.6    # ~constant world speed of currents

# particle counts (weighted onto trajectories by length)
N_A = 3200               # Type A — sharp, bright
N_B = 5200               # Type B — soft, atmospheric

# Type A look
A_BLUR = 1.1
A_GAIN = 2.1
# Type B look
B_BLUR = 4.5
B_GAIN = 1.05
# wide atmospheric bloom over both
WIDE_BLUR = 18.0
WIDE_GAIN = 0.65

DENS_GAMMA = 1.4         # cyan/blue spread (matches V3)
PAL_CMAP = v3.PAL_CMAP

# static base look (dim, but the string field stays readable as the flow path)
BASE_TRACE_W = 0.7
BASE_TRACE_ALPHA = 0.042
BASE_TRACE_GAIN = 1.55
BASE_GLOW = ([4, 11], [0.5, 0.34], 0.72)
PLAN_MAJOR_OP, PLAN_SECONDARY_OP = 0.42, 0.16


# ---------------------------------------------------------------------------
# Trajectories -> flow lookup tables
# ---------------------------------------------------------------------------

def resample_const(pts, k):
    x, y = pts[:, 0], pts[:, 1]
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    if d[-1] < 1e-6:
        return None, 0.0
    t = np.linspace(0, d[-1], k)
    return np.column_stack([np.interp(t, d, x), np.interp(t, d, y)]), float(d[-1])


def build_flow(traj_path, extent):
    xmin, xmax, ymin, ymax = extent
    df = pd.read_csv(traj_path, usecols=["frame", "track_id", "world_x", "world_y"])
    n_total = len(df); n_tracks = int(df.track_id.nunique())
    df = df[(df.world_x >= xmin) & (df.world_x <= xmax) &
            (df.world_y >= ymin) & (df.world_y <= ymax)].sort_values(["track_id", "frame"])

    polylines = []
    for _, g in df.groupby("track_id"):
        x = g.world_x.to_numpy(); y = g.world_y.to_numpy()
        if len(x) < 3:
            continue
        pl = v3.resample_smooth(x, y)
        if pl is not None:
            polylines.append(pl)

    # density field (for colour) — reuse V3
    D, xs, ys, vmax = v3.density_field(polylines, extent)

    samples, lengths, colors = [], [], []
    for pl in polylines:
        s, L = resample_const(pl, K_SAMPLES)
        if s is None or L < MIN_LEN_M:
            continue
        dv = v3.sample_density(D, xs, ys, s, vmax) ** DENS_GAMMA
        col = PAL_CMAP(dv)[:, :3]
        samples.append(s); lengths.append(L); colors.append(col)

    samples = np.array(samples)            # (T, K, 2)
    lengths = np.array(lengths)            # (T,)
    colors = np.array(colors)              # (T, K, 3)
    return samples, lengths, colors, (D, xs, ys, vmax), polylines, n_total, n_tracks


def assign_particles(lengths, n, F, rng):
    """Assign n particles to trajectories (prob ∝ length), with seamless integer
    cycle counts and roughly constant world speed."""
    p = lengths / lengths.sum()
    tid = rng.choice(len(lengths), size=n, p=p)
    phase0 = rng.random(n)
    v_per_frame = TARGET_SPEED_MS / FPS
    cyc_per_traj = np.maximum(1, np.round(v_per_frame * F / lengths)).astype(int)
    cyc = cyc_per_traj[tid]
    return tid, phase0, cyc


# ---------------------------------------------------------------------------
# Static base layer
# ---------------------------------------------------------------------------

def render_base(extent, polylines, dens, warped_rgb, W, H):
    D, xs, ys, vmax = dens
    xmin, xmax, ymin, ymax = extent
    # static trace strings (dim) via matplotlib at animation resolution
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("black")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin); ax.axis("off")
    segs, vals = [], []
    for p in polylines:
        if len(p) < 2:
            continue
        segs.append(np.stack([p[:-1], p[1:]], axis=1))
        vals.append(v3.sample_density(D, xs, ys, 0.5 * (p[:-1] + p[1:]), vmax))
    segs = np.concatenate(segs, 0); vals = np.concatenate(vals, 0) ** DENS_GAMMA
    rgba = PAL_CMAP(vals); rgba[:, 3] = BASE_TRACE_ALPHA
    ax.add_collection(LineCollection(segs, colors=rgba, linewidths=BASE_TRACE_W,
                      capstyle="round", antialiased=True))
    fig.canvas.draw()
    traces = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32)[..., :3] / 255.0
    plt.close(fig)
    Hh, Ww = traces.shape[:2]

    # architectural layers from V3, resized to (Ww, Hh)
    major, secondary, n_lines = v3.architectural_layers(warped_rgb)
    major = cv2.resize(major, (Ww, Hh), interpolation=cv2.INTER_AREA)
    secondary = cv2.resize(secondary, (Ww, Hh), interpolation=cv2.INTER_AREA)
    plan = (PLAN_MAJOR_OP * np.dstack([major] * 3) * v3.MAJOR_TINT +
            PLAN_SECONDARY_OP * np.dstack([secondary] * 3) * v3.SECONDARY_TINT)

    radii, weights, glow = BASE_GLOW
    b = np.zeros_like(traces)
    for r, w in zip(radii, weights):
        for c in range(3):
            b[..., c] += w * gaussian_filter(traces[..., c], r)
    base = BASE_TRACE_GAIN * traces + glow * b + plan
    return np.clip(base, 0, 1), (Ww, Hh), n_lines


# ---------------------------------------------------------------------------
# Per-frame particle compositing (additive glow on black, added to static base)
# ---------------------------------------------------------------------------

def world_to_px(pos, extent, W, H):
    xmin, xmax, ymin, ymax = extent
    px = ((pos[:, 0] - xmin) / (xmax - xmin) * (W - 1)).astype(np.int32)
    py = ((pos[:, 1] - ymin) / (ymax - ymin) * (H - 1)).astype(np.int32)
    np.clip(px, 0, W - 1, out=px); np.clip(py, 0, H - 1, out=py)
    return px, py


def splat(W, H, px, py, col, weight):
    acc = np.zeros((H, W, 3), dtype=np.float32)
    flat = py * W + px
    for c in range(3):
        np.add.at(acc[..., c].ravel(), flat, col[:, c] * weight)
    return acc


def render_frame(f, F, base, particles, samples, colors, extent, W, H):
    tidA, ph0A, cycA, tidB, ph0B, cycB = particles
    out = base.copy()

    for tid, ph0, cyc, blur, gain, wgt in (
        (tidA, ph0A, cycA, A_BLUR, A_GAIN, 1.0),
        (tidB, ph0B, cycB, B_BLUR, B_GAIN, 0.6)):
        phase = (ph0 + (f / F) * cyc) % 1.0
        idx = np.clip((phase * K_SAMPLES).astype(np.int32), 0, K_SAMPLES - 1)
        pos = samples[tid, idx]                     # (n,2)
        col = colors[tid, idx]                      # (n,3)
        px, py = world_to_px(pos, extent, W, H)
        acc = splat(W, H, px, py, col, wgt)
        for c in range(3):
            acc[..., c] = gaussian_filter(acc[..., c], blur)
        out += gain * acc

    # wide atmospheric bloom from combined particle energy (density -> glow)
    energy = out - base
    energy = np.clip(energy, 0, None)
    for c in range(3):
        out[..., c] += WIDE_GAIN * gaussian_filter(energy[..., c], WIDE_BLUR)

    out = out / (1.0 + 0.14 * out)
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=SECONDS)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--preview", action="store_true", help="lower particle count / faster")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    global N_A, N_B
    if args.preview:
        N_A, N_B = 1600, 2600

    warped_rgb, extent = warp_plan_to_world(DEF_PLAN, DEF_CALIB)
    W = WIDTH_PX
    xmin, xmax, ymin, ymax = extent
    H = int(round(W * (ymax - ymin) / (xmax - xmin)))
    H -= H % 2; W -= W % 2          # even dims required by yuv420p

    samples, lengths, colors, dens, polylines, n_total, n_tracks = build_flow(DEF_TRAJ, extent)
    n_flow = len(samples)
    print(f"[INFO] flow trajectories {n_flow} | tracks {n_tracks} | rows {n_total}")

    base, (W, H), n_lines = render_base(extent, polylines, dens, warped_rgb, W, H)

    F = int(round(args.seconds * args.fps))
    rng = np.random.default_rng(7)
    tidA, ph0A, cycA = assign_particles(lengths, N_A, F, rng)
    tidB, ph0B, cycB = assign_particles(lengths, N_B, F, rng)
    particles = (tidA, ph0A, cycA, tidB, ph0B, cycB)

    # still (frame 0) for quick inspection
    f0 = render_frame(0, F, base, particles, samples, colors, extent, W, H)
    imageio.imwrite(OUT_STILL, (f0 * 255).astype(np.uint8))
    print(f"[INFO] Saved {OUT_STILL} ({W}x{H})")

    writer = imageio.get_writer(OUT_MP4, fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=None,
                                output_params=["-pix_fmt", "yuv420p"])
    for f in range(F):
        frame = render_frame(f, F, base, particles, samples, colors, extent, W, H)
        writer.append_data((frame * 255).astype(np.uint8))
        if f % 25 == 0:
            print(f"   frame {f}/{F}")
    writer.close()
    print(f"[INFO] Saved {OUT_MP4}  ({F} frames, {args.fps} fps, {args.seconds:.0f}s loop)")

    write_report(n_total, n_tracks, n_flow, n_lines, F, args, (W, H), extent)


def write_report(n_total, n_tracks, n_flow, n_lines, F, args, size, extent):
    W, H = size; xmin, xmax, ymin, ymax = extent
    txt = f"""# Plaça Catalunya — Flow Currents (animated, over static Hero V3)

**Visualization-only animation built from existing tracked trajectories. No
retracking, recalibration, model inference, or source-data modification.**

The Trace + Dust Hero V3 is treated as a **static field** — the plan, linework and
trace strings are fully present in every frame and are never revealed/drawn over
time. Glowing particles flow **along the real trajectory geometry** (the
trajectories ARE the flow field), reading as continuous urban currents.

## Source data
`{DEF_TRAJ}` — tracks {n_tracks}, rows {n_total}.
- Flow trajectories used (≥ {MIN_LEN_M:.0f} m): **{n_flow}**.
- Architectural background: V3 layered drawing ({n_lines} major Hough lines +
  speckle-cleaned secondary detail).

## Layers
1. **Static architectural background** (V3 major + secondary linework, dim).
2. **Static trace strings** (width {BASE_TRACE_W}, alpha {BASE_TRACE_ALPHA}) —
   visible throughout, dimmed so the currents are the active layer.
3. **Animated flow particles** advecting along real trajectories:
   - **Type A** — sharp/bright ({N_A} particles, blur {A_BLUR}px, gain {A_GAIN}).
   - **Type B** — soft/atmospheric ({N_B} particles, blur {B_BLUR}px, gain {B_GAIN}).
4. **Glow** — particle energy is blurred wide (blur {WIDE_BLUR}px, gain {WIDE_GAIN})
   so dense corridors brighten through accumulation (natural pulsing, no flashing).

## Flow model (no random motion)
Each trajectory is arc-length resampled to {K_SAMPLES} points; a particle's
position is a lookup at `phase·K`. Phase advances as
`(phase0 + f/F · cycles) mod 1`, so particles travel strictly along the path,
continuously entering and exiting. **Colour** = local trajectory density on the
palette (γ = {DENS_GAMMA}; cyan/blue sparse → violet/magenta dense).

## Seamless loop
`cycles` per trajectory is an **integer**, so at frame F every particle returns to
its frame-0 phase → frame F ≡ frame 0 (seamless). Cycles ≈ `speed·F/length` with
`speed = {TARGET_SPEED_MS} m/s`, giving roughly constant world-speed currents
(longer paths get fewer cycles). Min 1 cycle.

## Output
- `{OUT_MP4.name}` — **{args.seconds:.0f}s seamless loop**, {args.fps} fps,
  {F} frames, {W}×{H}, H.264 (yuv420p).
- `{OUT_STILL.name}` — frame-0 still.

## Notes
- MP4 (not GIF): a 20–30 s HD loop as GIF would be enormous; H.264 keeps it small
  and high quality. Loops seamlessly on repeat.
- Particles are bound to the tracked trajectory network (no wandering / Brownian
  motion / explosions). Currents concentrate in real corridors because more
  trajectories (and thus more particles) run there.
- World extent X = [{xmin:.2f}, {xmax:.2f}] m, Y = [{ymin:.2f}, {ymax:.2f}] m.
"""
    OUT_REPORT.write_text(txt, encoding="utf-8")
    print(f"[INFO] Report saved: {OUT_REPORT}")


if __name__ == "__main__":
    main()
