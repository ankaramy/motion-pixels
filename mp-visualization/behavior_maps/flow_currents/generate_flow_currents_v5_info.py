"""
generate_flow_currents_v5_info.py
---------------------------------
MOTION PIXELS — Flow Currents V5 (Plaça Catalunya): compact footer + REAL video
time + 17 s, quick plan / slow flow, quieter gradient.

Movement style frozen from V3 (imported). V5 only changes timing + footer:
  - footer bar height reduced ~46% (≈6% of canvas);
  - nicer font (Roboto if present);
  - timer shows REAL source-video time (frame/fps), compressed into the 17 s;
  - 17 s total — plan appears quickly (~1 s), flow generation slowed;
  - gradient legend made quieter (lower opacity, thinner, dimmer labels).

VISUALIZATION ONLY. Built from existing tracked trajectories. No retracking,
recalibration, model inference, or source-data modification.

Usage:
    python generate_flow_currents_v5_info.py
    python generate_flow_currents_v5_info.py --preview
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
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
from PIL import Image
import imageio.v2 as imageio

BEHAVIOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BEHAVIOR_DIR))
from generate_bottleneck_map_test import warp_plan_to_world
FC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FC_DIR))
import generate_flow_currents_v3 as v3   # frozen movement pipeline

OUT_DIR = FC_DIR / "outputs"
OUT_GIF = OUT_DIR / "placa_catalunya_flow_currents_v5_info.gif"
OUT_MP4 = OUT_DIR / "placa_catalunya_flow_currents_v5_info.mp4"
OUT_STILL = OUT_DIR / "placa_catalunya_flow_currents_v5_info_still.png"
OUT_REPORT = OUT_DIR / "placa_catalunya_flow_currents_v5_info_report.md"


# ---------------------------------------------------------------------------
# V5 tunables
# ---------------------------------------------------------------------------

SECONDS = 17.0
FPS = 20
SRC_FPS = 30.0                # source video fps for real-time mapping

BAR_H = 70                    # ~6% of 1150 total (was 130)
GIF_WIDTH = 760
GIF_COLORS = 80
GIF_STRIDE = 2                # GIF samples every 2nd frame (~10 fps) for size

# timing fractions of the 17 s (plan quick, flow slow)
PLAN_FADE  = 1.0 / SECONDS
DRAW_START = 1.0 / SECONDS
DRAW_END   = 9.0 / SECONDS
DUST_START = 9.0 / SECONDS
GLOW_FULL_T = 15.5 / SECONDS

PLACE_TEXT = "Plaça Catalunya"
MAP_TEXT = "Flow Fields"
LOW_LABEL = "Low flow"
HIGH_LABEL = "High flow"

# font preference
FONT = "DejaVu Sans"
for _c in ["Roboto", "Inter", "IBM Plex Sans", "Helvetica Neue", "Segoe UI", "DejaVu Sans"]:
    try:
        fm.findfont(_c, fallback_to_default=False); FONT = _c; break
    except Exception:
        continue


def mmss(s):
    s = max(0, int(round(s)))
    return f"{s // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# Compact footer (real-time timer)
# ---------------------------------------------------------------------------

def render_bar(real_elapsed, real_total, W, H):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("black")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # thin outline only (compact, no thick border)
    ax.add_patch(Rectangle((0.004, 0.14), 0.992, 0.72, fill=False,
                 edgecolor="white", linewidth=0.6, alpha=0.26))

    def field(x, label, value, ha="left"):
        ax.text(x, 0.71, label, fontsize=6.5, color="#7e7e88", ha=ha, va="center",
                family=FONT)
        ax.text(x, 0.33, value, fontsize=10.5, color="#ededf2", ha=ha, va="center",
                family=FONT)

    field(0.020, "PLACE", PLACE_TEXT)
    field(0.250, "MAP", MAP_TEXT)

    # quieter colour-scale gradient (lower opacity, thinner, dimmer labels)
    gx0, gx1, gy0, gy1 = 0.520, 0.700, 0.40, 0.50
    grad = v3.PAL_CMAP(np.linspace(0, 1, 256)[None, :]).copy()
    grad[..., 3] = 0.55                                   # reduce opacity
    ax.imshow(grad, extent=[gx0, gx1, gy0, gy1], aspect="auto", zorder=2,
              interpolation="bilinear")
    ax.add_patch(Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, fill=False,
                 edgecolor="white", linewidth=0.4, alpha=0.20, zorder=3))
    ax.text(0.520, 0.71, "INTENSITY", fontsize=6.5, color="#7e7e88", ha="left",
            va="center", family=FONT)
    ax.text(gx0 - 0.007, 0.45, LOW_LABEL, fontsize=7.0, color="#9a9aa2", ha="right",
            va="center", family=FONT)
    ax.text(gx1 + 0.007, 0.45, HIGH_LABEL, fontsize=7.0, color="#9a9aa2", ha="left",
            va="center", family=FONT)

    field(0.980, "TIME ELAPSED", f"{mmss(real_elapsed)} / {mmss(real_total)}", ha="right")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    NA, NB = (1800, 2800) if args.preview else (v3.N_A, v3.N_B)

    # real source-video time range
    fr = pd.read_csv(v3.DEF_TRAJ, usecols=["frame"])
    f0, f1 = int(fr.frame.min()), int(fr.frame.max())
    real_total = (f1 - f0) / SRC_FPS

    warped_rgb, plan_extent = warp_plan_to_world(v3.DEF_PLAN, v3.DEF_CALIB)
    W = v3.CANVAS_W - (v3.CANVAS_W % 2); Hm = v3.CANVAS_H - (v3.CANVAS_H % 2)
    win = v3.window_extent(plan_extent, W, Hm)
    xmin, xmax, ymin_w, ymax_w = win

    polylines, dens, n_total, n_tracks = v3.load_geometry(plan_extent)
    rng = np.random.default_rng(7)
    seg_xy, seg_rgb, seg_appear = v3.build_segments(polylines, dens, rng)
    samples, lengths, colors = v3.build_flow(polylines, dens)
    n_flow = len(samples)
    plan, n_lines = v3.plan_window(warped_rgb, plan_extent, win, W, Hm)

    pprob = lengths / lengths.sum()
    tidA = rng.choice(n_flow, size=NA, p=pprob); ph0A = rng.random(NA)
    tidB = rng.choice(n_flow, size=NB, p=pprob); ph0B = rng.random(NB)

    F = int(round(SECONDS * FPS))
    H_total = Hm + BAR_H
    print(f"[INFO] flow {n_flow} | {W}x{H_total} (bar {BAR_H}, {BAR_H/H_total*100:.1f}%) | "
          f"F {F} @ {FPS}fps | font {FONT} | real {mmss(real_total)} (f{f0}-{f1}@{SRC_FPS:.0f})")

    gif_w = GIF_WIDTH - (GIF_WIDTH % 2); gif_h = int(round(gif_w * H_total / W)); gif_h -= gif_h % 2
    gif_frames = []
    mp4 = imageio.get_writer(OUT_MP4, fps=FPS, codec="libx264", quality=8,
                             macro_block_size=None, output_params=["-pix_fmt", "yuv420p"])

    full_mask = np.ones(len(seg_appear), bool)
    trace_full = None
    trail = np.zeros((Hm, W, 3), dtype=np.float32)
    still_saved = False

    for fi in range(F):
        t = fi / (F - 1)
        frame = plan * np.clip(t / PLAN_FADE, 0, 1)

        movement = np.zeros((Hm, W, 3), dtype=np.float32)
        if t >= DRAW_START:
            if t >= DRAW_END:
                if trace_full is None:
                    trace_full = v3.render_segment_buffer(seg_xy, seg_rgb, full_mask, win, W, Hm)
                traces = trace_full
            else:
                # re-map segment appearance into the V5 draw window for a slow draw
                local = (t - DRAW_START) / (DRAW_END - DRAW_START)
                appear_local = (seg_appear - v3.DRAW_START) / (v3.DRAW_END - v3.DRAW_START)
                traces = v3.render_segment_buffer(seg_xy, seg_rgb, appear_local <= local, win, W, Hm)
            movement += v3.TRACE_CORE_GAIN * traces

        if t >= DUST_START:
            pg = np.clip((t - DUST_START) / (0.85 - DUST_START), 0, 1)
            inst = np.zeros((Hm, W, 3), dtype=np.float32)
            for tid, ph0, blur, gain in ((tidA, ph0A, v3.A_BLUR, v3.A_GAIN),
                                         (tidB, ph0B, v3.B_BLUR, v3.B_GAIN)):
                phase = (ph0 + v3.TARGET_SPEED_MS * (fi / FPS) / lengths[tid]) % 1.0
                env = np.clip(phase / v3.FADE_IN, 0, 1) * np.clip((1 - phase) / v3.FADE_OUT, 0, 1)
                idx = np.clip((phase * v3.K_SAMPLES).astype(np.int32), 0, v3.K_SAMPLES - 1)
                pos = samples[tid, idx]; col = colors[tid, idx] * env[:, None]
                px = np.clip(((pos[:, 0] - xmin) / (xmax - xmin) * (W - 1)).astype(np.int32), 0, W - 1)
                py = np.clip(((pos[:, 1] - ymin_w) / (ymax_w - ymin_w) * (Hm - 1)).astype(np.int32), 0, Hm - 1)
                acc = v3.splat(W, Hm, px, py, col, gain * pg)
                for c in range(3):
                    acc[..., c] = cv2.GaussianBlur(acc[..., c], (0, 0), blur)
                inst += acc
            trail = trail * v3.TRAIL_DECAY + inst
            movement += v3.PARTICLE_CORE_GAIN * trail

        if t < DRAW_START:
            gs = 0.0
        elif t < DUST_START:
            gs = v3.lerp(*v3.GLOW_DRAW, (t - DRAW_START) / (DUST_START - DRAW_START))
        else:
            gs = v3.lerp(*v3.GLOW_DUST, (t - DUST_START) / (GLOW_FULL_T - DUST_START))

        out = frame + movement + v3.bloom(movement, gs)
        out = out / (1.0 + v3.ROLLOFF * out)
        movement_img = (np.clip(out, 0, 1) * 255).astype(np.uint8)

        bar = render_bar(t * real_total, real_total, W, BAR_H)
        full = np.vstack([movement_img, bar])

        mp4.append_data(full)
        if fi % GIF_STRIDE == 0:
            pim = Image.fromarray(full).resize((gif_w, gif_h), Image.LANCZOS).convert("RGB")
            gif_frames.append(pim.quantize(colors=GIF_COLORS, method=Image.MEDIANCUT,
                                           dither=Image.NONE))
        if not still_saved and t >= 0.97:
            Image.fromarray(full).save(OUT_STILL); still_saved = True
        if fi % 40 == 0:
            print(f"   frame {fi}/{F}")

    if not still_saved:
        Image.fromarray(full).save(OUT_STILL)
    mp4.close()
    gif_fps = FPS / GIF_STRIDE
    gif_frames[0].save(OUT_GIF, save_all=True, append_images=gif_frames[1:],
                       duration=int(round(1000 / gif_fps)), loop=0, optimize=True, disposal=2)
    gif_mb = OUT_GIF.stat().st_size / 1e6; mp4_mb = OUT_MP4.stat().st_size / 1e6
    print(f"[INFO] Saved GIF {gif_mb:.1f} MB ({len(gif_frames)} fr @ {gif_fps:.0f}fps) | MP4 {mp4_mb:.1f} MB")
    write_report(n_total, n_tracks, n_flow, n_lines, F, (W, H_total), gif_mb, mp4_mb,
                 f0, f1, real_total, gif_fps, len(gif_frames))


def write_report(n_total, n_tracks, n_flow, n_lines, F, size, gif_mb, mp4_mb,
                 f0, f1, real_total, gif_fps, n_gif):
    W, H = size
    txt = f"""# Plaça Catalunya — Flow Currents V5 (compact footer + real video time)

**Visualization-only animation from existing tracked trajectories. No retracking,
recalibration, model inference, or source-data modification.** Movement style
frozen from V3.

## Format
- GIF (primary): `{OUT_GIF.name}` — **{gif_mb:.1f} MB**, {GIF_WIDTH}px, {gif_fps:.0f} fps
  ({n_gif} frames), loops.
- MP4 (backup): `{OUT_MP4.name}` — **{mp4_mb:.1f} MB**, H.264, {W}×{H}, {FPS} fps.
- Still: `{OUT_STILL.name}` ({W}×{H}).
- Animation duration **{SECONDS:.0f} s**, **{F} frames @ {FPS} fps**.

## Real video time (timer)
- Source FPS: **{SRC_FPS:.0f}** · first frame **{f0}** · last frame **{f1}**.
- Real duration represented: **{mmss(real_total)}** ({real_total:.0f} s).
- The footer timer shows **real elapsed / real total** (`MM:SS / MM:SS`),
  i.e. the {real_total:.0f} s of source footage **compressed into the {SECONDS:.0f} s
  animation** (timer = animation-progress × real total, updating every frame).

## Timing (plan quick, flow slow)
- 0.0–1.0 s: plan / linework appears (quick, calm).
- 1.0–9.0 s: trace strings gradually form along real trajectories (slow, readable).
- 9.0–17.0 s: dust flushes through the strings, glow builds; final ~1.5 s holds
  near-complete while particles keep flowing.

## Footer (compact)
- Height **{BAR_H}px** (~{BAR_H/H*100:.1f}% of total — down ~46% from V4's 130px).
  Below the map; thin outline only; minimal vertical padding.
- Font: **{FONT}** (regular weight; small uppercase labels, larger values; no heavy bold).
- Fields: PLACE "{PLACE_TEXT}" · MAP "{MAP_TEXT}" · INTENSITY gradient
  ("{LOW_LABEL}"/"{HIGH_LABEL}") · TIME ELAPSED real `MM:SS / MM:SS`.

## Gradient legend (quieter)
Movement palette `{" → ".join(v3.PAL)}` at **opacity 0.55**, thin strip (~7px),
dimmed labels — readable but secondary; the flow field stays more vivid than the legend.

## Movement (frozen from V3)
{n_flow} flow paths (inflow-oriented, {v3.EXTRA_M:.0f} m off-screen lead-in);
particles A {v3.N_A} + B {v3.N_B}; bloom {v3.BLOOM_SIGMAS}; plan = V3 more_plan
({n_lines} major lines + secondary). Tracks {n_tracks}, rows {n_total}.

## Output paths
`{OUT_GIF.name}` (primary), `{OUT_MP4.name}` (backup), `{OUT_STILL.name}`, this report.
"""
    OUT_REPORT.write_text(txt, encoding="utf-8")
    print(f"[INFO] Report saved: {OUT_REPORT}")


if __name__ == "__main__":
    main()
