"""
generate_flow_currents_v4_info.py
---------------------------------
MOTION PIXELS — Flow Currents V4 (Plaça Catalunya) + INFORMATION BAR.

Same visual style as V3 (frozen): reduced glow, readable filaments, directional
inflow, 16:9 movement panel. V4 only:
  - slows the sequence to ~10 s,
  - adds a thin, elegant information bar below the movement panel with:
      Place · Map name · colour-scale gradient (cyan→blue→violet→magenta) ·
      animated elapsed-time counter.

The V3 movement render is reused unchanged (imported from
`generate_flow_currents_v3`); the bar is composited beneath it.

VISUALIZATION ONLY. Built from existing tracked trajectories. No retracking,
recalibration, model inference, or source-data modification.

Usage:
    python generate_flow_currents_v4_info.py
    python generate_flow_currents_v4_info.py --preview
"""

from pathlib import Path
import argparse
import sys

import numpy as np
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
OUT_GIF = OUT_DIR / "placa_catalunya_flow_currents_v4_info.gif"
OUT_MP4 = OUT_DIR / "placa_catalunya_flow_currents_v4_info.mp4"
OUT_STILL = OUT_DIR / "placa_catalunya_flow_currents_v4_info_still.png"
OUT_REPORT = OUT_DIR / "placa_catalunya_flow_currents_v4_info_report.md"


# ---------------------------------------------------------------------------
# V4 tunables (timing + bar only; movement style frozen from V3)
# ---------------------------------------------------------------------------

SECONDS = 10.0
FPS = 20                  # 20 keeps the 10 s GIF a reasonable size
BAR_H = 130               # ~10.7% of the 1210 px total height
GIF_WIDTH = 820
GIF_COLORS = 84

PLACE_TEXT = "Plaça Catalunya"
MAP_TEXT = "Flow Fields"
LOW_LABEL = "Low flow"
HIGH_LABEL = "High flow"

# pick a clean sans
FONT = "DejaVu Sans"
for _c in ["Inter", "Helvetica", "Arial", "IBM Plex Sans", "DejaVu Sans"]:
    try:
        fm.findfont(_c, fallback_to_default=False); FONT = _c; break
    except Exception:
        continue


def mmss(s):
    s = max(0, int(round(s)))
    return f"{s // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# Information bar (rendered per frame; timer updates)
# ---------------------------------------------------------------------------

def render_bar(t, total, W, H):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("black")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # thin white frame outline (quiet)
    ax.add_patch(Rectangle((0.004, 0.10), 0.992, 0.80, fill=False,
                 edgecolor="white", linewidth=0.8, alpha=0.40))

    def field(x, label, value, ha="left"):
        ax.text(x, 0.66, label, fontsize=7.5, color="#8a8a92", ha=ha, va="center",
                family=FONT)
        ax.text(x, 0.36, value, fontsize=13.5, color="#f2f2f5", ha=ha, va="center",
                family=FONT)

    field(0.022, "PLACE", PLACE_TEXT)
    field(0.250, "MAP", MAP_TEXT)

    # colour-scale gradient (same movement palette)
    gx0, gx1, gy0, gy1 = 0.515, 0.730, 0.34, 0.52
    grad = v3.PAL_CMAP(np.linspace(0, 1, 256)[None, :])
    ax.imshow(grad, extent=[gx0, gx1, gy0, gy1], aspect="auto", zorder=2,
              interpolation="bilinear")
    ax.add_patch(Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, fill=False,
                 edgecolor="white", linewidth=0.5, alpha=0.35, zorder=3))
    ax.text(0.515, 0.66, "INTENSITY", fontsize=7.5, color="#8a8a92", ha="left",
            va="center", family=FONT)
    ax.text(gx0 - 0.008, 0.43, LOW_LABEL, fontsize=8.5, color="#cfd0d6", ha="right",
            va="center", family=FONT)
    ax.text(gx1 + 0.008, 0.43, HIGH_LABEL, fontsize=8.5, color="#cfd0d6", ha="left",
            va="center", family=FONT)

    field(0.978, "TIME ELAPSED", f"{mmss(t)} / {mmss(total)}", ha="right")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


# ---------------------------------------------------------------------------
# Movement frame (V3 logic, reused)
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    NA, NB = (1800, 2800) if args.preview else (v3.N_A, v3.N_B)

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
    print(f"[INFO] flow {n_flow} | tracks {n_tracks} | {W}x{H_total} (bar {BAR_H}) | "
          f"F {F} @ {FPS}fps | font {FONT}")

    gif_w = GIF_WIDTH - (GIF_WIDTH % 2)
    gif_h = int(round(gif_w * H_total / W)); gif_h -= gif_h % 2
    gif_frames = []
    mp4 = imageio.get_writer(OUT_MP4, fps=FPS, codec="libx264", quality=8,
                             macro_block_size=None, output_params=["-pix_fmt", "yuv420p"])

    full_mask = np.ones(len(seg_appear), bool)
    trace_full = None
    trail = np.zeros((Hm, W, 3), dtype=np.float32)
    still_saved = False

    for fi in range(F):
        t = fi / (F - 1)
        frame = plan * np.clip(t / v3.PLAN_FADE, 0, 1)

        movement = np.zeros((Hm, W, 3), dtype=np.float32)
        if t >= v3.DRAW_START:
            if t >= v3.DRAW_END:
                if trace_full is None:
                    trace_full = v3.render_segment_buffer(seg_xy, seg_rgb, full_mask, win, W, Hm)
                traces = trace_full
            else:
                traces = v3.render_segment_buffer(seg_xy, seg_rgb, seg_appear <= t, win, W, Hm)
            movement += v3.TRACE_CORE_GAIN * traces

        if t >= v3.DUST_START:
            pg = np.clip((t - v3.DUST_START) / (0.85 - v3.DUST_START), 0, 1)
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

        if t < v3.DRAW_START:
            gs = 0.0
        elif t < v3.DUST_START:
            gs = v3.lerp(*v3.GLOW_DRAW, (t - v3.DRAW_START) / (v3.DUST_START - v3.DRAW_START))
        else:
            gs = v3.lerp(*v3.GLOW_DUST, (t - v3.DUST_START) / (v3.GLOW_FULL_T - v3.DUST_START))

        out = frame + movement + v3.bloom(movement, gs)
        out = out / (1.0 + v3.ROLLOFF * out)
        movement_img = (np.clip(out, 0, 1) * 255).astype(np.uint8)

        bar = render_bar(fi / FPS, SECONDS, W, BAR_H)
        full = np.vstack([movement_img, bar])

        mp4.append_data(full)
        pim = Image.fromarray(full).resize((gif_w, gif_h), Image.LANCZOS).convert("RGB")
        gif_frames.append(pim.quantize(colors=GIF_COLORS, method=Image.MEDIANCUT,
                                       dither=Image.NONE))
        if not still_saved and t >= 0.96:
            Image.fromarray(full).save(OUT_STILL); still_saved = True
        if fi % 20 == 0:
            print(f"   frame {fi}/{F}")

    if not still_saved:
        Image.fromarray(full).save(OUT_STILL)
    mp4.close()
    gif_frames[0].save(OUT_GIF, save_all=True, append_images=gif_frames[1:],
                       duration=int(round(1000 / FPS)), loop=0, optimize=True, disposal=2)
    gif_mb = OUT_GIF.stat().st_size / 1e6
    mp4_mb = OUT_MP4.stat().st_size / 1e6
    print(f"[INFO] Saved GIF {gif_mb:.1f} MB | MP4 {mp4_mb:.1f} MB | still")
    write_report(n_total, n_tracks, n_flow, n_lines, F, (W, H_total), gif_mb, mp4_mb)


def write_report(n_total, n_tracks, n_flow, n_lines, F, size, gif_mb, mp4_mb):
    W, H = size
    txt = f"""# Plaça Catalunya — Flow Currents V4 (with information bar)

**Visualization-only animation from existing tracked trajectories. No retracking,
recalibration, model inference, or source-data modification.** Movement style is
frozen from V3 (reduced glow, readable filaments, directional inflow, brighter
palette); V4 slows the sequence and adds an elegant information bar.

## Format
- GIF (primary): `{OUT_GIF.name}` — **{gif_mb:.1f} MB**, {GIF_WIDTH}px wide, {FPS} fps, loops.
- MP4 (backup): `{OUT_MP4.name}` — **{mp4_mb:.1f} MB**, H.264, {W}×{H}.
- Still: `{OUT_STILL.name}` ({W}×{H}).
- Duration **{SECONDS:.0f} s**, **{FPS} fps**, **{F} frames**, resolution **{W}×{H}**.
  (20 fps chosen so the 10 s GIF stays a reasonable size; MP4 is the smooth backup.)

## Timing
- 0.0–1.5 s: plan only (fade-in).
- 1.5–5.0 s: trace strings form along real trajectory geometry.
- 5.0–10.0 s: dust flushes through/between the strings, glow builds; final ~1 s
  holds the full flow field for reading.

## Information bar
- Height **{BAR_H}px** (~{BAR_H / H * 100:.1f}% of total) — sits BELOW the movement
  panel, never over the map. Black background, thin white outline, white text.
- Font: **{FONT}** (clean sans; bold avoided).
- Fields: **PLACE** "{PLACE_TEXT}" · **MAP** "{MAP_TEXT}" · **INTENSITY** colour
  gradient with "{LOW_LABEL}" / "{HIGH_LABEL}" · **TIME ELAPSED** animated
  `mm:ss / mm:ss` counter (updates every frame).
- Colour scale uses the movement palette `{" → ".join(v3.PAL)}` (cyan → blue →
  violet → magenta). No large legend / scientific colorbar.
- Hierarchy preserved: glowing movement > plan linework > information bar.

## Movement (frozen from V3)
{n_flow} flow paths (oriented toward the denser corridor, extrapolated
{v3.EXTRA_M:.0f} m off-screen for inflow); particles A {v3.N_A} + B {v3.N_B};
bloom sigmas {v3.BLOOM_SIGMAS}; plan = V3 more_plan ({n_lines} major lines +
secondary detail). Tracks {n_tracks}, rows {n_total}.

## Output paths
`{OUT_GIF.name}` (primary), `{OUT_MP4.name}` (backup), `{OUT_STILL.name}`, this report.
"""
    OUT_REPORT.write_text(txt, encoding="utf-8")
    print(f"[INFO] Report saved: {OUT_REPORT}")


if __name__ == "__main__":
    main()
