"""
animate_prediction_hero_v2.py
-----------------------------
anim_2 revision of the prediction-hero animation. Changes vs v1:
  1. shorter observed ground-truth (HIST_FRAMES 14), longer predicted tails
  2. much longer prediction-draw phase (total ~9 s)
  3. no on-map title
  4. predicted tails elongated via DRAW_STEPS=1300 -- this EXTRAPOLATES far
     beyond the model's training horizon (stress-test "fake length"); honest
     quantitative claims should use the in-horizon stills, not this.
  5. also emits, per scene:
       *_final.jpg          final composition still (dim plan backdrop, JPEG)
       *_traces.png         trajectories only, transparent background

Outputs -> out/anim_2/.  Reuses generate_prediction_hero.build_scene + the
behaviour-map HUD footer (1920x70, DARK) for the legend variants.

Usage:
    python animate_prediction_hero_v2.py            # espanya + esplanade
    python animate_prediction_hero_v2.py espanya
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
from PIL import Image
import imageio.v2 as imageio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_prediction_hero as gph  # noqa: E402
L = gph.L

# ── content overrides (applied to gph globals before build_scene) ─────────────
gph.HIST_FRAMES = 14        # shorter observed ground truth
gph.DRAW_STEPS = 1300       # much longer predicted tails (extrapolated / "faked")
gph.MIN_PRED_LEN = 22.0     # keep only long tails
gph.MIN_STRAIGHTNESS = 0.38
gph.MAX_TRACKS = 115

# ── canvas / timing ───────────────────────────────────────────────────────────
W = 1920
BAR_H = 70
GIF_WIDTH = 760
FPS = 30
GIF_FPS = 15
DUR = 9.0
DIM = gph.DIM
PAD_PX = gph.PAD_PX

# phase schedule (s): short GT in, long prediction grow
A0, A1 = 0.30, 1.50         # ground-truth grows in (1.2 s)
B0, B1 = 1.80, 8.30         # prediction grows out (6.5 s)

TH = dict(bg="black", sep="white", sep_lw=0.6, sep_a=0.26,
          lbl="#7e7e88", val="#ededf2", end="#9a9aa2")
FONT = "DejaVu Sans"
for _c in ["Roboto", "Inter", "IBM Plex Sans", "Helvetica Neue", "Segoe UI", "DejaVu Sans"]:
    try:
        fm.findfont(_c, fallback_to_default=False); FONT = _c; break
    except Exception:
        continue


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def even(n):
    n = int(round(n)); return n - (n % 2)


def crop_box(allpx, pw, ph):
    x0 = max(0, allpx[:, 0].min() - PAD_PX); x1 = min(pw, allpx[:, 0].max() + PAD_PX)
    y0 = max(0, allpx[:, 1].min() - PAD_PX); y1 = min(ph, allpx[:, 1].max() + PAD_PX)
    return x0, x1, y0, y1


def render_footer(place, map_text, right_text, width):
    fig = plt.figure(figsize=(width / 100, BAR_H / 100), dpi=100)
    fig.patch.set_facecolor(TH["bg"])
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(TH["bg"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(Rectangle((0.004, 0.14), 0.992, 0.72, fill=False,
                 edgecolor=TH["sep"], linewidth=TH["sep_lw"], alpha=TH["sep_a"]))

    def field(x, label, value, ha="left"):
        ax.text(x, 0.71, label, fontsize=6.5, color=TH["lbl"], ha=ha, va="center", family=FONT)
        ax.text(x, 0.33, value, fontsize=10.5, color=TH["val"], ha=ha, va="center", family=FONT)

    field(0.020, "PLACE", place)
    field(0.250, "MAP", map_text)
    ax.text(0.520, 0.71, "LAYERS", fontsize=6.5, color=TH["lbl"], ha="left", va="center", family=FONT)
    ax.plot([0.523, 0.543], [0.45, 0.45], color="#ffffff", lw=1.4, alpha=0.9, solid_capstyle="round")
    ax.text(0.549, 0.45, "Observed (ground truth)", fontsize=7.0, color=TH["end"],
            ha="left", va="center", family=FONT)
    ax.plot([0.690, 0.710], [0.45, 0.45], color=gph.PRED_COLOR, lw=1.4, alpha=0.95,
            linestyle=(0, (1.2, 1.4)))
    ax.text(0.716, 0.45, "Predicted (MODEL_X)", fontsize=7.0, color=TH["end"],
            ha="left", va="center", family=FONT)
    field(0.980, "MODEL", right_text, ha="right")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def build_frames(scene, traces, allpx):
    plan = cv2.cvtColor(cv2.imread(str(scene["plan"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    ph, pw = plan.shape[:2]
    x0, x1, y0, y1 = crop_box(allpx, pw, ph)
    fw = W; fh = even(fw * (y1 - y0) / (x1 - x0))
    fig = plt.figure(figsize=(fw / 100, fh / 100), dpi=100, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("black")
    gray = cv2.cvtColor(plan, cv2.COLOR_RGB2GRAY).astype(float) / 255.0
    ax.imshow(np.clip(np.stack([gray] * 3, -1) * DIM, 0, 1), interpolation="bilinear", zorder=0)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")

    arts = []
    for obs_px, pred_px in traces:
        glow, = ax.plot([], [], "-", color=gph.PRED_COLOR, lw=gph.PRED_GLOW_LW,
                        alpha=gph.PRED_GLOW_ALPHA, solid_capstyle="round", zorder=2)
        pred, = ax.plot([], [], color=gph.PRED_COLOR, lw=gph.PRED_LW,
                        alpha=gph.PRED_ALPHA, linestyle=gph.DASH, zorder=4)
        obs, = ax.plot([], [], "-", color=gph.HIST_COLOR, lw=gph.HIST_LW,
                       alpha=gph.HIST_ALPHA, solid_capstyle="round", zorder=5)
        mk, = ax.plot([], [], "o", color=gph.HIST_COLOR, ms=gph.MARKER_MS, alpha=0.7, zorder=6)
        arts.append((obs_px, pred_px, glow, pred, obs, mk))

    phase = ax.text(0.008, 0.93, "", transform=ax.transAxes, ha="left", va="top",
                    color="#bbbbbb", fontsize=10, alpha=0.0, family=FONT)

    nframes = int(round(DUR * FPS))
    frames = []
    for i in range(nframes):
        t = i / FPS
        hf = smoothstep((t - A0) / (A1 - A0)) if t > A0 else 0.0
        pf = smoothstep((t - B0) / (B1 - B0)) if t > B0 else 0.0
        for obs_px, pred_px, glow, pred, obs, mk in arts:
            ko = max(0, int(np.ceil(hf * len(obs_px))))
            if ko >= 2:
                obs.set_data(obs_px[:ko, 0], obs_px[:ko, 1])
                j = min(ko, len(obs_px)) - 1
                mk.set_data([obs_px[j, 0]], [obs_px[j, 1]])
            else:
                obs.set_data([], []); mk.set_data([], [])
            kp = max(0, int(np.ceil(pf * len(pred_px))))
            if kp >= 2:
                glow.set_data(pred_px[:kp, 0], pred_px[:kp, 1])
                pred.set_data(pred_px[:kp, 0], pred_px[:kp, 1])
            else:
                glow.set_data([], []); pred.set_data([], [])
        if t < B0:
            phase.set_text("GROUND TRUTH"); phase.set_alpha(0.5 * (hf if t < A1 else 1.0))
        else:
            phase.set_text("PREDICTION"); phase.set_alpha(0.5)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return frames


def render_still(scene, traces, allpx, mode, path):
    """mode 'jpg' = dim plan backdrop composition; 'png' = transparent traces only."""
    plan = cv2.cvtColor(cv2.imread(str(scene["plan"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    ph, pw = plan.shape[:2]
    x0, x1, y0, y1 = crop_box(allpx, pw, ph)
    fig_w = 16.0
    transparent = (mode == "png")
    fig = plt.figure(figsize=(fig_w, fig_w * (y1 - y0) / (x1 - x0)), dpi=200,
                     facecolor="none" if transparent else "black")
    ax = fig.add_axes([0, 0, 1, 1])
    if not transparent:
        ax.set_facecolor("black")
        gray = cv2.cvtColor(plan, cv2.COLOR_RGB2GRAY).astype(float) / 255.0
        ax.imshow(np.clip(np.stack([gray] * 3, -1) * DIM, 0, 1), interpolation="bilinear", zorder=0)
    for obs_px, pred_px in traces:
        ax.plot(pred_px[:, 0], pred_px[:, 1], "-", color=gph.PRED_COLOR, lw=gph.PRED_GLOW_LW,
                alpha=gph.PRED_GLOW_ALPHA, solid_capstyle="round", zorder=2)
        ax.plot(pred_px[:, 0], pred_px[:, 1], color=gph.PRED_COLOR, lw=gph.PRED_LW,
                alpha=gph.PRED_ALPHA, linestyle=gph.DASH, zorder=4)
        ax.plot(obs_px[:, 0], obs_px[:, 1], "-", color=gph.HIST_COLOR, lw=gph.HIST_LW,
                alpha=gph.HIST_ALPHA, solid_capstyle="round", zorder=5)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
    fig.savefig(path, dpi=200, transparent=transparent,
                facecolor="none" if transparent else "black",
                bbox_inches="tight", pad_inches=0,
                **({"pil_kwargs": {"quality": 92}} if mode == "jpg" else {}))
    plt.close(fig)


def encode(frames, footer, stem, out_dir):
    def to_gif(seq, path):
        gw = even(GIF_WIDTH); gh = even(gw * seq[0].shape[0] / seq[0].shape[1])
        small = [np.asarray(Image.fromarray(f).resize((gw, gh), Image.LANCZOS)) for f in seq[::2]]
        imageio.mimsave(path, small, fps=GIF_FPS, loop=0)

    def to_mp4(seq, path):
        wn = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8,
                                macro_block_size=16, pixelformat="yuv420p")
        for f in seq:
            wn.append_data(f)
        wn.close()

    legend = [np.vstack([fr, footer]) for fr in frames]
    to_mp4(frames, out_dir / f"{stem}_plain.mp4")
    to_gif(frames, out_dir / f"{stem}_plain.gif")
    to_mp4(legend, out_dir / f"{stem}_legend.mp4")
    to_gif(legend, out_dir / f"{stem}_legend.gif")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    keys = [which] if which in gph.SCENES else ["espanya", "esplanade"]
    out_dir = HERE / "out" / "anim_2"; out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L.set_seed()

    model_dir = gph.SWEEP / f"H{gph.MODEL_H}"
    sc = json.loads((model_dir / "scalers.json").read_text(encoding="utf-8"))
    f_sc = L.ColumnScaler.from_dict(sc["feature_scaler"])
    t_sc = L.ColumnScaler.from_dict(sc["target_scaler"])
    model = L.TrajectoryLSTM(L.N_FEAT).to(device)
    model.load_state_dict(torch.load(model_dir / "model_best.pth", map_location=device))
    model.eval()
    print(f"[INFO] model H{gph.MODEL_H} device={device} {DUR}s @ {FPS}fps "
          f"HIST={gph.HIST_FRAMES} DRAW={gph.DRAW_STEPS}")

    for k in keys:
        scene = gph.SCENES[k]
        print(f"\n=== {scene['title']} ===")
        traces, allpx, n = gph.build_scene(scene, model, f_sc, t_sc, device)
        stem = f"prediction_anim_{scene['recording']}"
        # stills
        render_still(scene, traces, allpx, "jpg", out_dir / f"{stem}_final.jpg")
        render_still(scene, traces, allpx, "png", out_dir / f"{stem}_traces.png")
        # animation
        frames = build_frames(scene, traces, allpx)
        footer = render_footer(scene["title"], "Trajectory Prediction",
                               f"MODEL_X · H{gph.MODEL_H}", frames[0].shape[1])
        encode(frames, footer, stem, out_dir)
        print(f"[OK]   {stem}: 4 videos + final.jpg + traces.png  ({n} pedestrians)")


if __name__ == "__main__":
    main()
