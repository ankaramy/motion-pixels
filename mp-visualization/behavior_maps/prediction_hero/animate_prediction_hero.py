"""
animate_prediction_hero.py
--------------------------
Animate the prediction-hero visual: first the OBSERVED ground-truth paths draw
in (white), then the MODEL_X PREDICTED tails grow out (dotted pink). ~7 s.

Reuses generate_prediction_hero.build_scene (same curated real rollouts), and
the unified behaviour-map HUD footer geometry (1920x70, DARK theme) for the
legend variant.

Emits, per scene, into out/anim/:
    *_legend.mp4 / *_legend.gif   (with HUD footer + Observed/Predicted legend)
    *_plain.mp4  / *_plain.gif    (map only, no footer)

Usage:
    python animate_prediction_hero.py                 # espanya + esplanade
    python animate_prediction_hero.py espanya
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

# ── canvas / timing standard (matches final behaviour maps) ───────────────────
W = 1920                      # frame width
BAR_H = 70                    # HUD footer height (px) — behaviour-map standard
GIF_WIDTH = 760
DUR = 7.0
FPS = 30                      # MP4 fps
GIF_FPS = 15                  # GIF fps (stride 2 of source)
DIM = gph.DIM
PAD_PX = gph.PAD_PX

# phase schedule (seconds)
A0, A1 = 0.40, 2.90           # ground-truth (history) grows in
B0, B1 = 3.20, 6.40           # prediction grows out
# hold A1..B0 and B1..DUR

# HUD theme (DARK) — identical to generate_final_behavior_maps
TH = dict(bg="black", sep="white", sep_lw=0.6, sep_a=0.26,
          lbl="#7e7e88", val="#ededf2", end="#9a9aa2", box_a=0.20)
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


def render_footer(place, map_text, right_text):
    """1920x70 DARK HUD footer with an Observed/Predicted line legend."""
    fig = plt.figure(figsize=(W / 100, BAR_H / 100), dpi=100)
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

    # legend in the standard 0.520+ region: two line swatches
    ax.text(0.520, 0.71, "LAYERS", fontsize=6.5, color=TH["lbl"], ha="left", va="center", family=FONT)
    ax.plot([0.523, 0.543], [0.45, 0.45], color="#ffffff", lw=1.4, alpha=0.9,
            solid_capstyle="round")
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


def build_frames(scene, model, f_sc, t_sc, device):
    traces, allpx, n = gph.build_scene(scene, model, f_sc, t_sc, device)

    plan = cv2.cvtColor(cv2.imread(str(scene["plan"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    ph, pw = plan.shape[:2]
    x0 = max(0, allpx[:, 0].min() - PAD_PX); x1 = min(pw, allpx[:, 0].max() + PAD_PX)
    y0 = max(0, allpx[:, 1].min() - PAD_PX); y1 = min(ph, allpx[:, 1].max() + PAD_PX)
    box_w, box_h = x1 - x0, y1 - y0

    fw = W
    fh = even(fw * box_h / box_w)
    fig = plt.figure(figsize=(fw / 100, fh / 100), dpi=100, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("black")
    gray = cv2.cvtColor(plan, cv2.COLOR_RGB2GRAY).astype(float) / 255.0
    ax.imshow(np.clip(np.stack([gray] * 3, -1) * DIM, 0, 1), interpolation="bilinear", zorder=0)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")

    # persistent artists per trace
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

    title = ax.text(0.992, 0.93, scene["title"], transform=ax.transAxes, ha="right",
                    va="top", color="white", fontsize=22, fontweight="light", alpha=0.92)
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
                mk.set_data([obs_px[min(ko, len(obs_px)) - 1, 0]],
                            [obs_px[min(ko, len(obs_px)) - 1, 1]])
            else:
                obs.set_data([], []); mk.set_data([], [])
            kp = max(0, int(np.ceil(pf * len(pred_px))))
            if kp >= 2:
                glow.set_data(pred_px[:kp, 0], pred_px[:kp, 1])
                pred.set_data(pred_px[:kp, 0], pred_px[:kp, 1])
            else:
                glow.set_data([], []); pred.set_data([], [])
        if t < B0:
            phase.set_text("GROUND TRUTH"); phase.set_alpha(0.55 * (hf if t < A1 else 1.0))
        else:
            phase.set_text("PREDICTION"); phase.set_alpha(0.55)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return frames, fh, n


def encode(frames, footer, scene, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = scene["recording"]
    fh = frames[0].shape[0]
    legend = [np.vstack([fr, footer]) for fr in frames]

    def to_gif(seq, path):
        gw = even(GIF_WIDTH)
        gh = even(gw * seq[0].shape[0] / seq[0].shape[1])
        small = [np.asarray(Image.fromarray(f).resize((gw, gh), Image.LANCZOS))
                 for f in seq[::2]]
        imageio.mimsave(path, small, fps=GIF_FPS, loop=0)

    def to_mp4(seq, path):
        wn = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8,
                                macro_block_size=16, pixelformat="yuv420p")
        for f in seq:
            wn.append_data(f)
        wn.close()

    stem = f"prediction_anim_{rec}"
    to_mp4(frames, out_dir / f"{stem}_plain.mp4")
    to_gif(frames, out_dir / f"{stem}_plain.gif")
    to_mp4(legend, out_dir / f"{stem}_legend.mp4")
    to_gif(legend, out_dir / f"{stem}_legend.gif")
    print(f"[OK]   {stem}_(plain|legend).(mp4|gif)  -> {out_dir}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    keys = [which] if which in gph.SCENES else ["espanya", "esplanade"]
    out_dir = HERE / "out" / "anim"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L.set_seed()

    model_dir = gph.SWEEP / f"H{gph.MODEL_H}"
    sc = json.loads((model_dir / "scalers.json").read_text(encoding="utf-8"))
    f_sc = L.ColumnScaler.from_dict(sc["feature_scaler"])
    t_sc = L.ColumnScaler.from_dict(sc["target_scaler"])
    model = L.TrajectoryLSTM(L.N_FEAT).to(device)
    model.load_state_dict(torch.load(model_dir / "model_best.pth", map_location=device))
    model.eval()
    print(f"[INFO] model H{gph.MODEL_H} device={device} {DUR}s @ {FPS}fps")

    for k in keys:
        scene = gph.SCENES[k]
        print(f"\n=== {scene['title']} ===")
        frames, fh, n = build_frames(scene, model, f_sc, t_sc, device)
        footer = render_footer(scene["title"], "Trajectory Prediction",
                               f"MODEL_X · H{gph.MODEL_H}")
        if footer.shape[1] != frames[0].shape[1]:
            footer = np.asarray(Image.fromarray(footer).resize(
                (frames[0].shape[1], BAR_H), Image.LANCZOS))
        encode(frames, footer, scene, out_dir)


if __name__ == "__main__":
    main()
