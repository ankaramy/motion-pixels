"""
generate_prediction_hero.py
---------------------------
Hero visual: a curated selection of REAL MODEL_X long-horizon predictions for a
Barcelona square, projected onto the calibrated plan.

Each pedestrian is drawn as:
    * thin solid white  = observed history (real past)
    * dotted pink       = the model's autoregressive future rollout

Nothing is faked: the dotted tails are produced by running a trained horizon
model forward (MODEL_X_HORIZON_SWEEP/H{MODEL_H}/model_best.pth), not by cutting
ground truth in half.

Long tails: selection (track qualification + ADE quality filter) uses a short
horizon SELECT_HORIZON, but each chosen seed is then rolled out DRAW_STEPS far
(can exceed the model's training horizon -- a stress-test style extrapolation,
same spirit as Plots_Wassim/Long_Horizon_Stress_Test) so predictions span the
scene. We curate to long, outward, non-collapsed, reasonably-directed windows.

Each scene emits three PNGs in out/:
    *_plan.png    dim satellite plan backdrop
    *_black.png   pure black backdrop
    *_traces.png  transparent background, trajectories only

HONESTY: qualitative visual over all qualifying tracks (not MODEL_X's held-out
split) -> not a generalization metric; curated "selected examples", not typical
accuracy (MODEL_X undershoots; DRAW_STEPS extrapolates beyond training horizon).

Usage:
    python generate_prediction_hero.py                 # both scenes
    python generate_prediction_hero.py espanya         # one scene
    python generate_prediction_hero.py catalunya
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── repo / library wiring ──────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODELX = REPO / "mp-core" / "trajectory-prediction" / "experiments" / "MODEL_X"
SWEEP = REPO / "mp-core" / "trajectory-prediction" / "experiments" / "MODEL_X_HORIZON_SWEEP"
sys.path.insert(0, str(MODELX))
import model_x_lib as L  # noqa: E402

# ── inputs ─────────────────────────────────────────────────────────────────────
MASTER_CSV = Path(r"C:\Users\OWNER\Desktop\new_datasets\Barcelona_v3_manual_master_dataset\master_dataset.csv")
ND = Path(r"C:\Users\OWNER\Desktop\new_datasets")

SCENES = {
    "espanya": {
        "recording": "placa_espanya_01", "title": "PLAÇA ESPANYA",
        "calib": ND / "placa_espanya_01" / "calibration" / "calib.json",
        "plan":  ND / "placa_espanya_01" / "plan" / "placa-espanya.png",
    },
    "catalunya": {
        "recording": "placa_catalunya_01", "title": "PLAÇA CATALUNYA",
        "calib": ND / "placa_catalunya_01" / "calibration" / "calib.json",
        "plan":  ND / "placa_catalunya_01" / "plan" / "placa-catalunya.png",
    },
    "esplanade": {
        "recording": "esplanade_espanya_01", "title": "ESPLANADE ESPANYA",
        "calib": ND / "esplanade_espanya_01" / "calibration" / "calib.json",
        "plan":  ND / "esplanade_espanya_01" / "plan" / "esplanade-espanya.png",
    },
}

# ── model / rollout ────────────────────────────────────────────────────────────
MODEL_H = 400          # which sweep checkpoint to load (best long-rollout behaviour)
SELECT_HORIZON = 200   # GT horizon for track qualification + ADE quality filter
DRAW_STEPS = 600       # rollout length actually drawn (long tails; extrapolates)
CAND_STRIDE = 25       # candidate windows sampled every N frames per track
CHUNK = 4096

# ── curation knobs ─────────────────────────────────────────────────────────────
ADE_MAX = 8.0          # keep windows whose rollout stays < this from GT (m, over SELECT_HORIZON)
MIN_PRED_LEN = 15.0    # drop collapsed stubs: keep tails travelling >= this (m)
MIN_STRAIGHTNESS = 0.4 # net_disp / path_len; drops spirals/curl-backs
MAX_TRACKS = 130       # cap pedestrians for legibility

# ── style (light) ──────────────────────────────────────────────────────────────
DIM = 0.34
CROP_TO_DATA = True
PAD_PX = 90
HIST_FRAMES = 40
HIST_COLOR = "#ffffff"
PRED_COLOR = "#ff2bd6"
HIST_LW = 0.5
HIST_ALPHA = 0.5
PRED_LW = 0.7
PRED_ALPHA = 0.85
PRED_GLOW_LW = 2.2
PRED_GLOW_ALPHA = 0.035
DASH = (0, (1.4, 4.2))
MARKER_MS = 1.0


def load_homography(path: Path) -> np.ndarray:
    c = json.loads(path.read_text(encoding="utf-8"))
    wp = np.array(c["world_points"], dtype=np.float32)
    pp = np.array(c["plan_points_px"], dtype=np.float32)
    H, _ = cv2.findHomography(wp, pp, cv2.RANSAC, 5.0)
    if H is None:
        raise SystemExit("[FATAL] findHomography returned None")
    return H.astype(np.float64)


def w2p(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        xy.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)


def build_scene(scene: dict, model, f_sc, t_sc, device):
    rec = scene["recording"]
    W = L.WINDOW_SIZE
    need = W + SELECT_HORIZON
    cols = ["recording_id", "trajectory_id", "timestep", "world_x", "world_y"] + L.FEAT_COLS + L.TARGET_COLS
    df = pd.read_csv(MASTER_CSV, usecols=lambda c: c in set(cols))
    df = df[df.recording_id == rec].copy()
    if df.empty:
        raise SystemExit(f"[FATAL] no rows for {rec}")
    wb = L.recording_world_bounds(df)
    lens = df.groupby("trajectory_id").size()
    qual_ids = [t for t in lens.index if lens[t] >= need]
    grp = {tid: g.sort_values("timestep").reset_index(drop=True)
           for tid, g in df.groupby("trajectory_id", sort=False)}
    print(f"[INFO] {rec}: {df.trajectory_id.nunique()} tracks, {len(qual_ids)} qualify "
          f"(>= {need} frames)")

    seeds, starts, gts, wb_arr, meta = L.build_eval_windows(
        df, qual_ids, wb, n_steps=SELECT_HORIZON, stride=CAND_STRIDE)
    M = len(seeds)
    print(f"[INFO] candidate windows: {M}; rolling out {DRAW_STEPS} steps...")

    preds = np.empty((M, DRAW_STEPS + 1, 2))
    for i in range(0, M, CHUNK):
        sl = slice(i, i + CHUNK)
        preds[sl] = L.rollout_world_batch(
            model, seeds[sl], starts[sl],
            {k: v[sl] for k, v in wb_arr.items()}, f_sc, t_sc, DRAW_STEPS, device)

    # quality (ADE over GT horizon) + shape stats (over full drawn tail)
    G = SELECT_HORIZON + 1
    ade = np.linalg.norm(preds[:, :G] - gts, axis=2)[:, 1:].mean(axis=1)
    seg = np.linalg.norm(np.diff(preds, axis=1), axis=2)
    plen = seg.sum(axis=1)
    netdisp = np.linalg.norm(preds[:, -1] - preds[:, 0], axis=1)
    straight = netdisp / np.maximum(plen, 1e-9)

    cand = pd.DataFrame({"tid": [m["trajectory_id"] for m in meta],
                         "start_idx": [m["start_idx"] for m in meta],
                         "widx": np.arange(M), "ade": ade,
                         "pred_len": plen, "straight": straight})
    cand = cand[cand.straight >= MIN_STRAIGHTNESS]

    chosen = []
    for tid, g in cand.groupby("tid"):
        ok = g[g.ade <= ADE_MAX]
        pick = (ok if len(ok) else g).sort_values("pred_len", ascending=False).iloc[0]
        chosen.append(pick)
    if not chosen:
        raise SystemExit("[FATAL] no windows survived curation")
    sel = pd.DataFrame(chosen)
    sel = sel[sel.pred_len >= MIN_PRED_LEN].sort_values("pred_len", ascending=False)
    if len(sel) > MAX_TRACKS:
        sel = sel.head(MAX_TRACKS)
    print(f"[INFO] selected {len(sel)} pedestrians "
          f"(median tail {sel.pred_len.median():.0f} m, max {sel.pred_len.max():.0f} m, "
          f"median ADE {sel.ade.median():.1f} m)")

    Hm = load_homography(scene["calib"])
    traces, allpx = [], []
    for _, r in sel.iterrows():
        g = grp[r.tid]; si = int(r.start_idx); anchor = si + W - 1
        h0 = max(0, anchor - HIST_FRAMES + 1)
        obs_px = w2p(Hm, g.loc[h0:anchor, ["world_x", "world_y"]].to_numpy(float))
        pred_px = w2p(Hm, preds[int(r.widx)])
        traces.append((obs_px, pred_px))
        allpx.append(obs_px); allpx.append(pred_px)
    return traces, np.vstack(allpx), len(sel)


def draw(traces, allpx, plan_path, title, subtitle, out, backdrop="black"):
    plan = cv2.cvtColor(cv2.imread(str(plan_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    ph, pw = plan.shape[:2]
    if CROP_TO_DATA:
        x0 = max(0, allpx[:, 0].min() - PAD_PX); x1 = min(pw, allpx[:, 0].max() + PAD_PX)
        y0 = max(0, allpx[:, 1].min() - PAD_PX); y1 = min(ph, allpx[:, 1].max() + PAD_PX)
    else:
        x0, x1, y0, y1 = 0, pw, 0, ph
    box_w, box_h = x1 - x0, y1 - y0

    transparent = (backdrop == "traces")
    fig_w = 16.0
    fig = plt.figure(figsize=(fig_w, fig_w * box_h / box_w),
                     facecolor="none" if transparent else "black")
    ax = fig.add_axes([0, 0, 1, 1])
    if not transparent:
        ax.set_facecolor("black")
        if backdrop == "plan":
            gray = cv2.cvtColor(plan, cv2.COLOR_RGB2GRAY).astype(float) / 255.0
            bg = np.clip(np.stack([gray] * 3, axis=-1) * DIM, 0, 1)
        else:
            bg = np.zeros((ph, pw, 3), float)
        ax.imshow(bg, interpolation="bilinear", zorder=0)

    for obs_px, pred_px in traces:
        ax.plot(pred_px[:, 0], pred_px[:, 1], "-", color=PRED_COLOR,
                lw=PRED_GLOW_LW, alpha=PRED_GLOW_ALPHA, solid_capstyle="round", zorder=2)
        ax.plot(pred_px[:, 0], pred_px[:, 1], color=PRED_COLOR, lw=PRED_LW,
                alpha=PRED_ALPHA, linestyle=DASH, zorder=4)
        ax.plot(obs_px[:, 0], obs_px[:, 1], "-", color=HIST_COLOR, lw=HIST_LW,
                alpha=HIST_ALPHA, solid_capstyle="round", zorder=5)
        ax.plot(obs_px[-1, 0], obs_px[-1, 1], "o", color=HIST_COLOR, ms=MARKER_MS,
                alpha=0.7, zorder=6)

    if not transparent:
        ax.text(0.987, 0.955, title, transform=ax.transAxes, ha="right", va="top",
                color="white", fontsize=20, fontweight="light", alpha=0.92)
        ax.text(0.987, 0.045, subtitle, transform=ax.transAxes, ha="right", va="bottom",
                color="#bbbbbb", fontsize=8.5, alpha=0.7)

    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
    fig.savefig(out, dpi=200, transparent=transparent,
                facecolor="none" if transparent else "black",
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[OK]   {out.name}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    keys = [which] if which in SCENES else list(SCENES)
    OUT_DIR = HERE / "out"; OUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L.set_seed()

    model_dir = SWEEP / f"H{MODEL_H}"
    sc = json.loads((model_dir / "scalers.json").read_text(encoding="utf-8"))
    f_sc = L.ColumnScaler.from_dict(sc["feature_scaler"])
    t_sc = L.ColumnScaler.from_dict(sc["target_scaler"])
    model = L.TrajectoryLSTM(L.N_FEAT).to(device)
    model.load_state_dict(torch.load(model_dir / "model_best.pth", map_location=device))
    model.eval()
    print(f"[INFO] model H{MODEL_H}  device={device}  draw={DRAW_STEPS} steps")

    for k in keys:
        scene = SCENES[k]
        print(f"\n=== {scene['title']} ===")
        traces, allpx, n = build_scene(scene, model, f_sc, t_sc, device)
        sub = f"MODEL_X · H{MODEL_H} · {DRAW_STEPS}-step prediction · {n} pedestrians · observed (white) → predicted (pink)"
        stem = f"prediction_hero_{scene['recording']}_H{MODEL_H}_d{DRAW_STEPS}"
        draw(traces, allpx, scene["plan"], scene["title"], sub, OUT_DIR / f"{stem}_plan.png", "plan")
        draw(traces, allpx, scene["plan"], scene["title"], sub, OUT_DIR / f"{stem}_black.png", "black")
        draw(traces, allpx, scene["plan"], scene["title"], sub, OUT_DIR / f"{stem}_traces.png", "traces")


if __name__ == "__main__":
    main()
