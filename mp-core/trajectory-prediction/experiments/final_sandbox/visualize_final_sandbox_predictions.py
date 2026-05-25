"""
visualize_final_sandbox_predictions.py
--------------------------------------
Run autoregressive rollout from the trained final-sandbox LSTM and draw
per-trajectory and contact-sheet plots in world coordinates. Predictions
are drawn so they start at the end of the seed window (no detached dots).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import (
    DATASET_CSV, MODEL_PTH, SCALER_PKL, ENCODED_CSV,
    PRED_PLOTS_DIR, PRED_SHEET,
    FEATURE_COLS, TARGET_COLS, WINDOW_SIZE, N_ROLLOUT,
    HIDDEN_SIZE, NUM_LAYERS, IDW_K,
)
from train_lstm_final_sandbox import TrajectoryLSTM


class SpatialInterpolator:
    """KDTree over the v2.1C encoded dataset. IDW-interpolates
    dist_to_obstacle and dist_to_boundary at any predicted (x, y).
    Mirrors the production sandbox (train_phase2b_final.py)."""

    def __init__(self, df: pd.DataFrame, k: int = IDW_K):
        coords = df[["world_x", "world_y"]].to_numpy(dtype=np.float64)
        self.tree     = cKDTree(coords)
        self.obs_vals = df["dist_to_obstacle"].to_numpy(dtype=np.float64)
        self.bnd_vals = df["dist_to_boundary"].to_numpy(dtype=np.float64)
        self.k        = k

    def query(self, x: float, y: float):
        dists, idxs = self.tree.query([x, y], k=self.k)
        if dists[0] < 1e-9:
            return (float(self.obs_vals[idxs[0]]),
                    float(self.bnd_vals[idxs[0]]))
        w = 1.0 / dists
        w = w / w.sum()
        return (float(np.dot(w, self.obs_vals[idxs])),
                float(np.dot(w, self.bnd_vals[idxs])))


def load_artifacts(device):
    with open(SCALER_PKL, "rb") as fh:
        bundle = pickle.load(fh)
    fs = bundle["feature_scaler"]; ts = bundle["target_scaler"]
    model = TrajectoryLSTM(len(FEATURE_COLS), HIDDEN_SIZE,
                           out_size=2, n_layers=NUM_LAYERS).to(device)
    model.load_state_dict(torch.load(MODEL_PTH, map_location=device))
    model.eval()
    return model, fs, ts


def rollout(model, fs, ts, track: pd.DataFrame, interp: SpatialInterpolator,
            device, target_mag: float = None
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Autoregressive rollout in world coords with KDTree-IDW spatial
    refresh at every step (production-sandbox formulation).

    Returns predicted (x, y) anchored at the seed's last position so the
    drawn line visually continues from the seed.
    """
    feats = track[FEATURE_COLS].to_numpy(dtype=np.float32)
    pos   = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    window  = fs.transform(feats[:WINDOW_SIZE]).copy()
    prev_x  = float(pos[WINDOW_SIZE - 1, 0])
    prev_y  = float(pos[WINDOW_SIZE - 1, 1])

    px, py = [prev_x], [prev_y]   # seed-end as anchor; line visually contiguous
    for _ in range(N_ROLLOUT):
        x_in = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            p_s = model(x_in).cpu().numpy()[0]
        du, dv = ts.inverse_transform(p_s[np.newaxis])[0]
        du, dv = float(du), float(dv)
        if target_mag is not None:
            m = float(np.hypot(du, dv))
            if m > 1e-9:
                s = target_mag / m
                du, dv = s * du, s * dv
        wx, wy = prev_x + du, prev_y + dv
        px.append(wx); py.append(wy)

        # Re-query KDTree at the predicted position — same logic the
        # production sandbox used. Without this the spatial features
        # freeze at seed values and the LSTM heading collapses.
        obs, bnd = interp.query(wx, wy)
        new_raw = np.array([wx, wy, du, dv, obs, bnd], dtype=np.float32)
        scaled = fs.transform(new_raw[np.newaxis])[0]
        window = np.vstack([window[1:], scaled])
        prev_x, prev_y = wx, wy

    return np.array(px, dtype=np.float64), np.array(py, dtype=np.float64)


def pick_tracks(df: pd.DataFrame, n: int = 10) -> List[int]:
    """Pick a diverse set of unique-shape tracks (skip duplicates by dup_idx)."""
    if "dup_idx" in df.columns:
        df = df[df["dup_idx"] == 0]
    sizes = df.groupby("track_id").size().sort_values(ascending=False)
    eligible = sizes[sizes >= WINDOW_SIZE + N_ROLLOUT].index.tolist()
    if len(eligible) == 0:
        return []
    # Score by motion + curvature so we pick tracks with visible shape.
    scores = []
    for tid in eligible:
        g = df[df["track_id"] == tid].sort_values("frame")
        x = g["world_x"].to_numpy(); y = g["world_y"].to_numpy()
        if len(x) < 3:
            continue
        path_len = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
        # heading change as a curvature proxy
        h = np.arctan2(np.diff(y), np.diff(x))
        curv = float(np.std(np.diff(np.unwrap(h)))) if len(h) > 1 else 0.0
        scores.append((tid, path_len, curv))
    # Pick a mix: top by curvature, top by path length, plus some short ones.
    by_curv = sorted(scores, key=lambda s: -s[2])
    by_len  = sorted(scores, key=lambda s: -s[1])
    by_short = sorted(scores, key=lambda s:  s[1])
    chosen, seen = [], set()
    for src in (by_curv, by_len, by_short):
        for tid, *_ in src:
            if tid in seen: continue
            seen.add(tid); chosen.append(tid)
            if len(chosen) >= n:
                break
        if len(chosen) >= n:
            break
    return chosen[:n]


def draw_track_panel(ax, full_gt, seed_xy, gt_future, pred_xy, tid):
    ax.plot(full_gt[:, 0], full_gt[:, 1], "-", color="#bdc3c7",
            lw=1.0, alpha=0.85, label="full GT")
    ax.plot(seed_xy[:, 0], seed_xy[:, 1], "-o", color="#2c3e50",
            lw=1.8, ms=4, label="seed (10)")
    ax.plot(gt_future[:, 0], gt_future[:, 1], "--", color="#7f8c8d",
            lw=1.5, dashes=(4, 2), label="GT future")
    ax.plot(pred_xy[:, 0], pred_xy[:, 1], "-s", color="#e74c3c",
            lw=1.8, ms=4, alpha=0.95, label="rollout")
    # Anchor marker: the seed's last point (also rollout start).
    ax.plot(seed_xy[-1, 0], seed_xy[-1, 1], "o", color="#27ae60",
            ms=9, mfc="none", mew=1.6, label="seed end / rollout start")

    # ADE / FDE on overlapping length.
    L = min(len(pred_xy) - 1, len(gt_future))
    if L > 0:
        # pred_xy[0] is seed-end (anchor); compare pred_xy[1:1+L] vs gt_future[:L]
        diffs = pred_xy[1:1 + L] - gt_future[:L]
        ade = float(np.mean(np.hypot(diffs[:, 0], diffs[:, 1])))
        fde = float(np.hypot(*(pred_xy[L] - gt_future[L - 1])))
    else:
        ade = fde = float("nan")
    ax.set_title(f"track {tid}   ADE={ade:.2f} m   FDE={fde:.2f} m",
                 fontsize=10)
    ax.set_xlabel("world_x (m)", fontsize=8)
    ax.set_ylabel("world_y (m)", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, lw=0.35, alpha=0.6)
    ax.legend(fontsize=7, loc="best")
    return ade, fde


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if not (MODEL_PTH.exists() and SCALER_PKL.exists()):
        raise SystemExit("[FATAL] model/scaler missing — run training first")
    PRED_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device: {device}")

    model, fs, ts = load_artifacts(device)
    df = pd.read_csv(DATASET_CSV)
    print(f"[INFO]  Dataset rows: {len(df):,}")

    # Build the KDTree from the v2.1C encoded CSV (per-point spatial values).
    enc = pd.read_csv(ENCODED_CSV)
    interp = SpatialInterpolator(enc, k=IDW_K)
    print(f"[INFO]  KDTree built on {len(enc):,} encoded rows "
          f"(k={IDW_K} IDW neighbours)")

    track_ids = pick_tracks(df, n=10)
    print(f"[INFO]  Picked tracks: {track_ids}")
    if not track_ids:
        raise SystemExit("[FATAL] no eligible tracks")

    rows = []
    for tid in track_ids:
        g = (df[df["track_id"] == tid].sort_values("frame")
             .iloc[: WINDOW_SIZE + N_ROLLOUT].reset_index(drop=True))
        pos = g[["world_x", "world_y"]].to_numpy(dtype=np.float32)
        seed_xy   = pos[:WINDOW_SIZE]
        gt_future = pos[WINDOW_SIZE:]
        full_gt   = pos
        px, py = rollout(model, fs, ts, g, interp, device, target_mag=None)
        pred_xy = np.column_stack([px, py])

        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        ade, fde = draw_track_panel(ax, full_gt, seed_xy, gt_future,
                                    pred_xy, tid)
        fig.tight_layout()
        out = PRED_PLOTS_DIR / f"track_{tid}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        rows.append({"track_id": int(tid), "ade": ade, "fde": fde,
                     "path": str(out)})
        print(f"[OK]    track {tid}  ADE={ade:.2f}  FDE={fde:.2f}")

    # Contact sheet 3×N
    n = len(track_ids); cols = 3
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols,
                             figsize=(6 * cols, 5.5 * rows_n),
                             squeeze=False)
    for i, tid in enumerate(track_ids):
        ax = axes[i // cols, i % cols]
        g = (df[df["track_id"] == tid].sort_values("frame")
             .iloc[: WINDOW_SIZE + N_ROLLOUT].reset_index(drop=True))
        pos = g[["world_x", "world_y"]].to_numpy(dtype=np.float32)
        px, py = rollout(model, fs, ts, g, interp, device, target_mag=None)
        draw_track_panel(ax, pos, pos[:WINDOW_SIZE], pos[WINDOW_SIZE:],
                         np.column_stack([px, py]), tid)
    for j in range(n, rows_n * cols):
        axes[j // cols, j % cols].set_axis_off()
    fig.suptitle(
        "Final sandbox — overfit-LSTM rollouts (seed black · GT-future "
        "dashed grey · rollout red)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(PRED_SHEET, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]    Contact sheet -> {PRED_SHEET.name}")

    # Per-track CSV summary
    pd.DataFrame(rows).to_csv(PRED_PLOTS_DIR / "track_metrics.csv",
                               index=False)
    print(f"[OK]    Track metrics -> {PRED_PLOTS_DIR.name}/track_metrics.csv")


if __name__ == "__main__":
    main()
