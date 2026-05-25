"""
overlay_final_sandbox_predictions_on_plan.py
--------------------------------------------
Project sandbox rollouts back onto the calibrated top-view image using the
same cv2.findHomography + cv2.perspectiveTransform path as the production
overlay script.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import (
    DATASET_CSV, MODEL_PTH, SCALER_PKL, ENCODED_CSV,
    CALIB_JSON, TOPVIEW_PNG, WALKABLE_PNG,
    PLAN_PLOTS_DIR, PLAN_SHEET,
    FEATURE_COLS, WINDOW_SIZE, N_ROLLOUT,
    HIDDEN_SIZE, NUM_LAYERS, IDW_K,
)
from visualize_final_sandbox_predictions import (
    load_artifacts, rollout, pick_tracks, SpatialInterpolator,
)


def load_H(path: Path) -> np.ndarray:
    c = json.loads(path.read_text(encoding="utf-8"))
    wp = np.array(c["world_points"],   dtype=np.float32)
    pp = np.array(c["plan_points_px"], dtype=np.float32)
    H, _ = cv2.findHomography(wp, pp, cv2.RANSAC, 5.0)
    if H is None:
        raise SystemExit("[FATAL] findHomography returned None")
    return H.astype(np.float64)


def w2p(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        xy.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)


def load_walkable_overlay(plan_rgb: np.ndarray) -> np.ndarray:
    """Optional faint walkable mask underneath. We just load the mask file
    directly; it's in world-grid pixels not plan pixels, so for this
    scene-overlay we instead skip warping it (avoids over-complication) and
    return None to indicate 'no mask layer'. Set ENABLE_MASK_LAYER below to
    enable a quick warp using the same approach as the production overlay
    script if you want it back."""
    return None


def draw_one(ax, plan_rgb, seed_xy_px, gt_full_px, gt_future_px,
             pred_px, tid):
    ax.imshow(plan_rgb, interpolation="bilinear")
    ax.plot(gt_full_px[:, 0], gt_full_px[:, 1], "-", color="#ffffff",
            lw=2.2, alpha=0.7)
    ax.plot(gt_full_px[:, 0], gt_full_px[:, 1], "-", color="#bdc3c7",
            lw=1.0, alpha=0.95, label="full GT")
    ax.plot(seed_xy_px[:, 0], seed_xy_px[:, 1], "-o", color="#1c2833",
            lw=1.8, ms=4, label="seed")
    ax.plot(gt_future_px[:, 0], gt_future_px[:, 1], "--", color="#7f8c8d",
            lw=1.6, dashes=(4, 2), label="GT future")
    ax.plot(pred_px[:, 0], pred_px[:, 1], "-s", color="#e74c3c",
            lw=1.9, ms=4, alpha=0.95, label="rollout")
    ax.plot(seed_xy_px[-1, 0], seed_xy_px[-1, 1], "o", color="#27ae60",
            ms=10, mfc="none", mew=1.8, label="seed end / rollout start")
    ax.set_title(f"track {tid}", fontsize=10, fontweight="bold")
    ax.set_xlabel("plan_x (px)"); ax.set_ylabel("plan_y (px)")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
    # Tight zoom around the union of seed + rollout + gt_future, with padding.
    all_xy = np.vstack([seed_xy_px, gt_future_px, pred_px])
    pad = 50
    x0 = max(0, all_xy[:, 0].min() - pad)
    x1 = min(plan_rgb.shape[1], all_xy[:, 0].max() + pad)
    y0 = max(0, all_xy[:, 1].min() - pad)
    y1 = min(plan_rgb.shape[0], all_xy[:, 1].max() + pad)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)   # image y inverted


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    for p in (DATASET_CSV, MODEL_PTH, SCALER_PKL, CALIB_JSON, TOPVIEW_PNG):
        if not p.exists():
            raise SystemExit(f"[FATAL] missing: {p}")
    PLAN_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device: {device}")

    model, fs, ts = load_artifacts(device)
    df = pd.read_csv(DATASET_CSV)
    enc = pd.read_csv(ENCODED_CSV)
    interp = SpatialInterpolator(enc, k=IDW_K)
    print(f"[INFO]  KDTree built on {len(enc):,} encoded rows")

    H = load_H(CALIB_JSON)
    print(f"[INFO]  H built from calibration ({CALIB_JSON.name})")

    plan = cv2.imread(str(TOPVIEW_PNG), cv2.IMREAD_COLOR)
    if plan is None:
        raise SystemExit(f"[FATAL] cannot read {TOPVIEW_PNG}")
    plan_rgb = cv2.cvtColor(plan, cv2.COLOR_BGR2RGB)
    print(f"[INFO]  Plan: {plan_rgb.shape[1]} × {plan_rgb.shape[0]} px")

    track_ids = pick_tracks(df, n=10)
    print(f"[INFO]  Picked tracks: {track_ids}")

    panels = []
    for tid in track_ids:
        g = (df[df["track_id"] == tid].sort_values("frame")
             .iloc[: WINDOW_SIZE + N_ROLLOUT].reset_index(drop=True))
        pos = g[["world_x", "world_y"]].to_numpy(dtype=np.float32)
        px_arr, py_arr = rollout(model, fs, ts, g, interp, device,
                                  target_mag=None)
        pred_world = np.column_stack([px_arr, py_arr])

        seed_px      = w2p(H, pos[:WINDOW_SIZE])
        full_gt_px   = w2p(H, pos)
        gt_future_px = w2p(H, pos[WINDOW_SIZE:])
        pred_px      = w2p(H, pred_world)

        # Drop any points that fall outside the image — keeps each panel sane.
        fig, ax = plt.subplots(figsize=(8.5, 9.5))
        draw_one(ax, plan_rgb, seed_px, full_gt_px, gt_future_px,
                 pred_px, tid)
        fig.tight_layout()
        out = PLAN_PLOTS_DIR / f"plan_overlay_track_{tid}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        panels.append((tid, seed_px, full_gt_px, gt_future_px, pred_px))
        print(f"[OK]    plan_overlay_track_{tid}.png")

    # Contact sheet
    n = len(panels); cols = 3
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols,
                             figsize=(7.5 * cols, 8.0 * rows_n),
                             squeeze=False)
    for i, (tid, sp, fp, gf, pp) in enumerate(panels):
        ax = axes[i // cols, i % cols]
        draw_one(ax, plan_rgb, sp, fp, gf, pp, tid)
    for j in range(n, rows_n * cols):
        axes[j // cols, j % cols].set_axis_off()
    fig.suptitle(
        "Final sandbox — predictions overlaid on calibrated top_view.png",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(PLAN_SHEET, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]    Plan contact sheet -> {PLAN_SHEET.name}")


if __name__ == "__main__":
    main()
