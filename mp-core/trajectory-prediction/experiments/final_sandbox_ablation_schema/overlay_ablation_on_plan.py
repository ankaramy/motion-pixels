"""
overlay_ablation_on_plan.py
---------------------------
Project the 4-model rollouts onto the calibrated top_view.png using the
production homography (cv2.findHomography(world_points, plan_points_px)).
Produces per-model and combined plan-overlay contact sheets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _paths import (
    DATASET_CSV, ENCODED_CSV, CALIB_JSON, TOPVIEW_PNG,
    PLAN_DIR, EXP, MODEL_DEFS, WINDOW_SIZE, HORIZON, IDW_K, PRESET_TRACKS,
)
from visualize_ablation_grid import (
    SpatialInterpolator3, load_model, rollout, gt_future_xy_headings,
)


def load_H(path: Path) -> np.ndarray:
    c = json.loads(path.read_text(encoding="utf-8"))
    wp = np.array(c["world_points"],   dtype=np.float32)
    pp = np.array(c["plan_points_px"], dtype=np.float32)
    H, _ = cv2.findHomography(wp, pp, cv2.RANSAC, 5.0)
    return H.astype(np.float64)


def w2p(H, xy):
    return cv2.perspectiveTransform(
        xy.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)


def crop(ax, all_xy, plan_shape, pad=50):
    x0 = max(0, all_xy[:, 0].min() - pad)
    x1 = min(plan_shape[1], all_xy[:, 0].max() + pad)
    y0 = max(0, all_xy[:, 1].min() - pad)
    y1 = min(plan_shape[0], all_xy[:, 1].max() + pad)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)


def per_model_plan_sheet(mdef, tracks, df, interp, extents, device,
                         H, plan_rgb, out_path: Path) -> None:
    model, fs, ts, features = load_model(mdef, device)
    panels = []
    for tid in tracks:
        g = (df[df["track_id"] == tid].sort_values("frame")
             .iloc[: WINDOW_SIZE + HORIZON].reset_index(drop=True))
        if len(g) < WINDOW_SIZE + HORIZON:
            continue
        pos = g[["world_x", "world_y"]].to_numpy(dtype=np.float64)
        seed_xy = pos[:WINDOW_SIZE]
        gt_future = pos[WINDOW_SIZE:]
        px, py, _ = rollout(model, fs, ts, features, g, interp,
                            extents["x_min"],
                            extents["x_max"] - extents["x_min"],
                            extents["y_min"],
                            extents["y_max"] - extents["y_min"],
                            device, HORIZON)
        pred_xy = np.column_stack([px, py])
        panels.append({
            "tid": tid,
            "seed_px":      w2p(H, seed_xy),
            "full_px":      w2p(H, pos),
            "gt_future_px": w2p(H, gt_future),
            "pred_px":      w2p(H, pred_xy),
        })

    n = len(panels); cols = 3
    rows_n = max(1, int(np.ceil(n / cols)))
    fig, axes = plt.subplots(rows_n, cols,
                              figsize=(7.5 * cols, 8.0 * rows_n),
                              squeeze=False)
    for i, d in enumerate(panels):
        ax = axes[i // cols, i % cols]
        ax.imshow(plan_rgb, interpolation="bilinear")
        ax.plot(d["full_px"][:, 0], d["full_px"][:, 1], "-",
                color="#bdc3c7", lw=1.0, alpha=0.95, label="full GT")
        ax.plot(d["seed_px"][:, 0], d["seed_px"][:, 1], "-o",
                color="#1c2833", lw=1.6, ms=3.6, label="seed")
        ax.plot(d["gt_future_px"][:, 0], d["gt_future_px"][:, 1], "--",
                color="#7f8c8d", lw=1.4, dashes=(4, 2), label="GT future")
        ax.plot(d["pred_px"][:, 0], d["pred_px"][:, 1], "-s",
                color=mdef["color"], lw=1.7, ms=3.8, alpha=0.95,
                label=mdef["label"])
        ax.plot(d["seed_px"][-1, 0], d["seed_px"][-1, 1], "o",
                color="#27ae60", ms=9, mfc="none", mew=1.6)
        ax.set_title(f"track {d['tid']}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right", framealpha=0.85)
        crop(ax, np.vstack([d["seed_px"], d["gt_future_px"], d["pred_px"]]),
             plan_rgb.shape)
    for j in range(n, rows_n * cols):
        axes[j // cols, j % cols].set_axis_off()
    fig.suptitle(f"Plan overlay — {mdef['label']}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def combined_plan_sheet(tracks, df, interp, extents, device, H, plan_rgb,
                        out_path: Path) -> None:
    # Pre-compute rollouts for every model × track.
    rolls = {mdef["name"]: {} for mdef in MODEL_DEFS}
    for mdef in MODEL_DEFS:
        model, fs, ts, features = load_model(mdef, device)
        for tid in tracks:
            g = (df[df["track_id"] == tid].sort_values("frame")
                 .iloc[: WINDOW_SIZE + HORIZON].reset_index(drop=True))
            if len(g) < WINDOW_SIZE + HORIZON:
                continue
            px, py, _ = rollout(model, fs, ts, features, g, interp,
                                extents["x_min"],
                                extents["x_max"] - extents["x_min"],
                                extents["y_min"],
                                extents["y_max"] - extents["y_min"],
                                device, HORIZON)
            rolls[mdef["name"]][tid] = w2p(H, np.column_stack([px, py]))

    n = len(tracks); cols = 3
    rows_n = max(1, int(np.ceil(n / cols)))
    fig, axes = plt.subplots(rows_n, cols,
                              figsize=(7.5 * cols, 8.0 * rows_n),
                              squeeze=False)
    for i, tid in enumerate(tracks):
        ax = axes[i // cols, i % cols]
        ax.imshow(plan_rgb, interpolation="bilinear")
        g = (df[df["track_id"] == tid].sort_values("frame")
             .iloc[: WINDOW_SIZE + HORIZON].reset_index(drop=True))
        pos = g[["world_x", "world_y"]].to_numpy(dtype=np.float64)
        seed_px      = w2p(H, pos[:WINDOW_SIZE])
        gt_future_px = w2p(H, pos[WINDOW_SIZE:])
        full_px      = w2p(H, pos)
        ax.plot(full_px[:, 0], full_px[:, 1], "-", color="#bdc3c7",
                lw=1.0, alpha=0.95, label="full GT")
        ax.plot(seed_px[:, 0], seed_px[:, 1], "-o", color="#1c2833",
                lw=1.6, ms=3.6, label="seed")
        ax.plot(gt_future_px[:, 0], gt_future_px[:, 1], "--",
                color="#7f8c8d", lw=1.4, dashes=(4, 2), label="GT future")
        all_xy_list = [seed_px, gt_future_px, full_px]
        for mdef in MODEL_DEFS:
            ppx = rolls[mdef["name"]].get(tid)
            if ppx is None: continue
            ax.plot(ppx[:, 0], ppx[:, 1], "-", color=mdef["color"],
                    lw=1.7, alpha=0.9, label=mdef["label"])
            all_xy_list.append(ppx)
        ax.plot(seed_px[-1, 0], seed_px[-1, 1], "o", color="#27ae60",
                ms=10, mfc="none", mew=1.8, label="seed end")
        ax.set_title(f"track {tid}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right", framealpha=0.85)
        crop(ax, np.vstack(all_xy_list), plan_rgb.shape)
    for j in range(n, rows_n * cols):
        axes[j // cols, j % cols].set_axis_off()
    fig.suptitle("Plan overlay — all 4 ablation models combined",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device: {device}")

    df  = pd.read_csv(DATASET_CSV)
    enc = pd.read_csv(ENCODED_CSV)
    interp = SpatialInterpolator3(enc, k=IDW_K)
    extents = pd.read_csv(EXP / "world_extents.csv", index_col=0
                          ).squeeze("columns").to_dict()
    extents = {k: float(v) for k, v in extents.items()}

    H = load_H(CALIB_JSON)
    plan = cv2.imread(str(TOPVIEW_PNG), cv2.IMREAD_COLOR)
    plan_rgb = cv2.cvtColor(plan, cv2.COLOR_BGR2RGB)
    print(f"[INFO]  Plan: {plan_rgb.shape[1]}×{plan_rgb.shape[0]} px")

    available = set(df["track_id"].unique().tolist())
    tracks = [t for t in PRESET_TRACKS if t in available]

    for mdef in MODEL_DEFS:
        out = PLAN_DIR / f"plan_contact_sheet_{mdef['name']}.png"
        per_model_plan_sheet(mdef, tracks, df, interp, extents, device,
                              H, plan_rgb, out)
        print(f"[OK]    {out.name}")

    combined = PLAN_DIR / "plan_contact_sheet_combined.png"
    combined_plan_sheet(tracks, df, interp, extents, device, H, plan_rgb,
                         combined)
    print(f"[OK]    {combined.name}")


if __name__ == "__main__":
    main()
