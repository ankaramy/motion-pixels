"""
make_plots.py — MODEL_X visualisations from the test-split evaluation.

Diagnostic panels (technical): observed history, GT future, prediction, current marker,
equal aspect, readable legend. Plus a clean presentation collage.

Reads evaluation/plot_windows.pkl, per_recording_metrics.csv, all_window_arrays.npz.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVAL = ROOT / "evaluation"
sys.path.insert(0, str(ROOT))

# ── diagnostic panel ───────────────────────────────────────────────────────────
def diag_panel(ax, w, show_legend=False):
    obs, gt, pred = w["obs"], w["gt"], w["pred"]
    ax.plot(obs[:, 0], obs[:, 1], "-o", color="#1f77b4", ms=2.5, lw=1.3, label="observed")
    ax.plot(gt[:, 0], gt[:, 1], "-o", color="#2ca02c", ms=2.5, lw=1.5, label="GT future")
    ax.plot(pred[:, 0], pred[:, 1], "--o", color="#d62728", ms=2.5, lw=1.5, label="prediction")
    ax.plot(obs[-1, 0], obs[-1, 1], "o", color="black", ms=6, label="current")
    ax.set_aspect("equal", "datalim")
    ax.set_title(f"{w['recording_id']}\nADE={w['ade']:.2f}m FDE={w['fde']:.2f}m"
                 + (" [TURN]" if w["is_turn"] else ""), fontsize=8)
    ax.tick_params(labelsize=6)
    if show_legend:
        ax.legend(fontsize=6, loc="best")


def grid6(windows, title, out):
    if not windows:
        print(f"[plot] skip {out.name} (no windows)"); return
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for i, ax in enumerate(axes.flat):
        if i < len(windows):
            diag_panel(ax, windows[i], show_legend=(i == 0))
        else:
            ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[plot] {out}")


def grid9(windows, title, out):
    if not windows:
        print(f"[plot] skip {out.name}"); return
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for i, ax in enumerate(axes.flat):
        if i < len(windows):
            diag_panel(ax, windows[i], show_legend=(i == 0))
        else:
            ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[plot] {out}")


# ── presentation collage (clean) ────────────────────────────────────────────────
def pres_panel(ax, w):
    obs, gt, pred = w["obs"], w["gt"], w["pred"]
    ax.plot(obs[:, 0], obs[:, 1], "-", color="black", lw=2.0, solid_capstyle="round")
    ax.plot(gt[:, 0], gt[:, 1], "-", color="#b0b0b0", lw=3.0, solid_capstyle="round")
    ax.plot(pred[:, 0], pred[:, 1], "-", color="#e0218a", lw=2.2, solid_capstyle="round")
    # clean arrow head on prediction
    if len(pred) >= 2:
        ax.annotate("", xy=pred[-1], xytext=pred[-2],
                    arrowprops=dict(arrowstyle="-|>", color="#e0218a", lw=2.0))
    ax.plot(obs[-1, 0], obs[-1, 1], "o", color="#333333", ms=4)  # small pebble
    ax.set_aspect("equal", "datalim"); ax.axis("off")


def presentation(windows, out):
    if not windows:
        print(f"[plot] skip {out.name}"); return
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < len(windows):
            pres_panel(ax, windows[i])
        else:
            ax.axis("off")
    # legend proxy
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="black", lw=2, label="observed"),
               Line2D([0], [0], color="#b0b0b0", lw=3, label="ground truth"),
               Line2D([0], [0], color="#e0218a", lw=2.2, label="prediction")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11, frameon=False)
    fig.suptitle("MODEL_X — predicted pedestrian trajectories (Barcelona test split)",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97]); fig.savefig(out, dpi=160); plt.close(fig)
    print(f"[plot] {out}")


def distributions():
    a = np.load(EVAL / "all_window_arrays.npz", allow_pickle=True)
    pred_len, gt_len = a["pred_len"], a["gt_len"]
    # 1) predicted path length distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(pred_len, bins=60, color="#d62728", alpha=0.6, label="predicted")
    ax.hist(gt_len, bins=60, color="#2ca02c", alpha=0.5, label="ground truth")
    ax.set_xlabel("path length over 10-step horizon (m)"); ax.set_ylabel("count")
    ax.legend(); ax.grid(alpha=.3); ax.set_title("MODEL_X path-length distribution")
    fig.tight_layout(); fig.savefig(HERE / "distributions" / "path_length_distribution.png", dpi=140)
    plt.close(fig)
    # 2) GT vs pred scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    m = max(gt_len.max(), pred_len.max())
    ax.scatter(gt_len, pred_len, s=4, alpha=0.15, color="#1f77b4")
    ax.plot([0, m], [0, m], "k--", lw=1, label="y=x")
    ax.set_xlabel("GT path length (m)"); ax.set_ylabel("predicted path length (m)")
    ax.set_aspect("equal"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("GT vs predicted path length")
    fig.tight_layout(); fig.savefig(HERE / "distributions" / "gt_vs_pred_length_scatter.png", dpi=140)
    plt.close(fig)
    print("[plot] distributions done")


def per_recording_bar():
    pr = pd.read_csv(EVAL / "per_recording_metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pr)); w = 0.38
    ax.bar(x - w/2, pr.ADE_mean, w, label="ADE (m)", color="#1f77b4")
    ax.bar(x + w/2, pr.FDE_mean, w, label="FDE (m)", color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(pr.recording_id, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("metres"); ax.legend(); ax.grid(alpha=.3, axis="y")
    ax.set_title("MODEL_X test ADE/FDE per recording")
    fig.tight_layout(); fig.savefig(HERE / "distributions" / "per_recording_ade_fde_bar.png", dpi=140)
    plt.close(fig)
    print("[plot] per-recording bar done")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with open(EVAL / "plot_windows.pkl", "rb") as fh:
        P = pickle.load(fh)
    grid6(P["best6"], "MODEL_X — Best 6 overall predictions (lowest ADE)",
          HERE / "best_6_overall" / "best_6_overall.png")
    grid6(P["worst6"], "MODEL_X — Worst 6 overall predictions (highest ADE)",
          HERE / "worst_6_overall" / "worst_6_overall.png")
    grid6(P["turn_best6"], "MODEL_X — Best 6 genuine-turn predictions (lowest ADE)",
          HERE / "best_6_genuine_turns" / "best_6_genuine_turns.png")
    grid9(P["rep9"], "MODEL_X — Representative predictions (ADE quantile spread)",
          HERE / "collages" / "representative_9panel.png")
    presentation(P["presentation"],
                 HERE / "collages" / "model_x_best_predictions_presentation.png")
    distributions()
    per_recording_bar()
    print("[plot] ALL DONE")


if __name__ == "__main__":
    main()
