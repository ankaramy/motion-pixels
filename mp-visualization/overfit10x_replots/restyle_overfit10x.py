"""
restyle_overfit10x.py
---------------------
VISUALIZATION-ONLY restyle of the OVERFIT10X model-comparison plots.

OVERFIT10X is a capacity / memorisation check — NOT thesis generalization
evidence. This script does not train, does not change metrics, does not animate
and does not touch source data. It re-hydrates the frozen model checkpoints
(the predictions are never stored as CSVs — they are deterministic rollouts of
the saved .pth files) and replots three selected trajectories in the clean,
official thesis aesthetic with Model C highlighted.

Selected tracks : 1001, 12007, 14007
Models          : A motion-only · B +position · C +spatial · D +entrance

Styling
  Ground Truth : dotted black, clearly visible, not too thick
  Model C      : solid red (#D7261E), visually dominant, clean
  Models A/B/D : solid light grey (#C8C8C8), same weight, subdued
  Seed/history : dark grey (#555555), subtle
  Background   : white · grid: very light dotted · axes: quiet

Outputs (mp-visualization/overfit10x_replots/)
  track_<id>_modelC_highlight.{png,svg}
  overfit10x_modelC_highlight_collage.{png,svg}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import torch

# ── Locate the OVERFIT10X experiment and import its frozen machinery ─────────
ROOT = Path(__file__).resolve().parents[2]            # motion-pixels/
EXP  = ROOT / "mp-core" / "trajectory-prediction" / "experiments" \
            / "schema_ablation_bridge_overfit10x"
sys.path.insert(0, str(EXP))

from run_overfit10x_ablation import (                  # noqa: E402
    COLORS, N_ROLLOUT, SEED, WINDOW_SIZE,
    build_kdt, resolve_feature_sets, set_seed, split_ids,
)
import visualize_overfit10x_ablation as viz            # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
TAG     = "capacity check — not generalization evidence"

# Selected trajectories (as specified)
TARGET_IDS = [1001, 12007, 14007]

# ── Style palette (user spec, layered on the official clean aesthetic) ───────
C_GT    = "#000000"   # ground truth — dotted black
C_MODEL = "#D7261E"   # Model C — solid red (focal)
C_OTHER = "#C8C8C8"   # Models A/B/D — light grey
C_SEED  = "#555555"   # seed / history — dark grey
C_CTX   = "#ededed"   # faint full-track context line
C_GRID  = "#e4e4e4"   # very light dotted grid
C_SPINE = "#cfcfcf"   # faded spines
C_TICK  = "#7a7a7a"   # quiet tick / label text

MODEL_C = "C_motion_position_spatial"

plt.rcParams.update({
    "font.family": ["Inter", "Helvetica Neue", "Arial", "sans-serif"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
})


def _style_axis(ax) -> None:
    ax.set_aspect("equal")
    ax.grid(True, color=C_GRID, ls=":", lw=0.8, zorder=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_SPINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=C_TICK, labelsize=8.5, length=3)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel("u (norm. world x)", color=C_TICK, fontsize=9.5)
    ax.set_ylabel("v (norm. world y)", color=C_TICK, fontsize=9.5)


def _square_bounds(traj_df, rollouts, wb, pad_frac=0.14):
    """Square data window centred on history+GT+predictions, generous padding."""
    us, vs = [], []
    for res in rollouts.values():
        pu, pv = viz.world_to_uv(res["pred_pos"], wb)
        gu, gv = viz.world_to_uv(res["gt_pos"], wb)
        us.extend([*pu, *gu]); vs.extend([*pv, *gv])
    seed = traj_df.iloc[:WINDOW_SIZE]
    us.extend(seed["u"].tolist()); vs.extend(seed["v"].tolist())
    umin, umax = min(us), max(us)
    vmin, vmax = min(vs), max(vs)
    span = max(umax - umin, vmax - vmin, 1e-6)
    half = 0.5 * span + pad_frac * span
    cu, cv = 0.5 * (umin + umax), 0.5 * (vmin + vmax)
    return cu - half, cu + half, cv - half, cv + half


def draw_track(ax, traj_df, rollouts, wb, traj_id, *, with_title=True):
    """Render one trajectory panel in the restyled aesthetic."""
    _style_axis(ax)

    # faint full-track context (quiet, behind everything)
    ax.plot(traj_df["u"], traj_df["v"], color=C_CTX, lw=1.0, zorder=1)

    # seed / history — dark grey, subtle
    sU = traj_df["u"].iloc[:WINDOW_SIZE]; sV = traj_df["v"].iloc[:WINDOW_SIZE]
    ax.plot(sU, sV, color=C_SEED, lw=2.2, solid_capstyle="round", zorder=3)
    ax.scatter(sU.iloc[-1], sV.iloc[-1], s=26, color=C_SEED, zorder=6)

    # GT rollout — dotted black
    gt_res = next(iter(rollouts.values()))
    gu, gv = viz.world_to_uv(gt_res["gt_pos"], wb)
    ax.plot(gu, gv, color=C_GT, lw=2.0, ls=(0, (1.6, 1.8)),
            dash_capstyle="round", zorder=4)

    # other models (A, B, D) — light grey, same weight, subdued
    for name, res in rollouts.items():
        if name == MODEL_C:
            continue
        pu, pv = viz.world_to_uv(res["pred_pos"], wb)
        ax.plot(pu, pv, color=C_OTHER, lw=1.8, solid_capstyle="round", zorder=5)

    # Model C — solid red, dominant, on top
    cres = rollouts[MODEL_C]
    cu, cv = viz.world_to_uv(cres["pred_pos"], wb)
    ax.plot(cu, cv, color=C_MODEL, lw=2.6, solid_capstyle="round", zorder=8)
    ax.scatter(cu[-1], cv[-1], s=30, color=C_MODEL, zorder=9)

    umin, umax, vmin, vmax = _square_bounds(traj_df, rollouts, wb)
    ax.set_xlim(umin, umax); ax.set_ylim(vmin, vmax)

    if with_title:
        ax.set_title(f"OVERFIT10X trajectory {traj_id}\nModel C highlighted",
                     fontsize=11, color="#222222", pad=10)

    # quiet annotation block, upper-left inside the plot
    ax.text(0.025, 0.975,
            f"Model C ADE = {cres['ade']:.3f} m\n{TAG}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, color="#333333", linespacing=1.4)
    return cres["ade"]


def legend_handles():
    return [
        Line2D([0], [0], color=C_SEED, lw=2.2, label="seed / history"),
        Line2D([0], [0], color=C_GT, lw=2.0, ls=(0, (1.6, 1.8)),
               label="ground truth"),
        Line2D([0], [0], color=C_OTHER, lw=1.8, label="models A · B · D"),
        Line2D([0], [0], color=C_MODEL, lw=2.6, label="Model C (+ spatial)"),
    ]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    print("[load] dataset + schema ...")
    df = pd.read_csv(EXP / "schema_ablation_bridge_overfit10x_dataset.csv")
    schema = json.loads((EXP / "schema_summary.json").read_text())
    wb = schema["world_bounds_used"]

    feature_sets, all_feat_cols, spatial_cols = resolve_feature_sets(df.columns)
    viz.FEATURE_SETS_FROZEN = feature_sets

    print("[load] frozen models + scalers ...")
    models, f_scs, t_scs = viz.load_models_and_scalers(
        df, EXP, device, feature_sets)

    train_ids, _, _ = split_ids(df)
    train_df = df[df["trajectory_id"].isin(train_ids)]
    kdt_base = build_kdt(train_df, spatial_cols)
    kdts = {n: (kdt_base if any(c in feature_sets[n] for c in spatial_cols)
                else None) for n in models}

    found = {}     # report bookkeeping
    rollouts_by_id = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for tid in TARGET_IDS:
        traj = (df[df["trajectory_id"] == tid]
                .sort_values("timestep").reset_index(drop=True))
        if traj.empty:
            print(f"[WARN] track {tid} not in dataset — skipped")
            found[tid] = None
            continue
        print(f"[rollout] track {tid} ...")
        rollouts = {}
        per_model = {}
        for name, model in models.items():
            r = viz.run_detailed_rollout(
                model, feature_sets[name], all_feat_cols, spatial_cols,
                f_scs[name], t_scs[name], wb, kdts[name], traj, device)
            rollouts[name] = r
            per_model[name] = float(r["ade"])
        rollouts_by_id[tid] = (traj, rollouts)
        found[tid] = per_model

        # ── single-track figure ──
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        draw_track(ax, traj, rollouts, wb, tid)
        ax.legend(handles=legend_handles(), fontsize=7.5, loc="lower right",
                  frameon=False)
        fig.tight_layout()
        for ext in ("png", "svg"):
            p = OUT_DIR / f"track_{tid}_modelC_highlight.{ext}"
            fig.savefig(p, dpi=200, bbox_inches="tight",
                        facecolor="white")
        plt.close(fig)
        print(f"[saved] track_{tid}_modelC_highlight.png/.svg")

    # ── 3-panel collage ──
    valid = [t for t in TARGET_IDS if t in rollouts_by_id]
    if valid:
        fig, axes = plt.subplots(1, len(valid), figsize=(6.0 * len(valid), 6.2))
        if len(valid) == 1:
            axes = [axes]
        for ax, tid in zip(axes, valid):
            traj, rollouts = rollouts_by_id[tid]
            draw_track(ax, traj, rollouts, wb, tid, with_title=True)
        fig.suptitle("OVERFIT10X — Model C highlighted   "
                     f"[{TAG}]", fontsize=12, color="#222222", y=1.02)
        fig.legend(handles=legend_handles(), fontsize=8.5, ncol=4,
                   loc="lower center", frameon=False,
                   bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout()
        for ext in ("png", "svg"):
            p = OUT_DIR / f"overfit10x_modelC_highlight_collage.{ext}"
            fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("[saved] overfit10x_modelC_highlight_collage.png/.svg")

    # expose bookkeeping for the report
    (OUT_DIR / "_found.json").write_text(
        json.dumps({str(k): v for k, v in found.items()}, indent=2))
    print("[done]")


if __name__ == "__main__":
    main()
