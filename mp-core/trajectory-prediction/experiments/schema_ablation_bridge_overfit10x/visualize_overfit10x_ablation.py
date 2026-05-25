"""
visualize_overfit10x_ablation.py
--------------------------------
OVERFIT10X — not thesis generalization evidence.

Same plotting code as visualize_bridge_ablation.py (local-bound overlay
+ 2x2 grid + population error / angular / turn-rate plots) but every
title and the summary md are tagged so the artefacts cannot be mistaken
for held-out generalisation results.

Outputs (inside the experiment folder):
  ablation_visuals/
    overlay/traj_<i>_id<tid>.png
    grid/traj_<i>_id<tid>.png
    error_over_time.png
    angular_error_over_time.png
    turn_rate_comparison.png
    overfit10x_visual_summary.md
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_overfit10x_ablation import (
    COLORS, FEATURE_SETS as FEATURE_SETS_BASE, N_ROLLOUT, SEED,
    TARGET_COLS, TRAIN_FRAC, VAL_FRAC, WINDOW_SIZE, SHIFT_THRESH_DEG,
    ColumnScaler, TrajectoryLSTM,
    build_kdt, gt_headings_from_df, gt_positions_from_seed,
    resolve_feature_sets, rollout, set_seed, split_ids, wrap_angle,
)


DEFAULT_DATA   = HERE / "schema_ablation_bridge_overfit10x_dataset.csv"
OVERFIT_TAG    = "OVERFIT10X — not thesis generalization evidence"
DEFAULT_SCHEMA = HERE / "schema_summary.json"
DEFAULT_OUT    = HERE / "ablation_visuals"


# ─────────────────────────────────────────────────────────────────────────────
# Visual style — mirrors the old experiment so the plots read the same.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_LABELS = {
    "A_motion_only":             "A — motion only",
    "B_motion_position":         "B — + position",
    "C_motion_position_spatial": "C — + spatial",
    "D_full_relational":         "D — + entrance",
    "E_full_affordance":         "E — + asymmetry",
}
GT_COLOR   = "#111111"
SEED_COLOR = "#888888"
N_SELECT   = 10

plt.rcParams.update({
    "font.family":  "sans-serif",
    "font.size":    9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────────────────────────────────────
# Re-hydrate models + scalers from the trainer's saved artefacts.
# ─────────────────────────────────────────────────────────────────────────────
def load_models_and_scalers(df: pd.DataFrame, ablation_dir: Path,
                            device: torch.device,
                            feature_sets: Dict[str, List[str]]
                            ) -> Tuple[Dict, Dict, Dict]:
    blob_path = ablation_dir / "scalers.pkl"
    blob = pickle.loads(blob_path.read_bytes()) if blob_path.exists() else None
    models, f_scs, t_scs = {}, {}, {}
    for name, feat_cols in feature_sets.items():
        # Build scalers
        if blob is not None and name in blob:
            b = blob[name]
            f_sc = ColumnScaler(); f_sc.cols = b["feat_cols"]
            f_sc.means = np.array(b["feat_means"]); f_sc.stds = np.array(b["feat_stds"])
            t_sc = ColumnScaler(); t_sc.cols = b["tgt_cols"]
            t_sc.means = np.array(b["tgt_means"]); t_sc.stds = np.array(b["tgt_stds"])
        else:
            # Fallback: refit on train split (deterministic via SEED).
            train_ids, _, _ = split_ids(df)
            train_df = df[df["trajectory_id"].isin(train_ids)]
            f_sc = ColumnScaler().fit(train_df, feat_cols)
            t_sc = ColumnScaler().fit(train_df, TARGET_COLS)

        model = TrajectoryLSTM(len(feat_cols))
        ckpt  = ablation_dir / name / "best_model.pth"
        if not ckpt.exists():
            raise SystemExit(f"[FATAL] missing checkpoint: {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.to(device).eval()
        models[name] = model
        f_scs[name]  = f_sc
        t_scs[name]  = t_sc
    return models, f_scs, t_scs


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory selection (diversity in u/v/obstacle/turn-rate)
# ─────────────────────────────────────────────────────────────────────────────
def select_trajectories(df: pd.DataFrame, test_ids: List, n: int) -> List:
    eligible = []
    for tid in test_ids:
        g = df[df["trajectory_id"] == tid].sort_values("timestep")
        if len(g) < WINDOW_SIZE + N_ROLLOUT + 5:
            continue
        eligible.append((
            tid,
            float(g["u"].mean()),
            float(g["v"].mean()),
            float(g["dist_to_obstacle_norm"].mean()),
            float(g["turn_rate"].abs().mean()),
        ))
    if not eligible:
        return test_ids[:n]
    arr = np.array([[e[1], e[2], e[3], e[4]] for e in eligible])
    rng = arr.max(0) - arr.min(0); rng[rng < 1e-9] = 1.0
    arr_n = (arr - arr.min(0)) / rng
    selected_idx = [0]
    for _ in range(min(n - 1, len(eligible) - 1)):
        d = np.min(np.linalg.norm(arr_n[None, :, :] - arr_n[selected_idx, :][:, None, :], axis=2), axis=0)
        selected_idx.append(int(np.argmax(d)))
    return [eligible[i][0] for i in selected_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Per-track rollout (returns positions, headings, ADE, FDE, step errors)
# ─────────────────────────────────────────────────────────────────────────────
def run_detailed_rollout(model, feat_cols, all_feat_cols, spatial_cols,
                         f_sc, t_sc, world_bnd, kdt, traj_df,
                         device) -> dict:
    seed_raw = traj_df.iloc[:WINDOW_SIZE][all_feat_cols].to_numpy(np.float32)
    gt_pos   = gt_positions_from_seed(traj_df, world_bnd, N_ROLLOUT)
    gt_h     = gt_headings_from_df(traj_df, N_ROLLOUT)
    pred_pos, pred_h = rollout(
        model, seed_raw, feat_cols, all_feat_cols, spatial_cols,
        f_sc, t_sc, world_bnd, kdt, N_ROLLOUT, device,
    )
    n = min(len(pred_pos), len(gt_pos))
    d = np.sqrt(((pred_pos[:n] - gt_pos[:n]) ** 2).sum(axis=1))[1:]
    n_h = min(len(pred_h), len(gt_h))
    ang = np.array([abs(wrap_angle(pred_h[i] - gt_h[i])) for i in range(n_h)])
    return {
        "pred_pos":  pred_pos, "gt_pos":  gt_pos,
        "pred_h":    pred_h,   "gt_h":    gt_h,
        "step_dist": d,        "step_ang": ang,
        "ade": float(d.mean()) if len(d) else float("nan"),
        "fde": float(d[-1])    if len(d) else float("nan"),
    }


def population_metrics(models, f_scs, t_scs, df, test_ids, world_bnd, kdts,
                       all_feat_cols, spatial_cols, device) -> Dict[str, dict]:
    result = {name: {"step_dist": [], "step_ang": []} for name in models}
    for tid in test_ids:
        traj = df[df["trajectory_id"] == tid].sort_values("timestep").reset_index(drop=True)
        if len(traj) < WINDOW_SIZE + N_ROLLOUT:
            continue
        for name, model in models.items():
            d = run_detailed_rollout(
                model, FEATURE_SETS_FROZEN[name], all_feat_cols, spatial_cols,
                f_scs[name], t_scs[name], world_bnd, kdts.get(name), traj, device,
            )
            result[name]["step_dist"].append(d["step_dist"])
            result[name]["step_ang"].append(d["step_ang"])

    out = {}
    for name in models:
        dists = result[name]["step_dist"]; angs = result[name]["step_ang"]
        max_s = max((len(x) for x in dists), default=0)
        if max_s == 0:
            out[name] = {"mean_dist": np.zeros(N_ROLLOUT), "std_dist": np.zeros(N_ROLLOUT),
                         "mean_ang":  np.zeros(N_ROLLOUT), "std_ang":  np.zeros(N_ROLLOUT)}
            continue
        D = np.full((len(dists), max_s), np.nan)
        A = np.full((len(angs),  max_s), np.nan)
        for i, (d, a) in enumerate(zip(dists, angs)):
            D[i, :len(d)] = d; A[i, :len(a)] = a
        out[name] = {
            "mean_dist": np.nanmean(D, axis=0),
            "std_dist":  np.nanstd(D,  axis=0),
            "mean_ang":  np.nanmean(A, axis=0),
            "std_ang":   np.nanstd(A,  axis=0),
        }
    return out


# Module-level slot for FEATURE_SETS resolved at runtime (with/without E).
FEATURE_SETS_FROZEN: Dict[str, List[str]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# uv conversion
# ─────────────────────────────────────────────────────────────────────────────
def world_to_uv(pos: np.ndarray, wb: dict) -> Tuple[np.ndarray, np.ndarray]:
    u = (pos[:, 0] - wb["xmin"]) / wb["xrng"]
    v = (pos[:, 1] - wb["ymin"]) / wb["yrng"]
    return u, v


# ─────────────────────────────────────────────────────────────────────────────
# Local-bounds helper (10% margin; expand smaller axis to square aspect)
# ─────────────────────────────────────────────────────────────────────────────
def _local_bounds(traj_df, rollouts_dict, wb, pad_frac: float = 0.10):
    all_u, all_v = [], []
    for res in rollouts_dict.values():
        pu, pv = world_to_uv(res["pred_pos"], wb)
        gu, gv = world_to_uv(res["gt_pos"],   wb)
        all_u.extend([*pu, *gu]); all_v.extend([*pv, *gv])
    seed = traj_df.iloc[:WINDOW_SIZE]
    all_u.extend(seed["u"].tolist()); all_v.extend(seed["v"].tolist())
    umin, umax = min(all_u), max(all_u)
    vmin, vmax = min(all_v), max(all_v)
    span = max(umax - umin, vmax - vmin, 1e-6)
    pad  = pad_frac * span
    cu, cv = 0.5 * (umin + umax), 0.5 * (vmin + vmax)
    half = 0.5 * span + pad
    return cu - half, cu + half, cv - half, cv + half


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — overlay (all 4-5 models + GT on one axis), local bounds
# ─────────────────────────────────────────────────────────────────────────────
def plot_overlay(traj_df, rollouts_dict, wb, out_path: Path, traj_id) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    # faint full GT trajectory for context
    ax.plot(traj_df["u"], traj_df["v"], color="#dddddd", lw=1.0, zorder=1)

    # seed
    sU = traj_df["u"].iloc[:WINDOW_SIZE]; sV = traj_df["v"].iloc[:WINDOW_SIZE]
    ax.plot(sU, sV, color=SEED_COLOR, lw=2.5, zorder=2, label="seed")
    ax.scatter(sU.iloc[0], sV.iloc[0], s=55, color=SEED_COLOR, zorder=5)

    # GT rollout
    gt_res = next(iter(rollouts_dict.values()))
    gu, gv = world_to_uv(gt_res["gt_pos"], wb)
    ax.plot(gu, gv, color=GT_COLOR, lw=2.5, ls="--", zorder=3,
            label="GT rollout")
    ax.scatter(gu[-1], gv[-1], s=45, color=GT_COLOR, marker="^", zorder=5)

    # predictions
    for name, res in rollouts_dict.items():
        pu, pv = world_to_uv(res["pred_pos"], wb)
        ax.plot(pu, pv, color=COLORS[name], lw=2.0, zorder=4,
                label=f"{MODEL_LABELS[name]}  ADE={res['ade']:.3f}m")
        ax.scatter(pu[-1], pv[-1], s=35, color=COLORS[name], marker="^", zorder=5)

    umin, umax, vmin, vmax = _local_bounds(traj_df, rollouts_dict, wb)
    ax.set_xlim(umin, umax); ax.set_ylim(vmin, vmax); ax.set_aspect("equal")
    ax.set_xlabel("u (norm. world x)"); ax.set_ylabel("v (norm. world y)")
    ax.set_title(f"Trajectory {traj_id} — all models vs GT\n[{OVERFIT_TAG}]",
                 fontsize=9)
    ax.legend(fontsize=7, loc="best", framealpha=0.75)
    ax.grid(alpha=0.15)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — 2×2 (or 2×3) grid, one model per panel
# ─────────────────────────────────────────────────────────────────────────────
def plot_grid(traj_df, rollouts_dict, wb, out_path: Path, traj_id) -> None:
    names = list(rollouts_dict.keys())
    ncols = 2; nrows = int(math.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                             squeeze=False)
    axes_flat = [ax for row in axes for ax in row]
    umin, umax, vmin, vmax = _local_bounds(traj_df, rollouts_dict, wb)
    gt_res = rollouts_dict[names[0]]
    gu, gv = world_to_uv(gt_res["gt_pos"], wb)

    for i, name in enumerate(names):
        ax = axes_flat[i]; res = rollouts_dict[name]
        ax.plot(traj_df["u"], traj_df["v"], color="#e0e0e0", lw=0.8, zorder=1)
        ax.plot(traj_df["u"].iloc[:WINDOW_SIZE], traj_df["v"].iloc[:WINDOW_SIZE],
                color=SEED_COLOR, lw=2.0, zorder=2)
        ax.plot(gu, gv, color=GT_COLOR, lw=2.0, ls="--", zorder=3, label="GT")
        ax.scatter(gu[-1], gv[-1], s=40, color=GT_COLOR, marker="^", zorder=5)
        pu, pv = world_to_uv(res["pred_pos"], wb)
        ax.plot(pu, pv, color=COLORS[name], lw=2.4, marker="o", ms=4, zorder=4)
        ax.scatter(pu[-1], pv[-1], s=40, color=COLORS[name], marker="^", zorder=5)
        ax.set_xlim(umin, umax); ax.set_ylim(vmin, vmax); ax.set_aspect("equal")
        ax.set_title(f"{MODEL_LABELS[name]}\nADE={res['ade']:.3f} m   "
                     f"FDE={res['fde']:.3f} m", fontsize=9)
        ax.set_xlabel("u"); ax.set_ylabel("v"); ax.grid(alpha=0.15)
    # hide any unused subplot
    for j in range(len(names), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(f"Trajectory {traj_id} — per-model comparison "
                 f"[{OVERFIT_TAG}]", fontsize=10, y=1.0)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plots 3 & 4 — error / angular-error over time (population)
# ─────────────────────────────────────────────────────────────────────────────
def plot_error_over_time(pop_metrics: dict, out_path: Path) -> None:
    steps = np.arange(1, N_ROLLOUT + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in pop_metrics:
        m  = pop_metrics[name]
        md = m["mean_dist"]; sd = m["std_dist"]
        ax.plot(steps[:len(md)], md, color=COLORS[name], lw=2.0,
                label=MODEL_LABELS[name])
        ax.fill_between(steps[:len(sd)], md - 0.5*sd, md + 0.5*sd,
                        color=COLORS[name], alpha=0.12)
    ax.set_xlabel("Rollout step"); ax.set_ylabel("Mean displacement error (m)")
    ax.set_title("Autoregressive drift — displacement error per step\n"
                 f"[{OVERFIT_TAG}]", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.set_xlim(1, N_ROLLOUT); ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_path.name}")


def plot_angular_error_over_time(pop_metrics: dict, out_path: Path) -> None:
    steps = np.arange(1, N_ROLLOUT + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in pop_metrics:
        m  = pop_metrics[name]
        ma = np.degrees(m["mean_ang"]); sa = np.degrees(m["std_ang"])
        ax.plot(steps[:len(ma)], ma, color=COLORS[name], lw=2.0,
                label=MODEL_LABELS[name])
        ax.fill_between(steps[:len(sa)], ma - 0.5*sa, ma + 0.5*sa,
                        color=COLORS[name], alpha=0.12)
    ax.set_xlabel("Rollout step"); ax.set_ylabel("Mean angular error (degrees)")
    ax.set_title("Directional accuracy — angular error per step\n"
                 f"[{OVERFIT_TAG}]", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.set_xlim(1, N_ROLLOUT); ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — turn-rate bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_turn_rate_bar(results_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(results_csv)
    gt_tr = df["mean_turn_rate_gt"].iloc[0]
    names  = ["GT"] + df["model"].tolist()
    values = [gt_tr] + df["mean_turn_rate_pred"].tolist()
    colors = ["#555"] + [COLORS[n] for n in df["model"]]
    labels = ["Ground truth"] + [MODEL_LABELS[n] for n in df["model"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(names)), values, color=colors, width=0.6,
                  edgecolor="white", lw=0.5)
    ax.axhline(gt_tr, color="#555", ls="--", lw=1.2, alpha=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Mean |turn_rate| per step (rad)")
    ax.set_title("Turn rate: predicted vs ground truth\n"
                 f"[{OVERFIT_TAG}]", fontsize=10)
    ax.set_ylim(0, max(values) * 1.2); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Markdown summary
# ─────────────────────────────────────────────────────────────────────────────
def write_visual_summary(sel_ids, df_results: pd.DataFrame, out_path: Path) -> None:
    md = []; A = md.append
    A("# schema_ablation_bridge_overfit10x — Visual Summary\n")
    A(f"> **{OVERFIT_TAG}.**  Train/val/test share trajectory content by "
      "construction; numbers below reflect MEMORISATION capacity, not "
      "held-out performance.\n")
    A("Same recipe as schema_ablation_bridge (real-metric du/dv, MSE-only "
      "loss, autoregressive rollout, KDTree spatial refresh) — trained on "
      "the 10× duplicated dataset.\n")
    A(f"Selected trajectories: `{sel_ids}`\n")
    A("## Models\n")
    A("| Tag | Label | Features |")
    A("|---|---|---|")
    for name, feats in FEATURE_SETS_FROZEN.items():
        A(f"| `{name}` | {MODEL_LABELS.get(name, name)} | "
          f"{', '.join(f'`{c}`' for c in feats)} |")
    A("")
    A("## Quantitative results (from ablation_results.csv)\n")
    A("| Model | ADE (m) | FDE (m) | Ang err (rad) | ang_ratio | curv_corr | "
      "path/GT | N |")
    A("|---|---|---|---|---|---|---|---|")
    for _, row in df_results.iterrows():
        A(f"| `{row['model']}` | {row['test_ade_m']:.4f} | "
          f"{row['test_fde_m']:.4f} | {row['mean_angular_err']:.4f} | "
          f"{row['angularity_ratio']:.3f} | {row['curvature_corr']:.3f} | "
          f"{row['path_length_ratio']:.3f} | {row['n_trajs']} |")
    A("")
    A("## Outputs\n")
    A("| File | Content |")
    A("|---|---|")
    A("| `overlay/traj_*.png` | All models vs GT on one axis (local bounds) |")
    A("| `grid/traj_*.png` | One model per panel vs GT |")
    A("| `error_over_time.png` | Per-step displacement drift, population mean |")
    A("| `angular_error_over_time.png` | Per-step heading error, population mean |")
    A("| `turn_rate_comparison.png` | Mean |turn_rate| vs GT |")
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[saved] {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    global FEATURE_SETS_FROZEN
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      type=Path, default=DEFAULT_DATA)
    parser.add_argument("--schema",       type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--ablation_dir", type=Path, default=HERE)
    parser.add_argument("--output_dir",   type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    print("[load] data ...")
    df = pd.read_csv(args.dataset)
    with open(args.schema) as f:
        schema = json.load(f)
    wb = schema["world_bounds_used"]

    feature_sets, all_feat_cols, spatial_cols = resolve_feature_sets(df.columns)
    FEATURE_SETS_FROZEN = feature_sets

    train_ids, val_ids, test_ids = split_ids(df)
    print(f"[split] {len(train_ids)} train · {len(val_ids)} val · "
          f"{len(test_ids)} test")

    print("[load] models + scalers ...")
    models, f_scs, t_scs = load_models_and_scalers(
        df, args.ablation_dir, device, feature_sets)

    print("[kdt]  rebuild over training data ...")
    train_df = df[df["trajectory_id"].isin(train_ids)]
    kdt_base = build_kdt(train_df, spatial_cols)
    kdts = {name: (kdt_base if any(c in feature_sets[name] for c in spatial_cols)
                   else None) for name in models}

    print("[select] representative trajectories ...")
    sel_ids = select_trajectories(df, test_ids, N_SELECT)
    print(f"         selected: {sel_ids}")

    print("[rollout] selected ...")
    traj_rollouts: Dict[int, dict] = {}
    for tid in sel_ids:
        traj = df[df["trajectory_id"] == tid].sort_values("timestep").reset_index(drop=True)
        traj_rollouts[tid] = {}
        for name, model in models.items():
            r = run_detailed_rollout(
                model, feature_sets[name], all_feat_cols, spatial_cols,
                f_scs[name], t_scs[name], wb, kdts[name], traj, device,
            )
            traj_rollouts[tid][name] = r

    out_dir     = args.output_dir
    overlay_dir = out_dir / "overlay"; grid_dir = out_dir / "grid"
    for d in (out_dir, overlay_dir, grid_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("[plot1] overlay ...")
    for i, tid in enumerate(sel_ids):
        traj = df[df["trajectory_id"] == tid].sort_values("timestep").reset_index(drop=True)
        plot_overlay(traj, traj_rollouts[tid], wb,
                     overlay_dir / f"traj_{i:02d}_id{tid}.png", tid)

    print("[plot2] grid ...")
    for i, tid in enumerate(sel_ids):
        traj = df[df["trajectory_id"] == tid].sort_values("timestep").reset_index(drop=True)
        plot_grid(traj, traj_rollouts[tid], wb,
                  grid_dir / f"traj_{i:02d}_id{tid}.png", tid)

    print("[pop ] population metrics ...")
    pop = population_metrics(models, f_scs, t_scs, df, test_ids, wb, kdts,
                             all_feat_cols, spatial_cols, device)

    print("[plot3] error over time ...")
    plot_error_over_time(pop, out_dir / "error_over_time.png")
    print("[plot4] angular error over time ...")
    plot_angular_error_over_time(pop, out_dir / "angular_error_over_time.png")
    print("[plot5] turn-rate bar ...")
    results_csv = args.ablation_dir / "ablation_results.csv"
    if results_csv.exists():
        plot_turn_rate_bar(results_csv, out_dir / "turn_rate_comparison.png")
        df_results = pd.read_csv(results_csv)
    else:
        df_results = pd.DataFrame()

    print("[summary] ...")
    write_visual_summary(sel_ids, df_results,
                         out_dir / "overfit10x_visual_summary.md")

    # ── Compact end-of-script sanity / collapse check ────────────────────
    print()
    print("=" * 66)
    print("[SANITY] schema_ablation_bridge_overfit10x rollout sanity "
          "[OVERFIT10X — not generalization evidence]")
    print("=" * 66)
    any_collapse = False
    for name in models:
        # population mean step magnitude (u/v space, last step in window)
        # we re-use pop["mean_dist"] only for the dispersion shape;
        # for movement magnitude derive from selected tracks instead.
        ratios = []
        for tid, packs in traj_rollouts.items():
            res = packs[name]
            p = res["pred_pos"]; g = res["gt_pos"]
            pn = min(len(p), len(g))
            if pn < 2:
                continue
            p_uv = np.column_stack([
                (p[:pn, 0] - wb["xmin"]) / wb["xrng"],
                (p[:pn, 1] - wb["ymin"]) / wb["yrng"]])
            g_uv = np.column_stack([
                (g[:pn, 0] - wb["xmin"]) / wb["xrng"],
                (g[:pn, 1] - wb["ymin"]) / wb["yrng"]])
            ps = np.linalg.norm(np.diff(p_uv, axis=0), axis=1).mean()
            gs = np.linalg.norm(np.diff(g_uv, axis=0), axis=1).mean()
            if gs > 1e-12:
                ratios.append(ps / gs)
        mvmt = float(np.mean(ratios)) if ratios else float("nan")
        collapse = bool(np.isfinite(mvmt) and mvmt < 0.25)
        if collapse: any_collapse = True
        print(f"  {MODEL_LABELS.get(name, name):<28} "
              f"mean pred/GT movement ratio = {mvmt:.4f}"
              f"   {'[STATIC COLLAPSE]' if collapse else ''}")
    print("=" * 66)
    if any_collapse:
        print("STATIC COLLAPSE DETECTED for at least one model.")
    else:
        print("No static collapse detected across selected trajectories.")
    print(f"\n[done] visuals in {out_dir}")


if __name__ == "__main__":
    main()
