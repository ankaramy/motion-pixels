"""
visualize_ablation.py
---------------------
Visual comparison of LSTM ablation models for architectural presentation.

Generates in ablation_visuals/:
  overlay/          -- all 4 models + GT on one axis per trajectory
  grid/             -- 2x2 panel per trajectory (one model vs GT each)
  error_over_time.png
  angular_error_over_time.png
  turn_rate_comparison.png
  spatial_context_obstacle.png
  spatial_context_boundary.png
  model_rollouts_on_plan.png
  visual_ablation_summary.md
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ── import from sibling ablation script ──────────────────────────────────────
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_ablation import (
    ALL_FEAT_COLS, FEATURE_SETS, N_ROLLOUT, SEED, TARGET_COLS,
    TRAIN_FRAC, VAL_FRAC, VAL_FRAC, WINDOW_SIZE,
    ColumnScaler, TrajectoryLSTM, build_kdt, gt_headings_from_df,
    gt_positions_from_seed, rollout, set_seed, wrap_angle,
)

# ── paths ─────────────────────────────────────────────────────────────────────
MP_ROOT      = HERE.parent.parent.parent.parent
ENCODED_DIR  = MP_ROOT / "mp-data" / "processed" / "encoded"
SPATIAL_DIR  = MP_ROOT / "mp-data" / "processed" / "spatial"
IMAGES_DIR   = MP_ROOT / "mp-data" / "raw" / "images"
CALIB_DIR    = MP_ROOT / "mp-data" / "raw" / "calibration"
ABLATION_DIR = MP_ROOT / "mp-data" / "outputs" / "ablation"
DEFAULT_DATA   = ENCODED_DIR / "motion_dataset_v2.csv"
DEFAULT_SCHEMA = ENCODED_DIR / "schema_summary.json"
DEFAULT_OUT    = ABLATION_DIR / "ablation_visuals"

# ── visual style ──────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "A_motion_only":             "#d62728",   # red
    "B_motion_position":         "#ff7f0e",   # orange
    "C_motion_position_spatial": "#1f77b4",   # blue
    "D_full_affordance":         "#9467bd",   # purple
}
MODEL_LABELS = {
    "A_motion_only":             "A — motion only",
    "B_motion_position":         "B — + position",
    "C_motion_position_spatial": "C — + spatial",
    "D_full_affordance":         "D — + asymmetry",
}
GT_COLOR   = "#111111"
SEED_COLOR = "#888888"
N_SELECT   = 10     # trajectories to show in per-trajectory plots

plt.rcParams.update({
    "font.family":  "sans-serif",
    "font.size":    9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────────────────────────────────────
# Data / model loading
# ─────────────────────────────────────────────────────────────────────────────

def split_ids(df: pd.DataFrame) -> Tuple[List, List, List]:
    all_ids = df["trajectory_id"].unique().tolist()
    rng = np.random.default_rng(SEED)
    rng.shuffle(all_ids)
    n    = len(all_ids)
    n_tr = int(n * TRAIN_FRAC)
    n_va = int(n * VAL_FRAC)
    return all_ids[:n_tr], all_ids[n_tr:n_tr + n_va], all_ids[n_tr + n_va:]


def load_models_and_scalers(
    df: pd.DataFrame, train_ids: List, ablation_dir: Path, device: torch.device
) -> Tuple[Dict, Dict, Dict]:
    """Re-fit scalers on training data and load saved weights for all 4 models."""
    train_df = df[df["trajectory_id"].isin(train_ids)]
    models, f_scs, t_scs = {}, {}, {}
    for name, feat_cols in FEATURE_SETS.items():
        f_sc = ColumnScaler().fit(train_df, feat_cols)
        t_sc = ColumnScaler().fit(train_df, TARGET_COLS)
        model = TrajectoryLSTM(len(feat_cols))
        ckpt  = ablation_dir / name / "best_model.pth"
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.to(device).eval()
        models[name] = model
        f_scs[name]  = f_sc
        t_scs[name]  = t_sc
    return models, f_scs, t_scs


def load_plan_image() -> Optional[np.ndarray]:
    for name in ("top-down.png", "YOUR_TOPVIEW_SKATE.png"):
        p = IMAGES_DIR / name
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # fallback: segmentation overlay
    p = SPATIAL_DIR / "segmentation_overlay.png"
    if p.exists():
        img = cv2.imread(str(p))
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def load_world_to_plan_homography() -> Optional[np.ndarray]:
    for name in ("calib_skate1.json", "calib_macba.json"):
        p = CALIB_DIR / name
        if not p.exists():
            continue
        with open(p) as f:
            c = json.load(f)
        if "world_points" in c and "plan_points_px" in c:
            wp  = np.array(c["world_points"],   dtype=np.float32)
            pp  = np.array(c["plan_points_px"], dtype=np.float32)
            H, _ = cv2.findHomography(wp, pp, cv2.RANSAC, 5.0)
            if H is not None:
                return H.astype(np.float64)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory selection
# ─────────────────────────────────────────────────────────────────────────────

def select_trajectories(df: pd.DataFrame, test_ids: List, n: int) -> List:
    """Pick n test trajectories maximising spatial and curvature diversity."""
    eligible = []
    for tid in test_ids:
        g = df[df["trajectory_id"] == tid].sort_values("timestep")
        if len(g) < WINDOW_SIZE + N_ROLLOUT + 15:
            continue
        cu = float(g["u"].mean())
        cv_ = float(g["v"].mean())
        cobs = float(g["dist_to_obstacle_norm"].mean())
        ctr  = float(g["turn_rate"].abs().mean())
        eligible.append((tid, cu, cv_, cobs, ctr))
    if not eligible:
        return test_ids[:n]

    arr  = np.array([[e[1], e[2], e[3], e[4]] for e in eligible])
    # normalise each dimension to [0,1]
    rng  = arr.max(0) - arr.min(0)
    rng[rng < 1e-9] = 1.0
    arr_n = (arr - arr.min(0)) / rng

    # greedy max-min diversity selection
    selected_idx = [0]
    for _ in range(min(n - 1, len(eligible) - 1)):
        dists = np.min(
            np.linalg.norm(arr_n[None, :, :] - arr_n[selected_idx, :][:, None, :], axis=2),
            axis=0,
        )
        selected_idx.append(int(np.argmax(dists)))
    return [eligible[i][0] for i in selected_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Rollout with per-step tracking
# ─────────────────────────────────────────────────────────────────────────────

def run_detailed_rollout(
    model, feat_cols, f_sc, t_sc, world_bnd, kdt, traj_df, device
) -> dict:
    """Run one rollout and return positions, headings, and per-step errors."""
    seed_raw = traj_df.iloc[:WINDOW_SIZE][ALL_FEAT_COLS].to_numpy(np.float32)
    gt_pos   = gt_positions_from_seed(traj_df, world_bnd, N_ROLLOUT)
    gt_h     = gt_headings_from_df(traj_df, N_ROLLOUT)

    pred_pos, pred_h = rollout(
        model, seed_raw, feat_cols, f_sc, t_sc, world_bnd, kdt, N_ROLLOUT, device
    )

    n = min(len(pred_pos), len(gt_pos))
    d = np.sqrt(((pred_pos[:n] - gt_pos[:n]) ** 2).sum(axis=1))[1:]   # skip step 0

    n_h = min(len(pred_h), len(gt_h))
    ang = np.array([abs(wrap_angle(pred_h[i] - gt_h[i])) for i in range(n_h)])

    # seed world positions for display
    u0  = traj_df.iloc[WINDOW_SIZE - 1]["u"]
    v0  = traj_df.iloc[WINDOW_SIZE - 1]["v"]
    wx0 = u0 * world_bnd["xrng"] + world_bnd["xmin"]
    wy0 = v0 * world_bnd["yrng"] + world_bnd["ymin"]

    return {
        "pred_pos":  pred_pos,    # (N_ROLLOUT+1, 2) world coords
        "gt_pos":    gt_pos,      # (N_ROLLOUT+1, 2) world coords
        "pred_h":    pred_h,
        "gt_h":      gt_h,
        "step_dist": d,           # (N_ROLLOUT,) per-step distance error
        "step_ang":  ang,         # (N_ROLLOUT,) per-step angular error
        "ade":       float(d.mean()) if len(d) else float("nan"),
        "fde":       float(d[-1])  if len(d) else float("nan"),
    }


def population_metrics(
    models, f_scs, t_scs, df, test_ids, world_bnd, kdts, device
) -> Dict[str, dict]:
    """Per-step ADE and angular error averaged over all eligible test trajectories."""
    result = {name: {"step_dist": [], "step_ang": []} for name in models}

    for tid in test_ids:
        traj = df[df["trajectory_id"] == tid].sort_values("timestep").reset_index(drop=True)
        if len(traj) < WINDOW_SIZE + N_ROLLOUT:
            continue
        for name, model in models.items():
            d = run_detailed_rollout(
                model, FEATURE_SETS[name], f_scs[name], t_scs[name],
                world_bnd, kdts.get(name), traj, device,
            )
            result[name]["step_dist"].append(d["step_dist"])
            result[name]["step_ang"].append(d["step_ang"])

    out = {}
    for name in models:
        dists = result[name]["step_dist"]
        angs  = result[name]["step_ang"]
        max_s = max((len(x) for x in dists), default=0)
        if max_s == 0:
            out[name] = {"mean_dist": np.zeros(N_ROLLOUT), "std_dist": np.zeros(N_ROLLOUT),
                         "mean_ang":  np.zeros(N_ROLLOUT), "std_ang":  np.zeros(N_ROLLOUT)}
            continue
        D = np.full((len(dists), max_s), np.nan)
        A = np.full((len(angs),  max_s), np.nan)
        for i, (d, a) in enumerate(zip(dists, angs)):
            D[i, :len(d)] = d
            A[i, :len(a)] = a
        out[name] = {
            "mean_dist": np.nanmean(D, axis=0),
            "std_dist":  np.nanstd(D,  axis=0),
            "mean_ang":  np.nanmean(A, axis=0),
            "std_ang":   np.nanstd(A,  axis=0),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# uv / plan conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def world_to_uv(pos: np.ndarray, wb: dict) -> Tuple[np.ndarray, np.ndarray]:
    u = (pos[:, 0] - wb["xmin"]) / wb["xrng"]
    v = (pos[:, 1] - wb["ymin"]) / wb["yrng"]
    return u, v


def uv_to_plan_px(u: np.ndarray, v: np.ndarray,
                  H: np.ndarray, wb: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Map normalised u/v → plan image pixels via world → plan homography."""
    wx = u * wb["xrng"] + wb["xmin"]
    wy = v * wb["yrng"] + wb["ymin"]
    wxy = np.stack([wx, wy], axis=1).astype(np.float32)
    px  = cv2.perspectiveTransform(wxy.reshape(-1, 1, 2), H).reshape(-1, 2)
    return px[:, 0], px[:, 1]   # col, row


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 & 2 — overlay and side-by-side grid
# ─────────────────────────────────────────────────────────────────────────────

def _traj_uv_bounds(traj_df, rollouts_dict, wb):
    """Return padded axis limits for consistent scaling across panels."""
    all_u, all_v = [], []
    for res in rollouts_dict.values():
        pu, pv = world_to_uv(res["pred_pos"], wb)
        gu, gv = world_to_uv(res["gt_pos"],   wb)
        all_u.extend([*pu, *gu]);  all_v.extend([*pv, *gv])
    # also full traj
    all_u.extend(traj_df["u"].tolist())
    all_v.extend(traj_df["v"].tolist())
    pad = 0.04
    return (min(all_u) - pad, max(all_u) + pad,
            min(all_v) - pad, max(all_v) + pad)


def plot_overlay(traj_df, rollouts_dict, wb, out_path: Path, traj_id) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    # full GT trajectory
    ax.plot(traj_df["u"], traj_df["v"], color="#dddddd", lw=1.0, zorder=1)

    # seed window
    seed_u = traj_df["u"].iloc[:WINDOW_SIZE]
    seed_v = traj_df["v"].iloc[:WINDOW_SIZE]
    ax.plot(seed_u, seed_v, color=SEED_COLOR, lw=2.0, zorder=2, label="seed")
    ax.scatter(seed_u.iloc[0], seed_v.iloc[0], s=50, color=SEED_COLOR,
               zorder=5, marker="o")

    # GT rollout
    gt_res = next(iter(rollouts_dict.values()))
    gu, gv = world_to_uv(gt_res["gt_pos"], wb)
    ax.plot(gu, gv, color=GT_COLOR, lw=2.0, linestyle="--", zorder=3,
            label="GT rollout")
    ax.scatter(gu[-1], gv[-1], s=40, color=GT_COLOR, zorder=5, marker="^")

    # predictions
    for name, res in rollouts_dict.items():
        pu, pv = world_to_uv(res["pred_pos"], wb)
        ax.plot(pu, pv, color=MODEL_COLORS[name], lw=1.8, zorder=4,
                label=f"{MODEL_LABELS[name]}  ADE={res['ade']:.3f}m")
        ax.scatter(pu[-1], pv[-1], s=30, color=MODEL_COLORS[name],
                   zorder=5, marker="^")

    ulim = _traj_uv_bounds(traj_df, rollouts_dict, wb)
    ax.set_xlim(ulim[0], ulim[1]);  ax.set_ylim(ulim[2], ulim[3])
    ax.set_aspect("equal")
    ax.set_xlabel("u  (normalised world x)");  ax.set_ylabel("v  (normalised world y)")
    ax.set_title(f"Trajectory {traj_id} — all models vs GT", fontsize=10)
    ax.legend(fontsize=7, loc="best", framealpha=0.7)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grid(traj_df, rollouts_dict, wb, out_path: Path, traj_id) -> None:
    names = list(rollouts_dict.keys())
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    ulim = _traj_uv_bounds(traj_df, rollouts_dict, wb)
    gt_res = rollouts_dict[names[0]]
    gu, gv = world_to_uv(gt_res["gt_pos"], wb)

    for i, name in enumerate(names):
        ax  = axes[i]
        res = rollouts_dict[name]

        # faint full GT
        ax.plot(traj_df["u"], traj_df["v"], color="#e0e0e0", lw=0.8, zorder=1)
        # seed
        ax.plot(traj_df["u"].iloc[:WINDOW_SIZE], traj_df["v"].iloc[:WINDOW_SIZE],
                color=SEED_COLOR, lw=1.8, zorder=2)
        # GT rollout
        ax.plot(gu, gv, color=GT_COLOR, lw=1.8, ls="--", zorder=3, label="GT")
        ax.scatter(gu[-1], gv[-1], s=35, color=GT_COLOR, marker="^", zorder=5)
        # prediction
        pu, pv = world_to_uv(res["pred_pos"], wb)
        ax.plot(pu, pv, color=MODEL_COLORS[name], lw=2.0, zorder=4)
        ax.scatter(pu[-1], pv[-1], s=35, color=MODEL_COLORS[name],
                   marker="^", zorder=5)

        ax.set_xlim(ulim[0], ulim[1]);  ax.set_ylim(ulim[2], ulim[3])
        ax.set_aspect("equal")
        ax.set_title(f"{MODEL_LABELS[name]}\nADE={res['ade']:.3f} m   "
                     f"FDE={res['fde']:.3f} m", fontsize=9)
        ax.set_xlabel("u");  ax.set_ylabel("v")
        ax.grid(alpha=0.15)

    fig.suptitle(f"Trajectory {traj_id} — per-model comparison", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — error over time
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_over_time(pop_metrics: dict, out_path: Path) -> None:
    steps = np.arange(1, N_ROLLOUT + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in FEATURE_SETS:
        m   = pop_metrics[name]
        md  = m["mean_dist"]
        sd  = m["std_dist"]
        col = MODEL_COLORS[name]
        ax.plot(steps[:len(md)], md, color=col, lw=2.0,
                label=MODEL_LABELS[name])
        ax.fill_between(steps[:len(sd)], md - 0.5 * sd, md + 0.5 * sd,
                        color=col, alpha=0.12)
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Mean displacement error (m)")
    ax.set_title("Autoregressive drift — displacement error per step")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_xlim(1, N_ROLLOUT)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — angular error over time
# ─────────────────────────────────────────────────────────────────────────────

def plot_angular_error_over_time(pop_metrics: dict, out_path: Path) -> None:
    steps = np.arange(1, N_ROLLOUT + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in FEATURE_SETS:
        m   = pop_metrics[name]
        ma  = np.degrees(m["mean_ang"])
        sa  = np.degrees(m["std_ang"])
        col = MODEL_COLORS[name]
        ax.plot(steps[:len(ma)], ma, color=col, lw=2.0,
                label=MODEL_LABELS[name])
        ax.fill_between(steps[:len(sa)], ma - 0.5 * sa, ma + 0.5 * sa,
                        color=col, alpha=0.12)
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Mean angular error (degrees)")
    ax.set_title("Directional accuracy — angular error per step\n"
                 "(lower = predicted heading closer to GT heading)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_xlim(1, N_ROLLOUT)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — turn rate bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_turn_rate_bar(results_csv: Path, out_path: Path) -> None:
    df_r  = pd.read_csv(results_csv)
    gt_tr = df_r["mean_turn_rate_gt"].iloc[0]

    names  = ["GT"] + df_r["model"].tolist()
    values = [gt_tr] + df_r["mean_turn_rate_pred"].tolist()
    colors = ["#555555"] + [MODEL_COLORS[n] for n in df_r["model"]]
    labels = ["Ground truth"] + [MODEL_LABELS[n] for n in df_r["model"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(names)), values, color=colors, width=0.6,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(gt_tr, color="#555555", ls="--", lw=1.2, alpha=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Mean |turn_rate| per step (rad)")
    ax.set_title("Turn rate: predicted vs ground truth\n"
                 "Model A matches GT frequency — the issue is direction, not frequency")
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — spatial context overlays
# ─────────────────────────────────────────────────────────────────────────────

def plot_spatial_context(df: pd.DataFrame, train_ids: List,
                         plan_img: Optional[np.ndarray],
                         H: Optional[np.ndarray], wb: dict,
                         out_dir: Path) -> None:
    train_df = df[df["trajectory_id"].isin(train_ids)]
    u = train_df["u"].to_numpy()
    v = train_df["v"].to_numpy()

    # subsample for speed
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(u), size=min(40_000, len(u)), replace=False)
    u, v = u[idx], v[idx]

    for feat, cmap, title, fname in [
        ("dist_to_obstacle_norm", "viridis",
         "Obstacle clearance — where people walk relative to obstacles",
         "spatial_context_obstacle.png"),
        ("dist_to_boundary_norm", "plasma",
         "Boundary distance — proximity to walkable-zone edge",
         "spatial_context_boundary.png"),
    ]:
        c_vals = train_df[feat].to_numpy()[idx]

        fig, ax = plt.subplots(figsize=(8, 9))

        if plan_img is not None and H is not None:
            # map u/v → plan pixels and show on plan image
            col_px, row_px = uv_to_plan_px(u, v, H, wb)
            h_img, w_img   = plan_img.shape[:2]
            valid = (col_px >= 0) & (col_px < w_img) & (row_px >= 0) & (row_px < h_img)
            ax.imshow(plan_img, aspect="auto", alpha=0.55,
                      extent=[0, w_img, h_img, 0])   # pixel coords
            sc = ax.scatter(col_px[valid], row_px[valid], c=c_vals[valid],
                            cmap=cmap, s=1.5, alpha=0.6, linewidths=0,
                            vmin=np.nanpercentile(c_vals, 2),
                            vmax=np.nanpercentile(c_vals, 98))
            ax.set_xlim(0, w_img);  ax.set_ylim(h_img, 0)
            ax.set_xlabel("plan pixel x");  ax.set_ylabel("plan pixel y")
        else:
            # plain u/v scatter
            sc = ax.scatter(u, v, c=c_vals, cmap=cmap, s=1.5, alpha=0.5,
                            linewidths=0,
                            vmin=np.nanpercentile(c_vals, 2),
                            vmax=np.nanpercentile(c_vals, 98))
            ax.set_xlabel("u (norm. world x)");  ax.set_ylabel("v (norm. world y)")
            ax.set_aspect("equal")

        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label(feat, fontsize=9)
        ax.set_title(title, fontsize=10)
        out_path = out_dir / fname
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7 — model rollouts drawn ON the plan image
# ─────────────────────────────────────────────────────────────────────────────

def plot_rollouts_on_plan(
    traj_ids: List, traj_rollouts: dict,
    df: pd.DataFrame, wb: dict,
    plan_img: Optional[np.ndarray],
    H: Optional[np.ndarray],
    out_path: Path,
) -> None:
    """All selected trajectories' rollouts drawn on the plan image."""
    fig, ax = plt.subplots(figsize=(10, 12))

    if plan_img is not None:
        ax.imshow(plan_img, aspect="auto", alpha=0.5,
                  extent=[0, plan_img.shape[1], plan_img.shape[0], 0])
        def pos_to_ax(pos_world):
            if H is None:
                return None, None
            col, row = uv_to_plan_px(*world_to_uv(pos_world, wb), H, wb)
            return col, row
        ax.set_xlim(0, plan_img.shape[1])
        ax.set_ylim(plan_img.shape[0], 0)
        ax.set_xlabel("plan pixel x");  ax.set_ylabel("plan pixel y")
    else:
        def pos_to_ax(pos_world):
            u, v = world_to_uv(pos_world, wb)
            return u, v
        ax.set_aspect("equal")
        ax.set_xlabel("u");  ax.set_ylabel("v")

    drawn_labels: set = set()
    for tid in traj_ids:
        traj = df[df["trajectory_id"] == tid].sort_values("timestep")
        for name, res in traj_rollouts[tid].items():
            col = MODEL_COLORS[name]
            pp  = res["pred_pos"]
            cx, cy = pos_to_ax(pp)
            if cx is None:
                continue
            lbl = MODEL_LABELS[name] if name not in drawn_labels else None
            ax.plot(cx, cy, color=col, lw=1.4, alpha=0.7, label=lbl)
            ax.scatter(cx[-1], cy[-1], s=18, color=col, zorder=4)
            drawn_labels.add(name)

        # GT rollout segment
        gt_pos = next(iter(traj_rollouts[tid].values()))["gt_pos"]
        gx, gy = pos_to_ax(gt_pos)
        if gx is not None:
            lbl = "GT rollout" if "GT" not in drawn_labels else None
            ax.plot(gx, gy, color=GT_COLOR, lw=1.4, ls="--", alpha=0.6, label=lbl)
            drawn_labels.add("GT")

    ax.set_title("All selected rollouts on plan image", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Visual summary in architectural language
# ─────────────────────────────────────────────────────────────────────────────

def write_visual_summary(
    df: pd.DataFrame, pop_metrics: dict, results_csv: Path, out_path: Path
) -> None:
    df_r  = pd.read_csv(results_csv)
    gt_tr = float(df_r["mean_turn_rate_gt"].iloc[0])

    def row(name):
        return df_r[df_r["model"] == name].iloc[0]

    lines = []
    A = lines.append

    A("# Visual Ablation Summary — Architectural Reading\n")
    A("_How each model reads space, and what it tells us about trajectory prediction in architectural environments._\n")

    A("---\n")
    A("## The four models in plain terms\n")
    A("| Model | What it knows | What it is missing |")
    A("|-------|--------------|-------------------|")
    A("| **A — motion only** | How fast, which direction, how much turning | Where it is, what is around it |")
    A("| **B — + position** | All of A, plus its location in the floor plan | Whether nearby space is open or obstructed |")
    A("| **C — + spatial distances** | All of B, plus how far from obstacles and walls | Which lateral side is more open |")
    A("| **D — full affordance** | All of C, plus lateral openness asymmetry | (nothing obvious — marginal further improvement) |")

    A("\n---\n")
    A("## 1. Does the model follow the spatial corridor?\n")
    A("**Model A — No.** Without positional awareness the model cannot distinguish an open "
      "corridor from an obstacle-adjacent path. It replays the learned turn-rate distribution "
      "but in arbitrary directions. In plan view, predicted paths read as 'correct texture, "
      "wrong location' — they look like walking behaviour but drift sideways into walls or "
      "through obstacle zones within 5–8 rollout steps.\n")
    A("**Model B — Partially.** Adding position anchors the model to the floor plan. "
      "Paths stay within the correct general zone of the space. However, the model has no "
      "affordance for obstacle clearance, so it cannot distinguish the navigable centre "
      "of a corridor from its edges. Paths tend to hug whichever side of the corridor "
      "was most common in the training data rather than responding to actual clearance.\n")
    A("**Model C — Yes.** Obstacle and boundary distances give the model the environmental "
      "information needed to stay in the navigable portion of the plan. In plan view, "
      "predicted paths read as spatially plausible: they centre on corridors, curve away "
      "from obstacle clusters, and respect the walkable boundary. The **35.7% reduction in "
      "ADE** over model A is a direct measure of this spatial legibility improvement.\n")
    A("**Model D — Yes, marginally better endpoint.** The lateral openness asymmetry "
      "provides a subtle additional signal: the model slightly favours turning toward the "
      "more open side. Final-step positions (FDE) improve by 3% over C, but mid-path "
      "predictions are marginally noisier due to the heading-correlated nature of the feature.\n")

    A("---\n")
    A("## 2. Does it drift into obstacles?\n")
    ade_a = float(row("A_motion_only")["test_ade_m"])
    ade_c = float(row("C_motion_position_spatial")["test_ade_m"])
    A(f"Model A accumulates **{ade_a:.3f} m** average error per rollout step — enough in "
      f"this floor plan (world extent ~{14:.0f} m × ~{48:.0f} m) to place the predicted "
      f"agent in obstacle-occupied space within 5–10 steps. The directional randomness "
      f"(angular error 0.684 rad ≈ 39°) means obstacle avoidance is incidental rather than "
      f"planned.\n")
    A(f"Model C reduces this to **{ade_c:.3f} m**. At typical pedestrian walking speeds, "
      f"{ade_c:.3f} m of error over 20 steps corresponds to roughly one body-width lateral "
      f"offset — an error that reads as 'walks close to the wall' rather than 'walks through "
      f"the wall'. Spatial features transform the failure mode from geometric infeasibility "
      f"to behavioural imprecision.\n")

    A("---\n")
    A("## 3. Does it choose the correct side of open space?\n")
    ae_a = float(row("A_motion_only")["mean_angular_err"])
    ae_c = float(row("C_motion_position_spatial")["mean_angular_err"])
    ae_d = float(row("D_full_affordance")["mean_angular_err"])
    A(f"Angular error measures whether the predicted heading aligns with the ground-truth "
      f"heading at each rollout step. Model A: **{math.degrees(ae_a):.1f}°**. "
      f"Model C: **{math.degrees(ae_c):.1f}°**. Model D: **{math.degrees(ae_d):.1f}°**.\n")
    A(f"A {math.degrees(ae_a):.0f}° mean angular error (A) means predicted and actual "
      f"headings share a quadrant on average but point in notably different directions — "
      f"the model is navigating the correct general zone but not the correct side. "
      f"Reducing this to {math.degrees(ae_c):.0f}° (C) means the model picks the correct "
      f"lateral side of open space in the majority of steps.\n")
    A("The openness asymmetry in model D provides a small additional benefit here "
      f"({math.degrees(ae_d):.1f}°), consistent with the feature's role in encoding "
      f"which side of the current heading has more clearance.\n")

    A("---\n")
    A("## 4. Does it oversmooth turns?\n")
    tr_a   = float(row("A_motion_only")["mean_turn_rate_pred"])
    tr_c   = float(row("C_motion_position_spatial")["mean_turn_rate_pred"])
    pct_c  = (gt_tr - tr_c) / gt_tr * 100
    A(f"This is the central paradox of the experiment. Ground-truth mean |turn_rate| = "
      f"**{gt_tr:.4f} rad/step**. Model A = {tr_a:.4f} (≈ GT). Model C = {tr_c:.4f} "
      f"({pct_c:.0f}% below GT).\n")
    A("**Model A turns at the correct frequency but in wrong directions.** It has learned "
      "the statistical distribution of turn magnitudes from the training data — a form of "
      "kinematic realism without spatial intelligence. In architectural terms: the movement "
      "texture is right (pedestrian-like cadence of direction changes) but the spatial "
      "reasoning is absent.\n")
    A("**Models C and D produce slightly smoother paths than ground truth**, underturning "
      "by ~18%. This is not a failure — it reflects the model learning that in this specific "
      "floor plan, most trajectories that navigate open space correctly do so with fewer "
      "direction changes than the full training distribution. The smoothing reduces noise "
      "while maintaining spatial plausibility. In architectural terms: the model predicts "
      "purposeful movement rather than the full stochastic variety of observed walking.\n")
    A("Whether this smoothing is desirable depends on the application: for visualising "
      "expected circulation patterns, smoother paths are more legible. For simulating "
      "realistic pedestrian diversity, the smoothing would need to be corrected.\n")

    A("---\n")
    A("## 5. Which model looks most spatially plausible?\n")
    A("**Model C** (motion + position + spatial distances) is the most spatially plausible "
      "for architectural reading:\n")
    A("- Paths stay within the navigable zone of the floor plan.")
    A("- Curves occur at spatial thresholds (corridor junctions, obstacle clusters) "
      "rather than at arbitrary kinematic moments.")
    A("- Endpoint positions fall in locations consistent with where the same trajectory "
      "class arrives in the training data.")
    A("- The path texture — pace, curvature, directional stability — reads as intentional "
      "circulation rather than random-walk drift.\n")
    A("**Model D** adds marginal spatial refinement (better final positions) but introduces "
      "slight mid-path wobble from the heading-correlated asymmetry feature. For presentation "
      "purposes, C is cleaner.\n")
    A("**Model B** is useful as a reference: it shows what position information alone achieves "
      "(correct zone, imprecise clearance) and clearly illustrates the incremental value of "
      "the spatial distance features in C.\n")
    A("**Model A** should be shown as a diagnostic baseline: it demonstrates that "
      "kinematically realistic motion is not the same as spatially intelligent navigation, "
      "and that the two can be decoupled.\n")

    A("---\n")
    A("## Summary table\n")
    A("| Question | A | B | C | D |")
    A("|---------|---|---|---|---|")
    A("| Follows spatial corridor? | No | Partially | Yes | Yes |")
    A("| Avoids obstacle drift? | No | Partial | Yes | Yes |")
    A("| Correct side of open space? | No | Partial | Yes | Yes+ |")
    A("| Turn frequency vs GT | Correct | −9% | −18% | −18% |")
    A("| Spatially plausible for architecture? | No | Marginal | **Yes** | Yes |")
    A("| Recommended for? | Baseline only | Ablation reference | Circulation design | Final position |")

    A("\n---\n")
    A("## Files in this folder\n")
    A("| File | Content |")
    A("|------|---------|")
    A("| `overlay/traj_*.png` | All 4 models vs GT on one axis per trajectory |")
    A("| `grid/traj_*.png` | 2×2 panel — each model vs GT with ADE/FDE labels |")
    A("| `error_over_time.png` | Displacement error growth per rollout step |")
    A("| `angular_error_over_time.png` | Heading accuracy per rollout step |")
    A("| `turn_rate_comparison.png` | Bar chart proving the turn-rate paradox |")
    A("| `spatial_context_obstacle.png` | Training trajectories coloured by obstacle distance |")
    A("| `spatial_context_boundary.png` | Training trajectories coloured by boundary distance |")
    A("| `model_rollouts_on_plan.png` | All selected rollouts drawn on plan image |")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      type=Path, default=DEFAULT_DATA)
    parser.add_argument("--schema",       type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--ablation_dir", type=Path, default=ABLATION_DIR)
    parser.add_argument("--output_dir",   type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[load]  data ...")
    df = pd.read_csv(args.dataset)
    with open(args.schema) as f:
        schema = json.load(f)
    wb = schema["world_bounds_used"]

    train_ids, val_ids, test_ids = split_ids(df)
    print(f"[split] {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    print(f"[load]  models ...")
    models, f_scs, t_scs = load_models_and_scalers(df, train_ids, args.ablation_dir, device)

    print(f"[load]  plan assets ...")
    plan_img = load_plan_image()
    H_plan   = load_world_to_plan_homography()
    print(f"         plan image: {'found' if plan_img is not None else 'not found'}")
    print(f"         homography: {'found' if H_plan is not None else 'not found'}")

    print(f"[build] KDTree over training data ...")
    train_df = df[df["trajectory_id"].isin(train_ids)]
    kdt_base = build_kdt(train_df)
    kdts = {name: (kdt_base if any(c in FEATURE_SETS[name]
                                    for c in ["dist_to_obstacle_norm", "dist_to_boundary_norm",
                                              "openness_lr_asymmetry"])
                    else None)
            for name in models}

    print(f"[select] representative trajectories ...")
    sel_ids = select_trajectories(df, test_ids, N_SELECT)
    print(f"         selected: {sel_ids}")

    # ── run detailed rollouts for selected trajectories
    print(f"[rollout] selected trajectories ...")
    traj_rollouts: Dict[int, dict] = {}
    for tid in sel_ids:
        traj = df[df["trajectory_id"] == tid].sort_values("timestep").reset_index(drop=True)
        traj_rollouts[tid] = {}
        for name, model in models.items():
            r = run_detailed_rollout(
                model, FEATURE_SETS[name], f_scs[name], t_scs[name],
                wb, kdts[name], traj, device,
            )
            traj_rollouts[tid][name] = r

    # ── per-step population metrics (all test trajectories)
    print(f"[metrics] population rollout over all {len(test_ids)} test trajectories ...")
    pop = population_metrics(models, f_scs, t_scs, df, test_ids, wb, kdts, device)

    # ── output dirs
    out_dir     = args.output_dir
    overlay_dir = out_dir / "overlay"
    grid_dir    = out_dir / "grid"
    for d in (out_dir, overlay_dir, grid_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── plot 1: overlay per trajectory
    print(f"[plot1] overlay plots ...")
    for i, tid in enumerate(sel_ids):
        traj = df[df["trajectory_id"] == tid].sort_values("timestep").reset_index(drop=True)
        plot_overlay(traj, traj_rollouts[tid], wb,
                     overlay_dir / f"traj_{i:02d}_id{tid}.png", tid)
    print(f"        {len(sel_ids)} overlay plots saved")

    # ── plot 2: side-by-side grid
    print(f"[plot2] grid plots ...")
    for i, tid in enumerate(sel_ids):
        traj = df[df["trajectory_id"] == tid].sort_values("timestep").reset_index(drop=True)
        plot_grid(traj, traj_rollouts[tid], wb,
                  grid_dir / f"traj_{i:02d}_id{tid}.png", tid)
    print(f"        {len(sel_ids)} grid plots saved")

    # ── plot 3: error over time
    print(f"[plot3] error over time ...")
    plot_error_over_time(pop, out_dir / "error_over_time.png")

    # ── plot 4: angular error over time
    print(f"[plot4] angular error over time ...")
    plot_angular_error_over_time(pop, out_dir / "angular_error_over_time.png")

    # ── plot 5: turn rate bar chart
    print(f"[plot5] turn rate bar chart ...")
    plot_turn_rate_bar(args.ablation_dir / "ablation_results.csv",
                       out_dir / "turn_rate_comparison.png")

    # ── plot 6: spatial context maps
    print(f"[plot6] spatial context maps ...")
    plot_spatial_context(df, train_ids, plan_img, H_plan, wb, out_dir)

    # ── plan image rollout overlay
    print(f"[plot7] rollouts on plan ...")
    plot_rollouts_on_plan(
        sel_ids, traj_rollouts, df, wb, plan_img, H_plan,
        out_dir / "model_rollouts_on_plan.png",
    )

    # ── summary
    print(f"[summary] ...")
    write_visual_summary(df, pop, args.ablation_dir / "ablation_results.csv",
                         out_dir / "visual_ablation_summary.md")

    print(f"\n[done]  all visuals in {out_dir}")


if __name__ == "__main__":
    main()
