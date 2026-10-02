"""
common.py — shared utilities for the Barcelona Model C training wrapper.

Reuses FROZEN components by importing them from the sandbox bridge experiment
(run_bridge_ablation.py). NOTHING in the frozen source is modified.

Imported frozen symbols (single source of truth for architecture + recipe):
    TrajectoryLSTM, ColumnScaler, WindowDataset, make_windows,
    rollout, gt_positions_from_seed, gt_headings_from_df,
    build_kdt, kdt_lookup, wrap_angle,
    WINDOW_SIZE, N_ROLLOUT, HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    BATCH_SIZE, MAX_EPOCHS, LR, LR_STEP, LR_GAMMA, PATIENCE, SEED
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- frozen source on path (read-only import) ---
MP_ROOT = Path(r"C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels")
FROZEN_DIR = MP_ROOT / "mp-core" / "trajectory-prediction" / "experiments" / "schema_ablation_bridge"
sys.path.insert(0, str(FROZEN_DIR))
import run_bridge_ablation as fz  # noqa: E402

# --- paths ---
DATASET_DIR = Path(r"C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_master_dataset")
MODEL_C_CSV = DATASET_DIR / "model_C_dataset.csv"
MASTER_CSV = DATASET_DIR / "master_dataset.csv"
MANIFEST = DATASET_DIR / "manifest.json"

OUT = Path(r"C:\Users\OWNER\Desktop\MotionPixels_Barcelona_Training_Visuals")
D_AUDIT = OUT / "00_dataset_audit"
D_TRAIN = OUT / "01_training_run"
D_VAL = OUT / "02_validation_rollouts"
D_TEST = OUT / "03_test_rollouts"
D_METRICS = OUT / "04_metrics"
D_THESIS = OUT / "05_thesis_figures"
D_REPORTS = OUT / "06_reports"
D_MODELS = OUT / "models"

# --- frozen recipe constants (re-exported, never re-typed) ---
WINDOW_SIZE = fz.WINDOW_SIZE
N_ROLLOUT = fz.N_ROLLOUT
FEAT_COLS = list(fz.FEATURE_SETS["C_motion_position_spatial"])   # 10 features
TARGET_COLS = list(fz.TARGET_COLS)                               # target_du, target_dv
SPATIAL_COLS = ["dist_to_obstacle_norm", "dist_to_boundary_norm"]
ALL_FEAT_COLS = list(FEAT_COLS)   # Model C feat_cols already contain u,v,heading_*

SPLIT_MAP = {
    "esplanade_espanya_01": "train",
    "placa_catalunya_01": "train",
    "placa_espanya_01": "train",
    "stairs_montjuic_01": "val",
    "red_bridge_combined_01": "test",
}
VAL_REC = "stairs_montjuic_01"
TEST_REC = "red_bridge_combined_01"

# thesis-readable style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 160, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3, "axes.titlesize": 11,
})

COL_SEED = "#444444"
COL_GT = "#1f77b4"
COL_PRED = "#d62728"


def load_world_bounds() -> dict:
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {r["recording_id"]: r["world_bounds"] for r in m["recordings"]}


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(MODEL_C_CSV)
    return df


def save_scalers(path: Path, f_sc, t_sc, feat_cols, tgt_cols) -> None:
    blob = {
        "feat_means": np.asarray(f_sc.means).tolist(),
        "feat_stds": np.asarray(f_sc.stds).tolist(),
        "feat_cols": list(feat_cols),
        "tgt_means": np.asarray(t_sc.means).tolist(),
        "tgt_stds": np.asarray(t_sc.stds).tolist(),
        "tgt_cols": list(tgt_cols),
    }
    path.write_bytes(pickle.dumps(blob))


def load_scalers(path: Path):
    blob = pickle.loads(Path(path).read_bytes())
    f_sc = fz.ColumnScaler()
    f_sc.means = np.asarray(blob["feat_means"]); f_sc.stds = np.asarray(blob["feat_stds"])
    f_sc.cols = blob["feat_cols"]
    t_sc = fz.ColumnScaler()
    t_sc.means = np.asarray(blob["tgt_means"]); t_sc.stds = np.asarray(blob["tgt_stds"])
    t_sc.cols = blob["tgt_cols"]
    return f_sc, t_sc


def load_model(ckpt_path: Path, device):
    import torch
    model = fz.TrajectoryLSTM(len(FEAT_COLS)).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def build_recording_kdt(df: pd.DataFrame, rec: str):
    rec_df = df[df["recording_id"] == rec]
    return fz.build_kdt(rec_df, SPATIAL_COLS)


def rollout_one_trajectory(model, traj_df: pd.DataFrame, world_bnd: dict,
                           f_sc, t_sc, kdt, device, max_steps: int = None):
    """Autoregressive rollout for one trajectory using the FROZEN rollout().
    Returns dict with seed/gt/pred world positions + ADE/FDE/angular metrics,
    or None if the trajectory is too short to roll out."""
    import math
    max_steps = N_ROLLOUT if max_steps is None else max_steps
    traj_df = traj_df.sort_values("timestep").reset_index(drop=True)
    future = len(traj_df) - WINDOW_SIZE
    if future < 1:
        return None
    n_steps = int(min(max_steps, future))

    seed_raw = traj_df.iloc[:WINDOW_SIZE][ALL_FEAT_COLS].to_numpy(np.float32)
    pred_pos, pred_h = fz.rollout(
        model, seed_raw, FEAT_COLS, ALL_FEAT_COLS, SPATIAL_COLS,
        f_sc, t_sc, world_bnd, kdt, n_steps, device)
    gt_pos = fz.gt_positions_from_seed(traj_df, world_bnd, n_steps)
    gt_h = fz.gt_headings_from_df(traj_df, n_steps)

    # seed/history world positions (the 10 observed frames)
    su = traj_df.iloc[:WINDOW_SIZE]["u"].to_numpy()
    sv = traj_df.iloc[:WINDOW_SIZE]["v"].to_numpy()
    seed_pos = np.column_stack([su * world_bnd["xrng"] + world_bnd["xmin"],
                                sv * world_bnd["yrng"] + world_bnd["ymin"]])

    n = min(len(pred_pos), len(gt_pos))
    # per-step Euclidean error, excluding the shared seed anchor (index 0)
    d = np.sqrt(((pred_pos[:n] - gt_pos[:n]) ** 2).sum(axis=1))[1:]
    if len(d) == 0:
        return None
    ade = float(d.mean()); fde = float(d[-1])
    n_h = min(len(pred_h), len(gt_h))
    ang = float(np.mean([abs(fz.wrap_angle(pred_h[i] - gt_h[i])) for i in range(n_h)])) if n_h else float("nan")
    tr_pred = float(np.mean([abs(fz.wrap_angle(pred_h[i] - pred_h[i-1])) for i in range(1, len(pred_h))])) if len(pred_h) > 1 else float("nan")
    tr_gt = float(np.mean([abs(fz.wrap_angle(gt_h[i] - gt_h[i-1])) for i in range(1, len(gt_h))])) if len(gt_h) > 1 else float("nan")

    return {
        "trajectory_id": str(traj_df["trajectory_id"].iloc[0]),
        "recording_id": str(traj_df["recording_id"].iloc[0]),
        "split": str(traj_df["split"].iloc[0]),
        "length": int(len(traj_df)),
        "n_steps": n_steps,
        "seed_pos": seed_pos, "gt_pos": gt_pos, "pred_pos": pred_pos,
        "per_step_err": d,
        "ade": ade, "fde": fde, "angular_error": ang,
        "mean_turn_pred": tr_pred, "mean_turn_gt": tr_gt,
    }


def plot_rollout(ax, r: dict, title: str = None):
    """Draw one rollout on an axis with seed/gt/pred + start/end markers."""
    seed, gt, pred = r["seed_pos"], r["gt_pos"], r["pred_pos"]
    ax.plot(seed[:, 0], seed[:, 1], "-", color=COL_SEED, lw=1.6, label="seed/history")
    ax.plot(gt[:, 0], gt[:, 1], "-o", color=COL_GT, lw=1.6, ms=2.5, label="ground truth")
    ax.plot(pred[:, 0], pred[:, 1], "--s", color=COL_PRED, lw=1.6, ms=2.5, label="prediction")
    ax.plot(seed[0, 0], seed[0, 1], "*", color="black", ms=12, label="start")
    ax.plot(gt[-1, 0], gt[-1, 1], "o", color=COL_GT, ms=9, mfc="none", mew=2, label="true end")
    ax.plot(pred[-1, 0], pred[-1, 1], "s", color=COL_PRED, ms=9, mfc="none", mew=2, label="pred end")
    ax.set_aspect("equal", adjustable="datalim")
    if title is None:
        title = (f"{r['trajectory_id']}  |  ADE={r['ade']:.2f}m  "
                 f"FDE={r['fde']:.2f}m  ang={np.degrees(r['angular_error']):.0f}deg")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("world_x (m)"); ax.set_ylabel("world_y (m)")
