"""
MOTION PIXELS - PHASE 5: V3 GRU vs LSTM Model C.

Minimal architecture change: LSTM cell -> GRU cell. EVERYTHING else (features,
targets, split, window, hidden size, layers, dropout, optimizer, lr, batch,
early stopping, loss, scalers, rollout, metrics) is identical to the Phase-4
V3 LSTM Model C, by reusing the FROZEN recipe symbols from run_bridge_ablation.py
(imported read-only). Nothing in the frozen recipe, Model C, Phase-4 outputs, or
datasets is modified.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# --- frozen recipe (read-only import; single source of truth for the recipe) ---
MP_ROOT = Path(r"C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels")
FROZEN_DIR = MP_ROOT / "mp-core" / "trajectory-prediction" / "experiments" / "schema_ablation_bridge"
sys.path.insert(0, str(FROZEN_DIR))
import run_bridge_ablation as fz  # noqa: E402

# --- paths ---
V3_DATASET = Path(r"C:\Users\OWNER\Desktop\new_datasets\Barcelona_v3_manual_master_dataset")
MODEL_C_CSV = V3_DATASET / "model_C_dataset.csv"
MANIFEST = V3_DATASET / "manifest.json"
PHASE4 = Path(r"C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\11_phase4_v3_model_c_training")
OUT = Path(r"C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\12_phase5_v3_gru_comparison")
D_CODE = OUT / "00_code_snapshot"
D_TRAIN = OUT / "01_training_run"
D_MODELS = OUT / "02_models"
D_METRICS = OUT / "03_metrics"
D_FIG = OUT / "04_figures"
D_REPORTS = OUT / "05_reports"

# --- recipe constants (re-exported from frozen, never re-typed) ---
WINDOW_SIZE = fz.WINDOW_SIZE
N_ROLLOUT = fz.N_ROLLOUT
HIDDEN_SIZE = fz.HIDDEN_SIZE
NUM_LAYERS = fz.NUM_LAYERS
DROPOUT = fz.DROPOUT
FEAT_COLS = list(fz.FEATURE_SETS["C_motion_position_spatial"])  # 10 features
TARGET_COLS = list(fz.TARGET_COLS)
SPATIAL_COLS = ["dist_to_obstacle_norm", "dist_to_boundary_norm"]
ALL_FEAT_COLS = list(FEAT_COLS)

SPLIT_MAP = {
    "esplanade_espanya_01": "train", "placa_catalunya_01": "train",
    "placa_espanya_01": "train", "stairs_montjuic_01": "val",
    "red_bridge_combined_01": "test",
}
VAL_REC = "stairs_montjuic_01"
TEST_REC = "red_bridge_combined_01"


# ─────────────────────────────────────────────────────────────────────────────
# GRU Model — the ONLY change vs frozen TrajectoryLSTM
# ─────────────────────────────────────────────────────────────────────────────
class TrajectoryGRU(nn.Module):
    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_feat, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.head = nn.Linear(HIDDEN_SIZE, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def load_world_bounds() -> dict:
    import json
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {r["recording_id"]: r["world_bounds"] for r in m["recordings"]}


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(MODEL_C_CSV)


def save_scalers(path: Path, f_sc, t_sc) -> None:
    import pickle
    blob = {"feat_means": np.asarray(f_sc.means).tolist(),
            "feat_stds": np.asarray(f_sc.stds).tolist(), "feat_cols": list(FEAT_COLS),
            "tgt_means": np.asarray(t_sc.means).tolist(),
            "tgt_stds": np.asarray(t_sc.stds).tolist(), "tgt_cols": list(TARGET_COLS)}
    path.write_bytes(pickle.dumps(blob))


def load_scalers(path: Path):
    import pickle
    blob = pickle.loads(Path(path).read_bytes())
    f = fz.ColumnScaler(); f.means = np.asarray(blob["feat_means"])
    f.stds = np.asarray(blob["feat_stds"]); f.cols = blob["feat_cols"]
    t = fz.ColumnScaler(); t.means = np.asarray(blob["tgt_means"])
    t.stds = np.asarray(blob["tgt_stds"]); t.cols = blob["tgt_cols"]
    return f, t


def load_gru(ckpt: Path, device):
    m = TrajectoryGRU(len(FEAT_COLS)).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)); m.eval()
    return m


def load_lstm(ckpt: Path, device):
    m = fz.TrajectoryLSTM(len(FEAT_COLS)).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)); m.eval()
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Rollout (one trajectory) — copied from the Barcelona common.py logic, which
# itself calls the FROZEN fz.rollout(). Works for any model with the same
# forward signature (LSTM or GRU).
# ─────────────────────────────────────────────────────────────────────────────
def rollout_one_trajectory(model, traj_df, world_bnd, f_sc, t_sc, kdt, device,
                           max_steps=None):
    max_steps = N_ROLLOUT if max_steps is None else max_steps
    traj_df = traj_df.sort_values("timestep").reset_index(drop=True)
    future = len(traj_df) - WINDOW_SIZE
    if future < 1:
        return None
    n_steps = int(min(max_steps, future))
    seed_raw = traj_df.iloc[:WINDOW_SIZE][ALL_FEAT_COLS].to_numpy(np.float32)
    pred_pos, pred_h = fz.rollout(model, seed_raw, FEAT_COLS, ALL_FEAT_COLS,
                                  SPATIAL_COLS, f_sc, t_sc, world_bnd, kdt, n_steps, device)
    gt_pos = fz.gt_positions_from_seed(traj_df, world_bnd, n_steps)
    gt_h = fz.gt_headings_from_df(traj_df, n_steps)
    su = traj_df.iloc[:WINDOW_SIZE]["u"].to_numpy()
    sv = traj_df.iloc[:WINDOW_SIZE]["v"].to_numpy()
    seed_pos = np.column_stack([su * world_bnd["xrng"] + world_bnd["xmin"],
                                sv * world_bnd["yrng"] + world_bnd["ymin"]])
    n = min(len(pred_pos), len(gt_pos))
    d = np.sqrt(((pred_pos[:n] - gt_pos[:n]) ** 2).sum(axis=1))[1:]
    if len(d) == 0:
        return None
    n_h = min(len(pred_h), len(gt_h))
    ang = float(np.mean([abs(fz.wrap_angle(pred_h[i] - gt_h[i])) for i in range(n_h)])) if n_h else float("nan")
    tr_pred = float(np.mean([abs(fz.wrap_angle(pred_h[i] - pred_h[i-1])) for i in range(1, len(pred_h))])) if len(pred_h) > 1 else float("nan")
    tr_gt = float(np.mean([abs(fz.wrap_angle(gt_h[i] - gt_h[i-1])) for i in range(1, len(gt_h))])) if len(gt_h) > 1 else float("nan")
    return {"trajectory_id": str(traj_df["trajectory_id"].iloc[0]),
            "recording_id": str(traj_df["recording_id"].iloc[0]),
            "split": str(traj_df["split"].iloc[0]), "length": int(len(traj_df)),
            "n_steps": n_steps, "seed_pos": seed_pos, "gt_pos": gt_pos,
            "pred_pos": pred_pos, "per_step_err": d, "ade": float(d.mean()),
            "fde": float(d[-1]), "angular_error": ang,
            "mean_turn_pred": tr_pred, "mean_turn_gt": tr_gt}


def teacher_forced(df, f_sc, t_sc, model, split, device):
    """Single-step teacher-forced prediction over ALL windows in a split.
    Copied from forensic.teacher_forced (frozen helpers)."""
    ids = df[df.split == split].trajectory_id.unique().tolist()
    X, Y = fz.make_windows(df, FEAT_COLS, ids)
    Xs = f_sc.transform(X.reshape(-1, len(FEAT_COLS))).reshape(X.shape).astype(np.float32)
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(Xs), 8192):
            xb = torch.tensor(Xs[i:i+8192], device=device)
            preds.append(model(xb).cpu().numpy())
    P = t_sc.inverse(np.concatenate(preds))
    pdu, pdv = P[:, 0], P[:, 1]
    tdu, tdv = Y[:, 0], Y[:, 1]
    step_err = np.hypot(pdu - tdu, pdv - tdv)
    moving = (np.hypot(tdu, tdv) > 1e-3) & (np.hypot(pdu, pdv) > 1e-3)
    ph = np.arctan2(pdv[moving], pdu[moving]); th = np.arctan2(tdv[moving], tdu[moving])
    ang = np.abs(np.arctan2(np.sin(ph - th), np.cos(ph - th)))
    return {"split": split, "n_windows": int(len(P)),
            "tf_step_err_m": float(step_err.mean()),
            "tf_step_median_m": float(np.median(step_err)),
            "tf_angular_deg": float(np.degrees(ang.mean())),
            "tf_angular_median_deg": float(np.degrees(np.median(ang)))}
