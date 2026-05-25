"""
run_bridge_ablation.py
----------------------
LSTM ablation that ports the OLD experiments/ablation/run_ablation.py
recipe onto the schema_ablation_bridge dataset. Predicts target_du and
target_dv only. MSE loss only. Truly autoregressive 20-step rollout.

Feature sets:
  A_motion_only        — motion kinematics only (6)
  B_motion_position    — + normalised world position (8)
  C_motion_position_spatial — + obstacle/boundary clearance (10)
  D_full_relational    — + entrance affinity (11)
  E_full_affordance    — + openness_lr_asymmetry (12)  IFF column present

Outputs (in this experiment folder):
  <set>/best_model.pth
  scalers.pkl                              (per-model feat + tgt scalers)
  loss_curves.png
  drift_curves.png
  rollout_plots/<set>_traj<i>.png
  ablation_results.csv
  ablation_summary.md
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent
DEFAULT_DATA   = HERE / "schema_ablation_bridge_dataset.csv"
DEFAULT_SCHEMA = HERE / "schema_summary.json"
DEFAULT_OUT    = HERE


# ─────────────────────────────────────────────────────────────────────────────
# Config (kept identical to the OLD ablation recipe)
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE   = 10
N_ROLLOUT     = 20
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
DROPOUT       = 0.2
BATCH_SIZE    = 512
MAX_EPOCHS    = 80
LR            = 1e-3
LR_STEP       = 25
LR_GAMMA      = 0.35
PATIENCE      = 15
SEED          = 42
N_KDT         = 5
N_PLOT_TRAJS  = 6
MIN_SPEED     = 1e-6
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15
SHIFT_THRESH_DEG = 15.0     # used only for angularity metric (NOT a loss)


# ─────────────────────────────────────────────────────────────────────────────
# Feature sets
# ─────────────────────────────────────────────────────────────────────────────
_BASE = ["du", "dv", "speed", "heading_sin", "heading_cos", "turn_rate"]

FEATURE_SETS: Dict[str, List[str]] = {
    "A_motion_only":              list(_BASE),
    "B_motion_position":          _BASE + ["u", "v"],
    "C_motion_position_spatial":  _BASE + ["u", "v",
                                           "dist_to_obstacle_norm",
                                           "dist_to_boundary_norm"],
    "D_full_relational":          _BASE + ["u", "v",
                                           "dist_to_obstacle_norm",
                                           "dist_to_boundary_norm",
                                           "entrance_affinity_norm"],
}

TARGET_COLS = ["target_du", "target_dv"]

# Superset of every feature any model might use (for raw seed lookups
# and KDTree rebuilds). The presence of `openness_lr_asymmetry` here is
# conditional: the orchestrator adds it at runtime if the dataset
# actually carries that column.
ALL_FEAT_COLS_BASE = [
    "du", "dv", "speed", "heading_sin", "heading_cos", "turn_rate",
    "u", "v",
    "dist_to_obstacle_norm", "dist_to_boundary_norm",
    "entrance_affinity_norm",
]

SPATIAL_COLS_BASE = [
    "dist_to_obstacle_norm", "dist_to_boundary_norm", "entrance_affinity_norm",
]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class TrajectoryLSTM(nn.Module):
    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feat, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.head = nn.Linear(HIDDEN_SIZE, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / windowing
# ─────────────────────────────────────────────────────────────────────────────
class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int: return len(self.X)
    def __getitem__(self, i: int):
        return self.X[i], self.Y[i]


def make_windows(df: pd.DataFrame, feat_cols: List[str],
                 traj_ids: List) -> Tuple[np.ndarray, np.ndarray]:
    Xs, Ys = [], []
    for tid in traj_ids:
        g = df[df["trajectory_id"] == tid].sort_values("timestep")
        feat = g[feat_cols].to_numpy(np.float32)
        tgt  = g[TARGET_COLS].to_numpy(np.float32)
        n = len(g)
        for i in range(n - WINDOW_SIZE):
            Xs.append(feat[i : i + WINDOW_SIZE])
            Ys.append(tgt[i + WINDOW_SIZE - 1])
    return (np.asarray(Xs, dtype=np.float32),
            np.asarray(Ys, dtype=np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Scaler (column-wise StandardScaler, numpy-backed for fast rollout)
# ─────────────────────────────────────────────────────────────────────────────
class ColumnScaler:
    def __init__(self) -> None:
        self.means: np.ndarray = None
        self.stds:  np.ndarray = None
        self.cols:  List[str]  = []

    def fit(self, df: pd.DataFrame, cols: List[str]) -> "ColumnScaler":
        self.cols  = cols
        data       = df[cols].to_numpy(np.float64)
        self.means = data.mean(axis=0)
        self.stds  = data.std(axis=0)
        self.stds[self.stds < 1e-9] = 1.0
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.means) / self.stds

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.stds + self.means

    def scale_col(self, col: str, val: float) -> float:
        i = self.cols.index(col)
        return (val - self.means[i]) / self.stds[i]


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (one feature set)
# ─────────────────────────────────────────────────────────────────────────────
def train_one(feat_cols: List[str],
              train_ids: List, val_ids: List, df: pd.DataFrame,
              f_sc: ColumnScaler, t_sc: ColumnScaler,
              out_dir: Path, device: torch.device):
    print(f"  building windows ...")
    X_tr, Y_tr = make_windows(df, feat_cols, train_ids)
    X_va, Y_va = make_windows(df, feat_cols, val_ids)

    X_tr = f_sc.transform(X_tr.reshape(-1, len(feat_cols))).reshape(X_tr.shape).astype(np.float32)
    X_va = f_sc.transform(X_va.reshape(-1, len(feat_cols))).reshape(X_va.shape).astype(np.float32)
    Y_tr = t_sc.transform(Y_tr).astype(np.float32)
    Y_va = t_sc.transform(Y_va).astype(np.float32)

    tr_loader = DataLoader(WindowDataset(X_tr, Y_tr), batch_size=BATCH_SIZE,
                           shuffle=True,  num_workers=0, pin_memory=False)
    va_loader = DataLoader(WindowDataset(X_va, Y_va), batch_size=BATCH_SIZE * 2,
                           shuffle=False, num_workers=0)

    model     = TrajectoryLSTM(len(feat_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, LR_STEP, LR_GAMMA)
    criterion = nn.MSELoss()

    best_val, no_improve, best_state = float("inf"), 0, None
    tr_hist, va_hist = [], []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= max(len(X_tr), 1)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_loss += criterion(model(xb), yb).item() * len(xb)
        va_loss /= max(len(X_va), 1)

        tr_hist.append(tr_loss); va_hist.append(va_loss)
        scheduler.step()

        if va_loss < best_val:
            best_val   = va_loss
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    ep {epoch:3d}  train={tr_loss:.5f}  val={va_loss:.5f}  "
                  f"best={best_val:.5f}")
        if no_improve >= PATIENCE:
            print(f"    early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "best_model.pth")
    return model, tr_hist, va_hist, best_val


# ─────────────────────────────────────────────────────────────────────────────
# KDTree spatial refresh
# ─────────────────────────────────────────────────────────────────────────────
def build_kdt(train_df: pd.DataFrame, spatial_cols: List[str]) -> dict:
    pts  = train_df[["u", "v"]].to_numpy(np.float32)
    vals = train_df[spatial_cols].to_numpy(np.float32)
    return {"tree": cKDTree(pts), "values": vals, "cols": spatial_cols}


def kdt_lookup(kdt: dict, u: float, v: float) -> np.ndarray:
    dists, idxs = kdt["tree"].query([[u, v]], k=N_KDT)
    dists, idxs = dists[0], idxs[0]
    if dists[0] < 1e-10:
        return kdt["values"][idxs[0]]
    w = 1.0 / dists
    w /= w.sum()
    return (w[:, None] * kdt["values"][idxs]).sum(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Autoregressive rollout (mirrors the OLD ablation rollout exactly)
# ─────────────────────────────────────────────────────────────────────────────
def rollout(model: TrajectoryLSTM,
            seed_raw: np.ndarray,                # (W, len(ALL_FEAT_COLS)) unscaled
            feat_cols: List[str],
            all_feat_cols: List[str],
            spatial_cols: List[str],
            f_sc: ColumnScaler, t_sc: ColumnScaler,
            world_bnd: dict, kdt: Optional[dict],
            n_steps: int,
            device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    u0 = seed_raw[-1, all_feat_cols.index("u")]
    v0 = seed_raw[-1, all_feat_cols.index("v")]
    wx = u0 * world_bnd["xrng"] + world_bnd["xmin"]
    wy = v0 * world_bnd["yrng"] + world_bnd["ymin"]

    hs0 = seed_raw[-1, all_feat_cols.index("heading_sin")]
    hc0 = seed_raw[-1, all_feat_cols.index("heading_cos")]
    prev_h = math.atan2(hs0, hc0)

    seed_feat = seed_raw[:, [all_feat_cols.index(c) for c in feat_cols]]
    seed_sc   = f_sc.transform(seed_feat).astype(np.float32)
    window    = torch.tensor(seed_sc, dtype=torch.float32, device=device)

    positions = [(wx, wy)]
    headings  = []

    for _ in range(n_steps):
        with torch.no_grad():
            pred_sc = model(window.unsqueeze(0))[0].cpu().numpy()    # (2,)
        du_m, dv_m = t_sc.inverse(pred_sc.reshape(1, 2))[0]

        wx += du_m; wy += dv_m
        positions.append((wx, wy))

        u_new   = float(np.clip((wx - world_bnd["xmin"]) / world_bnd["xrng"], 0.0, 1.0))
        v_new   = float(np.clip((wy - world_bnd["ymin"]) / world_bnd["yrng"], 0.0, 1.0))
        speed   = math.hypot(du_m, dv_m)
        heading = math.atan2(dv_m, du_m) if speed > MIN_SPEED else prev_h
        tr      = wrap_angle(heading - prev_h)
        hs, hc  = math.sin(heading), math.cos(heading)
        prev_h  = heading
        headings.append(heading)

        raw = {
            "du": du_m, "dv": dv_m, "speed": speed,
            "heading_sin": hs, "heading_cos": hc, "turn_rate": tr,
            "u": u_new, "v": v_new,
        }
        if kdt is not None:
            sp_vals = kdt_lookup(kdt, u_new, v_new)
            for col, val in zip(spatial_cols, sp_vals):
                raw[col] = val
        else:
            for col in spatial_cols:
                raw[col] = 0.0

        new_row = np.array([f_sc.scale_col(c, raw[c]) for c in feat_cols],
                           dtype=np.float32)
        new_t   = torch.tensor(new_row, device=device).unsqueeze(0)
        window  = torch.cat([window[1:], new_t], dim=0)

    return np.array(positions), np.array(headings)


def gt_positions_from_seed(traj_df: pd.DataFrame, world_bnd: dict,
                           n_steps: int) -> np.ndarray:
    last_u = traj_df.iloc[WINDOW_SIZE - 1]["u"]
    last_v = traj_df.iloc[WINDOW_SIZE - 1]["v"]
    wx0 = last_u * world_bnd["xrng"] + world_bnd["xmin"]
    wy0 = last_v * world_bnd["yrng"] + world_bnd["ymin"]
    rows = traj_df.iloc[WINDOW_SIZE : WINDOW_SIZE + n_steps]
    gt_wx = wx0 + rows["du"].cumsum().to_numpy()
    gt_wy = wy0 + rows["dv"].cumsum().to_numpy()
    return np.column_stack([
        np.concatenate([[wx0], gt_wx]),
        np.concatenate([[wy0], gt_wy]),
    ])


def gt_headings_from_df(traj_df: pd.DataFrame, n_steps: int) -> np.ndarray:
    rows = traj_df.iloc[WINDOW_SIZE : WINDOW_SIZE + n_steps]
    hs, hc = rows["heading_sin"].to_numpy(), rows["heading_cos"].to_numpy()
    return np.arctan2(hs, hc)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def _per_track_extras(p_uv: np.ndarray, g_uv: np.ndarray) -> dict:
    """Path-length, cum-heading, angularity, curvature_corr — observed-only."""
    out = {"path_length_ratio": float("nan"),
           "cum_heading_ratio": float("nan"),
           "angularity_ratio":  float("nan"),
           "curvature_corr":    float("nan")}
    if len(p_uv) < 2 or len(g_uv) < 2:
        return out
    n = min(len(p_uv), len(g_uv))
    p_steps = np.linalg.norm(np.diff(p_uv[:n], axis=0), axis=1)
    g_steps = np.linalg.norm(np.diff(g_uv[:n], axis=0), axis=1)
    if g_steps.sum() > 1e-9:
        out["path_length_ratio"] = float(p_steps.sum() / g_steps.sum())

    d_p = np.diff(p_uv[:n], axis=0); d_g = np.diff(g_uv[:n], axis=0)
    h_p = np.arctan2(d_p[:, 1], d_p[:, 0])
    h_g = np.arctan2(d_g[:, 1], d_g[:, 0])
    if len(h_p) > 1 and len(h_g) > 1:
        m  = min(len(h_p), len(h_g))
        tp = np.arctan2(np.sin(np.diff(h_p[:m])), np.cos(np.diff(h_p[:m])))
        tg = np.arctan2(np.sin(np.diff(h_g[:m])), np.cos(np.diff(h_g[:m])))
        cp, cg = float(np.abs(tp).sum()), float(np.abs(tg).sum())
        if cg > 1e-9:
            out["cum_heading_ratio"] = cp / cg
        thresh = np.deg2rad(SHIFT_THRESH_DEG)
        ang_p = float(np.mean(np.abs(tp) > thresh))
        ang_g = float(np.mean(np.abs(tg) > thresh))
        if ang_g > 1e-9:
            out["angularity_ratio"] = ang_p / ang_g
        sp, sg = np.abs(tp), np.abs(tg)
        if sp.std() > 1e-9 and sg.std() > 1e-9:
            out["curvature_corr"] = float(np.corrcoef(sp, sg)[0, 1])
    return out


def evaluate_on_test(model: TrajectoryLSTM, test_ids: List, df: pd.DataFrame,
                     feat_cols: List[str], all_feat_cols: List[str],
                     spatial_cols: List[str],
                     f_sc: ColumnScaler, t_sc: ColumnScaler,
                     world_bnd: dict, kdt: Optional[dict],
                     device: torch.device) -> dict:
    model.eval()
    ades, fdes = [], []
    drift_acc = np.zeros(N_ROLLOUT); drift_cnt = np.zeros(N_ROLLOUT)
    ang_errs, tr_preds, tr_gts = [], [], []
    pls, chs, angs, cvs = [], [], [], []

    for tid in test_ids:
        traj = df[df["trajectory_id"] == tid].sort_values("timestep").reset_index(drop=True)
        if len(traj) < WINDOW_SIZE + N_ROLLOUT:
            continue
        seed_raw = traj.iloc[:WINDOW_SIZE][all_feat_cols].to_numpy(np.float32)
        gt_pos   = gt_positions_from_seed(traj, world_bnd, N_ROLLOUT)
        gt_h     = gt_headings_from_df(traj, N_ROLLOUT)

        pred_pos, pred_h = rollout(
            model, seed_raw, feat_cols, all_feat_cols, spatial_cols,
            f_sc, t_sc, world_bnd, kdt, N_ROLLOUT, device,
        )

        n = min(len(pred_pos), len(gt_pos))
        d = np.sqrt(((pred_pos[:n] - gt_pos[:n]) ** 2).sum(axis=1))[1:]
        if len(d) == 0:
            continue
        ades.append(d.mean()); fdes.append(d[-1])
        for i, di in enumerate(d):
            if i < N_ROLLOUT:
                drift_acc[i] += di; drift_cnt[i] += 1

        n_h = min(len(pred_h), len(gt_h))
        if n_h > 0:
            ang_errs.append(np.mean(
                [abs(wrap_angle(pred_h[i] - gt_h[i])) for i in range(n_h)]))
        if len(pred_h) > 1:
            tr_preds.append(np.mean([abs(wrap_angle(pred_h[i] - pred_h[i-1]))
                                     for i in range(1, len(pred_h))]))
        if len(gt_h) > 1:
            tr_gts.append(np.mean([abs(wrap_angle(gt_h[i] - gt_h[i-1]))
                                   for i in range(1, len(gt_h))]))

        # Convert positions to normalised u/v for shape metrics.
        p_uv = np.column_stack([
            (pred_pos[:, 0] - world_bnd["xmin"]) / world_bnd["xrng"],
            (pred_pos[:, 1] - world_bnd["ymin"]) / world_bnd["yrng"],
        ])
        g_uv = np.column_stack([
            (gt_pos[:, 0]   - world_bnd["xmin"]) / world_bnd["xrng"],
            (gt_pos[:, 1]   - world_bnd["ymin"]) / world_bnd["yrng"],
        ])
        extras = _per_track_extras(p_uv, g_uv)
        if not np.isnan(extras["path_length_ratio"]): pls.append(extras["path_length_ratio"])
        if not np.isnan(extras["cum_heading_ratio"]): chs.append(extras["cum_heading_ratio"])
        if not np.isnan(extras["angularity_ratio"]):  angs.append(extras["angularity_ratio"])
        if not np.isnan(extras["curvature_corr"]):    cvs.append(extras["curvature_corr"])

    drift = np.where(drift_cnt > 0, drift_acc / drift_cnt, np.nan)
    return {
        "ade":                 float(np.mean(ades))      if ades       else float("nan"),
        "fde":                 float(np.mean(fdes))      if fdes       else float("nan"),
        "drift":               drift,
        "mean_angular_error":  float(np.mean(ang_errs))  if ang_errs   else float("nan"),
        "mean_turn_rate_pred": float(np.mean(tr_preds))  if tr_preds   else float("nan"),
        "mean_turn_rate_gt":   float(np.mean(tr_gts))    if tr_gts     else float("nan"),
        "path_length_ratio":   float(np.mean(pls))       if pls        else float("nan"),
        "cum_heading_ratio":   float(np.mean(chs))       if chs        else float("nan"),
        "angularity_ratio":    float(np.mean(angs))      if angs       else float("nan"),
        "curvature_corr":      float(np.mean(cvs))       if cvs        else float("nan"),
        "n_trajs_evaluated":   len(ades),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers (loss + drift only; rich plots live in visualize script)
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "A_motion_only":             "#e05c2a",
    "B_motion_position":         "#4c8cbf",
    "C_motion_position_spatial": "#3aaa5e",
    "D_full_relational":         "#8e44ad",
    "E_full_affordance":         "#d4a017",
}


def plot_loss_curves(histories: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, (tr, va, _) in histories.items():
        col = COLORS.get(name, "#333333")
        axes[0].plot(tr, color=col, lw=1.4, label=name)
        axes[1].plot(va, color=col, lw=1.4, label=name)
    for ax, title in zip(axes, ["Train MSE (scaled)", "Val MSE (scaled)"]):
        ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.set_title(title)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_path.name}")


def plot_drift_curves(metrics: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    steps = np.arange(1, N_ROLLOUT + 1)
    for name, m in metrics.items():
        ax.plot(steps, m["drift"], color=COLORS.get(name, "#333"), lw=1.8, label=name)
    ax.set_xlabel("rollout step"); ax.set_ylabel("ADE (metres)")
    ax.set_title("Per-step autoregressive drift")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary writers
# ─────────────────────────────────────────────────────────────────────────────
def write_csv(metrics: dict, histories: dict, out_path: Path) -> None:
    rows = []
    for name, m in metrics.items():
        _, _, best_val = histories[name]
        rows.append({
            "model":               name,
            "n_features":          len(FEATURE_SETS[name]),
            "best_val_mse":        round(best_val, 6),
            "test_ade_m":          round(m["ade"], 5),
            "test_fde_m":          round(m["fde"], 5),
            "mean_angular_err":    round(m["mean_angular_error"], 5),
            "mean_turn_rate_pred": round(m["mean_turn_rate_pred"], 5),
            "mean_turn_rate_gt":   round(m["mean_turn_rate_gt"], 5),
            "path_length_ratio":   round(m["path_length_ratio"], 5),
            "cum_heading_ratio":   round(m["cum_heading_ratio"], 5),
            "angularity_ratio":    round(m["angularity_ratio"], 5),
            "curvature_corr":      round(m["curvature_corr"], 5),
            "n_trajs":             m["n_trajs_evaluated"],
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[saved] {out_path.name}")


def write_summary(metrics: dict, histories: dict, out_path: Path) -> None:
    ranked = sorted(metrics.items(), key=lambda kv: kv[1]["ade"])
    gt_tr  = ranked[0][1]["mean_turn_rate_gt"]
    md = []
    A = md.append
    A("# schema_ablation_bridge — Results\n")
    A(f"Window={WINDOW_SIZE}  Horizon={N_ROLLOUT}  "
      f"Hidden={HIDDEN_SIZE} x {NUM_LAYERS}  Seed={SEED}\n")
    A("## Quantitative results\n")
    A("| Model | Val MSE | ADE (m) | FDE (m) | Ang err (rad) | "
      "Pred TR (rad) | GT TR (rad) | path/GT | cum_head/GT | ang_ratio | "
      "curv_corr | N |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, m in metrics.items():
        _, _, best_val = histories[name]
        def f(x):
            return f"{x:.4f}" if isinstance(x, float) and not math.isnan(x) else "—"
        A(f"| `{name}` | {f(best_val)} | {f(m['ade'])} | {f(m['fde'])} | "
          f"{f(m['mean_angular_error'])} | {f(m['mean_turn_rate_pred'])} | "
          f"{f(m['mean_turn_rate_gt'])} | {f(m['path_length_ratio'])} | "
          f"{f(m['cum_heading_ratio'])} | {f(m['angularity_ratio'])} | "
          f"{f(m['curvature_corr'])} | {m['n_trajs_evaluated']} |")
    A("")
    A("## Ranking by ADE (best first)\n")
    for rank, (name, m) in enumerate(ranked, 1):
        A(f"{rank}. `{name}` — ADE={m['ade']:.4f} m   FDE={m['fde']:.4f} m   "
          f"ang_err={math.degrees(m['mean_angular_error']):.1f}°")
    A("")
    A(f"GT mean |turn_rate| = **{gt_tr:.4f} rad / step**.\n")
    A("Notes: angular metrics here are *observed*, not optimised. "
      "Only MSE on (target_du, target_dv) was the training objective.\n")
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[saved] {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Public split helper (used by visualize too)
# ─────────────────────────────────────────────────────────────────────────────
def split_ids(df: pd.DataFrame, seed: int = SEED) -> Tuple[List, List, List]:
    all_ids = df["trajectory_id"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(all_ids)
    n    = len(all_ids)
    n_tr = int(n * TRAIN_FRAC)
    n_va = int(n * VAL_FRAC)
    return all_ids[:n_tr], all_ids[n_tr:n_tr+n_va], all_ids[n_tr+n_va:]


def resolve_feature_sets(df_cols) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    """Add E_full_affordance only if openness_lr_asymmetry is present."""
    feature_sets = {k: list(v) for k, v in FEATURE_SETS.items()}
    all_feat_cols = list(ALL_FEAT_COLS_BASE)
    spatial_cols  = list(SPATIAL_COLS_BASE)
    if "openness_lr_asymmetry" in df_cols:
        feature_sets["E_full_affordance"] = (
            feature_sets["D_full_relational"] + ["openness_lr_asymmetry"])
        all_feat_cols.append("openness_lr_asymmetry")
        spatial_cols.append("openness_lr_asymmetry")
        print("[info]  openness_lr_asymmetry present → model E enabled")
    else:
        print("[info]  openness_lr_asymmetry absent → model E skipped (graceful)")
    return feature_sets, all_feat_cols, spatial_cols


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    type=Path, default=DEFAULT_DATA)
    parser.add_argument("--schema",     type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    if not args.dataset.exists():
        raise SystemExit(f"[FATAL] missing dataset: {args.dataset}")
    if not args.schema.exists():
        raise SystemExit(f"[FATAL] missing schema:  {args.schema}")

    print(f"[load] {args.dataset.name}")
    df = pd.read_csv(args.dataset)
    print(f"       {len(df):,} rows · {df['trajectory_id'].nunique():,} tracks")
    with open(args.schema) as f:
        schema = json.load(f)
    world_bnd = schema["world_bounds_used"]
    print(f"[bnds] xrng={world_bnd['xrng']:.3f} m  yrng={world_bnd['yrng']:.3f} m")

    feature_sets, all_feat_cols, spatial_cols = resolve_feature_sets(df.columns)

    train_ids, val_ids, test_ids = split_ids(df)
    print(f"[split] {len(train_ids)} train · {len(val_ids)} val · "
          f"{len(test_ids)} test")

    train_df = df[df["trajectory_id"].isin(train_ids)]
    kdt_base = build_kdt(train_df, spatial_cols)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir = args.output_dir / "rollout_plots"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    histories: dict = {}; all_metrics: dict = {}
    scalers_blob: Dict[str, dict] = {}

    for name, feat_cols in feature_sets.items():
        print(f"\n{'='*60}\n[model] {name}  ({len(feat_cols)} features)\n{'='*60}")
        out_dir = args.output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        f_sc = ColumnScaler().fit(train_df, feat_cols)
        t_sc = ColumnScaler().fit(train_df, TARGET_COLS)
        needs_kdt = any(c in feat_cols for c in spatial_cols)
        kdt       = kdt_base if needs_kdt else None

        model, tr_hist, va_hist, best_val = train_one(
            feat_cols, train_ids, val_ids, df, f_sc, t_sc, out_dir, device)
        histories[name] = (tr_hist, va_hist, best_val)

        print("  evaluating on test set ...")
        metrics = evaluate_on_test(
            model, test_ids, df, feat_cols, all_feat_cols, spatial_cols,
            f_sc, t_sc, world_bnd, kdt, device)
        all_metrics[name] = metrics
        print(f"  ADE={metrics['ade']:.4f}m  FDE={metrics['fde']:.4f}m  "
              f"ang_err={metrics['mean_angular_error']:.4f}rad  "
              f"path/GT={metrics['path_length_ratio']:.3f}  "
              f"ang_ratio={metrics['angularity_ratio']:.3f}")

        scalers_blob[name] = {
            "feat_means": f_sc.means.tolist(), "feat_stds": f_sc.stds.tolist(),
            "feat_cols":  feat_cols,
            "tgt_means":  t_sc.means.tolist(), "tgt_stds":  t_sc.stds.tolist(),
            "tgt_cols":   TARGET_COLS,
            "needs_kdt":  needs_kdt,
        }

    # ── plots, csv, summary
    print("\n[plots] loss curves ...")
    plot_loss_curves(histories, args.output_dir / "loss_curves.png")
    print("[plots] drift curves ...")
    plot_drift_curves(all_metrics, args.output_dir / "drift_curves.png")

    # Save scaler blob for visualize to consume without retraining.
    (args.output_dir / "scalers.pkl").write_bytes(pickle.dumps(scalers_blob))
    print(f"[saved] scalers.pkl")

    write_csv(all_metrics, histories,
              args.output_dir / "ablation_results.csv")
    write_summary(all_metrics, histories,
                  args.output_dir / "ablation_summary.md")

    print(f"\n[done] results in {args.output_dir}")


if __name__ == "__main__":
    main()
