"""
run_ablation.py
---------------
LSTM ablation: 4 feature sets on motion_dataset_v2.csv.

Compares:
  A_motion_only            — motion kinematics only
  B_motion_position        — + normalised world position
  C_motion_position_spatial — + obstacle/boundary distances
  D_full_affordance        — + lateral openness asymmetry

Outputs (all inside --output_dir):
  <set>/best_model.pth
  loss_curves.png
  rollout_plots/<set>_traj<i>.png
  ablation_results.csv
  ablation_summary.md

Usage
-----
  python run_ablation.py
  python run_ablation.py --dataset PATH --schema PATH --output_dir DIR
"""

import argparse
import json
import math
import random
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ── paths ─────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent
MP_ROOT      = HERE.parent.parent.parent.parent
ENCODED_DIR  = MP_ROOT / "mp-data" / "processed" / "encoded"
DEFAULT_DATA   = ENCODED_DIR / "motion_dataset_v2.csv"
DEFAULT_SCHEMA = ENCODED_DIR / "schema_summary.json"
DEFAULT_OUT    = MP_ROOT / "mp-data" / "outputs" / "ablation"

# ── config ────────────────────────────────────────────────────────────────────
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
N_KDT         = 5        # KDTree neighbours for spatial interpolation
N_PLOT_TRAJS  = 6        # trajectories to plot per feature set
MIN_SPEED     = 1e-6
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15     # test = 1 - TRAIN - VAL

# ── feature sets ──────────────────────────────────────────────────────────────
FEATURE_SETS: Dict[str, List[str]] = {
    "A_motion_only": [
        "du", "dv", "speed", "heading_sin", "heading_cos", "turn_rate",
    ],
    "B_motion_position": [
        "du", "dv", "speed", "heading_sin", "heading_cos", "turn_rate",
        "u", "v",
    ],
    "C_motion_position_spatial": [
        "du", "dv", "speed", "heading_sin", "heading_cos", "turn_rate",
        "u", "v", "dist_to_obstacle_norm", "dist_to_boundary_norm",
    ],
    "D_full_affordance": [
        "du", "dv", "speed", "heading_sin", "heading_cos", "turn_rate",
        "u", "v", "dist_to_obstacle_norm", "dist_to_boundary_norm",
        "openness_lr_asymmetry",
    ],
}
TARGET_COLS    = ["target_du", "target_dv"]
# superset of features for raw seed access during rollout
ALL_FEAT_COLS = [
    "du", "dv", "speed", "heading_sin", "heading_cos", "turn_rate",
    "u", "v", "dist_to_obstacle_norm", "dist_to_boundary_norm",
    "openness_lr_asymmetry",
]
# spatial features resolvable via KDTree
SPATIAL_COLS = ["dist_to_obstacle_norm", "dist_to_boundary_norm", "openness_lr_asymmetry"]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


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
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.Y[i]


def make_windows(
    df: pd.DataFrame,
    feat_cols: List[str],
    traj_ids: List,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slide a window over each trajectory; target = next-step displacement."""
    Xs, Ys = [], []
    for tid in traj_ids:
        g = df[df["trajectory_id"] == tid].sort_values("timestep")
        feat = g[feat_cols].to_numpy(np.float32)
        tgt  = g[TARGET_COLS].to_numpy(np.float32)
        n    = len(g)
        for i in range(n - WINDOW_SIZE):
            Xs.append(feat[i : i + WINDOW_SIZE])
            # target_du/dv at step i+WINDOW_SIZE-1 IS du/dv at step i+WINDOW_SIZE
            Ys.append(tgt[i + WINDOW_SIZE - 1])
    return np.array(Xs, dtype=np.float32), np.array(Ys, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Scaling
# ─────────────────────────────────────────────────────────────────────────────

class ColumnScaler:
    """Per-column StandardScaler backed by numpy arrays for fast rollout use."""

    def __init__(self) -> None:
        self.means: np.ndarray = None
        self.stds:  np.ndarray = None
        self.cols:  List[str]  = []

    def fit(self, df: pd.DataFrame, cols: List[str]) -> "ColumnScaler":
        self.cols  = cols
        data       = df[cols].to_numpy(np.float64)
        self.means = data.mean(axis=0)
        self.stds  = data.std(axis=0)
        self.stds[self.stds < 1e-9] = 1.0   # guard zero-variance cols
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.means) / self.stds

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.stds + self.means

    def scale_col(self, col: str, val: float) -> float:
        i = self.cols.index(col)
        return (val - self.means[i]) / self.stds[i]


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_one(
    feat_cols: List[str],
    train_ids: List,
    val_ids:   List,
    df:        pd.DataFrame,
    f_sc:      ColumnScaler,
    t_sc:      ColumnScaler,
    out_dir:   Path,
    device:    torch.device,
) -> Tuple[TrajectoryLSTM, List[float], List[float]]:
    """Train LSTM for one feature set. Returns (model, train_losses, val_losses)."""

    print(f"  building windows ...")
    X_tr, Y_tr = make_windows(df, feat_cols, train_ids)
    X_va, Y_va = make_windows(df, feat_cols, val_ids)

    # scale features and targets
    X_tr = f_sc.transform(X_tr.reshape(-1, len(feat_cols))).reshape(X_tr.shape).astype(np.float32)
    X_va = f_sc.transform(X_va.reshape(-1, len(feat_cols))).reshape(X_va.shape).astype(np.float32)
    Y_tr = t_sc.transform(Y_tr).astype(np.float32)
    Y_va = t_sc.transform(Y_va).astype(np.float32)

    tr_loader = DataLoader(WindowDataset(X_tr, Y_tr), batch_size=BATCH_SIZE,
                           shuffle=True, num_workers=0, pin_memory=False)
    va_loader = DataLoader(WindowDataset(X_va, Y_va), batch_size=BATCH_SIZE * 2,
                           shuffle=False, num_workers=0)

    model     = TrajectoryLSTM(len(feat_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, LR_STEP, LR_GAMMA)
    criterion = nn.MSELoss()

    best_val   = float("inf")
    no_improve = 0
    best_state = None
    tr_hist, va_hist = [], []

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── train
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
        tr_loss /= len(X_tr)

        # ── val
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_loss += criterion(model(xb), yb).item() * len(xb)
        va_loss /= len(X_va)

        tr_hist.append(tr_loss)
        va_hist.append(va_loss)
        scheduler.step()

        if va_loss < best_val:
            best_val   = va_loss
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(f"    ep {epoch:3d}  train={tr_loss:.5f}  val={va_loss:.5f}  "
                  f"best={best_val:.5f}")

        if no_improve >= PATIENCE:
            print(f"    early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "best_model.pth")
    return model, tr_hist, va_hist, best_val


# ─────────────────────────────────────────────────────────────────────────────
# KDTree for spatial feature interpolation during rollout
# ─────────────────────────────────────────────────────────────────────────────

def build_kdt(train_df: pd.DataFrame) -> dict:
    pts  = train_df[["u", "v"]].to_numpy(np.float32)
    vals = train_df[SPATIAL_COLS].to_numpy(np.float32)
    return {"tree": cKDTree(pts), "values": vals, "cols": SPATIAL_COLS}


def kdt_lookup(kdt: dict, u: float, v: float) -> np.ndarray:
    dists, idxs = kdt["tree"].query([[u, v]], k=N_KDT)
    dists, idxs = dists[0], idxs[0]
    if dists[0] < 1e-10:
        return kdt["values"][idxs[0]]
    w = 1.0 / dists
    w /= w.sum()
    return (w[:, None] * kdt["values"][idxs]).sum(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Autoregressive rollout
# ─────────────────────────────────────────────────────────────────────────────

def rollout(
    model:      TrajectoryLSTM,
    seed_raw:   np.ndarray,    # (WINDOW_SIZE, len(ALL_FEAT_COLS)) unscaled
    feat_cols:  List[str],
    f_sc:       ColumnScaler,
    t_sc:       ColumnScaler,
    world_bnd:  dict,
    kdt:        Optional[dict],
    n_steps:    int,
    device:     torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Autoregressively predict n_steps ahead.
    Returns:
      positions (n_steps+1, 2) world coords in metres, starting from seed end
      headings  (n_steps,) predicted heading at each rollout step
    """
    model.eval()

    # ── starting world position from last seed step
    u0 = seed_raw[-1, ALL_FEAT_COLS.index("u")]
    v0 = seed_raw[-1, ALL_FEAT_COLS.index("v")]
    wx = u0 * world_bnd["xrng"] + world_bnd["xmin"]
    wy = v0 * world_bnd["yrng"] + world_bnd["ymin"]

    # ── initial heading from last seed step
    hs0 = seed_raw[-1, ALL_FEAT_COLS.index("heading_sin")]
    hc0 = seed_raw[-1, ALL_FEAT_COLS.index("heading_cos")]
    prev_h = math.atan2(hs0, hc0)

    # ── scale seed and build window tensor
    seed_feat = seed_raw[:, [ALL_FEAT_COLS.index(c) for c in feat_cols]]
    seed_sc   = f_sc.transform(seed_feat).astype(np.float32)
    window    = torch.tensor(seed_sc, dtype=torch.float32, device=device)  # (W, F)

    positions = [(wx, wy)]
    headings  = []

    for _ in range(n_steps):
        with torch.no_grad():
            pred_sc = model(window.unsqueeze(0))[0].cpu().numpy()   # (2,)

        # ── inverse-transform targets to metres
        du_m, dv_m = t_sc.inverse(pred_sc.reshape(1, 2))[0]

        # ── update world position
        wx += du_m
        wy += dv_m
        positions.append((wx, wy))

        # ── recompute derived features
        u_new   = float(np.clip((wx - world_bnd["xmin"]) / world_bnd["xrng"], 0.0, 1.0))
        v_new   = float(np.clip((wy - world_bnd["ymin"]) / world_bnd["yrng"], 0.0, 1.0))
        speed   = math.hypot(du_m, dv_m)
        heading = math.atan2(dv_m, du_m) if speed > MIN_SPEED else prev_h
        tr      = wrap_angle(heading - prev_h)
        hs      = math.sin(heading)
        hc      = math.cos(heading)
        prev_h  = heading
        headings.append(heading)

        # ── build raw new-step feature dict
        raw = {
            "du": du_m, "dv": dv_m, "speed": speed,
            "heading_sin": hs, "heading_cos": hc, "turn_rate": tr,
            "u": u_new, "v": v_new,
        }

        # ── spatial features via KDTree interpolation (C, D only)
        if kdt is not None:
            sp_vals = kdt_lookup(kdt, u_new, v_new)
            for col, val in zip(SPATIAL_COLS, sp_vals):
                raw[col] = val
        else:
            for col in SPATIAL_COLS:
                raw[col] = 0.0

        # ── scale new row and slide window
        new_row = np.array([f_sc.scale_col(c, raw[c]) for c in feat_cols],
                           dtype=np.float32)
        new_t   = torch.tensor(new_row, device=device).unsqueeze(0)   # (1, F)
        window  = torch.cat([window[1:], new_t], dim=0)               # (W, F)

    return np.array(positions), np.array(headings)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def gt_positions_from_seed(
    traj_df: pd.DataFrame,
    world_bnd: dict,
    n_steps: int,
) -> np.ndarray:
    """Integrate GT du/dv from last seed position for n_steps ahead."""
    last_u = traj_df.iloc[WINDOW_SIZE - 1]["u"]
    last_v = traj_df.iloc[WINDOW_SIZE - 1]["v"]
    wx0    = last_u * world_bnd["xrng"] + world_bnd["xmin"]
    wy0    = last_v * world_bnd["yrng"] + world_bnd["ymin"]

    rows  = traj_df.iloc[WINDOW_SIZE : WINDOW_SIZE + n_steps]
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


def evaluate_on_test(
    model:     TrajectoryLSTM,
    test_ids:  List,
    df:        pd.DataFrame,
    feat_cols: List[str],
    f_sc:      ColumnScaler,
    t_sc:      ColumnScaler,
    world_bnd: dict,
    kdt:       Optional[dict],
    device:    torch.device,
) -> dict:
    model.eval()
    ades, fdes = [], []
    drift_acc   = np.zeros(N_ROLLOUT)
    drift_cnt   = np.zeros(N_ROLLOUT)
    ang_errs    = []
    tr_preds    = []     # mean |turn_rate| per trajectory (predicted)
    tr_gts      = []     # mean |turn_rate| per trajectory (GT)

    for tid in test_ids:
        traj = df[df["trajectory_id"] == tid].sort_values("timestep").reset_index(drop=True)
        if len(traj) < WINDOW_SIZE + N_ROLLOUT:
            continue

        seed_raw = traj.iloc[:WINDOW_SIZE][ALL_FEAT_COLS].to_numpy(np.float32)
        gt_pos   = gt_positions_from_seed(traj, world_bnd, N_ROLLOUT)
        gt_h     = gt_headings_from_df(traj, N_ROLLOUT)

        pred_pos, pred_h = rollout(
            model, seed_raw, feat_cols, f_sc, t_sc, world_bnd, kdt, N_ROLLOUT, device,
        )

        n   = min(len(pred_pos), len(gt_pos))
        d   = np.sqrt(((pred_pos[:n] - gt_pos[:n]) ** 2).sum(axis=1))
        d   = d[1:]    # skip step-0 (same start)

        if len(d) == 0:
            continue

        ades.append(d.mean())
        fdes.append(d[-1])

        for i, di in enumerate(d):
            if i < N_ROLLOUT:
                drift_acc[i] += di
                drift_cnt[i] += 1

        # angular error
        n_h = min(len(pred_h), len(gt_h))
        if n_h > 0:
            ae = [abs(wrap_angle(pred_h[i] - gt_h[i])) for i in range(n_h)]
            ang_errs.append(np.mean(ae))

        # turn rate magnitude (straight-line proxy)
        if len(pred_h) > 1:
            tr_preds.append(np.mean([abs(wrap_angle(pred_h[i] - pred_h[i - 1]))
                                     for i in range(1, len(pred_h))]))
        if len(gt_h) > 1:
            tr_gts.append(np.mean([abs(wrap_angle(gt_h[i] - gt_h[i - 1]))
                                   for i in range(1, len(gt_h))]))

    drift = np.where(drift_cnt > 0, drift_acc / drift_cnt, np.nan)
    return {
        "ade":                 np.mean(ades)     if ades       else float("nan"),
        "fde":                 np.mean(fdes)     if fdes       else float("nan"),
        "drift":               drift,
        "mean_angular_error":  np.mean(ang_errs) if ang_errs   else float("nan"),
        "mean_turn_rate_pred": np.mean(tr_preds) if tr_preds   else float("nan"),
        "mean_turn_rate_gt":   np.mean(tr_gts)   if tr_gts     else float("nan"),
        "n_trajs_evaluated":   len(ades),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "A_motion_only":             "#e05c2a",
    "B_motion_position":         "#4c8cbf",
    "C_motion_position_spatial": "#3aaa5e",
    "D_full_affordance":         "#8e44ad",
}


def plot_loss_curves(histories: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for name, (tr, va, _) in histories.items():
        col = COLORS[name]
        axes[0].plot(tr, color=col, linewidth=1.4, label=name)
        axes[1].plot(va, color=col, linewidth=1.4, label=name)
    for ax, title in zip(axes, ["Train MSE (scaled)", "Val MSE (scaled)"]):
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.name}")


def plot_rollouts(
    models:    dict,
    test_ids:  List,
    df:        pd.DataFrame,
    feat_sets: Dict[str, List[str]],
    f_scs:     dict,
    t_scs:     dict,
    world_bnd: dict,
    kdts:      dict,
    device:    torch.device,
    out_dir:   Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # pick N_PLOT_TRAJS trajectories long enough for rollout
    eligible = [tid for tid in test_ids
                if len(df[df["trajectory_id"] == tid]) >= WINDOW_SIZE + N_ROLLOUT + 10]
    rng = np.random.default_rng(SEED)
    plot_ids = rng.choice(eligible, size=min(N_PLOT_TRAJS, len(eligible)), replace=False)

    for pi, tid in enumerate(plot_ids):
        traj = df[df["trajectory_id"] == tid].sort_values("timestep").reset_index(drop=True)
        gt_pos = gt_positions_from_seed(traj, world_bnd, N_ROLLOUT)

        # convert world metres to normalised u/v for display
        def to_uv(pos):
            u = (pos[:, 0] - world_bnd["xmin"]) / world_bnd["xrng"]
            v = (pos[:, 1] - world_bnd["ymin"]) / world_bnd["yrng"]
            return u, v

        fig, ax = plt.subplots(figsize=(6, 6))

        # ground truth full trajectory
        all_u = traj["u"].to_numpy()
        all_v = traj["v"].to_numpy()
        ax.plot(all_u, all_v, color="#cccccc", linewidth=1.2, zorder=1, label="GT full")

        # seed
        seed_u = all_u[:WINDOW_SIZE]
        seed_v = all_v[:WINDOW_SIZE]
        ax.plot(seed_u, seed_v, color="#333333", linewidth=2, zorder=2, label="seed")
        ax.scatter(seed_u[0], seed_v[0], s=40, color="#333333", zorder=3)

        # GT rollout segment
        gt_u, gt_v = to_uv(gt_pos)
        ax.plot(gt_u, gt_v, color="black", linewidth=1.8, linestyle="--",
                zorder=3, label="GT rollout")

        # predicted rollouts per model
        for name, model in models.items():
            seed_raw = traj.iloc[:WINDOW_SIZE][ALL_FEAT_COLS].to_numpy(np.float32)
            pred_pos, _ = rollout(
                model, seed_raw, feat_sets[name], f_scs[name], t_scs[name],
                world_bnd, kdts.get(name), N_ROLLOUT, device,
            )
            pu, pv = to_uv(pred_pos)
            ax.plot(pu, pv, color=COLORS[name], linewidth=1.6,
                    zorder=4, label=name, alpha=0.85)
            ax.scatter(pu[-1], pv[-1], s=25, color=COLORS[name], zorder=5)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xlabel("u (norm. world x)")
        ax.set_ylabel("v (norm. world y)")
        ax.set_title(f"Rollout — trajectory {tid}", fontsize=10)
        ax.legend(fontsize=6.5, loc="upper right")
        ax.grid(alpha=0.2)

        fname = out_dir / f"traj_{pi:02d}_id{tid}.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] rollout_plots/{fname.name}")


def plot_drift_curves(metrics: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    steps = np.arange(1, N_ROLLOUT + 1)
    for name, m in metrics.items():
        ax.plot(steps, m["drift"], color=COLORS[name], linewidth=1.8, label=name)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("ADE (metres)")
    ax.set_title("Per-step autoregressive drift")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary generation
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(metrics: dict, histories: dict, out_path: Path) -> None:
    # sort by ADE
    ranked = sorted(metrics.items(), key=lambda kv: kv[1]["ade"])
    gt_tr  = ranked[0][1]["mean_turn_rate_gt"]   # same for all models

    lines = []
    A = lines.append

    A("# LSTM Ablation Results\n")
    A(f"Window size: {WINDOW_SIZE} steps | Rollout: {N_ROLLOUT} steps | "
      f"Hidden: {HIDDEN_SIZE} x {NUM_LAYERS} layers\n")

    A("## Quantitative results\n")
    A("| Model | Val MSE | Test ADE (m) | Test FDE (m) | Angular err (rad) | "
      "Pred turn rate (rad) | GT turn rate (rad) | Trajs evaluated |")
    A("|-------|---------|-------------|-------------|-------------------|"
      "---------------------|-------------------|-----------------|")
    for name, m in metrics.items():
        _, _, best_val = histories[name]
        A(f"| `{name}` | {best_val:.5f} | {m['ade']:.4f} | {m['fde']:.4f} | "
          f"{m['mean_angular_error']:.4f} | {m['mean_turn_rate_pred']:.4f} | "
          f"{m['mean_turn_rate_gt']:.4f} | {m['n_trajs_evaluated']} |")
    A("")

    A("## Ranked by Test ADE (best first)\n")
    for rank, (name, m) in enumerate(ranked, 1):
        A(f"{rank}. `{name}` — ADE={m['ade']:.4f} m, FDE={m['fde']:.4f} m")
    A("")

    A("## Straight-line bias analysis\n")
    A(f"Ground-truth mean |turn_rate|: **{gt_tr:.4f} rad**  "
      f"(averaged over all test rollout windows)\n")
    for name, m in metrics.items():
        diff  = m["mean_turn_rate_pred"] - gt_tr
        label = "over-turns" if diff > 0.01 else ("under-turns (straighter)" if diff < -0.01 else "well-calibrated")
        A(f"- `{name}`: pred turn_rate={m['mean_turn_rate_pred']:.4f} rad  "
          f"(delta from GT: {diff:+.4f})  → **{label}**")
    A("")

    A("## Research question: do spatial features reduce straight-line drift?\n")

    ade_a = metrics["A_motion_only"]["ade"]
    ade_b = metrics.get("B_motion_position", {}).get("ade", float("nan"))
    ade_c = metrics.get("C_motion_position_spatial", {}).get("ade", float("nan"))
    ade_d = metrics.get("D_full_affordance", {}).get("ade", float("nan"))

    def pct(a, b):
        if math.isnan(a) or math.isnan(b) or b == 0:
            return float("nan")
        return (a - b) / b * 100

    A(f"ADE improvement from A (motion-only) baseline:")
    A(f"- B vs A (adding position): {pct(ade_a, ade_b):+.1f}%")
    A(f"- C vs A (adding spatial):  {pct(ade_a, ade_c):+.1f}%")
    A(f"- D vs A (full affordance): {pct(ade_a, ade_d):+.1f}%")
    A("")

    tr_a = metrics["A_motion_only"]["mean_turn_rate_pred"]
    tr_d = metrics["D_full_affordance"]["mean_turn_rate_pred"]
    A(f"Turn rate shift from A to D: {tr_d - tr_a:+.4f} rad "
      f"({'more curved' if tr_d > tr_a else 'more straight'} predictions with spatial features)")
    A(f"GT turn rate: {gt_tr:.4f} rad")
    A("")

    if not math.isnan(ade_c) and ade_c < ade_a:
        A("**Conclusion:** Spatial distance features (C) improve autoregressive ADE "
          "over motion-only (A). The model uses obstacle/boundary proximity to avoid "
          "collisions that motion dynamics alone cannot predict.")
    elif not math.isnan(ade_c) and ade_c >= ade_a:
        A("**Conclusion:** Spatial distance features (C) did not improve ADE over "
          "motion-only (A) in this experiment. This is consistent with the v1 diagnostic "
          "finding that `dist_to_obstacle_norm` is 95% correlated with `u`, providing "
          "little information beyond position. The spatial confound identified in "
          "diagnostics_summary.md appears to be the limiting factor.")
    A("")
    A("See `rollout_plots/` for qualitative rollout comparisons and "
      "`drift_curves.png` for per-step ADE growth.")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out_path.name}")


def write_csv(metrics: dict, histories: dict, out_path: Path) -> None:
    rows = []
    for name, m in metrics.items():
        _, _, best_val = histories[name]
        rows.append({
            "model":              name,
            "n_features":         len(FEATURE_SETS[name]),
            "best_val_mse":       round(best_val, 6),
            "test_ade_m":         round(m["ade"], 5),
            "test_fde_m":         round(m["fde"], 5),
            "mean_angular_err":   round(m["mean_angular_error"], 5),
            "mean_turn_rate_pred":round(m["mean_turn_rate_pred"], 5),
            "mean_turn_rate_gt":  round(m["mean_turn_rate_gt"], 5),
            "n_trajs":            m["n_trajs_evaluated"],
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[saved] {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    type=Path, default=DEFAULT_DATA)
    parser.add_argument("--schema",     type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ── load data
    print(f"[load]  {args.dataset.name}")
    df = pd.read_csv(args.dataset)
    print(f"        {len(df):,} rows  x  {df.shape[1]} cols")

    # ── world bounds from schema
    with open(args.schema) as f:
        schema = json.load(f)
    world_bnd = schema["world_bounds_used"]
    print(f"[bounds] xrng={world_bnd['xrng']:.3f} m  yrng={world_bnd['yrng']:.3f} m")

    # ── trajectory-level split
    all_ids = df["trajectory_id"].unique().tolist()
    rng = np.random.default_rng(SEED)
    rng.shuffle(all_ids)
    n = len(all_ids)
    n_tr = int(n * TRAIN_FRAC)
    n_va = int(n * VAL_FRAC)
    train_ids = all_ids[:n_tr]
    val_ids   = all_ids[n_tr : n_tr + n_va]
    test_ids  = all_ids[n_tr + n_va:]
    print(f"[split] {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test trajectories")

    train_df = df[df["trajectory_id"].isin(train_ids)]

    # ── KDTree over training data for spatial lookups during rollout
    kdt_base = build_kdt(train_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir = args.output_dir / "rollout_plots"

    histories: dict = {}
    all_metrics: dict = {}
    models_dict: dict = {}
    f_scs_dict:  dict = {}
    t_scs_dict:  dict = {}
    kdts_dict:   dict = {}

    for name, feat_cols in FEATURE_SETS.items():
        print(f"\n{'='*60}")
        print(f"[model] {name}  ({len(feat_cols)} features)")
        print(f"{'='*60}")
        out_dir = args.output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── fit scalers on training data for this feature set
        f_sc = ColumnScaler().fit(train_df, feat_cols)
        t_sc = ColumnScaler().fit(train_df, TARGET_COLS)

        # ── determine whether this model uses spatial KDTree
        needs_kdt = any(c in feat_cols for c in SPATIAL_COLS)
        kdt       = kdt_base if needs_kdt else None

        # ── train
        model, tr_hist, va_hist, best_val = train_one(
            feat_cols, train_ids, val_ids, df, f_sc, t_sc, out_dir, device,
        )

        histories[name] = (tr_hist, va_hist, best_val)
        models_dict[name] = model
        f_scs_dict[name]  = f_sc
        t_scs_dict[name]  = t_sc
        kdts_dict[name]   = kdt

        # ── evaluate on test set
        print(f"  evaluating on test set ...")
        metrics = evaluate_on_test(
            model, test_ids, df, feat_cols, f_sc, t_sc,
            world_bnd, kdt, device,
        )
        all_metrics[name] = metrics
        print(f"  ADE={metrics['ade']:.4f} m  FDE={metrics['fde']:.4f} m  "
              f"ang_err={metrics['mean_angular_error']:.4f} rad  "
              f"turn_rate_pred={metrics['mean_turn_rate_pred']:.4f}  "
              f"turn_rate_gt={metrics['mean_turn_rate_gt']:.4f}")

    # ── plots
    print("\n[plots] loss curves ...")
    plot_loss_curves(histories, args.output_dir / "loss_curves.png")

    print("[plots] drift curves ...")
    plot_drift_curves(all_metrics, args.output_dir / "drift_curves.png")

    print("[plots] rollout comparisons ...")
    plot_rollouts(
        models_dict, test_ids, df, FEATURE_SETS,
        f_scs_dict, t_scs_dict, world_bnd, kdts_dict, device, rollout_dir,
    )

    # ── outputs
    write_csv(all_metrics, histories, args.output_dir / "ablation_results.csv")
    write_summary(all_metrics, histories, args.output_dir / "ablation_summary.md")

    print(f"\n[done]  results in {args.output_dir}")


if __name__ == "__main__":
    main()
