"""
train_phase2c_fixed_spatial.py
-------------------------------
Phase 2C: Fix spatial feature integration during autoregressive rollout.

Problem (Phase 2A):
  During rollout, predicted positions were updated but dist_to_obstacle and
  dist_to_boundary were copied from ground-truth future frames.  This creates
  contradictory inputs: the position says "predicted location" but the spatial
  features say "ground-truth location".

Fix:
  At every rollout step, after computing (x_next, y_next), recompute
  dist_to_obstacle and dist_to_boundary from the predicted position using a
  KDTree spatial interpolator built from the encoded dataset.

Anchor points note:
  The original obstacle/boundary anchor clicks are NOT saved to disk (encode_space.py
  collects them interactively and never persists them).  Reconstruction approach:
  a KDTree is built over all 180,840 (world_x, world_y) rows from
  trajectories_encoded.csv.  For each predicted position during rollout, the
  5 nearest encoded points are found and their spatial feature values are
  combined with inverse-distance weighting.  This gives a smooth,
  physically-consistent estimate of both distances at any predicted location
  within or near the trajectory envelope.

Comparison:
  Model A  movement-only   world_x, world_y, delta_x, delta_y
  Model B  fixed-spatial   + dist_to_obstacle, dist_to_boundary
                             (recomputed from KDTree at every rollout step)

Output:
  mp-data/outputs/prediction/experiments/phase-2c/fixed_spatial_rollout/
    movement_only/  model.pth  scaler.pkl  loss_curve.png  prediction_visual.png
    fixed_spatial/  model.pth  scaler.pkl  loss_curve.png  prediction_visual.png
  mp-data/outputs/prediction/experiments/phase-2c/
    phase2c_fixed_spatial_vs_movement.png
"""

import pickle
import sys
import time
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

HERE    = Path(__file__).resolve().parent
MP_ROOT = HERE.parent.parent.parent.parent
MP_DATA = MP_ROOT / "mp-data"

# Use original encoded CSV (real data, not the 10x duplicate)
SRC_CSV   = MP_DATA / "processed" / "encoded" / "trajectories_encoded.csv"
PHASE2C   = MP_DATA / "outputs" / "prediction" / "experiments" / "phase-2c"
FSR_OUT   = PHASE2C / "fixed_spatial_rollout"
MOVE_OUT  = FSR_OUT / "movement_only"
SPAT_OUT  = FSR_OUT / "fixed_spatial"
COMP_PATH = PHASE2C / "phase2c_fixed_spatial_vs_movement.png"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
WINDOW_SIZE   = 10
HIDDEN_SIZE   = 256
EPOCHS        = 30
LEARNING_RATE = 1e-3
BATCH_SIZE    = 1024
PRINT_EVERY   = 5
MAX_SEQUENCES = 100_000
N_PERSONS     = 400
N_ROLLOUT     = 30
IDW_K         = 5      # neighbours for inverse-distance spatial interpolation

MOVEMENT_COLS = ["world_x", "world_y", "delta_x", "delta_y"]
SPATIAL_COLS  = ["world_x", "world_y", "delta_x", "delta_y",
                 "dist_to_obstacle", "dist_to_boundary"]
TARGET_COLS   = ["delta_x", "delta_y"]

# dist_to_entrance excluded: highly redundant with world_x/world_y (|r| ~ 0.7)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, output_size: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ---------------------------------------------------------------------------
# Spatial interpolator  (replaces missing anchor points)
# ---------------------------------------------------------------------------

class SpatialInterpolator:
    """KDTree over the full encoded dataset.

    For any predicted (x, y), returns dist_to_obstacle and dist_to_boundary
    via inverse-distance weighted interpolation over the k nearest encoded
    trajectory points.  Serves as a proxy for the original anchor-based
    distance function when anchor files are unavailable.
    """

    def __init__(self, df_encoded: pd.DataFrame, k: int = 5):
        coords = df_encoded[["world_x", "world_y"]].to_numpy(dtype=np.float64)
        self.tree     = cKDTree(coords)
        self.obs_vals = df_encoded["dist_to_obstacle"].to_numpy(dtype=np.float64)
        self.bnd_vals = df_encoded["dist_to_boundary"].to_numpy(dtype=np.float64)
        self.k        = k

    def query(self, x: float, y: float):
        """Return (dist_to_obstacle, dist_to_boundary) at world position (x, y)."""
        dists, idxs = self.tree.query([x, y], k=self.k)
        if dists[0] < 1e-9:          # exact dataset point
            return float(self.obs_vals[idxs[0]]), float(self.bnd_vals[idxs[0]])
        w = 1.0 / dists
        w /= w.sum()
        obs = float(np.dot(w, self.obs_vals[idxs]))
        bnd = float(np.dot(w, self.bnd_vals[idxs]))
        return obs, bnd


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["person_id", "frame_number"])
    df["delta_x"] = df.groupby("person_id")["world_x"].diff().fillna(0)
    df["delta_y"] = df.groupby("person_id")["world_y"].diff().fillna(0)
    return df


def build_sequences(df: pd.DataFrame, feature_cols: list):
    X, y = [], []
    for _, track in df.groupby("person_id"):
        track = track.reset_index(drop=True)
        if len(track) <= WINDOW_SIZE:
            continue
        feats   = track[feature_cols].to_numpy(dtype=np.float32)
        targets = track[TARGET_COLS].to_numpy(dtype=np.float32)
        for i in range(len(track) - WINDOW_SIZE):
            X.append(feats[i: i + WINDOW_SIZE])
            y.append(targets[i + WINDOW_SIZE])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Train one model
# ---------------------------------------------------------------------------

def train_one(X, y, out_dir, label, device):
    N, W, F = X.shape
    fs = MinMaxScaler()
    ts = MinMaxScaler()
    Xs = fs.fit_transform(X.reshape(-1, F)).reshape(N, W, F).astype(np.float32)
    ys = ts.fit_transform(y).astype(np.float32)

    loader = DataLoader(TensorDataset(torch.tensor(Xs), torch.tensor(ys)),
                        batch_size=BATCH_SIZE, shuffle=True)

    model = TrajectoryLSTM(F, HIDDEN_SIZE).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    crit  = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{label}]  {F} features  |  {n_params:,} params  |  "
          f"{len(loader)} batches/epoch  |  {N:,} seqs")
    print(f"[{label}]  Training {EPOCHS} epochs ...")

    t0 = time.time()
    history = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        history.append(avg)
        if ep % PRINT_EVERY == 0 or ep == 1:
            print(f"  Epoch {ep:>3d}/{EPOCHS}  loss {avg:.6f}")
    elapsed = time.time() - t0
    print(f"[{label}]  Done {elapsed:.1f}s  |  final loss {history[-1]:.6f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pth")
    bundle = {"feature_scaler": fs, "target_scaler": ts,
              "feature_cols": list(X.shape), "target_cols": TARGET_COLS,
              "window_size": WINDOW_SIZE}
    with open(out_dir / "scaler.pkl", "wb") as fh:
        pickle.dump(bundle, fh)

    color = "#c0392b" if "movement" in label.lower() else "#2980b9"
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, EPOCHS + 1), history, lw=1.5, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalised delta)")
    ax.set_title(f"Phase 2C — {label}  |  {EPOCHS} ep  |  {N:,} seqs")
    ax.grid(True, lw=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    return model, fs, ts, history, elapsed


# ---------------------------------------------------------------------------
# Rollout — movement-only
# ---------------------------------------------------------------------------

def rollout_movement(model, fs, ts, track: pd.DataFrame, device):
    """Standard delta rollout: no spatial recomputation needed."""
    model.eval()
    feature_cols = MOVEMENT_COLS
    feats     = track[feature_cols].to_numpy(dtype=np.float32)
    positions = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    T = len(feats)

    feats_scaled = fs.transform(feats)
    window = feats_scaled[:WINDOW_SIZE].copy()
    prev_x, prev_y = positions[WINDOW_SIZE - 1]
    pred_x, pred_y = [], []

    for step in range(N_ROLLOUT):
        x_in = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            p_s = model(x_in).cpu().numpy()[0]
        dx, dy = ts.inverse_transform(p_s[np.newaxis])[0]
        wx = float(prev_x) + float(dx)
        wy = float(prev_y) + float(dy)
        pred_x.append(wx)
        pred_y.append(wy)

        new_raw = np.array([wx, wy, float(dx), float(dy)], dtype=np.float32)
        window = np.vstack([window[1:], fs.transform(new_raw[np.newaxis])[0]])
        prev_x, prev_y = wx, wy

    return np.array(pred_x), np.array(pred_y), positions


# ---------------------------------------------------------------------------
# Rollout — fixed spatial  (THE KEY FIX)
# ---------------------------------------------------------------------------

def rollout_fixed_spatial(model, fs, ts, track: pd.DataFrame,
                          interp: SpatialInterpolator, device):
    """Delta rollout with KDTree-recomputed spatial features.

    At every step:
      1. Predict (dx, dy)
      2. Accumulate: x_next = x + dx, y_next = y + dy
      3. Recompute dist_to_obstacle, dist_to_boundary from KDTree at (x_next, y_next)
      4. Feed [x_next, y_next, dx, dy, obs, bnd] as next window row
    """
    model.eval()
    feature_cols = SPATIAL_COLS
    feats     = track[feature_cols].to_numpy(dtype=np.float32)
    positions = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    T = len(feats)

    feats_scaled = fs.transform(feats)
    window = feats_scaled[:WINDOW_SIZE].copy()
    prev_x, prev_y = positions[WINDOW_SIZE - 1]
    pred_x, pred_y = [], []

    for step in range(N_ROLLOUT):
        x_in = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            p_s = model(x_in).cpu().numpy()[0]
        dx, dy = ts.inverse_transform(p_s[np.newaxis])[0]
        wx = float(prev_x) + float(dx)
        wy = float(prev_y) + float(dy)
        pred_x.append(wx)
        pred_y.append(wy)

        # --- THE FIX: recompute spatial features from predicted position ---
        obs, bnd = interp.query(wx, wy)

        new_raw = np.array([wx, wy, float(dx), float(dy),
                            obs, bnd], dtype=np.float32)
        window = np.vstack([window[1:], fs.transform(new_raw[np.newaxis])[0]])
        prev_x, prev_y = wx, wy

    return np.array(pred_x), np.array(pred_y), positions


# ---------------------------------------------------------------------------
# Per-model prediction visual
# ---------------------------------------------------------------------------

def save_prediction_visual(pred_x, pred_y, positions, out_path, label):
    seed_x = positions[:WINDOW_SIZE, 0]
    seed_y = positions[:WINDOW_SIZE, 1]
    gt_x   = positions[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 0]
    gt_y   = positions[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(np.r_[seed_x, gt_x], np.r_[seed_y, gt_y],
             color="#27ae60", lw=1.5, alpha=0.7, label="GT path")
    ax1.plot(seed_x, seed_y, "-o", color="#e67e22", ms=4, lw=2.2, label="Seed")
    ax1.plot(pred_x, pred_y, "r--o", ms=3, alpha=0.85, label=f"{label} pred")
    ax1.legend(fontsize=8)
    ax1.set_title(f"{label} — 2D [delta rollout]")
    ax1.set_xlabel("world_x (m)")
    ax1.set_ylabel("world_y (m)")
    ax1.grid(True, lw=0.4, alpha=0.6)

    steps = range(N_ROLLOUT)
    ax2.plot(steps, gt_x,   "b-",  label="GT world_x")
    ax2.plot(steps, pred_x, "r--", label="pred world_x")
    ax2.plot(steps, gt_y,   "b:",  label="GT world_y")
    ax2.plot(steps, pred_y, "r-.", label="pred world_y")
    ax2.legend(fontsize=8)
    ax2.set_title(f"{label} — X/Y over rollout steps")
    ax2.set_xlabel("Rollout step")
    ax2.set_ylabel("World coordinate (m)")
    ax2.grid(True, lw=0.4, alpha=0.6)

    fig.suptitle(
        f"Phase 2C — {label}  |  delta prediction  |  {N_ROLLOUT}-step rollout\n"
        "[NOT generalisation evidence]",
        fontsize=10, color="#7f0000",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO]  Prediction visual: {out_path}")


# ---------------------------------------------------------------------------
# Comparison figure
# ---------------------------------------------------------------------------

def _draw_panel(ax, seed_x, seed_y, gt_x, gt_y, pred_x, pred_y,
                title, subtitle, badge, xlim, ylim, drift_m):
    ax.set_facecolor("#fafafa")
    full_x = np.r_[seed_x, gt_x]
    full_y = np.r_[seed_y, gt_y]
    ax.plot(full_x, full_y, color="#27ae60", lw=1.5, alpha=0.6, zorder=2)
    ax.plot(full_x, full_y, "o", color="#27ae60", ms=3.5, alpha=0.5, zorder=2)
    ax.plot(seed_x, seed_y, color="#e67e22", lw=2.8, zorder=4)
    ax.plot(seed_x, seed_y, "o", color="#e67e22", ms=5, zorder=4)
    ax.plot(seed_x[-1], seed_y[-1], "o", color="#e67e22", ms=9,
            zorder=5, markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(pred_x, pred_y, color="#e74c3c", lw=2.2, ls="--", zorder=6)
    ax.plot(pred_x, pred_y, "o", color="#e74c3c", ms=4, alpha=0.85, zorder=6)
    ax.plot(pred_x[0], pred_y[0], "o", color="#e74c3c", ms=9,
            zorder=7, markeredgecolor="white", markeredgewidth=1.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("world_x  (m)", fontsize=10)
    ax.set_ylabel("world_y  (m)", fontsize=10)
    ax.grid(True, lw=0.35, alpha=0.5, color="#cccccc")
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold",
                 pad=10, color="#2c3e50")
    ax.text(0.025, 0.975, badge, transform=ax.transAxes, fontsize=8.5,
            va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#ecf0f1",
                      edgecolor="#bdc3c7", alpha=0.95))
    dc = "#e74c3c" if drift_m > 0.3 else "#27ae60"
    ax.text(0.975, 0.025, f"endpoint drift  {drift_m:.2f} m",
            transform=ax.transAxes, fontsize=8.5, va="bottom", ha="right",
            color=dc, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#fef9f9" if drift_m > 0.3 else "#eafaf1",
                      edgecolor=dc, alpha=0.92))


def save_comparison(seed_x, seed_y, gt_x, gt_y,
                    pred_xa, pred_ya, drift_a, loss_a,
                    pred_xb, pred_yb, drift_b, loss_b):
    all_x = np.concatenate([seed_x, gt_x, pred_xa, pred_xb])
    all_y = np.concatenate([seed_y, gt_y, pred_ya, pred_yb])
    pad   = max(0.15, (all_x.max() - all_x.min()) * 0.08)
    xlim  = (float(all_x.min()) - pad, float(all_x.max()) + pad)
    ylim  = (float(all_y.min()) - pad, float(all_y.max()) + pad)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#f5f5f5")

    _draw_panel(
        ax1, seed_x, seed_y, gt_x, gt_y, pred_xa, pred_ya,
        title    = "Movement-only LSTM",
        subtitle = f"4 inputs  |  loss {loss_a:.6f}",
        badge    = "world_x  world_y  delta_x  delta_y",
        xlim=xlim, ylim=ylim, drift_m=drift_a,
    )
    _draw_panel(
        ax2, seed_x, seed_y, gt_x, gt_y, pred_xb, pred_yb,
        title    = "Fixed-spatial LSTM",
        subtitle = f"6 inputs  |  loss {loss_b:.6f}  |  KDTree recomputed",
        badge    = "+ dist_to_obstacle  dist_to_boundary\n(recomputed at each rollout step)",
        xlim=xlim, ylim=ylim, drift_m=drift_b,
    )

    handles = [
        mpatches.Patch(facecolor="#27ae60", label="Real path (GT)"),
        mpatches.Patch(facecolor="#e67e22", label="Seed / history  (10 steps)"),
        mpatches.Patch(facecolor="#e74c3c", label="Predicted rollout  (30 steps)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10.5,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.005), edgecolor="#cccccc")
    fig.suptitle(
        "Phase 2C  —  Fixed Spatial Rollout vs Movement-only\n"
        "LSTM  |  delta prediction  |  same track  |  30-step horizon\n"
        "Spatial fix: dist_to_obstacle & dist_to_boundary recomputed via KDTree "
        "at each predicted position\n"
        "[NOT generalisation evidence  —  controlled ablation only]",
        fontsize=10.5, color="#7f0000", y=1.02,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.98])
    fig.savefig(COMP_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK]  Comparison figure: {COMP_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device : {device}")

    if not SRC_CSV.exists():
        sys.exit(f"[ERROR] {SRC_CSV} not found. Run encode_space.py first.")

    # ── Load encoded data ──────────────────────────────────────────────────
    df_full = pd.read_csv(SRC_CSV)
    if "track_id" in df_full.columns and "person_id" not in df_full.columns:
        df_full = df_full.rename(columns={"track_id": "person_id"})
    if "frame_idx" in df_full.columns and "frame_number" not in df_full.columns:
        df_full = df_full.rename(columns={"frame_idx": "frame_number"})
    print(f"[INFO]  Loaded {len(df_full):,} rows  |  "
          f"{df_full['person_id'].nunique()} persons")

    required = ["world_x", "world_y", "dist_to_obstacle", "dist_to_boundary"]
    if miss := set(required) - set(df_full.columns):
        sys.exit(f"[ERROR] Missing columns: {miss}")

    # ── Build spatial interpolator from FULL dataset before filtering ──────
    # Use all 180K points so the KDTree covers the entire trajectory envelope.
    print("[INFO]  Building spatial interpolator (KDTree over full dataset) ...")
    interp = SpatialInterpolator(df_full, k=IDW_K)
    print(f"[INFO]  KDTree built on {len(df_full):,} points  |  k={IDW_K}")

    # ── Validate interpolator accuracy on a random sample ─────────────────
    rng = np.random.default_rng(0)
    val_idx = rng.choice(len(df_full), min(500, len(df_full)), replace=False)
    val_df  = df_full.iloc[val_idx]
    obs_errs, bnd_errs = [], []
    for _, row in val_df.iterrows():
        obs_hat, bnd_hat = interp.query(row["world_x"], row["world_y"])
        obs_errs.append(abs(obs_hat - row["dist_to_obstacle"]))
        bnd_errs.append(abs(bnd_hat - row["dist_to_boundary"]))
    print(f"[INFO]  Interpolation validation (n=500):")
    print(f"          dist_to_obstacle  MAE = {np.mean(obs_errs):.4f} m  "
          f"max = {np.max(obs_errs):.4f} m")
    print(f"          dist_to_boundary  MAE = {np.mean(bnd_errs):.4f} m  "
          f"max = {np.max(bnd_errs):.4f} m")

    # ── Pre-filter persons ─────────────────────────────────────────────────
    all_pids = df_full["person_id"].unique()
    chosen   = np.random.default_rng(42).choice(
        all_pids, min(N_PERSONS, len(all_pids)), replace=False)
    df = df_full[df_full["person_id"].isin(chosen)].copy()
    print(f"[INFO]  Pre-filtered to {len(chosen)} persons  ({len(df):,} rows)")

    df = add_deltas(df)

    # ── Fix comparison track ───────────────────────────────────────────────
    min_len    = WINDOW_SIZE + N_ROLLOUT + 1
    candidates = [(pid, grp.reset_index(drop=True))
                  for pid, grp in df.groupby("person_id")
                  if len(grp) >= min_len]
    if not candidates:
        sys.exit("[ERROR] No track is long enough for rollout.")
    comp_pid, comp_track = candidates[len(candidates) // 2]
    print(f"[INFO]  Comparison track : person_id={comp_pid}  "
          f"length={len(comp_track)}")

    # ── Build sequences for both models ───────────────────────────────────
    print("\n[INFO]  Building sequences ...")
    X_move, y_move = build_sequences(df, MOVEMENT_COLS)
    X_spat, y_spat = build_sequences(df, SPATIAL_COLS)
    print(f"[INFO]  Movement-only : {len(X_move):,} seqs  ({X_move.shape})")
    print(f"[INFO]  Fixed-spatial : {len(X_spat):,} seqs  ({X_spat.shape})")

    # Subsample — same random indices for both so they train on the same examples
    if len(X_move) > MAX_SEQUENCES:
        idx = np.random.default_rng(42).choice(len(X_move), MAX_SEQUENCES, replace=False)
        X_move, y_move = X_move[idx], y_move[idx]
        X_spat, y_spat = X_spat[idx], y_spat[idx]
        print(f"[INFO]  Subsampled to {len(X_move):,} sequences (same index for both)")

    # ── Train movement-only ─────────────────────────────────────────────────
    print(f"\n{'='*54}")
    print("  Training  Movement-only LSTM")
    print(f"{'='*54}")
    m_model, m_fs, m_ts, m_hist, m_elapsed = train_one(
        X_move, y_move, MOVE_OUT, "Movement-only", device)

    m_pred_x, m_pred_y, positions = rollout_movement(
        m_model, m_fs, m_ts, comp_track, device)
    save_prediction_visual(m_pred_x, m_pred_y, positions,
                           MOVE_OUT / "prediction_visual.png", "Movement-only")

    gt_end  = positions[WINDOW_SIZE + N_ROLLOUT - 1]
    m_drift = float(np.sqrt(
        (m_pred_x[-1] - gt_end[0]) ** 2 + (m_pred_y[-1] - gt_end[1]) ** 2))
    print(f"[Movement-only]  endpoint drift : {m_drift:.3f} m")

    # ── Train fixed-spatial ─────────────────────────────────────────────────
    print(f"\n{'='*54}")
    print("  Training  Fixed-spatial LSTM")
    print(f"{'='*54}")
    s_model, s_fs, s_ts, s_hist, s_elapsed = train_one(
        X_spat, y_spat, SPAT_OUT, "Fixed-spatial", device)

    s_pred_x, s_pred_y, _ = rollout_fixed_spatial(
        s_model, s_fs, s_ts, comp_track, interp, device)
    save_prediction_visual(s_pred_x, s_pred_y, positions,
                           SPAT_OUT / "prediction_visual.png", "Fixed-spatial")

    s_drift = float(np.sqrt(
        (s_pred_x[-1] - gt_end[0]) ** 2 + (s_pred_y[-1] - gt_end[1]) ** 2))
    print(f"[Fixed-spatial]  endpoint drift : {s_drift:.3f} m")

    # ── Comparison figure ───────────────────────────────────────────────────
    seed_x = positions[:WINDOW_SIZE, 0]
    seed_y = positions[:WINDOW_SIZE, 1]
    gt_x   = positions[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 0]
    gt_y   = positions[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 1]
    PHASE2C.mkdir(parents=True, exist_ok=True)
    save_comparison(seed_x, seed_y, gt_x, gt_y,
                    m_pred_x, m_pred_y, m_drift, m_hist[-1],
                    s_pred_x, s_pred_y, s_drift, s_hist[-1])

    # ── Report ──────────────────────────────────────────────────────────────
    drift_improved  = s_drift < m_drift
    drift_delta     = m_drift - s_drift
    loss_delta      = s_hist[-1] - m_hist[-1]

    print(f"\n{'='*60}")
    print("  PHASE 2C — FIXED SPATIAL ROLLOUT RESULTS")
    print(f"{'='*60}")
    print(f"  {'Model':<22}  {'Final Loss':>12}  {'Runtime':>10}  {'Drift':>10}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*10}  {'-'*10}")
    print(f"  {'Movement-only':<22}  {m_hist[-1]:>12.6f}  {m_elapsed:>8.1f}s  "
          f"{m_drift:>8.3f} m")
    print(f"  {'Fixed-spatial':<22}  {s_hist[-1]:>12.6f}  {s_elapsed:>8.1f}s  "
          f"{s_drift:>8.3f} m")
    print(f"  {'-'*22}  {'-'*12}  {'-'*10}  {'-'*10}")
    print(f"\n  Loss delta  (spatial - movement) : {loss_delta:+.6f}")
    print(f"  Drift delta (movement - spatial) : {drift_delta:+.3f} m  "
          f"({'spatial better' if drift_improved else 'movement better'})")
    print(f"\n  Spatial recomputation method : KDTree IDW (k={IDW_K}) "
          f"over {len(df_full):,} encoded points")
    print(f"  Anchor points availability   : NOT saved to disk")
    print(f"  Fallback used                : KDTree nearest-neighbour "
          f"interpolation from trajectories_encoded.csv")
    print(f"\n  Did fixed spatial rollout reduce drift?  "
          f"{'YES  ({:.3f} m improvement)'.format(drift_delta) if drift_improved else 'NO'}")
    print(f"\n  Conclusion:")
    if drift_improved and abs(drift_delta) > 0.05:
        print("    Recomputing spatial features at each rollout step made")
        print("    spatial inputs useful. The fixed-spatial model achieved")
        print(f"   lower endpoint drift ({s_drift:.3f} m vs {m_drift:.3f} m).")
        print("    Spatial context, when kept consistent with predicted")
        print("    position, contributes real signal to the prediction.")
    elif drift_improved:
        print("    Marginal improvement from spatial recomputation.")
        print(f"   Drift reduced by only {drift_delta:.3f} m.")
        print("    Spatial features are consistent but add little signal")
        print("    beyond what movement features already capture.")
    else:
        print("    Spatial recomputation did not reduce drift.")
        print(f"   Fixed-spatial: {s_drift:.3f} m vs movement-only: {m_drift:.3f} m.")
        print("    Possible reasons: 30 epochs insufficient for spatial")
        print("    feature learning; KDTree interpolation introduces noise;")
        print("    movement features already capture trajectory structure.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
