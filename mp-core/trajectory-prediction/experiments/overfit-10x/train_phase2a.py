"""
train_phase2a.py
----------------
Phase 2A: Does spatial information improve prediction?

Two LSTM models are trained on the same data with the same delta prediction
target.  The only difference is the input feature set:

  Model A  movement-only   world_x, world_y, delta_x, delta_y
  Model B  spatial-aware   + dist_to_obstacle, dist_to_boundary,
                             dist_to_entrance

Quick-sanity settings: 100 K sequences, 30 epochs, batch 1024.
Both models use the same track, seed, and prediction horizon for comparison.

Outputs
-------
  mp-data/outputs/prediction/experiments/phase-2a/
    movement_only/  model.pth  scaler.pkl  loss_curve.png  prediction_visual.png
    spatial_aware/  model.pth  scaler.pkl  loss_curve.png  prediction_visual.png
    phase2a_movement_vs_spatial.png

NOT final thesis evaluation. NOT generalisation evidence.
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
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

HERE    = Path(__file__).resolve().parent
MP_ROOT = HERE.parent.parent.parent.parent
MP_DATA = MP_ROOT / "mp-data"

SRC_CSV     = (MP_DATA / "outputs" / "prediction" / "experiments"
               / "overfit-10x" / "trajectories_overfit.csv")
PHASE2A_OUT = MP_DATA / "outputs" / "prediction" / "experiments" / "phase-2a"
MOVE_OUT    = PHASE2A_OUT / "movement_only"
SPAT_OUT    = PHASE2A_OUT / "spatial_aware"
COMP_PATH   = PHASE2A_OUT / "phase2a_movement_vs_spatial.png"

# ---------------------------------------------------------------------------
# Quick-sanity settings
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

MOVEMENT_COLS = ["world_x", "world_y", "delta_x", "delta_y"]
SPATIAL_COLS  = ["world_x", "world_y", "delta_x", "delta_y",
                 "dist_to_obstacle", "dist_to_boundary", "dist_to_entrance"]
TARGET_COLS   = ["delta_x", "delta_y"]   # Phase 1B delta prediction


# ---------------------------------------------------------------------------
# Model  (architecture unchanged — only input_size differs)
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

def train_one(df, feature_cols, out_dir, label, device):
    """Build sequences, train, persist model + scaler + loss curve."""
    print(f"\n[{label}]  Building sequences ...")
    X, y = build_sequences(df, feature_cols)
    print(f"[{label}]  {len(X):,} seqs  ->  subsample {MAX_SEQUENCES:,}")
    if len(X) > MAX_SEQUENCES:
        idx = np.random.default_rng(42).choice(len(X), MAX_SEQUENCES, replace=False)
        X, y = X[idx], y[idx]

    fs = MinMaxScaler()
    ts = MinMaxScaler()
    N, W, F = X.shape
    Xs = fs.fit_transform(X.reshape(-1, F)).reshape(N, W, F).astype(np.float32)
    ys = ts.fit_transform(y).astype(np.float32)

    loader = DataLoader(TensorDataset(torch.tensor(Xs), torch.tensor(ys)),
                        batch_size=BATCH_SIZE, shuffle=True)

    model = TrajectoryLSTM(len(feature_cols), HIDDEN_SIZE).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    crit  = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{label}]  {len(feature_cols)} features  |  {n_params:,} params  |  "
          f"{len(loader)} batches/epoch")
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

    # Persist
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pth")
    bundle = {"feature_scaler": fs, "target_scaler": ts,
              "feature_cols": feature_cols, "target_cols": TARGET_COLS,
              "window_size": WINDOW_SIZE}
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(bundle, f)

    # Loss curve
    color = "#c0392b" if "movement" in label.lower() else "#2980b9"
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, EPOCHS + 1), history, lw=1.5, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalised delta)")
    ax.set_title(f"Phase 2A — {label}  |  {EPOCHS} ep  |  {len(X):,} seqs")
    ax.grid(True, lw=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    return model, fs, ts, history, elapsed


# ---------------------------------------------------------------------------
# Rollout  (delta prediction — same logic as Phase 1B)
# ---------------------------------------------------------------------------

def rollout(model, fs, ts, track: pd.DataFrame, feature_cols: list, device):
    """30-step autoregressive delta rollout on comp_track."""
    model.eval()
    feats     = track[feature_cols].to_numpy(dtype=np.float32)   # (T, F)
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

        src = min(WINDOW_SIZE + step, T - 1)
        # Start from GT values at this step, then override predicted quantities
        new_vals = {col: float(feats[src, i]) for i, col in enumerate(feature_cols)}
        new_vals["world_x"] = wx
        new_vals["world_y"] = wy
        new_vals["delta_x"] = float(dx)
        new_vals["delta_y"] = float(dy)
        new_raw = np.array([new_vals[c] for c in feature_cols], dtype=np.float32)
        window = np.vstack([window[1:], fs.transform(new_raw[np.newaxis])[0]])
        prev_x, prev_y = wx, wy

    return np.array(pred_x), np.array(pred_y), positions


# ---------------------------------------------------------------------------
# Per-model prediction visual
# ---------------------------------------------------------------------------

def save_prediction_visual(model, fs, ts, track, feature_cols,
                            out_path, label, device):
    pred_x, pred_y, positions = rollout(model, fs, ts, track, feature_cols, device)
    seed_x = positions[:WINDOW_SIZE, 0]
    seed_y = positions[:WINDOW_SIZE, 1]
    gt_x   = positions[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 0]
    gt_y   = positions[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(np.r_[seed_x, gt_x], np.r_[seed_y, gt_y],
             "g-o", ms=3, alpha=0.7, label="GT path")
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
        f"Phase 2A — {label}  |  delta prediction  |  {N_ROLLOUT}-step rollout\n"
        "[NOT generalisation evidence]",
        fontsize=10, color="#7f0000",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO]  Prediction visual: {out_path}")
    return pred_x, pred_y


# ---------------------------------------------------------------------------
# Comparison figure
# ---------------------------------------------------------------------------

def _draw_panel(ax, seed_x, seed_y, gt_x, gt_y, pred_x, pred_y,
                title, subtitle, badge, xlim, ylim, drift_m, n_feats):
    ax.set_facecolor("#fafafa")
    all_x = np.r_[seed_x, gt_x]
    all_y = np.r_[seed_y, gt_y]
    ax.plot(all_x, all_y, color="#27ae60", lw=1.5, alpha=0.6, zorder=2)
    ax.plot(all_x, all_y, "o", color="#27ae60", ms=3.5, alpha=0.5, zorder=2)
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
    ax.set_title(f"{title}\n{subtitle}",
                 fontsize=11, fontweight="bold", pad=10, color="#2c3e50")
    ax.text(0.025, 0.975, badge,
            transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
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
                    pred_xa, pred_ya, pred_xb, pred_yb,
                    drift_a, drift_b, loss_a, loss_b):
    all_x = np.concatenate([seed_x, gt_x, pred_xa, pred_xb])
    all_y = np.concatenate([seed_y, gt_y, pred_ya, pred_yb])
    pad   = max(0.12, (all_x.max() - all_x.min()) * 0.08)
    xlim  = (float(all_x.min()) - pad, float(all_x.max()) + pad)
    ylim  = (float(all_y.min()) - pad, float(all_y.max()) + pad)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#f5f5f5")

    _draw_panel(
        ax1, seed_x, seed_y, gt_x, gt_y, pred_xa, pred_ya,
        title    = "Movement-only Model",
        subtitle = (f"Inputs: world_x, world_y, dx, dy  "
                    f"|  loss {loss_a:.6f}"),
        badge    = "4 input features",
        xlim=xlim, ylim=ylim, drift_m=drift_a, n_feats=4,
    )
    _draw_panel(
        ax2, seed_x, seed_y, gt_x, gt_y, pred_xb, pred_yb,
        title    = "Spatial-aware Model",
        subtitle = (f"+ dist_obstacle, dist_boundary, dist_entrance  "
                    f"|  loss {loss_b:.6f}"),
        badge    = "7 input features",
        xlim=xlim, ylim=ylim, drift_m=drift_b, n_feats=7,
    )

    handles = [
        mpatches.Patch(facecolor="#27ae60", label="Real path (GT)"),
        mpatches.Patch(facecolor="#e67e22", label="Seed / history window  (10 steps)"),
        mpatches.Patch(facecolor="#e74c3c", label="Predicted rollout  (30 steps)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10.5,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.005), edgecolor="#cccccc")

    fig.suptitle(
        "Phase 2A — Does Spatial Information Improve Prediction?\n"
        "LSTM  |  delta prediction target  |  same track  |  same seed  "
        "|  30-step horizon\n"
        "[NOT generalisation evidence — controlled ablation only]",
        fontsize=11, color="#7f0000", y=1.015,
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
        sys.exit(f"[ERROR] {SRC_CSV} not found. Run make_overfit_dataset.py first.")

    # ── Load & filter data ─────────────────────────────────────────────────
    df = pd.read_csv(SRC_CSV)
    if "track_id" in df.columns and "person_id" not in df.columns:
        df = df.rename(columns={"track_id": "person_id"})
    if "frame_idx" in df.columns and "frame_number" not in df.columns:
        df = df.rename(columns={"frame_idx": "frame_number"})
    print(f"[INFO]  Loaded {len(df):,} rows  |  {df['person_id'].nunique()} persons")

    all_pids = df["person_id"].unique()
    chosen   = np.random.default_rng(42).choice(
        all_pids, min(N_PERSONS, len(all_pids)), replace=False)
    df = df[df["person_id"].isin(chosen)].copy()
    print(f"[INFO]  Pre-filtered to {len(chosen)} persons  ({len(df):,} rows)")

    # Check only static columns (deltas are computed by add_deltas below)
    static_cols = [c for c in SPATIAL_COLS if c not in ("delta_x", "delta_y")]
    missing = set(static_cols) - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Missing columns: {missing}")

    df = add_deltas(df)

    # ── Fix comparison track before training (same for both models) ────────
    min_len    = WINDOW_SIZE + N_ROLLOUT + 1
    candidates = [(pid, grp.reset_index(drop=True))
                  for pid, grp in df.groupby("person_id")
                  if len(grp) >= min_len]
    if not candidates:
        sys.exit("[ERROR] No track is long enough for rollout.")
    comp_pid, comp_track = candidates[len(candidates) // 2]
    print(f"[INFO]  Comparison track : person_id={comp_pid}  "
          f"length={len(comp_track)}")

    # ── Train both models ──────────────────────────────────────────────────
    model_a, fs_a, ts_a, hist_a, time_a = train_one(
        df, MOVEMENT_COLS, MOVE_OUT, "Movement-only", device)

    model_b, fs_b, ts_b, hist_b, time_b = train_one(
        df, SPATIAL_COLS, SPAT_OUT, "Spatial-aware", device)

    # ── Per-model prediction visuals ───────────────────────────────────────
    pred_xa, pred_ya = save_prediction_visual(
        model_a, fs_a, ts_a, comp_track, MOVEMENT_COLS,
        MOVE_OUT / "prediction_visual.png", "Movement-only", device)

    pred_xb, pred_yb = save_prediction_visual(
        model_b, fs_b, ts_b, comp_track, SPATIAL_COLS,
        SPAT_OUT / "prediction_visual.png", "Spatial-aware", device)

    # ── Comparison figure ──────────────────────────────────────────────────
    positions = comp_track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    seed_x = positions[:WINDOW_SIZE, 0]
    seed_y = positions[:WINDOW_SIZE, 1]
    gt_x   = positions[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 0]
    gt_y   = positions[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 1]

    origin_x, origin_y = float(positions[WINDOW_SIZE, 0]), float(positions[WINDOW_SIZE, 1])
    drift_a = float(np.hypot(pred_xa[-1] - origin_x, pred_ya[-1] - origin_y))
    drift_b = float(np.hypot(pred_xb[-1] - origin_x, pred_yb[-1] - origin_y))

    save_comparison(seed_x, seed_y, gt_x, gt_y,
                    pred_xa, pred_ya, pred_xb, pred_yb,
                    drift_a, drift_b, hist_a[-1], hist_b[-1])

    # ── Final report ───────────────────────────────────────────────────────
    sep = "=" * 60
    drift_delta = drift_a - drift_b
    loss_delta  = hist_a[-1] - hist_b[-1]
    spatial_helped = (drift_b <= drift_a) and (hist_b[-1] <= hist_a[-1] * 1.05)
    conclusion = "YES" if spatial_helped else "NO"

    print(f"\n{sep}")
    print("  PHASE 2A — RESULTS")
    print(sep)
    print(f"  {'Model':<22}  {'Final loss':>12}  {'Runtime':>9}  {'Drift':>8}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*9}  {'-'*8}")
    print(f"  {'Movement-only':<22}  {hist_a[-1]:>12.6f}  {time_a:>8.1f}s"
          f"  {drift_a:>7.3f}m")
    print(f"  {'Spatial-aware':<22}  {hist_b[-1]:>12.6f}  {time_b:>8.1f}s"
          f"  {drift_b:>7.3f}m")
    print(f"  {'-'*22}  {'-'*12}  {'-'*9}  {'-'*8}")
    print(f"  Loss improvement    : {loss_delta:+.6f}  "
          f"({'spatial better' if loss_delta > 0 else 'movement better'})")
    print(f"  Drift improvement   : {drift_delta:+.3f} m  "
          f"({'spatial better' if drift_delta > 0 else 'movement better'})")
    print(f"\n  Did spatial features help?  -->  {conclusion}")
    print(sep)


if __name__ == "__main__":
    main()
