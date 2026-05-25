"""
train_gru_overfit.py
---------------------
Trains a GRU on the 10x-duplicated dataset for comparison against the LSTM.

GRUs have fewer parameters than LSTMs (no separate cell state) and typically
train faster.  This experiment checks whether a simpler recurrent architecture
can still memorise the duplicated training set.

This is NOT a generalisation test.  Do not cite these results in the thesis.

Usage
-----
  python train_gru_overfit.py
"""

import pickle
import sys
from pathlib import Path

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

EXPR_OUT = MP_DATA / "outputs" / "prediction" / "experiments" / "overfit-10x"
GRU_OUT  = EXPR_OUT / "gru"
CSV_PATH = EXPR_OUT / "trajectories_overfit.csv"

MODEL_PATH  = GRU_OUT / "overfit_gru.pth"
SCALER_PATH = GRU_OUT / "overfit_gru_scaler.pkl"
PLOT_PATH   = GRU_OUT / "loss_curve_gru.png"
VISUAL_PATH = GRU_OUT / "prediction_visual_gru.png"

# ---------------------------------------------------------------------------
# Hyper-parameters  (matched to LSTM experiment for fair comparison)
# ---------------------------------------------------------------------------
# QUICK SANITY TEST mode — not final overfit evidence
WINDOW_SIZE    = 10
HIDDEN_SIZE    = 256
EPOCHS         = 30
LEARNING_RATE  = 1e-3
BATCH_SIZE     = 1024
PRINT_EVERY    = 5
MAX_SEQUENCES  = 100_000   # cap for quick-test; set to None for full run

FEATURE_COLS = [
    "world_x", "world_y",
    "dist_to_obstacle", "dist_to_boundary", "dist_to_entrance",
    "frame_number", "delta_x", "delta_y",
]
# PHASE 1B: predict displacement, not absolute position
TARGET_COLS = ["delta_x", "delta_y"]
MIN_DISPLACEMENT_M = 0.0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TrajectoryGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# ---------------------------------------------------------------------------
# Data helpers  (identical to train_lstm_overfit.py)
# ---------------------------------------------------------------------------

def build_sequences(df: pd.DataFrame):
    df = df.copy().sort_values(["person_id", "frame_number"])
    df["delta_x"] = df.groupby("person_id")["world_x"].diff().fillna(0)
    df["delta_y"] = df.groupby("person_id")["world_y"].diff().fillna(0)

    X, y = [], []
    for _, track in df.groupby("person_id"):
        track = track.reset_index(drop=True)
        if len(track) <= WINDOW_SIZE:
            continue
        feats   = track[FEATURE_COLS].to_numpy(dtype=np.float32)
        targets = track[TARGET_COLS].to_numpy(dtype=np.float32)
        for i in range(len(track) - WINDOW_SIZE):
            X.append(feats[i: i + WINDOW_SIZE])
            y.append(targets[i + WINDOW_SIZE])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def normalise(X, y):
    feat_scaler   = MinMaxScaler()
    target_scaler = MinMaxScaler()
    N, W, F = X.shape
    X_scaled  = feat_scaler.fit_transform(X.reshape(-1, F)).reshape(N, W, F)
    y_scaled  = target_scaler.fit_transform(y)
    return (
        X_scaled.astype(np.float32),
        y_scaled.astype(np.float32),
        {"feature_scaler": feat_scaler, "target_scaler": target_scaler,
         "feature_cols": FEATURE_COLS, "target_cols": TARGET_COLS,
         "window_size": WINDOW_SIZE},
    )


def train_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimiser.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimiser.step()
        total += loss.item()
    return total / len(loader)


def generate_prediction_visual(model, scaler_bundle, df_raw, out_path, label, device,
                               n_rollout=30):
    """Autoregressive delta rollout: predict (dx,dy), accumulate into position."""
    model.eval()
    fs = scaler_bundle["feature_scaler"]
    ts = scaler_bundle["target_scaler"]

    min_len = WINDOW_SIZE + n_rollout + 1
    candidates = [(pid, grp) for pid, grp in df_raw.groupby("person_id")
                  if len(grp) >= min_len]
    if not candidates:
        print("[WARN] No track long enough for prediction visual — skipping.")
        return

    _, track = candidates[len(candidates) // 2]
    track = track.sort_values("frame_number").reset_index(drop=True).copy()
    track["delta_x"] = track["world_x"].diff().fillna(0)
    track["delta_y"] = track["world_y"].diff().fillna(0)

    feats     = track[FEATURE_COLS].to_numpy(dtype=np.float32)           # (T, 8)
    positions = track[["world_x", "world_y"]].to_numpy(dtype=np.float32) # (T, 2) — for plotting
    T = len(feats)

    feats_scaled = fs.transform(feats)
    window = feats_scaled[:WINDOW_SIZE].copy()

    prev_x, prev_y = positions[WINDOW_SIZE - 1, 0], positions[WINDOW_SIZE - 1, 1]
    pred_x, pred_y = [], []

    for step in range(n_rollout):
        x_in = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            p_scaled = model(x_in).cpu().numpy()[0]
        # Inverse-transform gives predicted (delta_x, delta_y)
        pred_dx, pred_dy = ts.inverse_transform(p_scaled[np.newaxis])[0]
        # Accumulate delta into absolute position
        wx = float(prev_x) + float(pred_dx)
        wy = float(prev_y) + float(pred_dy)
        pred_x.append(wx)
        pred_y.append(wy)

        src = min(WINDOW_SIZE + step, T - 1)
        new_raw = np.array([
            wx, wy,
            feats[src, 2], feats[src, 3], feats[src, 4],  # spatial context from GT
            feats[src, 5],                                  # frame_number
            float(pred_dx),                                 # feed back predicted delta_x
            float(pred_dy),                                 # feed back predicted delta_y
        ], dtype=np.float32)
        new_scaled = fs.transform(new_raw[np.newaxis])[0]
        window = np.vstack([window[1:], new_scaled])
        prev_x, prev_y = wx, wy

    gt_x = positions[WINDOW_SIZE: WINDOW_SIZE + n_rollout, 0]
    gt_y = positions[WINDOW_SIZE: WINDOW_SIZE + n_rollout, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    seed_x = positions[:WINDOW_SIZE, 0]
    seed_y = positions[:WINDOW_SIZE, 1]
    ax1.plot(seed_x, seed_y, "g-o", ms=3, label="Seed (GT)", alpha=0.8)
    ax1.plot(gt_x,   gt_y,   "b-o", ms=3, label="GT continuation", alpha=0.7)
    ax1.plot(pred_x, pred_y, "r--o", ms=3, label=f"{label} autoregressive (Δ)", alpha=0.8)
    ax1.legend(fontsize=8)
    ax1.set_title(f"{label} — 2D trajectory  [delta rollout]")
    ax1.set_xlabel("world_x (m)")
    ax1.set_ylabel("world_y (m)")
    ax1.grid(True, lw=0.4, alpha=0.6)

    steps = list(range(n_rollout))
    ax2.plot(steps, gt_x,   "b-",  label="GT world_x")
    ax2.plot(steps, pred_x, "r--", label=f"{label} pred world_x")
    ax2.plot(steps, gt_y,   "b:",  label="GT world_y")
    ax2.plot(steps, pred_y, "r-.", label=f"{label} pred world_y")
    ax2.legend(fontsize=8)
    ax2.set_title(f"{label} — X/Y over rollout steps")
    ax2.set_xlabel("Rollout step")
    ax2.set_ylabel("World coordinate (m)")
    ax2.grid(True, lw=0.4, alpha=0.6)

    fig.suptitle(
        f"PHASE 1B — DELTA PREDICTION: {label} autoregressive {n_rollout}-step rollout\n"
        "[MEMORISATION TEST — NOT generalisation evidence]",
        fontsize=10, color="#7f0000",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Prediction visual saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not CSV_PATH.exists():
        sys.exit(
            f"[ERROR] {CSV_PATH.name} not found.\n"
            "        Run make_overfit_dataset.py first."
        )

    df = pd.read_csv(CSV_PATH)
    if "track_id" in df.columns and "person_id" not in df.columns:
        df = df.rename(columns={"track_id": "person_id"})
    if "frame_idx" in df.columns and "frame_number" not in df.columns:
        df = df.rename(columns={"frame_idx": "frame_number"})

    print(f"[INFO] Loaded {len(df):,} rows  |  {df['person_id'].nunique()} unique persons")

    missing = [c for c in FEATURE_COLS if c not in ("delta_x", "delta_y") and c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] Missing columns: {missing}")

    # Quick-test: cap persons before sequence building to keep build time short.
    if MAX_SEQUENCES is not None:
        all_pids = df["person_id"].unique()
        n_persons = min(400, len(all_pids))
        chosen = np.random.default_rng(42).choice(all_pids, n_persons, replace=False)
        df = df[df["person_id"].isin(chosen)].copy()
        print(f"[INFO] Quick-test: pre-filtered to {n_persons} persons "
              f"({len(df):,} rows)")

    print("[INFO] Building sequences ...")
    X, y = build_sequences(df)
    print(f"[INFO] Sequences (full): {len(X):,}  |  X {X.shape}  y {y.shape}")

    if MAX_SEQUENCES is not None and len(X) > MAX_SEQUENCES:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), MAX_SEQUENCES, replace=False)
        X, y = X[idx], y[idx]
        print(f"[INFO] Subsampled to {len(X):,} sequences  [QUICK SANITY TEST]")

    X_scaled, y_scaled, scaler_bundle = normalise(X, y)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_scaled), torch.tensor(y_scaled)),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}  |  Batches/epoch: {len(loader)}")

    model     = TrajectoryGRU(len(FEATURE_COLS), HIDDEN_SIZE).to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parameters: {total_params:,}")
    print(f"\n[PHASE 1B — DELTA PREDICTION] GRU — {EPOCHS} epochs, batch={BATCH_SIZE}, "
          f"seq={len(X):,}\n")

    import time
    t0 = time.time()
    loss_history = []
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_epoch(model, loader, optimiser, criterion, device)
        loss_history.append(avg_loss)
        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3d}/{EPOCHS}  |  Loss: {avg_loss:.6f}")
    elapsed = time.time() - t0

    GRU_OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler_bundle, f)

    generate_prediction_visual(model, scaler_bundle, df, VISUAL_PATH, "GRU", device)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, EPOCHS + 1), loss_history, linewidth=1.2, color="#2980b9")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalised)")
    ax.set_title(f"GRU — PHASE 1B DELTA — {EPOCHS} epochs, "
                 f"hidden={HIDDEN_SIZE}, {len(X):,} seqs")
    ax.grid(True, linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)

    print(f"\n{'='*52}")
    print("  GRU — PHASE 1B DELTA PREDICTION COMPLETE")
    print(f"{'='*52}")
    print(f"  Sequences used : {len(X):,}")
    print(f"  Batches/epoch  : {len(loader)}")
    print(f"  Runtime        : {elapsed:.1f}s  ({elapsed/EPOCHS:.2f}s/epoch)")
    print(f"  Final loss     : {loss_history[-1]:.6f}")
    print(f"  Best loss      : {min(loss_history):.6f}  "
          f"(epoch {loss_history.index(min(loss_history))+1})")
    print(f"  Saved          : {GRU_OUT}")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()
