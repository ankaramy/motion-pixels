"""
train_lstm_overfit.py
---------------------
Trains a larger LSTM on the 10x-duplicated dataset for an extended run.

Goal: drive training loss toward zero to confirm the model has sufficient
capacity to memorise the training distribution.

This is NOT a generalisation test.  Do not cite these results in the thesis.

Usage
-----
  python train_lstm_overfit.py
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

EXPR_OUT  = MP_DATA / "outputs" / "prediction" / "experiments" / "overfit-10x"
LSTM_OUT  = EXPR_OUT / "lstm"
CSV_PATH  = EXPR_OUT / "trajectories_overfit.csv"

MODEL_PATH  = LSTM_OUT / "overfit_lstm.pth"
SCALER_PATH = LSTM_OUT / "overfit_lstm_scaler.pkl"
PLOT_PATH   = LSTM_OUT / "loss_curve_lstm.png"

# ---------------------------------------------------------------------------
# Hyper-parameters (intentionally set to favour memorisation)
# ---------------------------------------------------------------------------
WINDOW_SIZE   = 10
HIDDEN_SIZE   = 256    # 2x main pipeline — more capacity
EPOCHS        = 500    # 5x main pipeline — time to memorise
LEARNING_RATE = 1e-3
BATCH_SIZE    = 32     # smaller — more gradient steps per epoch
PRINT_EVERY   = 50

FEATURE_COLS = [
    "world_x", "world_y",
    "dist_to_obstacle", "dist_to_boundary", "dist_to_entrance",
    "frame_number", "delta_x", "delta_y",
]
TARGET_COLS = ["world_x", "world_y"]

# Disabled: include near-static sequences to maximise training signal
MIN_DISPLACEMENT_M = 0.0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ---------------------------------------------------------------------------
# Data helpers  (identical logic to main train_model.py)
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
        feats  = track[FEATURE_COLS].to_numpy(dtype=np.float32)
        targets = track[TARGET_COLS].to_numpy(dtype=np.float32)
        for i in range(len(track) - WINDOW_SIZE):
            window = feats[i: i + WINDOW_SIZE]
            if MIN_DISPLACEMENT_M > 0:
                disp = np.sqrt(
                    (feats[i + WINDOW_SIZE - 1, 0] - feats[i, 0]) ** 2 +
                    (feats[i + WINDOW_SIZE - 1, 1] - feats[i, 1]) ** 2
                )
                if disp < MIN_DISPLACEMENT_M:
                    continue
            X.append(window)
            y.append(targets[i + WINDOW_SIZE])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def normalise(X, y):
    feat_scaler   = MinMaxScaler()
    target_scaler = MinMaxScaler()
    N, W, F = X.shape
    X_flat       = X.reshape(-1, F)
    X_scaled     = feat_scaler.fit_transform(X_flat).reshape(N, W, F)
    y_scaled     = target_scaler.fit_transform(y)
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
        sys.exit(f"[ERROR] Missing columns: {missing}\n        Run encode_space.py then make_overfit_dataset.py.")

    print("[INFO] Building sequences ...")
    X, y = build_sequences(df)
    print(f"[INFO] Sequences: {len(X):,}  |  X {X.shape}  y {y.shape}")

    X_scaled, y_scaled, scaler_bundle = normalise(X, y)
    X_tensor = torch.tensor(X_scaled)
    y_tensor = torch.tensor(y_scaled)
    loader   = DataLoader(TensorDataset(X_tensor, y_tensor),
                          batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model     = TrajectoryLSTM(len(FEATURE_COLS), HIDDEN_SIZE).to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parameters: {total_params:,}")
    print(f"\n[INFO] Training LSTM for {EPOCHS} epochs (overfit target) ...\n")

    loss_history = []
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_epoch(model, loader, optimiser, criterion, device)
        loss_history.append(avg_loss)
        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d}/{EPOCHS}  |  Loss: {avg_loss:.6f}")

    LSTM_OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler_bundle, f)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, EPOCHS + 1), loss_history, linewidth=1.2, color="#c0392b")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalised)")
    ax.set_title(f"LSTM Overfit — {EPOCHS} epochs, hidden={HIDDEN_SIZE}, 10x duplicated data")
    ax.grid(True, linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)

    print(f"\n{'='*52}")
    print("  LSTM OVERFIT COMPLETE")
    print(f"{'='*52}")
    print(f"  Final loss   : {loss_history[-1]:.6f}")
    print(f"  Best loss    : {min(loss_history):.6f}  (epoch {loss_history.index(min(loss_history))+1})")
    print(f"  Sequences    : {len(X):,}")
    print(f"  Saved        : {LSTM_OUT}")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()
