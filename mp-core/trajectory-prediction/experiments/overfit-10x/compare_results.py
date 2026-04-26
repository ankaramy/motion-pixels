"""
compare_results.py
------------------
Loads the saved loss curves from both overfit experiments and produces a
side-by-side comparison plot.

Reads loss values by re-running inference over the training set; does NOT
reload the raw loss history (which was not persisted separately).  Instead,
it computes final training MSE from the saved models so the comparison is
always reproducible.

Usage
-----
  python compare_results.py
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

LSTM_MODEL  = EXPR_OUT / "lstm" / "overfit_lstm.pth"
LSTM_SCALER = EXPR_OUT / "lstm" / "overfit_lstm_scaler.pkl"
LSTM_CURVE  = EXPR_OUT / "lstm" / "loss_curve_lstm.png"

GRU_MODEL   = EXPR_OUT / "gru"  / "overfit_gru.pth"
GRU_SCALER  = EXPR_OUT / "gru"  / "overfit_gru_scaler.pkl"
GRU_CURVE   = EXPR_OUT / "gru"  / "loss_curve_gru.png"

CSV_PATH    = EXPR_OUT / "trajectories_overfit.csv"
OUT_PLOT    = EXPR_OUT / "comparison_loss_curves.png"

WINDOW_SIZE  = 10
HIDDEN_SIZE  = 256
FEATURE_COLS = [
    "world_x", "world_y",
    "dist_to_obstacle", "dist_to_boundary", "dist_to_entrance",
    "frame_number", "delta_x", "delta_y",
]
TARGET_COLS = ["world_x", "world_y"]


# ---------------------------------------------------------------------------
# Architectures  (must match the training scripts exactly)
# ---------------------------------------------------------------------------

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TrajectoryGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_sequences(df):
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


def eval_mse(model, X_t, y_t, device, batch_size=256):
    model.eval()
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size)
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            total += criterion(model(xb), yb).item()
    return total / len(loader)


def load_loss_png_title(png_path: Path) -> str:
    """Extract epoch count from the saved loss curve filename for the label."""
    return png_path.stem.replace("_", " ")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    missing = [p for p in (LSTM_MODEL, LSTM_SCALER, GRU_MODEL, GRU_SCALER, CSV_PATH) if not p.exists()]
    if missing:
        sys.exit(
            "[ERROR] Missing files — run both training scripts first:\n" +
            "".join(f"  {p}\n" for p in missing)
        )

    df = pd.read_csv(CSV_PATH)
    if "track_id" in df.columns and "person_id" not in df.columns:
        df = df.rename(columns={"track_id": "person_id"})
    if "frame_idx" in df.columns and "frame_number" not in df.columns:
        df = df.rename(columns={"frame_idx": "frame_number"})

    print(f"[INFO] Dataset: {len(df):,} rows  |  {df['person_id'].nunique()} persons")
    print("[INFO] Building sequences ...")

    X_raw, y_raw = build_sequences(df)
    print(f"[INFO] Sequences: {len(X_raw):,}")

    # Use LSTM scaler for consistent comparison (both scalers were fit on identical data)
    with open(LSTM_SCALER, "rb") as f:
        lstm_bundle = pickle.load(f)
    with open(GRU_SCALER, "rb") as f:
        gru_bundle = pickle.load(f)

    def scale(bundle, X, y):
        N, W, F = X.shape
        Xs = bundle["feature_scaler"].transform(X.reshape(-1, F)).reshape(N, W, F)
        ys = bundle["target_scaler"].transform(y)
        return torch.tensor(Xs.astype(np.float32)), torch.tensor(ys.astype(np.float32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    lstm_model = TrajectoryLSTM(len(FEATURE_COLS), HIDDEN_SIZE).to(device)
    lstm_model.load_state_dict(torch.load(LSTM_MODEL, map_location=device))

    gru_model = TrajectoryGRU(len(FEATURE_COLS), HIDDEN_SIZE).to(device)
    gru_model.load_state_dict(torch.load(GRU_MODEL, map_location=device))

    X_lstm, y_lstm = scale(lstm_bundle, X_raw, y_raw)
    X_gru,  y_gru  = scale(gru_bundle,  X_raw, y_raw)

    print("[INFO] Evaluating ...")
    lstm_mse = eval_mse(lstm_model, X_lstm, y_lstm, device)
    gru_mse  = eval_mse(gru_model,  X_gru,  y_gru,  device)

    # ── Side-by-side loss curve images ──────────────────────────────────────
    lstm_img = plt.imread(str(LSTM_CURVE)) if LSTM_CURVE.exists() else None
    gru_img  = plt.imread(str(GRU_CURVE))  if GRU_CURVE.exists()  else None

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "Overfit-10x: LSTM vs GRU  |  Training MSE (normalised)\n"
        "[NOT generalisation evidence — memorisation test only]",
        fontsize=11, color="#7f0000",
    )

    for ax, img, label, mse in [
        (axes[0], lstm_img, "LSTM", lstm_mse),
        (axes[1], gru_img,  "GRU",  gru_mse),
    ]:
        if img is not None:
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{label}  |  final training MSE: {mse:.6f}", fontsize=10)
        else:
            ax.text(0.5, 0.5, f"{label}\nloss curve not found", ha="center", va="center")
            ax.axis("off")

    fig.tight_layout()
    EXPR_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOT, dpi=150)
    plt.close(fig)

    print(f"\n{'='*52}")
    print("  COMPARISON RESULTS")
    print(f"{'='*52}")
    print(f"  LSTM final training MSE : {lstm_mse:.6f}")
    print(f"  GRU  final training MSE : {gru_mse:.6f}")
    winner = "LSTM" if lstm_mse < gru_mse else "GRU"
    print(f"  Lower training MSE      : {winner}")
    print(f"\n  Comparison plot saved   : {OUT_PLOT}")
    print(f"\n  NOTE: These are training-set MSE values on 10x duplicated data.")
    print(f"        They indicate memorisation capacity, NOT real-world accuracy.")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()
