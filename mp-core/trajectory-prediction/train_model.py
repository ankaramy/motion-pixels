"""
train_model.py
--------------
Trains a single-layer LSTM to predict the next (world_x, world_y) position of
a pedestrian given the last 10 frames of their trajectory.

Pipeline
--------
  1. Load trajectories_encoded.csv
  2. Build sliding-window sequences per person
  3. Normalise features with MinMaxScaler  →  saved as scaler.pkl
  4. Train LSTM (PyTorch) for 100 epochs
  5. Save model  →  trajectory_model.pth
  6. Save loss curve  →  loss_curve.png

Input features (8 per timestep)
--------------------------------
  world_x, world_y, dist_to_obstacle, dist_to_boundary, dist_to_entrance,
  frame_number, delta_x, delta_y

Target (2 values)
-----------------
  world_x, world_y  at the frame immediately after the window

Usage
-----
  python train_model.py
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE        = Path(__file__).resolve().parent
MP_ROOT     = HERE.parent.parent
MP_DATA     = MP_ROOT / "mp-data"
PROCESSED   = MP_DATA / "processed" / "encoded"
PRED_OUT    = MP_DATA / "outputs" / "prediction"

CSV_PATH    = PROCESSED / "trajectories_encoded.csv"
MODEL_PATH  = PRED_OUT  / "trajectory_model.pth"
SCALER_PATH = PRED_OUT  / "scaler.pkl"
PLOT_PATH   = PRED_OUT  / "loss_curve.png"

# ---------------------------------------------------------------------------
# Hyper-parameters  (change these to experiment)
# ---------------------------------------------------------------------------
WINDOW_SIZE  = 10       # how many past frames the model sees at once
HIDDEN_SIZE  = 128      # number of units in the LSTM layer
LEARNING_RATE = 1e-3
EPOCHS       = 100
BATCH_SIZE   = 64
PRINT_EVERY  = 10       # print loss every N epochs

# The 8 input features (order must stay consistent with scaler columns)
FEATURE_COLS = [
    "world_x",
    "world_y",
    "dist_to_obstacle",
    "dist_to_boundary",
    "dist_to_entrance",
    "frame_number",
    "delta_x",    # change in world_x from the previous frame
    "delta_y",    # change in world_y from the previous frame
]

# Sequences where the person barely moves are unhelpful for learning direction.
# Any window whose total path length is below this threshold is dropped.
MIN_DISPLACEMENT_M = 0.3   # metres

# The 2 target columns (position we want the model to predict)
TARGET_COLS = ["world_x", "world_y"]


# ---------------------------------------------------------------------------
# Step 1 — Build sliding-window sequences
# ---------------------------------------------------------------------------

def build_sequences(df: pd.DataFrame):
    """
    For every person, sort their rows by frame, compute velocity features,
    then slide a window of WINDOW_SIZE frames across their trajectory.

    Velocity features
    -----------------
    delta_x / delta_y — change in world position from one frame to the next.
    For the very first frame of each track there is no previous frame, so the
    delta is set to 0.  The model quickly learns to ignore this edge value
    because it only ever appears at the start of a track.

    Displacement filter
    -------------------
    Windows where the person barely moved (total path < MIN_DISPLACEMENT_M)
    are dropped.  Still bodies give the model no direction signal and push
    predictions toward the dataset mean.

    Returns
    -------
    X : np.ndarray  shape (num_samples, WINDOW_SIZE, 8)
        Input sequences (raw / unscaled).
    y : np.ndarray  shape (num_samples, 2)
        Target position (world_x, world_y) at the frame after each window.
    """
    X_list, y_list = [], []
    skipped = 0

    # Process each person independently so windows never cross persons
    for _, group in df.groupby("person_id"):
        # Sort by frame so the sequence is chronological
        track = group.sort_values("frame_number").reset_index(drop=True)

        # --- Compute per-frame velocity ---
        wx = track["world_x"].to_numpy(dtype=np.float64)
        wy = track["world_y"].to_numpy(dtype=np.float64)

        # delta[0] = 0 (no previous frame); delta[t] = pos[t] - pos[t-1]
        delta_x = np.zeros(len(track), dtype=np.float64)
        delta_y = np.zeros(len(track), dtype=np.float64)
        delta_x[1:] = wx[1:] - wx[:-1]
        delta_y[1:] = wy[1:] - wy[:-1]

        track = track.copy()
        track["delta_x"] = delta_x
        track["delta_y"] = delta_y

        features = track[FEATURE_COLS].to_numpy(dtype=np.float64)   # (T, 8)
        targets  = track[TARGET_COLS].to_numpy(dtype=np.float64)     # (T, 2)

        # Slide a window: frames [t … t+W-1] → predict frame [t+W]
        for t in range(len(track) - WINDOW_SIZE):
            window = features[t : t + WINDOW_SIZE]   # (W, 8)

            # --- Displacement filter ---
            # Total path length = sum of per-step distances inside the window
            step_dist = np.sqrt(
                window[:, 6] ** 2 + window[:, 7] ** 2   # delta_x=col6, delta_y=col7
            )
            if step_dist.sum() < MIN_DISPLACEMENT_M:
                skipped += 1
                continue

            X_list.append(window)
            y_list.append(targets[t + WINDOW_SIZE])   # shape (2,)

    if not X_list:
        sys.exit(
            "[ERROR] No sequences could be built.\n"
            f"        Need at least {WINDOW_SIZE + 1} consecutive frames per person\n"
            f"        with >{MIN_DISPLACEMENT_M} m of movement.\n"
            "        Check that trajectories_encoded.csv is populated."
        )

    print(f"[INFO] Sequences filtered out (displacement < {MIN_DISPLACEMENT_M} m): {skipped:,}")

    X = np.array(X_list, dtype=np.float32)   # (N, W, 8)
    y = np.array(y_list, dtype=np.float32)   # (N, 2)
    return X, y


# ---------------------------------------------------------------------------
# Step 2 — Normalise with MinMaxScaler
# ---------------------------------------------------------------------------

def normalise(X: np.ndarray, y: np.ndarray):
    """
    Scale every feature and target column to [0, 1].

    We fit a single scaler on the flattened feature matrix so that the same
    per-column statistics are applied consistently.  The scaler is saved to
    disk so predict_trajectory.py can invert the transform later.

    Returns
    -------
    X_scaled : same shape as X
    y_scaled : same shape as y
    scaler   : fitted MinMaxScaler (also saved to SCALER_PATH)
    """
    N, W, F = X.shape   # num_samples, window_size, num_features

    # Reshape to 2-D so sklearn can process it, then reshape back
    X_flat   = X.reshape(-1, F)          # (N*W, F)

    # Fit one scaler on features, one on targets
    feat_scaler   = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_scaled_flat = feat_scaler.fit_transform(X_flat)          # (N*W, 6)
    y_scaled      = target_scaler.fit_transform(y)             # (N, 2)

    X_scaled = X_scaled_flat.reshape(N, W, F)

    # Bundle both scalers into one dict and save
    scaler_bundle = {
        "feature_scaler": feat_scaler,
        "target_scaler":  target_scaler,
        "feature_cols":   FEATURE_COLS,
        "target_cols":    TARGET_COLS,
        "window_size":    WINDOW_SIZE,
    }
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler_bundle, f)
    print(f"[INFO] Scaler saved: {SCALER_PATH.name}")

    return X_scaled.astype(np.float32), y_scaled.astype(np.float32), scaler_bundle


# ---------------------------------------------------------------------------
# Step 3 — LSTM model definition
# ---------------------------------------------------------------------------

class TrajectoryLSTM(nn.Module):
    """
    A minimal LSTM that reads a sequence of pedestrian states and predicts
    where the person will be one step ahead.

    Architecture
    ------------
      Input  →  LSTM (128 hidden units)  →  Linear  →  (world_x, world_y)

    The LSTM processes all WINDOW_SIZE timesteps; we take only the final
    hidden state as a summary of the whole sequence, then pass it through a
    linear layer to produce the two-coordinate prediction.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 2):
        super().__init__()

        # LSTM layer: input_size features per timestep, hidden_size memory cells
        # batch_first=True means input tensors are (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = 1,
            batch_first = True,
        )

        # Fully connected output layer: maps 128 hidden values → 2 coordinates
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x : (batch_size, window_size, input_size)
        Returns predicted (world_x, world_y) : (batch_size, 2)
        """
        # lstm_out : (batch, seq, hidden) — all hidden states for each timestep
        # (h_n, c_n) are the final hidden and cell states (we don't need them)
        lstm_out, _ = self.lstm(x)

        # Take only the last timestep's hidden state — it summarises the window
        last_hidden = lstm_out[:, -1, :]   # (batch, hidden_size)

        # Map to 2D output
        prediction = self.fc(last_hidden)  # (batch, 2)
        return prediction


# ---------------------------------------------------------------------------
# Step 4 — Training loop
# ---------------------------------------------------------------------------

def train(model, loader, optimiser, criterion, device):
    """Run one full pass over the training data and return the mean loss."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        # Move tensors to the chosen device (CPU or GPU)
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimiser.zero_grad()          # clear gradients from previous step
        predictions = model(X_batch)   # forward pass
        loss = criterion(predictions, y_batch)   # compare to targets
        loss.backward()                # compute gradients
        optimiser.step()               # update weights

        total_loss += loss.item() * len(X_batch)   # accumulate weighted loss

    return total_loss / len(loader.dataset)   # return average loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Load data ---
    if not CSV_PATH.exists():
        sys.exit(
            f"[ERROR] {CSV_PATH.name} not found.\n"
            "        Run encode_space.py first to generate it."
        )

    df = pd.read_csv(CSV_PATH)

    # Normalise column names if this came from trajectory-extraction format
    rename_map = {}
    if "track_id" in df.columns and "person_id" not in df.columns:
        rename_map["track_id"] = "person_id"
    if "frame_idx" in df.columns and "frame_number" not in df.columns:
        rename_map["frame_idx"] = "frame_number"
    elif "frame" in df.columns and "frame_number" not in df.columns:
        rename_map["frame"] = "frame_number"
    if "wx" in df.columns and "world_x" not in df.columns:
        rename_map["wx"] = "world_x"
    if "wy" in df.columns and "world_y" not in df.columns:
        rename_map["wy"] = "world_y"
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"[INFO] Normalised column names: {rename_map}")

    # Check that encode_space.py was actually run (delta cols are computed here,
    # so only check the columns that must exist in the CSV already)
    CSV_REQUIRED = [c for c in FEATURE_COLS if c not in ("delta_x", "delta_y")]
    missing = [c for c in CSV_REQUIRED if c not in df.columns]
    if missing:
        sys.exit(
            f"[ERROR] Missing columns in CSV: {missing}\n"
            "        Run encode_space.py first to add distance columns."
        )

    # Drop rows where any CSV feature is NaN (happens when a phase was skipped).
    # delta_x / delta_y are computed later inside build_sequences, not in the CSV.
    n_before = len(df)
    df = df.dropna(subset=CSV_REQUIRED + TARGET_COLS)
    if len(df) < n_before:
        print(f"[WARN] Dropped {n_before - len(df)} rows with NaN values.")

    print(f"[INFO] Loaded {len(df):,} rows  |  "
          f"{df['person_id'].nunique()} unique persons")

    # --- Build sequences ---
    print(f"[INFO] Building sliding windows (window size = {WINDOW_SIZE}) …")
    X, y = build_sequences(df)
    print(f"[INFO] Sequences built: {len(X):,}  |  "
          f"X shape: {X.shape}  |  y shape: {y.shape}")

    # --- Normalise ---
    print("[INFO] Normalising features …")
    X_scaled, y_scaled, _ = normalise(X, y)

    # --- Create PyTorch Dataset and DataLoader ---
    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.tensor(X_scaled)   # (N, W, 6)
    y_tensor = torch.tensor(y_scaled)   # (N, 2)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Device selection ---
    # Use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on: {device}")

    # --- Initialise model, loss, optimiser ---
    model     = TrajectoryLSTM(
        input_size  = len(FEATURE_COLS),   # 8
        hidden_size = HIDDEN_SIZE,          # 128
        output_size = len(TARGET_COLS),     # 2
    ).to(device)

    criterion = nn.MSELoss()                         # mean-squared error
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {total_params:,}")

    # --- Training loop ---
    print(f"\n[INFO] Training for {EPOCHS} epochs …\n")
    loss_history = []

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train(model, loader, optimiser, criterion, device)
        loss_history.append(avg_loss)

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d}/{EPOCHS}  |  Loss: {avg_loss:.6f}")

    # --- Save model ---
    PRED_OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n[INFO] Model saved: {MODEL_PATH.name}")

    # --- Save loss curve ---
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, EPOCHS + 1), loss_history, linewidth=1.5, color="#2a6dd9")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalised)")
    ax.set_title("LSTM Training Loss")
    ax.grid(True, linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"[INFO] Loss curve saved: {PLOT_PATH.name}")

    # --- Final summary ---
    print("\n" + "=" * 50)
    print("  TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Final loss      : {loss_history[-1]:.6f}")
    print(f"  Best loss       : {min(loss_history):.6f}  (epoch {loss_history.index(min(loss_history)) + 1})")
    print(f"  Sequences used  : {len(X):,}")
    print(f"  Saved files     : {MODEL_PATH.name}, {SCALER_PATH.name}, {PLOT_PATH.name}")
    print("=" * 50)


if __name__ == "__main__":
    main()
