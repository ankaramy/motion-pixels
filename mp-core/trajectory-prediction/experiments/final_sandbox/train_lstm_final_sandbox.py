"""
train_lstm_final_sandbox.py
---------------------------
Overfit-style LSTM training on the duplicated sandbox dataset.

Predicts next-step (delta_x, delta_y) from a 10-frame window. The aim is
clear memorisation for visual review; not generalisation.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import (
    DATASET_CSV, MODEL_PTH, SCALER_PKL, LOSS_PNG, TRAIN_MD,
    FEATURE_COLS, TARGET_COLS, WINDOW_SIZE,
    HIDDEN_SIZE, NUM_LAYERS, EPOCHS, BATCH_SIZE, LR,
)


class TrajectoryLSTM(nn.Module):
    def __init__(self, in_size, hidden, out_size=2, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(in_size, hidden, n_layers, batch_first=True)
        self.fc   = nn.Linear(hidden, out_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def build_windows(df: pd.DataFrame):
    feats   = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    targets = df[TARGET_COLS].to_numpy(dtype=np.float32)
    tids    = df["track_id"].to_numpy()

    X, Y = [], []
    for tid in np.unique(tids):
        mask = tids == tid
        f = feats[mask]; t = targets[mask]
        for i in range(len(f) - WINDOW_SIZE):
            X.append(f[i : i + WINDOW_SIZE])
            Y.append(t[i + WINDOW_SIZE])
    return np.asarray(X), np.asarray(Y)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if not DATASET_CSV.exists():
        raise SystemExit(f"[FATAL] dataset missing: {DATASET_CSV}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device: {device}")

    df = pd.read_csv(DATASET_CSV)
    print(f"[INFO]  Dataset rows: {len(df):,} · "
          f"unique tracks: {df['track_id'].nunique():,}")

    X, Y = build_windows(df)
    print(f"[INFO]  Built {len(X):,} (window=10 → target) sequences")

    # Scale features and targets separately.
    fs = MinMaxScaler().fit(X.reshape(-1, X.shape[-1]))
    ts = MinMaxScaler().fit(Y)
    Xs = fs.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    Ys = ts.transform(Y)

    Xt = torch.tensor(Xs, dtype=torch.float32)
    Yt = torch.tensor(Ys, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt, Yt),
                        batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0)

    model = TrajectoryLSTM(len(FEATURE_COLS), HIDDEN_SIZE,
                           out_size=2, n_layers=NUM_LAYERS).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()

    losses = []
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        running = 0.0; n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item(); n_batches += 1
        ep_loss = running / max(1, n_batches)
        losses.append(ep_loss)
        print(f"[INFO]  epoch {ep:02d}/{EPOCHS}  "
              f"loss={ep_loss:.6f}  elapsed={time.time()-t0:.1f}s")

    torch.save(model.state_dict(), MODEL_PTH)
    with open(SCALER_PKL, "wb") as fh:
        pickle.dump({"feature_scaler": fs, "target_scaler": ts,
                     "feature_cols": FEATURE_COLS,
                     "target_cols":  TARGET_COLS,
                     "window_size":  WINDOW_SIZE,
                     "hidden":       HIDDEN_SIZE,
                     "n_layers":     NUM_LAYERS}, fh)
    print(f"[OK]    Saved model -> {MODEL_PTH.name}")
    print(f"[OK]    Saved scaler -> {SCALER_PKL.name}")

    # Loss curve.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, EPOCHS + 1), losses, "-o", lw=1.6, ms=4,
            color="#2980b9")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss (scaled)")
    ax.set_title(f"Final sandbox — overfit LSTM training\n"
                 f"{len(X):,} sequences · hidden={HIDDEN_SIZE} · "
                 f"epochs={EPOCHS}")
    ax.grid(True, lw=0.4, alpha=0.6)
    fig.tight_layout(); fig.savefig(LOSS_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]    Saved loss curve -> {LOSS_PNG.name}")

    md = []
    md.append("# Final Sandbox — Training Summary")
    md.append("")
    md.append(f"- Dataset: `{DATASET_CSV.name}` ({len(df):,} rows)")
    md.append(f"- Sequences (window={WINDOW_SIZE}): **{len(X):,}**")
    md.append(f"- Device: {device}")
    md.append(f"- Architecture: LSTM hidden={HIDDEN_SIZE} layers={NUM_LAYERS}")
    md.append(f"- Optimiser: Adam lr={LR} · batch={BATCH_SIZE} · "
              f"epochs={EPOCHS}")
    md.append("")
    md.append("## Loss per epoch")
    md.append("")
    md.append("| epoch | loss |")
    md.append("|---|---|")
    for i, l in enumerate(losses, 1):
        md.append(f"| {i} | {l:.6f} |")
    md.append("")
    md.append(f"- First epoch loss: **{losses[0]:.6f}**")
    md.append(f"- Final epoch loss: **{losses[-1]:.6f}**")
    md.append(f"- Reduction: "
              f"**{(losses[0] - losses[-1]) / losses[0] * 100:.1f}%**")
    TRAIN_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK]    Wrote {TRAIN_MD.name}")


if __name__ == "__main__":
    main()
