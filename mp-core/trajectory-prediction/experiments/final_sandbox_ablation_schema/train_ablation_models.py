"""
train_ablation_models.py
------------------------
Train 4 LSTM ablation models (A/B/C/D) on the schema dataset. Each model
uses its own feature subset; all share architecture (LSTM 256x2) and
horizon (10 → next-step target).
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _paths import (
    DATASET_CSV, MODELS_DIR, MODEL_DEFS, TARGET_COLS,
    WINDOW_SIZE, HIDDEN_SIZE, NUM_LAYERS, EPOCHS, BATCH_SIZE, LR,
)


class TrajectoryLSTM(nn.Module):
    def __init__(self, in_size: int, hidden: int = HIDDEN_SIZE,
                 out_size: int = 2, n_layers: int = NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(in_size, hidden, n_layers, batch_first=True)
        self.fc   = nn.Linear(hidden, out_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def build_windows(df: pd.DataFrame, feature_cols: List[str]
                   ) -> Tuple[np.ndarray, np.ndarray]:
    feats   = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df[TARGET_COLS].to_numpy(dtype=np.float32)
    tids    = df["track_id"].to_numpy()
    X, Y = [], []
    for tid in np.unique(tids):
        m = (tids == tid)
        f = feats[m]; t = targets[m]
        for i in range(len(f) - WINDOW_SIZE):
            X.append(f[i: i + WINDOW_SIZE])
            Y.append(t[i + WINDOW_SIZE - 1])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def train_one_model(mdef: Dict, df: pd.DataFrame, device) -> Dict:
    name = mdef["name"]; features = mdef["features"]
    print(f"\n========== train {name} ({len(features)} features) ==========")
    print(f"        features: {features}")

    out_dir = MODELS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    X, Y = build_windows(df, features)
    print(f"[INFO]  {len(X):,} sequences")
    fs = MinMaxScaler().fit(X.reshape(-1, X.shape[-1]))
    ts = MinMaxScaler().fit(Y)
    Xs = fs.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    Ys = ts.transform(Y)

    Xt = torch.tensor(Xs, dtype=torch.float32)
    Yt = torch.tensor(Ys, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt, Yt),
                        batch_size=BATCH_SIZE, shuffle=True)

    model = TrajectoryLSTM(len(features)).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()

    losses: List[float] = []
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        running = 0.0; nb = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            running += loss.item(); nb += 1
        ep_loss = running / max(1, nb)
        losses.append(ep_loss)
        print(f"        epoch {ep:02d}/{EPOCHS}  loss={ep_loss:.6f}  "
              f"elapsed={time.time()-t0:.0f}s")

    torch.save(model.state_dict(), out_dir / "model.pth")
    with open(out_dir / "scaler.pkl", "wb") as fh:
        pickle.dump({"feature_scaler": fs, "target_scaler": ts,
                     "feature_cols": features,
                     "target_cols": TARGET_COLS}, fh)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, EPOCHS + 1), losses, "-o", lw=1.6, ms=4,
            color=mdef["color"])
    ax.set_title(f"{mdef['label']} — training loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss (scaled)")
    ax.grid(True, lw=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]    {name}: saved model.pth, scaler.pkl, loss_curve.png "
          f"(first loss {losses[0]:.6f} → final {losses[-1]:.6f})")
    return {"name": name, "first_loss": losses[0],
            "final_loss": losses[-1], "epochs": EPOCHS,
            "n_features": len(features), "n_sequences": int(len(X))}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if not DATASET_CSV.exists():
        raise SystemExit(f"[FATAL] missing dataset: {DATASET_CSV}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device: {device}")

    df = pd.read_csv(DATASET_CSV)
    print(f"[INFO]  Dataset: {len(df):,} rows · "
          f"{df['track_id'].nunique():,} tracks")

    summaries = []
    for mdef in MODEL_DEFS:
        summaries.append(train_one_model(mdef, df, device))

    # Combined loss curve
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mdef in MODEL_DEFS:
        name = mdef["name"]
        # Re-read losses from each model's training (we don't persist them
        # individually — just regenerate the per-model PNG; for combined
        # we recompute by reading the summaries instead).
        pass
    # Persist a tiny JSON of summaries.
    pd.DataFrame(summaries).to_csv(MODELS_DIR / "training_summaries.csv",
                                    index=False)
    print(f"\n[OK]    training_summaries.csv written")


if __name__ == "__main__":
    main()
