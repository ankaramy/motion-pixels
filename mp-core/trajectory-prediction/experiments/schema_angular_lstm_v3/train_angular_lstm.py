"""
train_angular_lstm.py (v2)
--------------------------
Train an angular-aware autoregressive LSTM on the v2 relational schema.

v2 changes vs v1:
  * Shorter EPOCHS (40), validation-loss EARLY STOPPING (patience=8),
    BEST validation checkpoint is restored before saving.
  * Freeze-collapse penalty:
        pred_mag = ||(pred_du, pred_dv)||
        freeze   = mean(relu(FREEZE_MAG_THRESH - pred_mag))
    Computed on the UN-standardized prediction (i.e. the scaled-by-
    MOTION_SCALE schema space).
  * Total loss = W_POSITION * pos + W_TURN * turn + W_HEADING * head
                  + W_FREEZE * freeze

Architecture is unchanged: LSTM(14 -> 256 x 2) -> Linear(3).
"""

from __future__ import annotations

import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _paths import (
    SCHEMA_CSV, MODEL_PTH, SCALER_PKL, LOSS_PNG, TRAIN_SUMMARY_CSV,
    WINDOW_SIZE, HIDDEN_SIZE, NUM_LAYERS, EPOCHS, PATIENCE,
    BATCH_SIZE, LR, TRAIN_FRAC,
    W_POSITION, W_TURN, W_HEADING, W_FREEZE, W_PATH, FREEZE_MAG_THRESH,
    MOTION_SCALE,
    FEATURE_COLS, TARGET_COLS,
)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class WindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame,
                 feat_scaler: StandardScaler,
                 tgt_scaler: StandardScaler,
                 window: int = WINDOW_SIZE):
        self.window = window
        Xs, Ys, RawY = [], [], []

        feats_all = feat_scaler.transform(df[FEATURE_COLS].to_numpy())
        tgts_all  = tgt_scaler.transform(df[TARGET_COLS].to_numpy())
        raw_tgts  = df[TARGET_COLS].to_numpy()

        idx = 0
        for _, g in df.groupby("track_id", sort=False):
            n = len(g)
            if n < window:
                idx += n
                continue
            f = feats_all[idx: idx + n]
            t = tgts_all [idx: idx + n]
            r = raw_tgts [idx: idx + n]
            for i in range(n - window):
                Xs.append(f[i: i + window])
                Ys.append(t[i + window - 1])
                RawY.append(r[i + window - 1])
            idx += n

        self.X    = np.asarray(Xs,   dtype=np.float32)
        self.Y    = np.asarray(Ys,   dtype=np.float32)
        self.RawY = np.asarray(RawY, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (torch.from_numpy(self.X[i]),
                torch.from_numpy(self.Y[i]),
                torch.from_numpy(self.RawY[i]))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class AngularLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN_SIZE,
                 layers: int = NUM_LAYERS, n_targets: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden,
                            num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, n_targets)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def angular_heading_loss(pred_raw_duv: torch.Tensor,
                         tgt_raw_duv:  torch.Tensor,
                         eps: float = 1e-8) -> torch.Tensor:
    pn = pred_raw_duv / (pred_raw_duv.norm(dim=-1, keepdim=True) + eps)
    tn = tgt_raw_duv  / (tgt_raw_duv .norm(dim=-1, keepdim=True) + eps)
    cos_sim = (pn * tn).sum(dim=-1)
    return (1.0 - cos_sim).mean()


def path_length_loss(pred_raw_duv: torch.Tensor,
                     tgt_raw_duv:  torch.Tensor,
                     eps: float = 1e-8) -> torch.Tensor:
    """
    L1 between predicted and target step magnitudes. Discourages
    systematic under-movement that's invisible to the position MSE.
    """
    pred_mag = torch.sqrt((pred_raw_duv ** 2).sum(dim=-1) + eps)
    gt_mag   = torch.sqrt((tgt_raw_duv  ** 2).sum(dim=-1) + eps)
    return (pred_mag - gt_mag).abs().mean()


def freeze_penalty(pred_raw_duv: torch.Tensor,
                   thresh: float = FREEZE_MAG_THRESH) -> torch.Tensor:
    """
    Weak penalty against zero-motion predictions. Computed in the
    UN-standardized, motion-scaled space (i.e. the same units as the
    schema's du/dv columns), so thresh corresponds to "predicted move
    smaller than thresh in scaled u/v per step".
    """
    pred_mag = pred_raw_duv.norm(dim=-1)
    return torch.relu(thresh - pred_mag).mean()


# --------------------------------------------------------------------------- #
# Train / validate
# --------------------------------------------------------------------------- #
def split_tracks(df: pd.DataFrame, train_frac: float = TRAIN_FRAC,
                 seed: int = 0):
    rng = np.random.default_rng(seed)
    tids = df["track_id"].unique().tolist()
    rng.shuffle(tids)
    cut = int(round(len(tids) * train_frac))
    train_tids = set(tids[:cut])
    val_tids   = set(tids[cut:])
    return (df[df["track_id"].isin(train_tids)].copy().reset_index(drop=True),
            df[df["track_id"].isin(val_tids)  ].copy().reset_index(drop=True))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if not SCHEMA_CSV.exists():
        raise SystemExit(f"[FATAL] missing input: {SCHEMA_CSV}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO]  Device: {device}")
    print(f"[INFO]  Loading {SCHEMA_CSV.name}")
    df = pd.read_csv(SCHEMA_CSV)
    print(f"[INFO]  {len(df):,} rows · "
          f"{df['track_id'].nunique():,} tracks")
    print(f"[INFO]  MOTION_SCALE={MOTION_SCALE}  "
          f"W_POSITION={W_POSITION}  W_TURN={W_TURN}  W_HEADING={W_HEADING}  "
          f"W_FREEZE={W_FREEZE}  W_PATH={W_PATH}  "
          f"FREEZE_MAG_THRESH={FREEZE_MAG_THRESH}  PATIENCE={PATIENCE}")

    train_df, val_df = split_tracks(df)
    print(f"[INFO]  Train tracks: {train_df['track_id'].nunique():,}   "
          f"Val tracks: {val_df['track_id'].nunique():,}")

    feat_scaler = StandardScaler().fit(train_df[FEATURE_COLS].to_numpy())
    tgt_scaler  = StandardScaler().fit(train_df[TARGET_COLS].to_numpy())
    SCALER_PKL.write_bytes(pickle.dumps({
        "feature_scaler": feat_scaler,
        "target_scaler":  tgt_scaler,
        "feature_cols":   FEATURE_COLS,
        "target_cols":    TARGET_COLS,
        "window":         WINDOW_SIZE,
        "motion_scale":   MOTION_SCALE,
    }))
    print(f"[OK]    Wrote {SCALER_PKL.name}")

    train_ds = WindowDataset(train_df, feat_scaler, tgt_scaler, WINDOW_SIZE)
    val_ds   = WindowDataset(val_df,   feat_scaler, tgt_scaler, WINDOW_SIZE)
    print(f"[INFO]  Train windows: {len(train_ds):,}   "
          f"Val windows: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False)

    model = AngularLSTM(n_features=len(FEATURE_COLS),
                        hidden=HIDDEN_SIZE,
                        layers=NUM_LAYERS,
                        n_targets=len(TARGET_COLS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    tgt_mean = torch.tensor(tgt_scaler.mean_,  dtype=torch.float32,
                            device=device)
    tgt_std  = torch.tensor(tgt_scaler.scale_, dtype=torch.float32,
                            device=device)

    history = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    epochs_since_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        tr_pos = tr_turn = tr_head = tr_freeze = tr_path = tr_total = 0.0
        n_batches = 0
        for X, Y, RawY in train_loader:
            X, Y, RawY = X.to(device), Y.to(device), RawY.to(device)
            pred = model(X)
            pred_raw = pred * tgt_std + tgt_mean

            pos_l    = mse(pred[:, :2],  Y[:, :2])
            turn_l   = mse(pred[:, 2:3], Y[:, 2:3])
            head_l   = angular_heading_loss(pred_raw[:, :2], RawY[:, :2])
            freeze_l = freeze_penalty(pred_raw[:, :2])
            path_l   = path_length_loss(pred_raw[:, :2], RawY[:, :2])
            loss     = (W_POSITION * pos_l
                        + W_TURN     * turn_l
                        + W_HEADING  * head_l
                        + W_FREEZE   * freeze_l
                        + W_PATH     * path_l)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_pos    += pos_l.item();    tr_turn   += turn_l.item()
            tr_head   += head_l.item();   tr_freeze += freeze_l.item()
            tr_path   += path_l.item();   tr_total  += loss.item()
            n_batches += 1
        tr_pos /= n_batches; tr_turn /= n_batches
        tr_head /= n_batches; tr_freeze /= n_batches
        tr_path /= n_batches; tr_total /= n_batches

        # ---- Validate ----
        model.eval()
        va_pos = va_turn = va_head = va_freeze = va_path = va_total = 0.0
        n_vb = 0
        with torch.no_grad():
            for X, Y, RawY in val_loader:
                X, Y, RawY = X.to(device), Y.to(device), RawY.to(device)
                pred = model(X)
                pred_raw = pred * tgt_std + tgt_mean
                pos_l    = mse(pred[:, :2],  Y[:, :2])
                turn_l   = mse(pred[:, 2:3], Y[:, 2:3])
                head_l   = angular_heading_loss(pred_raw[:, :2], RawY[:, :2])
                freeze_l = freeze_penalty(pred_raw[:, :2])
                path_l   = path_length_loss(pred_raw[:, :2], RawY[:, :2])
                loss     = (W_POSITION * pos_l
                            + W_TURN     * turn_l
                            + W_HEADING  * head_l
                            + W_FREEZE   * freeze_l
                            + W_PATH     * path_l)
                va_pos    += pos_l.item();    va_turn   += turn_l.item()
                va_head   += head_l.item();   va_freeze += freeze_l.item()
                va_path   += path_l.item();   va_total  += loss.item()
                n_vb += 1
        if n_vb:
            va_pos /= n_vb; va_turn /= n_vb
            va_head /= n_vb; va_freeze /= n_vb
            va_path /= n_vb; va_total /= n_vb

        history.append({
            "epoch": epoch,
            "train_position": tr_pos, "train_turn": tr_turn,
            "train_heading":  tr_head, "train_freeze": tr_freeze,
            "train_path":     tr_path, "train_total": tr_total,
            "val_position":   va_pos,  "val_turn":    va_turn,
            "val_heading":    va_head, "val_freeze":  va_freeze,
            "val_path":       va_path, "val_total":   va_total,
        })

        improved = va_total < best_val - 1e-6
        marker   = ""
        if improved:
            best_val   = va_total
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
            marker = "  *BEST*"
        else:
            epochs_since_improve += 1

        print(f"[E{epoch:03d}] "
              f"train pos={tr_pos:.4f} turn={tr_turn:.4f} "
              f"head={tr_head:.4f} fr={tr_freeze:.4f} path={tr_path:.4f} "
              f"total={tr_total:.4f}  ||  "
              f"val pos={va_pos:.4f} turn={va_turn:.4f} "
              f"head={va_head:.4f} fr={va_freeze:.4f} path={va_path:.4f} "
              f"total={va_total:.4f}{marker}")

        if epochs_since_improve >= PATIENCE:
            print(f"[EARLY] no val improvement for {PATIENCE} epochs "
                  f"(best @ epoch {best_epoch}, val_total={best_val:.4f}). "
                  "Stopping.")
            break

    # ---- Restore best validation checkpoint --------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[INFO]  Restored best validation weights from "
              f"epoch {best_epoch} (val_total={best_val:.4f})")
    else:
        print("[WARN]  No best-state recorded — saving final weights as-is.")

    torch.save({
        "state_dict": model.state_dict(),
        "n_features": len(FEATURE_COLS),
        "hidden":     HIDDEN_SIZE,
        "layers":     NUM_LAYERS,
        "n_targets":  len(TARGET_COLS),
        "feature_cols": FEATURE_COLS,
        "target_cols":  TARGET_COLS,
        "window":     WINDOW_SIZE,
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "motion_scale": MOTION_SCALE,
    }, MODEL_PTH)
    print(f"[OK]    Wrote {MODEL_PTH.name}")

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(TRAIN_SUMMARY_CSV, index=False)
    print(f"[OK]    Wrote {TRAIN_SUMMARY_CSV.name}")

    # Loss curve.
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(hist_df["epoch"], hist_df["train_total"], label="train")
    ax[0].plot(hist_df["epoch"], hist_df["val_total"],   label="val")
    if best_epoch:
        ax[0].axvline(best_epoch, color="grey", lw=0.7, ls="--",
                      label=f"best val (E{best_epoch})")
    ax[0].set_title("Total loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    for k, col in [("position", "train_position"),
                   ("turn",     "train_turn"),
                   ("heading",  "train_heading"),
                   ("freeze",   "train_freeze"),
                   ("path",     "train_path")]:
        ax[1].plot(hist_df["epoch"], hist_df[col], label=f"train {k}")
    ax[1].set_title("Loss components (train)")
    ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(LOSS_PNG, dpi=110)
    plt.close(fig)
    print(f"[OK]    Wrote {LOSS_PNG.name}")

    print(f"[DONE]  best_epoch={best_epoch}  best_val_total={best_val:.4f}")


if __name__ == "__main__":
    main()
