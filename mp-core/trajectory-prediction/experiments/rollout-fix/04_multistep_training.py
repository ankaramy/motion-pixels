"""
04_multistep_training.py
------------------------
Experiment D — train a SEPARATE small LSTM to predict the cumulative
displacement over K future steps (K=5), then divide by K at inference to obtain
a per-step delta.

Hypothesis:
  Predicting an aggregated K-step displacement gives the model a larger,
  less-noisy regression target — the per-step shrinkage induced by MSE
  toward the dataset mean should be milder when the target's |magnitude|
  is K× larger than a single step.

Constraints honoured:
  - Same LSTM architecture (256-unit, 1 layer)
  - Same input features (6 spatial)
  - Trains and saves SEPARATELY to:
      mp-data/outputs/prediction_rollout_fix/models/lstm_k5/
  - Does NOT overwrite anything in mp-data/outputs/prediction/.
"""

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

HERE     = Path(__file__).resolve().parent
MP_ROOT  = HERE.parent.parent.parent.parent
MP_DATA  = MP_ROOT / "mp-data"
OVERFIT  = MP_ROOT / "mp-core" / "trajectory-prediction" / "experiments" / "overfit-10x"
sys.path.insert(0, str(OVERFIT))

from train_phase2b_final import (   # noqa: E402
    TrajectoryLSTM, SpatialInterpolator, add_deltas,
    FEATURE_COLS, WINDOW_SIZE, HIDDEN_SIZE, IDW_K,
)

SRC_CSV  = MP_DATA / "processed" / "encoded" / "trajectories_encoded.csv"
OUT      = MP_DATA / "outputs" / "prediction_rollout_fix"
MODELS   = OUT / "models" / "lstm_k5"
PLOTS    = OUT / "plots"
DEBUG    = OUT / "debug"
REPORTS  = OUT / "reports"

K               = 5
EPOCHS          = 20
BATCH_SIZE      = 1024
LR              = 1e-3
MAX_SEQUENCES   = 60_000
N_PERSONS       = 300
N_ROLLOUT       = 30
PROBE_PIDS      = [310, 50, 100, 200, 1, 2]
MIN_LEN         = WINDOW_SIZE + N_ROLLOUT
TARGET_COLS_K   = ["sum_dx_k", "sum_dy_k"]


# ── Sequence builder for K-step target ─────────────────────────────────────
def build_sequences_k(df, k=K):
    X, y = [], []
    for _, track in df.groupby("person_id"):
        track = track.reset_index(drop=True)
        if len(track) < WINDOW_SIZE + k:
            continue
        feats = track[FEATURE_COLS].to_numpy(dtype=np.float32)
        du = track["delta_x"].to_numpy(dtype=np.float32)
        dv = track["delta_y"].to_numpy(dtype=np.float32)
        for i in range(len(track) - WINDOW_SIZE - k + 1):
            X.append(feats[i: i + WINDOW_SIZE])
            target = np.array([
                float(du[i + WINDOW_SIZE: i + WINDOW_SIZE + k].sum()),
                float(dv[i + WINDOW_SIZE: i + WINDOW_SIZE + k].sum()),
            ], dtype=np.float32)
            y.append(target)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Rollout using K-step model with per-step division ──────────────────────
def rollout_k_div(model, fs, ts, track, interp, device, k=K):
    feats     = track[FEATURE_COLS].to_numpy(dtype=np.float32)
    positions = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    feats_s   = fs.transform(feats)
    window = feats_s[:WINDOW_SIZE].copy()
    prev_x = float(positions[WINDOW_SIZE - 1, 0])
    prev_y = float(positions[WINDOW_SIZE - 1, 1])

    pred_du, pred_dv, pred_x, pred_y = [], [], [], []
    for _ in range(N_ROLLOUT):
        x_in = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            p_s = model(x_in).cpu().numpy()[0]
        Kdu, Kdv = ts.inverse_transform(p_s[np.newaxis])[0]
        du = float(Kdu) / k
        dv = float(Kdv) / k

        wx, wy = prev_x + du, prev_y + dv
        pred_du.append(du); pred_dv.append(dv)
        pred_x.append(wx);  pred_y.append(wy)

        obs, bnd = interp.query(wx, wy)
        new_raw = np.array([wx, wy, du, dv, obs, bnd], dtype=np.float32)
        window = np.vstack([window[1:], fs.transform(new_raw[np.newaxis])[0]])
        prev_x, prev_y = wx, wy

    return (np.array(pred_du), np.array(pred_dv),
            np.array(pred_x), np.array(pred_y))


def metrics(du, dv, x, y, gx, gy, gdu, gdv):
    ade = float(np.mean(np.hypot(x - gx, y - gy)))
    fde = float(np.hypot(x[-1] - gx[-1], y[-1] - gy[-1]))
    cum_pred = float(np.sum(np.hypot(du, dv)))
    cum_gt   = float(np.sum(np.hypot(gdu, gdv)))
    return {"ade": ade, "fde": fde, "cum_pred": cum_pred, "cum_gt": cum_gt,
            "cum_ratio": cum_pred / cum_gt if cum_gt > 1e-9 else float("nan")}


def gt_future(track):
    pos = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    seed_end = pos[WINDOW_SIZE - 1]
    fx = pos[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 0]
    fy = pos[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 1]
    gdu = np.r_[fx[0] - seed_end[0], np.diff(fx)]
    gdv = np.r_[fy[0] - seed_end[1], np.diff(fy)]
    return fx, fy, gdu, gdv


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    DEBUG.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device : {device}")

    df_full = pd.read_csv(SRC_CSV)
    if "track_id" in df_full.columns and "person_id" not in df_full.columns:
        df_full = df_full.rename(columns={"track_id": "person_id"})
    if "frame_idx" in df_full.columns and "frame_number" not in df_full.columns:
        df_full = df_full.rename(columns={"frame_idx": "frame_number"})

    interp = SpatialInterpolator(df_full, k=IDW_K)
    df = add_deltas(df_full)

    rng = np.random.default_rng(42)
    all_pids = df_full["person_id"].unique()
    chosen = rng.choice(all_pids, min(N_PERSONS, len(all_pids)), replace=False)
    df_train = df[df["person_id"].isin(chosen)].copy()
    print(f"[INFO]  Building K={K} sequences from {len(chosen)} persons ...")
    X, y = build_sequences_k(df_train, k=K)
    print(f"[INFO]  {len(X):,} sequences with K-step targets")

    if len(X) > MAX_SEQUENCES:
        idx = rng.choice(len(X), MAX_SEQUENCES, replace=False)
        X, y = X[idx], y[idx]
    print(f"[INFO]  Training on {len(X):,} sequences")

    N, W, F = X.shape
    fs = MinMaxScaler()
    ts = MinMaxScaler()
    Xs = fs.fit_transform(X.reshape(-1, F)).reshape(N, W, F).astype(np.float32)
    ys = ts.fit_transform(y).astype(np.float32)
    print(f"[INFO]  Target K-step ranges (m): du=[{y[:,0].min():.3f},{y[:,0].max():.3f}]  "
          f"dv=[{y[:,1].min():.3f},{y[:,1].max():.3f}]")

    loader = DataLoader(TensorDataset(torch.tensor(Xs), torch.tensor(ys)),
                        batch_size=BATCH_SIZE, shuffle=True)
    model = TrajectoryLSTM(F, HIDDEN_SIZE).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()

    history = []
    t0 = time.time()
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
        if ep == 1 or ep % 5 == 0 or ep == EPOCHS:
            print(f"  ep {ep:>3}/{EPOCHS}  loss={avg:.6f}")
    print(f"[INFO]  Trained in {time.time()-t0:.1f}s  final loss={history[-1]:.6f}")

    torch.save(model.state_dict(), MODELS / "model.pth")
    with open(MODELS / "scaler.pkl", "wb") as fh:
        pickle.dump({"feature_scaler": fs, "target_scaler": ts,
                     "feature_cols": FEATURE_COLS,
                     "target_cols": TARGET_COLS_K,
                     "K": K, "window_size": WINDOW_SIZE}, fh)
    print(f"[OK]    Model saved to {MODELS}")

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, EPOCHS + 1), history, color="#16a085", lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (scaled)")
    ax.set_title(f"Multi-step LSTM (K={K}) loss curve")
    ax.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(PLOTS / "k5_loss_curve.png", dpi=150)
    plt.close(fig)

    # ── Evaluate on probe PIDs ───────────────────────────────────────────────
    eligible = set(df.groupby("person_id").size()[
        lambda s: s >= MIN_LEN].index.tolist())
    probe = [p for p in PROBE_PIDS if p in eligible]

    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Multi-step LSTM rollout (K={K}, divide-by-K)", fontsize=12,
                 fontweight="bold")

    for ax, pid in zip(axes.flat, probe):
        track = (df[df["person_id"] == pid].reset_index(drop=True)
                 .iloc[:MIN_LEN].copy())
        gx, gy, gdu, gdv = gt_future(track)
        du, dv, x, y = rollout_k_div(model, fs, ts, track, interp, device, k=K)
        m = metrics(du, dv, x, y, gx, gy, gdu, gdv)
        rows.append({"pid": pid, **m})

        ax.plot(gx, gy, "k-o", ms=3, lw=1.5, alpha=0.7, label="GT")
        ax.plot(x, y, "r-^", ms=3, lw=1.3, alpha=0.85, label=f"K=5/K pred")
        ax.set_title(f"pid {pid}  ADE={m['ade']:.2f}  cum={m['cum_ratio']:.2f}",
                     fontsize=10)
        ax.set_xlabel("world_x (m)"); ax.set_ylabel("world_y (m)")
        ax.legend(fontsize=8); ax.grid(True, lw=0.3, alpha=0.5)
        ax.set_aspect("equal", adjustable="datalim")
        print(f"  pid={pid}  cum_ratio={m['cum_ratio']:.3f}  "
              f"ADE={m['ade']:.3f}  FDE={m['fde']:.3f}")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOTS / "k5_rollout_per_pid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    df_m = pd.DataFrame(rows)
    df_m.to_csv(DEBUG / "k5_metrics.csv", index=False)

    md = [
        f"# Multi-step (K={K}) LSTM rollout — experiment D",
        "",
        "Train a new LSTM whose target is the cumulative K-step displacement,",
        "then divide by K at inference for per-step delta.  Same architecture",
        "(256-unit LSTM) and same six spatial features as the production model.",
        "",
        f"- K (target horizon)  : **{K}** steps",
        f"- Persons used        : {N_PERSONS}",
        f"- Sequences trained on: {len(X):,} (cap {MAX_SEQUENCES:,})",
        f"- Epochs              : {EPOCHS}",
        f"- Final train loss    : {history[-1]:.6f}",
        f"- Model dir           : `{MODELS.relative_to(MP_ROOT).as_posix()}`",
        "",
        "## Per-trajectory metrics",
        "",
        "| pid | ADE (m) | FDE (m) | cum ratio (pred/GT) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        md.append(f"| {r['pid']} | {r['ade']:.3f} | {r['fde']:.3f} | {r['cum_ratio']:.3f} |")

    md += [
        "",
        f"- Mean ADE        : **{np.mean([r['ade'] for r in rows]):.3f} m**",
        f"- Mean FDE        : **{np.mean([r['fde'] for r in rows]):.3f} m**",
        f"- Mean cum-ratio  : **{np.mean([r['cum_ratio'] for r in rows]):.3f}**",
        "",
        f"![loss]({(PLOTS / 'k5_loss_curve.png').as_posix()})",
        f"![rollouts]({(PLOTS / 'k5_rollout_per_pid.png').as_posix()})",
        "",
        "## Interpretation",
        "",
        "Training on a larger (K-step) regression target effectively gives the",
        "model a larger signal-to-noise ratio for motion magnitude.  Dividing",
        "by K at inference recovers per-step deltas without changing the",
        "architecture or the rollout structure.",
        "",
        "---",
        "*Generated by `experiments/rollout-fix/04_multistep_training.py`*",
    ]
    out = REPORTS / "k5_multistep_report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK]    Report : {out}")


if __name__ == "__main__":
    main()
