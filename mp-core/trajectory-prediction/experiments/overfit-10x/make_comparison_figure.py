"""
make_comparison_figure.py
-------------------------
Creates comparison_phase1A_vs_phase1B.png.

Phase 1A (absolute prediction) model is retrained here from scratch — the
original weights were overwritten when Phase 1B training ran.
Phase 1B (delta prediction) model and scaler are loaded from disk.

Both rollouts use the same track, the same seed window, and the same 30-step
prediction horizon so the comparison is spatially fair.

Output
------
  mp-data/outputs/prediction/experiments/overfit-10x/
      comparison_phase1A_vs_phase1B.png
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

EXPR_OUT = MP_DATA / "outputs" / "prediction" / "experiments" / "overfit-10x"
LSTM_OUT = EXPR_OUT / "lstm"
CSV_PATH = EXPR_OUT / "trajectories_overfit.csv"

PHASE1B_MODEL  = LSTM_OUT / "overfit_lstm.pth"
PHASE1B_SCALER = LSTM_OUT / "overfit_lstm_scaler.pkl"
OUT_PATH       = EXPR_OUT / "comparison_phase1A_vs_phase1B.png"

# ── Hyper-parameters — identical to training scripts ──────────────────────
WINDOW_SIZE   = 10
HIDDEN_SIZE   = 256
EPOCHS        = 30
LEARNING_RATE = 1e-3
BATCH_SIZE    = 1024
MAX_SEQUENCES = 100_000
N_PERSONS     = 400
N_ROLLOUT     = 30

FEATURE_COLS = [
    "world_x", "world_y",
    "dist_to_obstacle", "dist_to_boundary", "dist_to_entrance",
    "frame_number", "delta_x", "delta_y",
]


# ── Model ─────────────────────────────────────────────────────────────────

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── Data helpers ──────────────────────────────────────────────────────────

def build_sequences(df, target_cols):
    df = df.copy().sort_values(["person_id", "frame_number"])
    df["delta_x"] = df.groupby("person_id")["world_x"].diff().fillna(0)
    df["delta_y"] = df.groupby("person_id")["world_y"].diff().fillna(0)
    X, y = [], []
    for _, track in df.groupby("person_id"):
        track = track.reset_index(drop=True)
        if len(track) <= WINDOW_SIZE:
            continue
        feats   = track[FEATURE_COLS].to_numpy(dtype=np.float32)
        targets = track[target_cols].to_numpy(dtype=np.float32)
        for i in range(len(track) - WINDOW_SIZE):
            X.append(feats[i: i + WINDOW_SIZE])
            y.append(targets[i + WINDOW_SIZE])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_phase1a(df, device):
    """Train LSTM with absolute (x, y) targets — Phase 1A setup."""
    print("[Phase 1A] Building sequences (absolute targets) ...")
    X, y = build_sequences(df, ["world_x", "world_y"])
    print(f"           {len(X):,} sequences -> subsample to {MAX_SEQUENCES:,}")
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
    model = TrajectoryLSTM(len(FEATURE_COLS), HIDDEN_SIZE).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    crit  = nn.MSELoss()

    print(f"[Phase 1A] Training {EPOCHS} epochs ...")
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
        if ep % 10 == 0 or ep == 1:
            print(f"           epoch {ep:>3d}/{EPOCHS}  loss {epoch_loss/len(loader):.6f}")
    print(f"[Phase 1A] Done in {time.time()-t0:.1f}s")
    return model, fs, ts


# ── Rollout helpers ───────────────────────────────────────────────────────

def rollout_absolute(model, fs, ts, feats, positions, device):
    """Phase 1A: predict (x, y) directly, feed back as next position."""
    model.eval()
    window    = fs.transform(feats)[:WINDOW_SIZE].copy()
    prev_x, prev_y = positions[WINDOW_SIZE - 1]
    T = len(feats)
    pred_x, pred_y = [], []
    for step in range(N_ROLLOUT):
        with torch.no_grad():
            p_s = model(torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
                        ).cpu().numpy()[0]
        wx, wy = ts.inverse_transform(p_s[np.newaxis])[0]
        pred_x.append(float(wx))
        pred_y.append(float(wy))
        src = min(WINDOW_SIZE + step, T - 1)
        new_raw = np.array([wx, wy,
                            feats[src, 2], feats[src, 3], feats[src, 4],
                            feats[src, 5],
                            float(wx) - float(prev_x),
                            float(wy) - float(prev_y)], dtype=np.float32)
        window = np.vstack([window[1:], fs.transform(new_raw[np.newaxis])[0]])
        prev_x, prev_y = wx, wy
    return pred_x, pred_y


def rollout_delta(model, fs, ts, feats, positions, device):
    """Phase 1B: predict (dx, dy), accumulate into position."""
    model.eval()
    window    = fs.transform(feats)[:WINDOW_SIZE].copy()
    prev_x, prev_y = positions[WINDOW_SIZE - 1]
    T = len(feats)
    pred_x, pred_y = [], []
    for step in range(N_ROLLOUT):
        with torch.no_grad():
            p_s = model(torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
                        ).cpu().numpy()[0]
        dx, dy = ts.inverse_transform(p_s[np.newaxis])[0]
        wx = float(prev_x) + float(dx)
        wy = float(prev_y) + float(dy)
        pred_x.append(wx)
        pred_y.append(wy)
        src = min(WINDOW_SIZE + step, T - 1)
        new_raw = np.array([wx, wy,
                            feats[src, 2], feats[src, 3], feats[src, 4],
                            feats[src, 5],
                            float(dx), float(dy)], dtype=np.float32)
        window = np.vstack([window[1:], fs.transform(new_raw[np.newaxis])[0]])
        prev_x, prev_y = wx, wy
    return pred_x, pred_y


# ── Figure helpers ────────────────────────────────────────────────────────

def draw_panel(ax, seed_x, seed_y, gt_x, gt_y, pred_x, pred_y,
               title, subtitle, badge, xlim, ylim, drift_m=None):
    ax.set_facecolor("#fafafa")

    # Real path: full GT (seed + continuation), green
    ax.plot(np.concatenate([seed_x, gt_x]),
            np.concatenate([seed_y, gt_y]),
            color="#27ae60", linewidth=1.5, alpha=0.6,
            zorder=2, solid_capstyle="round")
    ax.plot(np.concatenate([seed_x, gt_x]),
            np.concatenate([seed_y, gt_y]),
            "o", color="#27ae60", ms=3.5, alpha=0.55, zorder=2)

    # Seed / history window, orange — drawn on top of GT
    ax.plot(seed_x, seed_y, color="#e67e22", linewidth=2.8,
            zorder=4, solid_capstyle="round")
    ax.plot(seed_x, seed_y, "o", color="#e67e22", ms=5, zorder=4)
    # End-of-seed marker
    ax.plot(seed_x[-1], seed_y[-1], "o", color="#e67e22", ms=9,
            zorder=5, markeredgecolor="white", markeredgewidth=1.5)

    # Predicted rollout, red dashed
    ax.plot(pred_x, pred_y, color="#e74c3c", linewidth=2.2,
            linestyle="--", zorder=6, dash_capstyle="round")
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

    # Badge (top-left)
    ax.text(0.025, 0.975, badge, transform=ax.transAxes,
            fontsize=8.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#ecf0f1",
                      edgecolor="#bdc3c7", alpha=0.95))

    # Drift annotation (bottom-right)
    if drift_m is not None:
        ax.text(0.975, 0.025,
                f"endpoint drift  {drift_m:.2f} m",
                transform=ax.transAxes,
                fontsize=8, va="bottom", ha="right",
                color="#e74c3c" if drift_m > 0.3 else "#27ae60",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#fef9f9" if drift_m > 0.3 else "#eafaf1",
                          edgecolor="#e74c3c" if drift_m > 0.3 else "#27ae60",
                          alpha=0.92))


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    if not CSV_PATH.exists():
        sys.exit(f"[ERROR] {CSV_PATH} not found. Run make_overfit_dataset.py first.")
    if not PHASE1B_MODEL.exists() or not PHASE1B_SCALER.exists():
        sys.exit("[ERROR] Phase 1B model/scaler not found. Run train_lstm_overfit.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {device}")

    # ── Load and filter data (same logic as training scripts) ─────────────
    df = pd.read_csv(CSV_PATH)
    if "track_id" in df.columns and "person_id" not in df.columns:
        df = df.rename(columns={"track_id": "person_id"})
    if "frame_idx" in df.columns and "frame_number" not in df.columns:
        df = df.rename(columns={"frame_idx": "frame_number"})
    print(f"[INFO] Loaded {len(df):,} rows  |  {df['person_id'].nunique()} persons")

    all_pids = df["person_id"].unique()
    chosen   = np.random.default_rng(42).choice(all_pids, min(N_PERSONS, len(all_pids)),
                                                replace=False)
    df = df[df["person_id"].isin(chosen)].copy()
    print(f"[INFO] Pre-filtered to {len(chosen)} persons  ({len(df):,} rows)")

    # ── Select the comparison track ───────────────────────────────────────
    df_s = df.copy().sort_values(["person_id", "frame_number"])
    df_s["delta_x"] = df_s.groupby("person_id")["world_x"].diff().fillna(0)
    df_s["delta_y"] = df_s.groupby("person_id")["world_y"].diff().fillna(0)

    min_len    = WINDOW_SIZE + N_ROLLOUT + 1
    candidates = [(pid, grp.reset_index(drop=True))
                  for pid, grp in df_s.groupby("person_id")
                  if len(grp) >= min_len]
    if not candidates:
        sys.exit("[ERROR] No track is long enough for the comparison.")

    pid, track = candidates[len(candidates) // 2]
    print(f"[INFO] Selected track : person_id={pid}  length={len(track)}")

    feats     = track[FEATURE_COLS].to_numpy(dtype=np.float32)
    positions = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)

    # ── Phase 1A — retrain (original model was overwritten by Phase 1B) ───
    model_1a, fs_1a, ts_1a = train_phase1a(df, device)

    # ── Phase 1B — load from disk ─────────────────────────────────────────
    print("\n[Phase 1B] Loading model from disk ...")
    with open(PHASE1B_SCALER, "rb") as f:
        bundle_1b = pickle.load(f)
    model_1b = TrajectoryLSTM(len(FEATURE_COLS), HIDDEN_SIZE).to(device)
    model_1b.load_state_dict(torch.load(PHASE1B_MODEL, map_location=device,
                                        weights_only=True))
    fs_1b = bundle_1b["feature_scaler"]
    ts_1b = bundle_1b["target_scaler"]
    print("[Phase 1B] Loaded.")

    # ── Run both rollouts on the same track ───────────────────────────────
    print("\n[INFO] Running rollouts ...")
    pred_x_1a, pred_y_1a = rollout_absolute(model_1a, fs_1a, ts_1a,
                                             feats, positions, device)
    pred_x_1b, pred_y_1b = rollout_delta(model_1b, fs_1b, ts_1b,
                                          feats, positions, device)

    seed_x = positions[:WINDOW_SIZE, 0]
    seed_y = positions[:WINDOW_SIZE, 1]
    gt_x   = positions[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 0]
    gt_y   = positions[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 1]

    # Endpoint drift from the first GT rollout point
    origin_x, origin_y = positions[WINDOW_SIZE, 0], positions[WINDOW_SIZE, 1]
    drift_1a = float(np.sqrt((pred_x_1a[-1] - origin_x)**2 +
                             (pred_y_1a[-1] - origin_y)**2))
    drift_1b = float(np.sqrt((pred_x_1b[-1] - origin_x)**2 +
                             (pred_y_1b[-1] - origin_y)**2))
    print(f"\n  Phase 1A endpoint drift : {drift_1a:.3f} m")
    print(f"  Phase 1B endpoint drift : {drift_1b:.3f} m")

    # ── Shared axis limits (union of all paths, padded) ───────────────────
    all_x = np.concatenate([seed_x, gt_x, pred_x_1a, pred_x_1b])
    all_y = np.concatenate([seed_y, gt_y, pred_y_1a, pred_y_1b])
    pad   = max(0.12, (all_x.max() - all_x.min()) * 0.08)
    xlim  = (all_x.min() - pad, all_x.max() + pad)
    ylim  = (all_y.min() - pad, all_y.max() + pad)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#f5f5f5")

    draw_panel(ax1, seed_x, seed_y, gt_x, gt_y,
               pred_x_1a, pred_y_1a,
               title    = "Phase 1A — Absolute Position Prediction",
               subtitle = "Problem: autoregressive drift",
               badge    = "Model predicts  (x, y)  directly",
               xlim=xlim, ylim=ylim,
               drift_m=drift_1a)

    # Arrow pointing to where Phase 1A prediction ends up
    ax1.annotate(
        "drift away",
        xy    =(pred_x_1a[-1], pred_y_1a[-1]),
        xytext=(pred_x_1a[-1] - (xlim[1]-xlim[0])*0.12,
                pred_y_1a[-1] + (ylim[1]-ylim[0])*0.06),
        fontsize=9, color="#c0392b", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.8),
    )

    draw_panel(ax2, seed_x, seed_y, gt_x, gt_y,
               pred_x_1b, pred_y_1b,
               title    = "Phase 1B — Delta Prediction",
               subtitle = "Improvement: stable local rollout",
               badge    = "Model predicts  (Δx, Δy),  position accumulated",
               xlim=xlim, ylim=ylim,
               drift_m=drift_1b)

    # Tick mark showing Phase 1B stays close
    ax2.annotate(
        "stays close  [reduced]",
        xy    =(pred_x_1b[-1], pred_y_1b[-1]),
        xytext=(pred_x_1b[-1] + (xlim[1]-xlim[0])*0.06,
                pred_y_1b[-1] - (ylim[1]-ylim[0])*0.08),
        fontsize=9, color="#27ae60", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.8),
    )

    # Shared legend
    legend_handles = [
        mpatches.Patch(facecolor="#27ae60", label="Real path (GT)"),
        mpatches.Patch(facecolor="#e67e22", label="Seed / history window  (10 steps)"),
        mpatches.Patch(facecolor="#e74c3c", label="Predicted rollout  (30 steps)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=10.5, framealpha=0.95,
               bbox_to_anchor=(0.5, 0.005),
               edgecolor="#cccccc")

    fig.suptitle(
        "Motion Pixels — Overfit Experiment\n"
        "Phase 1A vs Phase 1B  |  LSTM  |  same track · same seed · 30-step horizon\n"
        "[MEMORISATION TEST — NOT generalisation evidence]",
        fontsize=11, color="#7f0000", y=1.015,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.98])

    EXPR_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"\n[OK] Saved: {OUT_PATH}")
    if drift_1a > 0:
        print(f"     Drift reduction: {drift_1a:.3f}m -> {drift_1b:.3f}m"
              f"  ({(1 - drift_1b/drift_1a)*100:.0f}% less)")


if __name__ == "__main__":
    main()
