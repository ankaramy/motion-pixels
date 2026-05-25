"""
train_phase2b_final.py
-----------------------
Phase 2B-Final: Fair model comparison with correctly integrated spatial features.

Phase 2B used movement-only inputs.
Phase 2C proved that spatial features work when recomputed during rollout.
This script combines both: LSTM, GRU, and TCN with spatial inputs + KDTree
rollout recomputation.

Feature set (all models):
  world_x, world_y, delta_x, delta_y, dist_to_obstacle, dist_to_boundary

Rollout rule (all models):
  At every predicted step, dist_to_obstacle and dist_to_boundary are
  recomputed from the predicted position via KDTree inverse-distance weighted
  interpolation over the full encoded dataset.
  Ground-truth spatial values are NEVER copied into rollout windows.

Anchor points note:
  encode_space.py does not persist anchor files. Fallback: KDTree over
  all 180,840 rows of trajectories_encoded.csv (MAE ~0 m on in-distribution
  positions, validated in Phase 2C).

Output: mp-data/outputs/prediction/experiments/phase-2b-final/
  lstm/  gru/  tcn/  — each: model.pth  scaler.pkl  loss_curve.png
                              prediction_visual.png
  phase2b_final_spatial_model_comparison.png
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
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

HERE    = Path(__file__).resolve().parent
MP_ROOT = HERE.parent.parent.parent.parent
MP_DATA = MP_ROOT / "mp-data"

SRC_CSV   = MP_DATA / "processed" / "encoded" / "trajectories_encoded.csv"
OUT_ROOT  = MP_DATA / "outputs" / "prediction" / "experiments" / "phase-2b-final"
COMP_PATH = OUT_ROOT / "phase2b_final_spatial_model_comparison.png"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
WINDOW_SIZE   = 10
HIDDEN_SIZE   = 256     # LSTM / GRU
TCN_CHANNELS  = 128     # TCN per-layer channels
EPOCHS        = 30
LEARNING_RATE = 1e-3
BATCH_SIZE    = 1024
PRINT_EVERY   = 5
MAX_SEQUENCES = 100_000
N_PERSONS     = 400
N_ROLLOUT     = 30
IDW_K         = 5       # neighbours for spatial interpolation

FEATURE_COLS = ["world_x", "world_y", "delta_x", "delta_y",
                "dist_to_obstacle", "dist_to_boundary"]
TARGET_COLS  = ["delta_x", "delta_y"]
# dist_to_entrance excluded: Pearson |r| ~ 0.7 with world_x/world_y — redundant.


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, output_size: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TrajectoryGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, output_size: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class _Chomp1d(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, :, :-self.n].contiguous()


class _TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            _Chomp1d(pad), nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            _Chomp1d(pad), nn.ReLU(),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + (x if self.downsample is None
                                        else self.downsample(x)))


class TrajectoryTCN(nn.Module):
    """TCN with dilated causal convolutions.

    4 levels, dilation 1/2/4/8, kernel=3 → receptive field 31 > WINDOW_SIZE.
    """
    def __init__(self, input_size: int, num_channels: int = 128,
                 output_size: int = 2, num_levels: int = 4, kernel_size: int = 3):
        super().__init__()
        layers = []
        for i in range(num_levels):
            in_ch = input_size if i == 0 else num_channels
            layers.append(_TemporalBlock(in_ch, num_channels, kernel_size,
                                         dilation=2 ** i))
        self.tcn = nn.Sequential(*layers)
        self.fc  = nn.Linear(num_channels, output_size)

    def forward(self, x):
        return self.fc(self.tcn(x.transpose(1, 2))[:, :, -1])


# ---------------------------------------------------------------------------
# Spatial interpolator
# ---------------------------------------------------------------------------

class SpatialInterpolator:
    """KDTree over the full encoded dataset.

    For any predicted (x, y), returns (dist_to_obstacle, dist_to_boundary)
    via inverse-distance weighted interpolation over the k nearest encoded
    trajectory points. Validated in Phase 2C: MAE ~0 m on in-distribution
    positions because the dataset is dense enough that exact matches exist.
    """

    def __init__(self, df: pd.DataFrame, k: int = 5):
        coords = df[["world_x", "world_y"]].to_numpy(dtype=np.float64)
        self.tree     = cKDTree(coords)
        self.obs_vals = df["dist_to_obstacle"].to_numpy(dtype=np.float64)
        self.bnd_vals = df["dist_to_boundary"].to_numpy(dtype=np.float64)
        self.k        = k

    def query(self, x: float, y: float):
        dists, idxs = self.tree.query([x, y], k=self.k)
        if dists[0] < 1e-9:
            return float(self.obs_vals[idxs[0]]), float(self.bnd_vals[idxs[0]])
        w = 1.0 / dists
        w /= w.sum()
        return float(np.dot(w, self.obs_vals[idxs])), float(np.dot(w, self.bnd_vals[idxs]))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["person_id", "frame_number"])
    df["delta_x"] = df.groupby("person_id")["world_x"].diff().fillna(0)
    df["delta_y"] = df.groupby("person_id")["world_y"].diff().fillna(0)
    return df


def build_sequences(df: pd.DataFrame):
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


# ---------------------------------------------------------------------------
# Train one model  (receives already-subsampled X, y)
# ---------------------------------------------------------------------------

def train_one(X, y, model_factory, out_dir, label, device):
    N, W, F = X.shape
    fs = MinMaxScaler()
    ts = MinMaxScaler()
    Xs = fs.fit_transform(X.reshape(-1, F)).reshape(N, W, F).astype(np.float32)
    ys = ts.fit_transform(y).astype(np.float32)

    loader = DataLoader(TensorDataset(torch.tensor(Xs), torch.tensor(ys)),
                        batch_size=BATCH_SIZE, shuffle=True)

    model    = model_factory().to(device)
    opt      = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    crit     = nn.MSELoss()
    n_params = sum(p.numel() for p in model.parameters())

    print(f"[{label}]  {n_params:,} params  |  {len(loader)} batches/epoch  |  "
          f"{N:,} seqs")
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

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pth")
    bundle = {"feature_scaler": fs, "target_scaler": ts,
              "feature_cols": FEATURE_COLS, "target_cols": TARGET_COLS,
              "window_size": WINDOW_SIZE}
    with open(out_dir / "scaler.pkl", "wb") as fh:
        pickle.dump(bundle, fh)

    color_map = {"LSTM": "#c0392b", "GRU": "#2980b9", "TCN": "#8e44ad"}
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, EPOCHS + 1), history, lw=1.5,
            color=color_map.get(label, "#333"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalised delta)")
    ax.set_title(f"Phase 2B-Final — {label}  |  {EPOCHS} ep  |  {N:,} seqs  "
                 f"|  6 spatial features")
    ax.grid(True, lw=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    return model, fs, ts, history, elapsed, n_params


# ---------------------------------------------------------------------------
# Rollout with KDTree spatial recomputation
# ---------------------------------------------------------------------------

def rollout(model, fs, ts, track: pd.DataFrame,
            interp: SpatialInterpolator, device):
    """Autoregressive delta rollout with spatial features recomputed at each step.

    At every step:
      1. Predict (dx, dy) from current window
      2. Accumulate: x_next = x + dx,  y_next = y + dy
      3. Recompute dist_to_obstacle, dist_to_boundary via KDTree at (x_next, y_next)
      4. Build next window row: [x_next, y_next, dx, dy, obs, bnd]
    """
    model.eval()
    feats     = track[FEATURE_COLS].to_numpy(dtype=np.float32)
    positions = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    T = len(feats)

    feats_scaled = fs.transform(feats)
    window   = feats_scaled[:WINDOW_SIZE].copy()
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

        # Recompute spatial features from predicted position — never copy GT
        obs, bnd = interp.query(wx, wy)
        new_raw = np.array([wx, wy, float(dx), float(dy), obs, bnd],
                           dtype=np.float32)
        window = np.vstack([window[1:], fs.transform(new_raw[np.newaxis])[0]])
        prev_x, prev_y = wx, wy

    return np.array(pred_x), np.array(pred_y), positions


# ---------------------------------------------------------------------------
# Per-model prediction visual
# ---------------------------------------------------------------------------

def save_prediction_visual(pred_x, pred_y, positions, out_path, label):
    seed_x = positions[:WINDOW_SIZE, 0]
    seed_y = positions[:WINDOW_SIZE, 1]
    gt_x   = positions[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 0]
    gt_y   = positions[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(np.r_[seed_x, gt_x], np.r_[seed_y, gt_y],
             color="#27ae60", lw=1.5, alpha=0.7, label="GT path")
    ax1.plot(seed_x, seed_y, "-o", color="#e67e22", ms=4, lw=2.2, label="Seed")
    ax1.plot(pred_x, pred_y, "r--o", ms=3, alpha=0.85, label=f"{label} pred")
    ax1.legend(fontsize=8)
    ax1.set_title(f"{label} — 2D [spatial delta rollout]")
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
        f"Phase 2B-Final — {label}  |  spatial features + KDTree rollout  "
        f"|  {N_ROLLOUT}-step horizon\n"
        "[NOT generalisation evidence]",
        fontsize=10, color="#7f0000",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO]  Prediction visual: {out_path}")


# ---------------------------------------------------------------------------
# 3-panel comparison figure
# ---------------------------------------------------------------------------

def _draw_panel(ax, seed_x, seed_y, gt_x, gt_y, pred_x, pred_y,
                title, subtitle, badge, xlim, ylim, drift_m, panel_color):
    ax.set_facecolor("#fafafa")
    full_x = np.r_[seed_x, gt_x]
    full_y = np.r_[seed_y, gt_y]
    ax.plot(full_x, full_y, color="#27ae60", lw=1.5, alpha=0.6, zorder=2)
    ax.plot(full_x, full_y, "o", color="#27ae60", ms=3.5, alpha=0.5, zorder=2)
    ax.plot(seed_x, seed_y, color="#e67e22", lw=2.8, zorder=4)
    ax.plot(seed_x, seed_y, "o", color="#e67e22", ms=5, zorder=4)
    ax.plot(seed_x[-1], seed_y[-1], "o", color="#e67e22", ms=9, zorder=5,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(pred_x, pred_y, color="#e74c3c", lw=2.2, ls="--", zorder=6)
    ax.plot(pred_x, pred_y, "o", color="#e74c3c", ms=4, alpha=0.85, zorder=6)
    ax.plot(pred_x[0], pred_y[0], "o", color="#e74c3c", ms=9, zorder=7,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("world_x  (m)", fontsize=9)
    ax.set_ylabel("world_y  (m)", fontsize=9)
    ax.grid(True, lw=0.35, alpha=0.5, color="#cccccc")
    ax.set_title(f"{title}\n{subtitle}", fontsize=10, fontweight="bold",
                 pad=8, color="#2c3e50")
    for sp in ax.spines.values():
        sp.set_edgecolor(panel_color)
        sp.set_linewidth(2.0)
    ax.text(0.025, 0.975, badge, transform=ax.transAxes, fontsize=7.5,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1",
                      edgecolor="#bdc3c7", alpha=0.95))
    dc = "#e74c3c" if drift_m > 0.3 else "#27ae60"
    ax.text(0.975, 0.025, f"drift  {drift_m:.2f} m",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
            color=dc, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#fef9f9" if drift_m > 0.3 else "#eafaf1",
                      edgecolor=dc, alpha=0.92))


def save_comparison(seed_x, seed_y, gt_x, gt_y, results):
    all_x = np.concatenate([seed_x, gt_x] + [r["pred_x"] for r in results])
    all_y = np.concatenate([seed_y, gt_y] + [r["pred_y"] for r in results])
    pad   = max(0.15, (all_x.max() - all_x.min()) * 0.08)
    xlim  = (float(all_x.min()) - pad, float(all_x.max()) + pad)
    ylim  = (float(all_y.min()) - pad, float(all_y.max()) + pad)

    panel_colors = {"LSTM": "#c0392b", "GRU": "#2980b9", "TCN": "#8e44ad"}

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor("#f5f5f5")

    for ax, r in zip(axes, results):
        _draw_panel(
            ax, seed_x, seed_y, gt_x, gt_y, r["pred_x"], r["pred_y"],
            title       = r["label"],
            subtitle    = f"loss {r['loss']:.6f}  |  {r['runtime']:.0f}s",
            badge       = r["badge"],
            xlim=xlim, ylim=ylim, drift_m=r["drift"],
            panel_color = panel_colors.get(r["label"], "#333"),
        )

    handles = [
        mpatches.Patch(facecolor="#27ae60", label="Real path (GT)"),
        mpatches.Patch(facecolor="#e67e22", label="Seed / history  (10 steps)"),
        mpatches.Patch(facecolor="#e74c3c", label="Predicted rollout  (30 steps)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10.5,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.005),
               edgecolor="#cccccc")
    fig.suptitle(
        "Phase 2B-Final  —  LSTM  vs  GRU  vs  TCN\n"
        "Inputs: world_x, world_y, delta_x, delta_y, dist_to_obstacle, "
        "dist_to_boundary\n"
        "Rollout: spatial features recomputed via KDTree at every predicted "
        "position  |  30-step horizon\n"
        "[NOT generalisation evidence  —  controlled comparison only]",
        fontsize=10.5, color="#7f0000", y=1.02,
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
        sys.exit(f"[ERROR] {SRC_CSV} not found. Run encode_space.py first.")

    # ── Load full encoded dataset ──────────────────────────────────────────
    df_full = pd.read_csv(SRC_CSV)
    if "track_id" in df_full.columns and "person_id" not in df_full.columns:
        df_full = df_full.rename(columns={"track_id": "person_id"})
    if "frame_idx" in df_full.columns and "frame_number" not in df_full.columns:
        df_full = df_full.rename(columns={"frame_idx": "frame_number"})
    print(f"[INFO]  Loaded {len(df_full):,} rows  |  "
          f"{df_full['person_id'].nunique()} persons")

    required_static = ["world_x", "world_y", "dist_to_obstacle", "dist_to_boundary"]
    if miss := set(required_static) - set(df_full.columns):
        sys.exit(f"[ERROR] Missing columns: {miss}")

    # ── Spatial interpolator — built on full dataset before person filter ──
    print("[INFO]  Building KDTree spatial interpolator ...")
    interp = SpatialInterpolator(df_full, k=IDW_K)
    print(f"[INFO]  KDTree ready  ({len(df_full):,} points, k={IDW_K})")

    # ── Pre-filter persons ─────────────────────────────────────────────────
    all_pids = df_full["person_id"].unique()
    chosen   = np.random.default_rng(42).choice(
        all_pids, min(N_PERSONS, len(all_pids)), replace=False)
    df = df_full[df_full["person_id"].isin(chosen)].copy()
    print(f"[INFO]  Pre-filtered to {len(chosen)} persons  ({len(df):,} rows)")

    df = add_deltas(df)

    # ── Fix comparison track (same for all 3 models) ───────────────────────
    min_len    = WINDOW_SIZE + N_ROLLOUT + 1
    candidates = [(pid, grp.reset_index(drop=True))
                  for pid, grp in df.groupby("person_id")
                  if len(grp) >= min_len]
    if not candidates:
        sys.exit("[ERROR] No track is long enough for rollout.")
    comp_pid, comp_track = candidates[len(candidates) // 2]
    print(f"[INFO]  Comparison track : person_id={comp_pid}  "
          f"length={len(comp_track)}")

    # ── Build sequences once; subsample with fixed seed for all models ─────
    print("\n[INFO]  Building sequences ...")
    X_all, y_all = build_sequences(df)
    print(f"[INFO]  {len(X_all):,} sequences  ({X_all.shape})")

    if len(X_all) > MAX_SEQUENCES:
        idx_sub = np.random.default_rng(42).choice(
            len(X_all), MAX_SEQUENCES, replace=False)
        X_train, y_train = X_all[idx_sub], y_all[idx_sub]
    else:
        X_train, y_train = X_all, y_all
    print(f"[INFO]  Training on {len(X_train):,} sequences\n")

    # ── Model registry ─────────────────────────────────────────────────────
    n_feats = len(FEATURE_COLS)
    models_cfg = [
        ("LSTM", OUT_ROOT / "lstm",
         lambda: TrajectoryLSTM(n_feats, HIDDEN_SIZE)),
        ("GRU",  OUT_ROOT / "gru",
         lambda: TrajectoryGRU(n_feats, HIDDEN_SIZE)),
        ("TCN",  OUT_ROOT / "tcn",
         lambda: TrajectoryTCN(n_feats, TCN_CHANNELS)),
    ]

    results = []
    for label, out_dir, factory in models_cfg:
        print(f"\n{'='*54}")
        print(f"  Training {label}")
        print(f"{'='*54}")
        model, fs, ts, history, elapsed, n_params = train_one(
            X_train.copy(), y_train.copy(), factory, out_dir, label, device)

        pred_x, pred_y, positions = rollout(
            model, fs, ts, comp_track, interp, device)
        save_prediction_visual(pred_x, pred_y, positions,
                               out_dir / "prediction_visual.png", label)

        gt_end = positions[WINDOW_SIZE + N_ROLLOUT - 1]
        drift  = float(np.sqrt(
            (pred_x[-1] - gt_end[0]) ** 2 + (pred_y[-1] - gt_end[1]) ** 2))

        results.append({
            "label":    label,
            "pred_x":   pred_x,
            "pred_y":   pred_y,
            "drift":    drift,
            "loss":     history[-1],
            "runtime":  elapsed,
            "badge":    f"{n_params:,} params",
            "n_params": n_params,
            "history":  history,
        })
        print(f"[{label}]  endpoint drift : {drift:.3f} m")

    # ── Comparison figure ───────────────────────────────────────────────────
    positions = comp_track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    seed_x = positions[:WINDOW_SIZE, 0]
    seed_y = positions[:WINDOW_SIZE, 1]
    gt_x   = positions[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 0]
    gt_y   = positions[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 1]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    save_comparison(seed_x, seed_y, gt_x, gt_y, results)

    # ── Summary report ──────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  PHASE 2B-FINAL RESULTS  (spatial features + KDTree rollout)")
    print(f"{'='*68}")
    print(f"  {'Model':<8}  {'Final Loss':>12}  {'Runtime':>10}  "
          f"{'Drift':>10}  {'Params':>12}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*12}")
    for r in results:
        print(f"  {r['label']:<8}  {r['loss']:>12.6f}  {r['runtime']:>8.1f}s  "
              f"{r['drift']:>8.3f} m  {r['n_params']:>12,}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*12}")

    best_drift = min(results, key=lambda r: r["drift"])
    best_loss  = min(results, key=lambda r: r["loss"])
    fastest    = min(results, key=lambda r: r["runtime"])

    print(f"\n  Lowest endpoint drift : {best_drift['label']}  "
          f"({best_drift['drift']:.3f} m)")
    print(f"  Lowest training loss  : {best_loss['label']}  "
          f"({best_loss['loss']:.6f})")
    print(f"  Fastest training      : {fastest['label']}  "
          f"({fastest['runtime']:.1f}s)")

    # Behaviour assessment
    print(f"\n  Rollout behaviour assessment:")
    for r in results:
        d = r["drift"]
        if d < 0.15:
            behaviour = "stable — near ground truth"
        elif d < 0.5:
            behaviour = "mild drift"
        elif d < 1.5:
            behaviour = "drifts noticeably"
        else:
            behaviour = "drifts severely"
        print(f"    {r['label']:<8}  drift {d:.3f} m  ->  {behaviour}")

    lstm_r = next(r for r in results if r["label"] == "LSTM")
    lstm_best = best_drift["label"] == "LSTM"
    print(f"\n  Does LSTM remain best with spatial recomputation?  "
          f"{'YES' if lstm_best else 'NO — ' + best_drift['label'] + ' wins'}")
    print(f"\n  Comparison figure : {COMP_PATH}")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
