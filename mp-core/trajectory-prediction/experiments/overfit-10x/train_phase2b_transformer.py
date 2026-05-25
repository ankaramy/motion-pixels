"""
train_phase2b_transformer.py
----------------------------
Adds a small Transformer Encoder benchmark to Phase 2B.

Loads the already-trained LSTM / GRU / TCN checkpoints from phase-2b/ and
trains a new Transformer model under identical conditions. Produces:

  phase-2b/transformer/
    model.pth  scaler.pkl  loss_curve.png  prediction_visual.png

  phase-2b/phase2b_model_comparison_with_transformer.png
    4-panel figure: LSTM | GRU | TCN | Transformer

Does NOT modify any existing phase-2b outputs.

Transformer architecture
------------------------
  input_proj   : Linear(4, 64)
  pos_enc      : sinusoidal positional encoding
  encoder      : 2-layer TransformerEncoder  (d_model=64, nhead=4, ffn=128)
  head         : Linear(64, 2)  on last-sequence token
  params       : ~67 K
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

SRC_CSV     = (MP_DATA / "outputs" / "prediction" / "experiments"
               / "overfit-10x" / "trajectories_overfit.csv")
PHASE2B_OUT = MP_DATA / "outputs" / "prediction" / "experiments" / "phase-2b"
TRANS_OUT   = PHASE2B_OUT / "transformer"
COMP_PATH   = PHASE2B_OUT / "phase2b_model_comparison_with_transformer.png"

# ---------------------------------------------------------------------------
# Settings  (identical to train_phase2b.py)
# ---------------------------------------------------------------------------
WINDOW_SIZE   = 10
HIDDEN_SIZE   = 256      # LSTM / GRU
TCN_CHANNELS  = 128      # TCN
D_MODEL       = 64       # Transformer
N_HEADS       = 4
NUM_LAYERS    = 2
FFN_DIM       = 128
EPOCHS        = 30
LEARNING_RATE = 1e-3
BATCH_SIZE    = 1024
PRINT_EVERY   = 5
MAX_SEQUENCES = 100_000
N_PERSONS     = 400
N_ROLLOUT     = 30

FEATURE_COLS = ["world_x", "world_y", "delta_x", "delta_y"]
TARGET_COLS  = ["delta_x", "delta_y"]

# Known results from the Phase 2B run (used for comparison figure labels only)
PHASE2B_KNOWN = {
    "LSTM": {"loss": 0.000205, "runtime": 32.2,  "n_params": 268_802},
    "GRU":  {"loss": 0.000200, "runtime": 37.9,  "n_params": 201_730},
    "TCN":  {"loss": 0.000192, "runtime": 57.9,  "n_params": 347_522},
}


# ---------------------------------------------------------------------------
# Model definitions  (LSTM / GRU / TCN must match train_phase2b.py exactly)
# ---------------------------------------------------------------------------

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TrajectoryGRU(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, output_size=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class _Chomp1d(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, :, :-self.n].contiguous()


class _TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
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
        return self.relu(self.net(x) + (x if self.downsample is None else self.downsample(x)))


class TrajectoryTCN(nn.Module):
    def __init__(self, input_size=4, num_channels=128, output_size=2,
                 num_levels=4, kernel_size=3):
        super().__init__()
        layers = []
        for i in range(num_levels):
            in_ch = input_size if i == 0 else num_channels
            layers.append(_TemporalBlock(in_ch, num_channels, kernel_size, dilation=2 ** i))
        self.tcn = nn.Sequential(*layers)
        self.fc  = nn.Linear(num_channels, output_size)

    def forward(self, x):
        return self.fc(self.tcn(x.transpose(1, 2))[:, :, -1])


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (no learned parameters)."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TrajectoryTransformer(nn.Module):
    """Small Transformer Encoder for delta trajectory prediction.

    Pipeline: input_proj -> pos_enc -> 2-layer encoder -> last token -> fc
    """
    def __init__(self, input_size: int = 4, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128, output_size: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = _PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        # x: (batch, seq, features)
        x = self.pos_enc(self.input_proj(x))
        x = self.encoder(x)
        return self.fc(x[:, -1, :])   # last-sequence token


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
# Rollout
# ---------------------------------------------------------------------------

def rollout(model, fs, ts, track: pd.DataFrame, device):
    """30-step autoregressive delta rollout on the fixed comparison track."""
    model.eval()
    feats     = track[FEATURE_COLS].to_numpy(dtype=np.float32)
    positions = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    T = len(feats)

    feats_scaled = fs.transform(feats)
    window = feats_scaled[:WINDOW_SIZE].copy()
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

        src = min(WINDOW_SIZE + step, T - 1)
        new_raw = np.array([wx, wy, float(dx), float(dy)], dtype=np.float32)
        window = np.vstack([window[1:], fs.transform(new_raw[np.newaxis])[0]])
        prev_x, prev_y = wx, wy

    return np.array(pred_x), np.array(pred_y), positions


def endpoint_drift(pred_x, pred_y, positions):
    gt_end = positions[WINDOW_SIZE + N_ROLLOUT - 1]
    return float(np.sqrt((pred_x[-1] - gt_end[0]) ** 2 + (pred_y[-1] - gt_end[1]) ** 2))


# ---------------------------------------------------------------------------
# Train Transformer
# ---------------------------------------------------------------------------

def train_transformer(X, y, device):
    N, W, F = X.shape
    fs = MinMaxScaler()
    ts = MinMaxScaler()
    Xs = fs.fit_transform(X.reshape(-1, F)).reshape(N, W, F).astype(np.float32)
    ys = ts.fit_transform(y).astype(np.float32)

    loader = DataLoader(TensorDataset(torch.tensor(Xs), torch.tensor(ys)),
                        batch_size=BATCH_SIZE, shuffle=True)

    model = TrajectoryTransformer(
        input_size=len(FEATURE_COLS), d_model=D_MODEL, nhead=N_HEADS,
        num_layers=NUM_LAYERS, dim_feedforward=FFN_DIM,
    ).to(device)
    opt  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    crit = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Transformer]  {n_params:,} params  |  {len(loader)} batches/epoch  |  "
          f"{N:,} seqs")
    print(f"[Transformer]  d_model={D_MODEL}  nhead={N_HEADS}  "
          f"layers={NUM_LAYERS}  ffn={FFN_DIM}")
    print(f"[Transformer]  Training {EPOCHS} epochs ...")

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
    print(f"[Transformer]  Done {elapsed:.1f}s  |  final loss {history[-1]:.6f}")

    # Persist
    TRANS_OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), TRANS_OUT / "model.pth")
    bundle = {"feature_scaler": fs, "target_scaler": ts,
              "feature_cols": FEATURE_COLS, "target_cols": TARGET_COLS,
              "window_size": WINDOW_SIZE}
    with open(TRANS_OUT / "scaler.pkl", "wb") as fh:
        pickle.dump(bundle, fh)

    # Loss curve
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, EPOCHS + 1), history, lw=1.5, color="#16a085")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalised delta)")
    ax.set_title(f"Phase 2B — Transformer  |  {EPOCHS} ep  |  {N:,} seqs  |  "
                 f"d_model={D_MODEL}, heads={N_HEADS}, layers={NUM_LAYERS}")
    ax.grid(True, lw=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(TRANS_OUT / "loss_curve.png", dpi=150)
    plt.close(fig)

    return model, fs, ts, history, elapsed, n_params


# ---------------------------------------------------------------------------
# Per-model prediction visual
# ---------------------------------------------------------------------------

def save_prediction_visual(model, fs, ts, track, out_path, label, device):
    pred_x, pred_y, positions = rollout(model, fs, ts, track, device)
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
    ax1.set_title(f"{label} — 2D [delta rollout]")
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
        f"Phase 2B — {label}  |  movement-only  |  {N_ROLLOUT}-step rollout\n"
        "[NOT generalisation evidence]",
        fontsize=10, color="#7f0000",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO]  Prediction visual: {out_path}")
    return pred_x, pred_y


# ---------------------------------------------------------------------------
# Load existing Phase 2B model and run rollout
# ---------------------------------------------------------------------------

def load_existing(label, out_dir, model_factory, comp_track, device):
    """Load saved checkpoint, run rollout, return (pred_x, pred_y, drift)."""
    model_path  = out_dir / "model.pth"
    scaler_path = out_dir / "scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        sys.exit(
            f"[ERROR] {label} checkpoint not found in {out_dir}\n"
            "        Run train_phase2b.py first."
        )
    with open(scaler_path, "rb") as fh:
        bundle = pickle.load(fh)
    fs = bundle["feature_scaler"]
    ts = bundle["target_scaler"]

    model = model_factory().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    pred_x, pred_y, positions = rollout(model, fs, ts, comp_track, device)
    drift = endpoint_drift(pred_x, pred_y, positions)
    print(f"[{label}]  loaded checkpoint  |  drift {drift:.3f} m")
    return pred_x, pred_y, drift


# ---------------------------------------------------------------------------
# 4-panel comparison figure
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
    ax.plot(seed_x[-1], seed_y[-1], "o", color="#e67e22", ms=9,
            zorder=5, markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(pred_x, pred_y, color="#e74c3c", lw=2.2, ls="--", zorder=6)
    ax.plot(pred_x, pred_y, "o", color="#e74c3c", ms=4, alpha=0.85, zorder=6)
    ax.plot(pred_x[0], pred_y[0], "o", color="#e74c3c", ms=9,
            zorder=7, markeredgecolor="white", markeredgewidth=1.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("world_x  (m)", fontsize=9)
    ax.set_ylabel("world_y  (m)", fontsize=9)
    ax.grid(True, lw=0.35, alpha=0.5, color="#cccccc")
    ax.set_title(f"{title}\n{subtitle}", fontsize=10, fontweight="bold",
                 pad=8, color="#2c3e50")
    for spine in ax.spines.values():
        spine.set_edgecolor(panel_color)
        spine.set_linewidth(2.0)
    ax.text(0.025, 0.975, badge, transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1",
                      edgecolor="#bdc3c7", alpha=0.95))
    dc = "#e74c3c" if drift_m > 0.3 else "#27ae60"
    ax.text(0.975, 0.025, f"drift  {drift_m:.2f} m", transform=ax.transAxes,
            fontsize=8, va="bottom", ha="right", color=dc, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#fef9f9" if drift_m > 0.3 else "#eafaf1",
                      edgecolor=dc, alpha=0.92))


def save_comparison(seed_x, seed_y, gt_x, gt_y, results):
    all_x = np.concatenate([seed_x, gt_x] + [r["pred_x"] for r in results])
    all_y = np.concatenate([seed_y, gt_y] + [r["pred_y"] for r in results])
    pad   = max(0.15, (all_x.max() - all_x.min()) * 0.08)
    xlim  = (float(all_x.min()) - pad, float(all_x.max()) + pad)
    ylim  = (float(all_y.min()) - pad, float(all_y.max()) + pad)

    panel_colors = {
        "LSTM":        "#c0392b",
        "GRU":         "#2980b9",
        "TCN":         "#8e44ad",
        "Transformer": "#16a085",
    }

    fig, axes = plt.subplots(1, 4, figsize=(26, 7))
    fig.patch.set_facecolor("#f5f5f5")

    for ax, r in zip(axes, results):
        _draw_panel(
            ax, seed_x, seed_y, gt_x, gt_y, r["pred_x"], r["pred_y"],
            title       = r["label"],
            subtitle    = f"loss {r['loss']:.6f}  |  {r['runtime']:.0f}s",
            badge       = r["badge"],
            xlim=xlim, ylim=ylim, drift_m=r["drift"],
            panel_color = panel_colors.get(r["label"], "#333333"),
        )

    handles = [
        mpatches.Patch(facecolor="#27ae60", label="Real path (GT)"),
        mpatches.Patch(facecolor="#e67e22", label="Seed / history  (10 steps)"),
        mpatches.Patch(facecolor="#e74c3c", label="Predicted rollout  (30 steps)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10.5,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.003), edgecolor="#cccccc")
    fig.suptitle(
        "Phase 2B  —  LSTM  |  GRU  |  TCN  |  Transformer\n"
        "Movement-only inputs  |  delta prediction  |  same track  |  30-step horizon\n"
        "[NOT generalisation evidence  —  controlled comparison only]",
        fontsize=11, color="#7f0000", y=1.015,
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
        sys.exit(f"[ERROR] {SRC_CSV} not found. Run make_overfit_dataset.py first.")

    # ── Load & pre-filter (same pipeline as train_phase2b.py) ──────────────
    df = pd.read_csv(SRC_CSV)
    if "track_id" in df.columns and "person_id" not in df.columns:
        df = df.rename(columns={"track_id": "person_id"})
    if "frame_idx" in df.columns and "frame_number" not in df.columns:
        df = df.rename(columns={"frame_idx": "frame_number"})
    print(f"[INFO]  Loaded {len(df):,} rows  |  {df['person_id'].nunique()} persons")

    all_pids = df["person_id"].unique()
    chosen   = np.random.default_rng(42).choice(
        all_pids, min(N_PERSONS, len(all_pids)), replace=False)
    df = df[df["person_id"].isin(chosen)].copy()
    print(f"[INFO]  Pre-filtered to {len(chosen)} persons  ({len(df):,} rows)")

    static_cols = [c for c in FEATURE_COLS if c not in ("delta_x", "delta_y")]
    if missing := set(static_cols) - set(df.columns):
        sys.exit(f"[ERROR] Missing columns: {missing}")

    df = add_deltas(df)

    # ── Fix comparison track (same selection logic → same person as Phase 2B) ─
    min_len    = WINDOW_SIZE + N_ROLLOUT + 1
    candidates = [(pid, grp.reset_index(drop=True))
                  for pid, grp in df.groupby("person_id")
                  if len(grp) >= min_len]
    if not candidates:
        sys.exit("[ERROR] No track is long enough for rollout.")
    comp_pid, comp_track = candidates[len(candidates) // 2]
    print(f"[INFO]  Comparison track : person_id={comp_pid}  length={len(comp_track)}")

    # ── Build & subsample sequences (Transformer trains on same subset) ─────
    print("\n[INFO]  Building sequences ...")
    X_all, y_all = build_sequences(df)
    print(f"[INFO]  {len(X_all):,} sequences")
    if len(X_all) > MAX_SEQUENCES:
        idx_sub = np.random.default_rng(42).choice(
            len(X_all), MAX_SEQUENCES, replace=False)
        X_train, y_train = X_all[idx_sub], y_all[idx_sub]
    else:
        X_train, y_train = X_all, y_all
    print(f"[INFO]  Training on {len(X_train):,} sequences\n")

    # ── Load existing LSTM / GRU / TCN checkpoints ─────────────────────────
    print("=" * 52)
    print("  Loading Phase 2B checkpoints")
    print("=" * 52)

    existing_cfgs = [
        ("LSTM", PHASE2B_OUT / "lstm",
         lambda: TrajectoryLSTM(len(FEATURE_COLS), HIDDEN_SIZE)),
        ("GRU",  PHASE2B_OUT / "gru",
         lambda: TrajectoryGRU(len(FEATURE_COLS), HIDDEN_SIZE)),
        ("TCN",  PHASE2B_OUT / "tcn",
         lambda: TrajectoryTCN(len(FEATURE_COLS), TCN_CHANNELS)),
    ]

    results = []
    for label, out_dir, factory in existing_cfgs:
        pred_x, pred_y, drift = load_existing(label, out_dir, factory, comp_track, device)
        known = PHASE2B_KNOWN[label]
        results.append({
            "label":    label,
            "pred_x":   pred_x,
            "pred_y":   pred_y,
            "drift":    drift,
            "loss":     known["loss"],
            "runtime":  known["runtime"],
            "badge":    f"{known['n_params']:,} params",
            "n_params": known["n_params"],
        })

    # ── Train Transformer ───────────────────────────────────────────────────
    print(f"\n{'='*52}")
    print("  Training Transformer")
    print(f"{'='*52}")
    t_model, t_fs, t_ts, t_history, t_elapsed, t_nparams = train_transformer(
        X_train.copy(), y_train.copy(), device)

    t_pred_x, t_pred_y = save_prediction_visual(
        t_model, t_fs, t_ts, comp_track,
        TRANS_OUT / "prediction_visual.png", "Transformer", device)

    positions = comp_track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    t_drift   = endpoint_drift(t_pred_x, t_pred_y, positions)

    results.append({
        "label":    "Transformer",
        "pred_x":   t_pred_x,
        "pred_y":   t_pred_y,
        "drift":    t_drift,
        "loss":     t_history[-1],
        "runtime":  t_elapsed,
        "badge":    f"{t_nparams:,} params",
        "n_params": t_nparams,
    })

    # ── Comparison figure ───────────────────────────────────────────────────
    seed_x = positions[:WINDOW_SIZE, 0]
    seed_y = positions[:WINDOW_SIZE, 1]
    gt_x   = positions[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 0]
    gt_y   = positions[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 1]
    save_comparison(seed_x, seed_y, gt_x, gt_y, results)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  PHASE 2B + TRANSFORMER RESULTS")
    print(f"{'='*70}")
    print(f"  {'Model':<14}  {'Final Loss':>12}  {'Runtime':>10}  "
          f"{'Drift':>10}  {'Params':>12}")
    print(f"  {'-'*14}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*12}")
    for r in results:
        marker = " <--" if r["label"] == "Transformer" else ""
        print(f"  {r['label']:<14}  {r['loss']:>12.6f}  {r['runtime']:>8.1f}s  "
              f"{r['drift']:>8.3f} m  {r['n_params']:>12,}{marker}")
    print(f"  {'-'*14}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*12}")

    lstm_r  = next(r for r in results if r["label"] == "LSTM")
    trans_r = next(r for r in results if r["label"] == "Transformer")
    best    = min(results, key=lambda r: r["drift"])

    drift_delta = trans_r["drift"] - lstm_r["drift"]
    loss_delta  = trans_r["loss"]  - lstm_r["loss"]

    print(f"\n  Transformer vs LSTM")
    print(f"    Loss delta  : {loss_delta:+.6f}  "
          f"({'Transformer better' if loss_delta < 0 else 'LSTM better'})")
    print(f"    Drift delta : {drift_delta:+.3f} m  "
          f"({'Transformer better' if drift_delta < 0 else 'LSTM better'})")
    print(f"\n  Best autoregressive stability : {best['label']}  "
          f"(drift {best['drift']:.3f} m)")
    beat = trans_r["drift"] < lstm_r["drift"]
    print(f"\n  Did Transformer beat LSTM?  -->  {'YES' if beat else 'NO'}")
    print(f"\n  Comparison figure : {COMP_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
