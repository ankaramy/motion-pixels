"""
visualize_angular_rollouts.py
-----------------------------
True autoregressive rollout of the trained angular LSTM.

Procedure (per track)
---------------------
1. Pull a seed of `WINDOW_SIZE` real frames.
2. Predict `HORIZON` future steps. Each step:
     - feed the current window through the model
     - un-scale the predicted (du, dv, turn_rate)
     - recompute u, v at the new position
     - recompute du, dv, speed_rel, heading_sin/cos, turn_rate_rel
     - re-flag is_stop / is_shift
     - refresh spatial relational features by KDTree-nearest lookup of
       the new (u, v) against the schema-space cloud
     - append the new row and slide the window
3. GT future is held aside and ONLY used for comparison plots.

Outputs
-------
- rollout_predictions.csv          (long format: track_id, step, u, v, ...)
- contact_sheet_angular_lstm.png   (grid of preset tracks)
- per_track_plots/track_<id>.png   (one combined plot per track)
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _paths import (
    SCHEMA_CSV, MODEL_PTH, SCALER_PKL,
    CONTACT_PNG, PER_TRACK_DIR, ROLLOUT_CSV,
    WINDOW_SIZE, HORIZON, HIDDEN_SIZE, NUM_LAYERS,
    STOP_THRESH_REL, SHIFT_THRESH_DEG,
    PRESET_TRACKS, FEATURE_COLS, TARGET_COLS,
)


# --------------------------------------------------------------------------- #
# Model class (mirrors training script — kept here to avoid import cycle)
# --------------------------------------------------------------------------- #
class AngularLSTM(nn.Module):
    def __init__(self, n_features, hidden=HIDDEN_SIZE,
                 layers=NUM_LAYERS, n_targets=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden,
                            num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, n_targets)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# --------------------------------------------------------------------------- #
# Spatial feature refresher
# --------------------------------------------------------------------------- #
class SpatialField:
    """
    KDTree over the schema-space (u, v) cloud. Looks up the nearest
    real sample to a predicted (u, v) and returns its relational
    spatial features. This is the rollout-time replacement for
    re-running the encoder.
    """

    SPATIAL_COLS = [
        "obstacle_clearance_pct",
        "boundary_clearance_pct",
        "entrance_affinity_pct",
        "local_space_openness",
    ]

    def __init__(self, df: pd.DataFrame, k: int = 5):
        self.k = k
        self.uv = df[["u", "v"]].to_numpy(dtype=np.float64)
        self.tree = cKDTree(self.uv)
        self.values = df[self.SPATIAL_COLS].to_numpy(dtype=np.float64)

    def lookup(self, u: float, v: float) -> dict:
        # Inverse-distance weighted average across k nearest samples.
        dists, idxs = self.tree.query([u, v], k=self.k)
        dists = np.atleast_1d(dists)
        idxs  = np.atleast_1d(idxs)
        w = 1.0 / (dists + 1e-6)
        w = w / w.sum()
        vals = (self.values[idxs] * w[:, None]).sum(axis=0)
        return dict(zip(self.SPATIAL_COLS, vals))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def speed_rel_from_table(speed_table: np.ndarray, mag: float) -> float:
    """
    Approximate the dataset speed_rel rank for a fresh predicted magnitude
    using the sorted training-set magnitudes via np.searchsorted.
    """
    idx = np.searchsorted(speed_table, mag, side="right")
    return float(idx) / float(max(1, len(speed_table)))


def build_window_row(u: float, v: float,
                     du: float, dv: float,
                     turn_raw: float,
                     speed_table: np.ndarray,
                     spatial: SpatialField) -> dict:
    """Compute the 14 FEATURE_COLS for a freshly produced step."""
    mag = float(np.hypot(du, dv))
    speed_rel = speed_rel_from_table(speed_table, mag)
    heading = float(np.arctan2(dv, du))

    sp = spatial.lookup(u, v)
    return {
        "u": u, "v": v,
        "du": du, "dv": dv,
        "speed_rel": speed_rel,
        "heading_sin": float(np.sin(heading)),
        "heading_cos": float(np.cos(heading)),
        "turn_rate_rel": float(np.clip(turn_raw / np.pi, -1.0, 1.0)),
        "is_stop":  float(speed_rel < STOP_THRESH_REL),
        "is_shift": float(abs(turn_raw) * 180.0 / np.pi > SHIFT_THRESH_DEG),
        "obstacle_clearance_pct": sp["obstacle_clearance_pct"],
        "boundary_clearance_pct": sp["boundary_clearance_pct"],
        "entrance_affinity_pct":  sp["entrance_affinity_pct"],
        "local_space_openness":   sp["local_space_openness"],
    }


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #
def rollout_track(track_df: pd.DataFrame,
                  model: AngularLSTM,
                  feat_scaler, tgt_scaler,
                  speed_table: np.ndarray,
                  spatial: SpatialField,
                  device: str) -> pd.DataFrame:
    """
    Returns a long-format dataframe with columns:
        track_id, step, kind (seed | pred | gt), u, v, du, dv, ...
    """
    track_df = track_df.sort_values("frame_idx").reset_index(drop=True)
    if len(track_df) < WINDOW_SIZE + 1:
        return pd.DataFrame()

    tid = int(track_df["track_id"].iloc[0])

    # Seed window: first WINDOW_SIZE real frames.
    seed = track_df.iloc[:WINDOW_SIZE].copy()
    gt_future = track_df.iloc[WINDOW_SIZE: WINDOW_SIZE + HORIZON].copy()

    rows = []
    for i, r in seed.iterrows():
        rows.append({"track_id": tid, "step": i, "kind": "seed",
                     "u": r["u"], "v": r["v"],
                     "du": r["du"], "dv": r["dv"]})

    # Build the rolling window of FEATURE_COLS values.
    window_feats = seed[FEATURE_COLS].to_numpy(dtype=np.float64).tolist()

    last_u = float(seed["u"].iloc[-1])
    last_v = float(seed["v"].iloc[-1])
    last_heading = float(np.arctan2(seed["dv"].iloc[-1],
                                     seed["du"].iloc[-1]))

    model.eval()
    with torch.no_grad():
        for step in range(HORIZON):
            X_arr = np.asarray(window_feats[-WINDOW_SIZE:], dtype=np.float64)
            X_scl = feat_scaler.transform(X_arr).astype(np.float32)
            X_t   = torch.from_numpy(X_scl).unsqueeze(0).to(device)
            pred  = model(X_t).cpu().numpy()[0]              # standardized

            # Un-scale to (du, dv, turn_rate_rel).
            pred_raw = (pred * tgt_scaler.scale_) + tgt_scaler.mean_
            pdu, pdv, pturn = float(pred_raw[0]), float(pred_raw[1]), float(pred_raw[2])

            new_u = float(np.clip(last_u + pdu, 0.0, 1.0))
            new_v = float(np.clip(last_v + pdv, 0.0, 1.0))
            # Convert turn_rate_rel back to radians, accumulate heading.
            turn_raw = float(np.clip(pturn, -1.0, 1.0)) * np.pi
            new_heading = wrap(last_heading + turn_raw)

            new_row = build_window_row(new_u, new_v, pdu, pdv,
                                        turn_raw, speed_table, spatial)
            window_feats.append([new_row[c] for c in FEATURE_COLS])

            rows.append({"track_id": tid,
                         "step": WINDOW_SIZE + step,
                         "kind": "pred",
                         "u": new_u, "v": new_v,
                         "du": pdu, "dv": pdv})

            last_u, last_v, last_heading = new_u, new_v, new_heading

    # Append GT future for comparison.
    for j, (_, r) in enumerate(gt_future.iterrows()):
        rows.append({"track_id": tid, "step": WINDOW_SIZE + j, "kind": "gt",
                     "u": r["u"], "v": r["v"],
                     "du": r["du"], "dv": r["dv"]})

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_track(ax, roll: pd.DataFrame, title: str):
    seed = roll[roll["kind"] == "seed"].sort_values("step")
    pred = roll[roll["kind"] == "pred"].sort_values("step")
    gt   = roll[roll["kind"] == "gt"  ].sort_values("step")

    ax.plot(seed["u"], seed["v"], "o-", color="#2c3e50",
            label="seed",      ms=3, lw=1.2)
    if not gt.empty:
        ax.plot(gt["u"], gt["v"], "o--", color="#16a085",
                label="GT future", ms=3, lw=1.0, alpha=0.85)
    ax.plot(pred["u"], pred["v"], "o-", color="#c0392b",
            label="pred",      ms=3, lw=1.4)

    # Connect last seed to first prediction so the joint is visible.
    if not seed.empty and not pred.empty:
        ax.plot([seed["u"].iloc[-1], pred["u"].iloc[0]],
                [seed["v"].iloc[-1], pred["v"].iloc[0]],
                "-", color="#c0392b", lw=1.0, alpha=0.6)

    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if not SCHEMA_CSV.exists():
        raise SystemExit(f"[FATAL] missing schema: {SCHEMA_CSV}")
    if not MODEL_PTH.exists():
        raise SystemExit(f"[FATAL] missing model: {MODEL_PTH}")
    if not SCALER_PKL.exists():
        raise SystemExit(f"[FATAL] missing scalers: {SCALER_PKL}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO]  Device: {device}")

    df = pd.read_csv(SCHEMA_CSV)
    scalers = pickle.loads(SCALER_PKL.read_bytes())
    feat_scaler = scalers["feature_scaler"]
    tgt_scaler  = scalers["target_scaler"]

    ckpt = torch.load(MODEL_PTH, map_location=device, weights_only=False)
    model = AngularLSTM(n_features=ckpt["n_features"],
                        hidden=ckpt["hidden"],
                        layers=ckpt["layers"],
                        n_targets=ckpt["n_targets"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[INFO]  Loaded model ({ckpt['n_features']} feats -> "
          f"{ckpt['n_targets']} tgts)")

    # Pre-compute sorted magnitude table for speed_rel lookup at rollout time.
    speed_table = np.sort(
        np.hypot(df["du"].to_numpy(), df["dv"].to_numpy()))

    # Spatial KDTree over the whole dataset.
    spatial = SpatialField(df, k=5)

    # Decide which tracks to roll out. Prefer PRESET_TRACKS that exist;
    # fall back to the longest-N tracks.
    avail = set(df["track_id"].unique().tolist())
    chosen = [t for t in PRESET_TRACKS if t in avail]
    if len(chosen) < len(PRESET_TRACKS):
        extra = (df.groupby("track_id").size()
                  .sort_values(ascending=False).index.tolist())
        for t in extra:
            if t not in chosen:
                chosen.append(int(t))
            if len(chosen) >= len(PRESET_TRACKS):
                break
    print(f"[INFO]  Rolling out {len(chosen)} tracks: {chosen}")

    PER_TRACK_DIR.mkdir(parents=True, exist_ok=True)
    all_rollouts = []
    per_track_data = {}

    for tid in chosen:
        gtrack = df[df["track_id"] == tid].copy()
        roll = rollout_track(gtrack, model, feat_scaler, tgt_scaler,
                              speed_table, spatial, device)
        if roll.empty:
            print(f"[WARN]  track {tid}: too short, skipped")
            continue
        all_rollouts.append(roll)
        per_track_data[tid] = roll

        # Per-track combined plot.
        fig, ax = plt.subplots(figsize=(5, 5))
        plot_track(ax, roll, f"track {tid} — seed→pred (H={HORIZON})")
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        out = PER_TRACK_DIR / f"track_{tid}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)

    if not all_rollouts:
        raise SystemExit("[FATAL] no rollouts produced")

    pd.concat(all_rollouts, ignore_index=True).to_csv(ROLLOUT_CSV,
                                                       index=False)
    print(f"[OK]    Wrote {ROLLOUT_CSV.name}")

    # Contact sheet.
    n = len(per_track_data)
    ncols = 5 if n >= 5 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(3.2 * ncols, 3.2 * nrows),
                              squeeze=False)
    for ax in axes.flatten():
        ax.set_visible(False)
    for i, (tid, roll) in enumerate(per_track_data.items()):
        ax = axes[i // ncols][i % ncols]
        ax.set_visible(True)
        plot_track(ax, roll, f"track {tid}")
    # Single shared legend.
    handles = [
        plt.Line2D([], [], color="#2c3e50", marker="o", lw=1.2, ms=3,
                   label="seed"),
        plt.Line2D([], [], color="#16a085", marker="o", lw=1.0, ms=3,
                   linestyle="--", label="GT future"),
        plt.Line2D([], [], color="#c0392b", marker="o", lw=1.4, ms=3,
                   label="pred"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.01), fontsize=9)
    fig.suptitle(
        f"Angular LSTM autoregressive rollout · window={WINDOW_SIZE} · "
        f"horizon={HORIZON}", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(CONTACT_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]    Wrote {CONTACT_PNG.name}")


if __name__ == "__main__":
    main()
