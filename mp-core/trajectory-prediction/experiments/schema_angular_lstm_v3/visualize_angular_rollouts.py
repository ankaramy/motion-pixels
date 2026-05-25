"""
visualize_angular_rollouts.py (v2)
----------------------------------
True autoregressive rollout of the trained angular LSTM.

v2 difference vs v1:
  * Predicted du, dv from the model live in the MOTION_SCALE-scaled
    schema space. u/v positions are still in [0, 1], so when we apply
    the predicted step we divide by MOTION_SCALE:
        new_u = last_u + pdu / MOTION_SCALE
        new_v = last_v + pdv / MOTION_SCALE
  * Sanity check at the end:
        mean predicted step magnitude  (raw u/v space)
        mean GT       step magnitude   (raw u/v space)
        pred / GT movement ratio
        STATIC COLLAPSE DETECTED   if ratio < 0.25
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
    MOTION_SCALE,
)


# --------------------------------------------------------------------------- #
# Model class
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
# Spatial KDTree refresher
# --------------------------------------------------------------------------- #
class SpatialField:
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
        dists, idxs = self.tree.query([u, v], k=self.k)
        dists = np.atleast_1d(dists)
        idxs  = np.atleast_1d(idxs)
        w = 1.0 / (dists + 1e-6)
        w = w / w.sum()
        vals = (self.values[idxs] * w[:, None]).sum(axis=0)
        return dict(zip(self.SPATIAL_COLS, vals))


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def speed_rel_from_table(speed_table: np.ndarray, mag: float) -> float:
    idx = np.searchsorted(speed_table, mag, side="right")
    return float(idx) / float(max(1, len(speed_table)))


def build_window_row(u: float, v: float,
                     du_scaled: float, dv_scaled: float,
                     turn_raw: float,
                     speed_table: np.ndarray,
                     spatial: SpatialField) -> dict:
    """
    Compute the 14 FEATURE_COLS for a freshly produced step.

    du_scaled, dv_scaled are in MOTION_SCALE-units (same space as the schema
    CSV's du, dv columns and as speed_table).
    """
    mag = float(np.hypot(du_scaled, dv_scaled))
    speed_rel = speed_rel_from_table(speed_table, mag)
    heading = float(np.arctan2(dv_scaled, du_scaled))   # scale-invariant

    sp = spatial.lookup(u, v)
    return {
        "u": u, "v": v,
        "du": du_scaled, "dv": dv_scaled,
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
    track_df = track_df.sort_values("frame_idx").reset_index(drop=True)
    if len(track_df) < WINDOW_SIZE + 1:
        return pd.DataFrame()

    tid = int(track_df["track_id"].iloc[0])

    seed = track_df.iloc[:WINDOW_SIZE].copy()
    gt_future = track_df.iloc[WINDOW_SIZE: WINDOW_SIZE + HORIZON].copy()

    rows = []
    for i, r in seed.iterrows():
        rows.append({"track_id": tid, "step": i, "kind": "seed",
                     "u": r["u"], "v": r["v"],
                     "du": r["du"], "dv": r["dv"]})

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
            pred  = model(X_t).cpu().numpy()[0]

            pred_raw = (pred * tgt_scaler.scale_) + tgt_scaler.mean_
            pdu_scaled, pdv_scaled, pturn = (float(pred_raw[0]),
                                             float(pred_raw[1]),
                                             float(pred_raw[2]))

            # Position update — undo MOTION_SCALE so we stay in [0, 1].
            new_u = float(np.clip(last_u + pdu_scaled / MOTION_SCALE, 0.0, 1.0))
            new_v = float(np.clip(last_v + pdv_scaled / MOTION_SCALE, 0.0, 1.0))
            turn_raw = float(np.clip(pturn, -1.0, 1.0)) * np.pi
            new_heading = wrap(last_heading + turn_raw)

            new_row = build_window_row(new_u, new_v,
                                       pdu_scaled, pdv_scaled,
                                       turn_raw, speed_table, spatial)
            window_feats.append([new_row[c] for c in FEATURE_COLS])

            rows.append({"track_id": tid,
                         "step": WINDOW_SIZE + step,
                         "kind": "pred",
                         "u": new_u, "v": new_v,
                         "du": pdu_scaled, "dv": pdv_scaled})

            last_u, last_v, last_heading = new_u, new_v, new_heading

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

    if not seed.empty and not pred.empty:
        ax.plot([seed["u"].iloc[-1], pred["u"].iloc[0]],
                [seed["v"].iloc[-1], pred["v"].iloc[0]],
                "-", color="#c0392b", lw=1.0, alpha=0.6)

    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


# --------------------------------------------------------------------------- #
# Per-track metrics + zoomed plotting (for contact_sheet_zoomed.png)
# --------------------------------------------------------------------------- #
def per_track_metrics(roll: pd.DataFrame) -> dict:
    """
    Compute per-track ADE, FDE, movement_ratio, heading_error_deg,
    angularity_ratio in raw u/v ([0, 1]) space. Returns NaNs when a
    metric cannot be computed for the track.
    """
    pred = roll[roll["kind"] == "pred"].sort_values("step")[["u", "v"]].to_numpy()
    gt   = roll[roll["kind"] == "gt"  ].sort_values("step")[["u", "v"]].to_numpy()

    out = {"ADE": float("nan"), "FDE": float("nan"),
           "movement_ratio": float("nan"),
           "heading_error_deg": float("nan"),
           "angularity_ratio": float("nan")}
    if len(pred) < 2 or len(gt) < 2:
        return out

    m = min(len(pred), len(gt))
    p = pred[:m]; g = gt[:m]

    step_err = np.linalg.norm(p - g, axis=1)
    out["ADE"] = float(step_err.mean())
    out["FDE"] = float(step_err[-1])

    p_steps = np.linalg.norm(np.diff(p, axis=0), axis=1)
    g_steps = np.linalg.norm(np.diff(g, axis=0), axis=1)
    if g_steps.mean() > 1e-12:
        out["movement_ratio"] = float(p_steps.mean() / g_steps.mean())

    d_p = np.diff(p, axis=0); d_g = np.diff(g, axis=0)
    h_p = np.arctan2(d_p[:, 1], d_p[:, 0])
    h_g = np.arctan2(d_g[:, 1], d_g[:, 0])
    mh = min(len(h_p), len(h_g))
    if mh > 0:
        diff = wrap(h_p[:mh] - h_g[:mh])
        out["heading_error_deg"] = float(np.mean(np.abs(np.degrees(diff))))
    if mh > 1:
        t_p = wrap(np.diff(h_p[:mh]))
        t_g = wrap(np.diff(h_g[:mh]))
        thresh = np.deg2rad(SHIFT_THRESH_DEG)
        ang_p = float(np.mean(np.abs(t_p) > thresh))
        ang_g = float(np.mean(np.abs(t_g) > thresh))
        if ang_g > 1e-9:
            out["angularity_ratio"] = ang_p / ang_g
    return out


def plot_track_zoomed(ax, roll: pd.DataFrame, title: str):
    """
    Per-track plot with auto-zoom (10 % margin on the local span) and
    thicker, higher-contrast styling for the zoomed contact sheet.
    """
    seed = roll[roll["kind"] == "seed"].sort_values("step")
    pred = roll[roll["kind"] == "pred"].sort_values("step")
    gt   = roll[roll["kind"] == "gt"  ].sort_values("step")

    ax.plot(seed["u"], seed["v"], "-", color="#2c3e50",
            label="seed",      lw=3.0)
    if not gt.empty:
        ax.plot(gt["u"], gt["v"], "--", color="#16a085",
                label="GT future", lw=3.0)
    ax.plot(pred["u"], pred["v"], "o-", color="#c0392b",
            label="pred",      lw=3.0, ms=4)

    # Bridge last seed → first pred so the joint is visible after zoom.
    if not seed.empty and not pred.empty:
        ax.plot([seed["u"].iloc[-1], pred["u"].iloc[0]],
                [seed["v"].iloc[-1], pred["v"].iloc[0]],
                "-", color="#c0392b", lw=2.0, alpha=0.6)

    # Local bounds across seed + GT + pred with a 10 % margin.
    xs = np.concatenate([seed["u"].to_numpy(),
                         pred["u"].to_numpy(),
                         gt["u"].to_numpy() if not gt.empty else np.array([])])
    ys = np.concatenate([seed["v"].to_numpy(),
                         pred["v"].to_numpy(),
                         gt["v"].to_numpy() if not gt.empty else np.array([])])
    if xs.size and ys.size:
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
        x_span = max(xmax - xmin, 1e-6)
        y_span = max(ymax - ymin, 1e-6)
        # Use the larger span so the aspect stays square and tiny tracks
        # don't get squashed to a line.
        span = max(x_span, y_span)
        margin = 0.10 * span
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        half = 0.5 * span + margin
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


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
    print(f"[INFO]  MOTION_SCALE={MOTION_SCALE}")

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
          f"{ckpt['n_targets']} tgts)  best_epoch="
          f"{ckpt.get('best_epoch', '?')}")

    # speed_table built from the SCALED du/dv in the schema CSV — matches
    # the units of the model's predictions.
    speed_table = np.sort(
        np.hypot(df["du"].to_numpy(), df["dv"].to_numpy()))

    spatial = SpatialField(df, k=5)

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

        fig, ax = plt.subplots(figsize=(5, 5))
        plot_track(ax, roll, f"track {tid} — seed→pred (H={HORIZON})")
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        out = PER_TRACK_DIR / f"track_{tid}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)

    if not all_rollouts:
        raise SystemExit("[FATAL] no rollouts produced")

    full = pd.concat(all_rollouts, ignore_index=True)
    full.to_csv(ROLLOUT_CSV, index=False)
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
        f"Angular LSTM v3 autoregressive rollout · window={WINDOW_SIZE} · "
        f"horizon={HORIZON} · motion_scale={MOTION_SCALE}", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(CONTACT_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]    Wrote {CONTACT_PNG.name}")

    # ---- Zoomed contact sheet (auto local bounds + per-track metrics) -----
    CONTACT_ZOOMED_PNG = CONTACT_PNG.with_name("contact_sheet_zoomed.png")
    fig_z, axes_z = plt.subplots(nrows, ncols,
                                  figsize=(3.6 * ncols, 3.6 * nrows),
                                  squeeze=False)
    for ax in axes_z.flatten():
        ax.set_visible(False)
    for i, (tid, roll) in enumerate(per_track_data.items()):
        ax = axes_z[i // ncols][i % ncols]
        ax.set_visible(True)
        tm = per_track_metrics(roll)
        title = (
            f"track {tid}\n"
            f"ADE={tm['ADE']:.4f}  FDE={tm['FDE']:.4f}\n"
            f"mvmt={tm['movement_ratio']:.2f}  "
            f"head={tm['heading_error_deg']:.1f}°  "
            f"ang={tm['angularity_ratio']:.2f}"
        )
        plot_track_zoomed(ax, roll, title)
    handles_z = [
        plt.Line2D([], [], color="#2c3e50",                lw=3.0, label="seed"),
        plt.Line2D([], [], color="#16a085", linestyle="--", lw=3.0, label="GT future"),
        plt.Line2D([], [], color="#c0392b", marker="o", ms=4, lw=3.0, label="pred"),
    ]
    fig_z.legend(handles=handles_z, loc="lower center", ncol=3,
                 bbox_to_anchor=(0.5, -0.01), fontsize=10)
    fig_z.suptitle(
        f"Angular LSTM v3 — ZOOMED rollouts (auto local bounds, 10% margin) · "
        f"window={WINDOW_SIZE} · horizon={HORIZON}", fontsize=11)
    fig_z.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig_z.savefig(CONTACT_ZOOMED_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig_z)
    print(f"[OK]    Wrote {CONTACT_ZOOMED_PNG.name}")

    # ---- Rollout sanity check ---------------------------------------------
    # Computed in the raw u/v ([0, 1]) space, per-track then aggregated.
    SHIFT_RAD = np.deg2rad(SHIFT_THRESH_DEG)

    pred_mags_all = []
    gt_mags_all   = []
    per_track_path_ratio = []
    per_track_ang_ratio  = []
    per_track_head_err   = []
    per_track_curv_corr  = []

    for tid, roll in per_track_data.items():
        p = roll[roll["kind"] == "pred"].sort_values("step")[["u", "v"]].to_numpy()
        g = roll[roll["kind"] == "gt"  ].sort_values("step")[["u", "v"]].to_numpy()
        if len(p) < 2 or len(g) < 2:
            continue
        m = min(len(p), len(g))
        p = p[:m]; g = g[:m]

        # Step magnitudes.
        p_steps = np.linalg.norm(np.diff(p, axis=0), axis=1)
        g_steps = np.linalg.norm(np.diff(g, axis=0), axis=1)
        pred_mags_all.extend(p_steps.tolist())
        gt_mags_all.extend(g_steps.tolist())

        # Path-length ratio (per-track).
        pl_p = float(p_steps.sum()); pl_g = float(g_steps.sum())
        if pl_g > 1e-9:
            per_track_path_ratio.append(pl_p / pl_g)

        # Headings + turn-rates from successive (u, v).
        d_p = np.diff(p, axis=0); d_g = np.diff(g, axis=0)
        h_p = np.arctan2(d_p[:, 1], d_p[:, 0])
        h_g = np.arctan2(d_g[:, 1], d_g[:, 0])
        mh = min(len(h_p), len(h_g))
        if mh > 0:
            diff = wrap(h_p[:mh] - h_g[:mh])
            per_track_head_err.append(float(np.mean(np.abs(np.degrees(diff)))))
        if mh > 1:
            t_p = wrap(np.diff(h_p[:mh]))
            t_g = wrap(np.diff(h_g[:mh]))
            ang_p = float(np.mean(np.abs(t_p) > SHIFT_RAD))
            ang_g = float(np.mean(np.abs(t_g) > SHIFT_RAD))
            if ang_g > 1e-9:
                per_track_ang_ratio.append(ang_p / ang_g)
            sp = np.abs(t_p); sg = np.abs(t_g)
            if sp.std() > 1e-9 and sg.std() > 1e-9:
                per_track_curv_corr.append(float(np.corrcoef(sp, sg)[0, 1]))

    if pred_mags_all and gt_mags_all:
        mean_pred = float(np.mean(pred_mags_all))
        mean_gt   = float(np.mean(gt_mags_all))
        ratio = mean_pred / mean_gt if mean_gt > 1e-12 else float("nan")
    else:
        mean_pred = mean_gt = ratio = float("nan")

    def _mean_or_nan(xs):
        return float(np.mean(xs)) if xs else float("nan")

    path_length_ratio = _mean_or_nan(per_track_path_ratio)
    angularity_ratio  = _mean_or_nan(per_track_ang_ratio)
    heading_error_deg = _mean_or_nan(per_track_head_err)
    curvature_corr    = _mean_or_nan(per_track_curv_corr)

    print()
    print("=" * 60)
    print("[SANITY] Rollout movement + angular check "
          "(raw u/v space, per track, then averaged):")
    print(f"  mean predicted step magnitude : {mean_pred:.6f}")
    print(f"  mean GT        step magnitude : {mean_gt:.6f}")
    print(f"  pred / GT movement ratio      : {ratio:.4f}")
    print(f"  path_length_ratio             : {path_length_ratio:.4f}")
    print(f"  angularity_ratio              : {angularity_ratio:.4f}")
    print(f"  heading_error_deg             : {heading_error_deg:.4f}")
    print(f"  curvature_corr                : {curvature_corr:.4f}")
    collapse = bool(np.isfinite(ratio) and ratio < 0.25)
    print(f"  static collapse detected      : {collapse}")
    if collapse:
        print()
        print("STATIC COLLAPSE DETECTED")
    print("=" * 60)


if __name__ == "__main__":
    main()
