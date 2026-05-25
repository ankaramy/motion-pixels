"""
vis_phase2b_final_plan_view.py
------------------------------
Plan-view visualisation for Phase 2B-Final multi-trajectory evaluation.

Reads the 50 trajectory IDs from multi_eval_results.csv, re-runs the
LSTM and GRU rollouts (identical logic to eval_phase2b_final_multi.py),
and produces two figures:

  phase2b_final_multi_plan_view.png
      Two side-by-side panels (LSTM | GRU) showing all 50 rollouts on the
      same world-coordinate axes.  Ground truth = light green, seed = orange,
      prediction = red.  The 5 highest-drift trajectories per model are drawn
      with a heavier, more saturated red line.

  phase2b_final_top10_worst_plan_view.png
      Same layout but restricted to the 10 highest-drift trajectories per
      model, so failure cases are visible without clutter.

No retraining.  No changes to results.
"""

import pickle
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE    = Path(__file__).resolve().parent
MP_ROOT = HERE.parent.parent.parent.parent
MP_DATA = MP_ROOT / "mp-data"

sys.path.insert(0, str(HERE))
from train_phase2b_final import (          # noqa: E402
    TrajectoryLSTM, TrajectoryGRU, SpatialInterpolator,
    add_deltas, FEATURE_COLS, WINDOW_SIZE, HIDDEN_SIZE, IDW_K,
)

SRC_CSV   = MP_DATA / "processed" / "encoded" / "trajectories_encoded.csv"
PHASE_DIR = MP_DATA / "outputs" / "prediction" / "experiments" / "phase-2b-final"
EVAL_CSV  = PHASE_DIR / "multi_eval_results.csv"
LSTM_DIR  = PHASE_DIR / "lstm"
GRU_DIR   = PHASE_DIR / "gru"
FIG_ALL   = PHASE_DIR / "phase2b_final_multi_plan_view.png"
FIG_WORST = PHASE_DIR / "phase2b_final_top10_worst_plan_view.png"

N_ROLLOUT      = 30
MIN_LEN        = WINDOW_SIZE + N_ROLLOUT
N_HIGHLIGHT    = 5    # strongest-red lines in the full figure
N_WORST        = 10   # tracks in the worst-cases figure


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_cls, model_dir: Path, n_feats: int, device):
    pth = model_dir / "model.pth"
    sca = model_dir / "scaler.pkl"
    if not pth.exists() or not sca.exists():
        sys.exit(f"[ERROR] Missing artefacts in {model_dir}")
    model = model_cls(n_feats, HIDDEN_SIZE).to(device)
    model.load_state_dict(torch.load(pth, map_location=device))
    model.eval()
    with open(sca, "rb") as fh:
        bundle = pickle.load(fh)
    return model, bundle["feature_scaler"], bundle["target_scaler"]


# ---------------------------------------------------------------------------
# Rollout  (verbatim from eval_phase2b_final_multi to stay consistent)
# ---------------------------------------------------------------------------

def rollout(model, fs, ts, track: pd.DataFrame,
            interp: SpatialInterpolator, device):
    feats     = track[FEATURE_COLS].to_numpy(dtype=np.float32)
    positions = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)

    feats_scaled = fs.transform(feats)
    window = feats_scaled[:WINDOW_SIZE].copy()
    prev_x, prev_y = positions[WINDOW_SIZE - 1]
    pred_x, pred_y = [], []

    for _ in range(N_ROLLOUT):
        x_in = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            p_s = model(x_in).cpu().numpy()[0]
        dx, dy = ts.inverse_transform(p_s[np.newaxis])[0]
        wx = float(prev_x) + float(dx)
        wy = float(prev_y) + float(dy)
        pred_x.append(wx)
        pred_y.append(wy)

        obs, bnd = interp.query(wx, wy)
        new_raw = np.array([wx, wy, float(dx), float(dy), obs, bnd],
                           dtype=np.float32)
        window = np.vstack([window[1:], fs.transform(new_raw[np.newaxis])[0]])
        prev_x, prev_y = wx, wy

    return np.array(pred_x), np.array(pred_y), positions


# ---------------------------------------------------------------------------
# Collect all rollouts
# ---------------------------------------------------------------------------

def collect_rollouts(df, eval_df, lstm_model, lstm_fs, lstm_ts,
                     gru_model, gru_fs, gru_ts, interp, device):
    """
    Return a list of dicts, one per trajectory, with:
      pid, seed_xy, gt_xy, lstm_xy, gru_xy, lstm_drift, gru_drift
    """
    results = []
    for row in eval_df.itertuples(index=False):
        pid = int(row.trajectory_id)
        track = (df[df["person_id"] == pid]
                 .reset_index(drop=True)
                 .iloc[:MIN_LEN]
                 .copy())
        if len(track) < MIN_LEN:
            print(f"[WARN]  pid={pid} has only {len(track)} frames — skipping")
            continue

        positions = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
        seed_xy = positions[:WINDOW_SIZE]
        gt_xy   = positions[WINDOW_SIZE:]

        lx, ly, _ = rollout(lstm_model, lstm_fs, lstm_ts, track, interp, device)
        gx, gy, _ = rollout(gru_model,  gru_fs,  gru_ts,  track, interp, device)

        results.append({
            "pid":        pid,
            "seed_xy":    seed_xy,
            "gt_xy":      gt_xy,
            "lstm_xy":    np.column_stack([lx, ly]),
            "gru_xy":     np.column_stack([gx, gy]),
            "lstm_drift": float(row.lstm_drift),
            "gru_drift":  float(row.gru_drift),
        })

    return results


# ---------------------------------------------------------------------------
# Shared axis limits
# ---------------------------------------------------------------------------

def compute_limits(results, pad_frac=0.06):
    all_x, all_y = [], []
    for r in results:
        for key in ("seed_xy", "gt_xy", "lstm_xy", "gru_xy"):
            all_x.append(r[key][:, 0])
            all_y.append(r[key][:, 1])
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    rng_x = all_x.max() - all_x.min()
    rng_y = all_y.max() - all_y.min()
    pad   = max(rng_x, rng_y) * pad_frac
    return (all_x.min() - pad, all_x.max() + pad,
            all_y.min() - pad, all_y.max() + pad)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

COLOUR_GT   = "#27ae60"   # light green
COLOUR_SEED = "#e67e22"   # orange
COLOUR_PRED = "#e74c3c"   # red
COLOUR_PRED_WORST = "#8b0000"  # dark red for worst trajectories


def draw_panel(ax, results, drift_key, pred_key,
               highlight_pids, xlim, ylim, title):
    """Draw all trajectories on a single axes."""
    ax.set_facecolor("#f8f8f8")

    for r in results:
        is_worst = r["pid"] in highlight_pids

        lw_pred  = 1.8 if is_worst else 0.8
        alpha_gt = 0.35 if not is_worst else 0.50
        alpha_pr = 0.70 if is_worst else 0.30
        c_pred   = COLOUR_PRED_WORST if is_worst else COLOUR_PRED
        zorder   = 4 if is_worst else 2

        # Ground truth future
        ax.plot(r["gt_xy"][:, 0], r["gt_xy"][:, 1],
                color=COLOUR_GT, lw=0.8, alpha=alpha_gt, zorder=1)
        # Seed/history
        # connect seed end → first gt point for visual continuity
        ax.plot(r["seed_xy"][:, 0], r["seed_xy"][:, 1],
                color=COLOUR_SEED, lw=0.8, alpha=alpha_gt, zorder=2)
        # Prediction
        ax.plot(r[pred_key][:, 0], r[pred_key][:, 1],
                color=c_pred, lw=lw_pred, alpha=alpha_pr, zorder=zorder)

    # Re-draw worst on top so they're not buried
    for r in results:
        if r["pid"] not in highlight_pids:
            continue
        ax.plot(r["gt_xy"][:, 0], r["gt_xy"][:, 1],
                color=COLOUR_GT, lw=1.2, alpha=0.7, zorder=5)
        ax.plot(r["seed_xy"][:, 0], r["seed_xy"][:, 1],
                color=COLOUR_SEED, lw=1.2, alpha=0.7, zorder=5)
        ax.plot(r[pred_key][:, 0], r[pred_key][:, 1],
                color=COLOUR_PRED_WORST, lw=1.8, alpha=0.9, zorder=6)

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel("world_x  (m)", fontsize=9)
    ax.set_ylabel("world_y  (m)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", color="#2c3e50")
    ax.grid(True, lw=0.3, alpha=0.4, color="#cccccc")
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)


def make_legend():
    return [
        mlines.Line2D([], [], color=COLOUR_GT,         lw=1.5, label="Ground truth (future)"),
        mlines.Line2D([], [], color=COLOUR_SEED,        lw=1.5, label="Seed / history  (10 steps)"),
        mlines.Line2D([], [], color=COLOUR_PRED,        lw=1.0, label="Predicted rollout"),
        mlines.Line2D([], [], color=COLOUR_PRED_WORST,  lw=2.0, label=f"Highest-drift trajectories"),
    ]


# ---------------------------------------------------------------------------
# Figure 1  —  all 50
# ---------------------------------------------------------------------------

def save_all_figure(results, xlim, ylim, n_highlight, out_path):
    lstm_sorted = sorted(results, key=lambda r: r["lstm_drift"], reverse=True)
    gru_sorted  = sorted(results, key=lambda r: r["gru_drift"],  reverse=True)
    lstm_worst  = {r["pid"] for r in lstm_sorted[:n_highlight]}
    gru_worst   = {r["pid"] for r in gru_sorted[:n_highlight]}

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#f0f0f0")

    draw_panel(ax_l, results, "lstm_drift", "lstm_xy",
               lstm_worst, xlim, ylim,
               f"LSTM — {len(results)} rollout trajectories\n"
               f"(mean drift {np.mean([r['lstm_drift'] for r in results]):.2f} m  "
               f"|  worst {lstm_sorted[0]['lstm_drift']:.2f} m)")

    draw_panel(ax_r, results, "gru_drift", "gru_xy",
               gru_worst, xlim, ylim,
               f"GRU — {len(results)} rollout trajectories\n"
               f"(mean drift {np.mean([r['gru_drift'] for r in results]):.2f} m  "
               f"|  worst {gru_sorted[0]['gru_drift']:.2f} m)")

    fig.legend(handles=make_legend(), loc="lower center", ncol=4,
               fontsize=9, framealpha=0.95, bbox_to_anchor=(0.5, 0.01),
               edgecolor="#cccccc")
    fig.suptitle(
        "Phase 2B-Final  —  Plan-view: LSTM vs GRU  |  50-trajectory rollout\n"
        "10-step seed  →  30-step prediction  |  spatial features recomputed "
        "via KDTree at every predicted position",
        fontsize=11, y=1.00,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK]    All-50 plan view : {out_path}")


# ---------------------------------------------------------------------------
# Figure 2  —  top-N worst
# ---------------------------------------------------------------------------

def save_worst_figure(results, xlim, ylim, n_worst, out_path):
    lstm_sorted = sorted(results, key=lambda r: r["lstm_drift"], reverse=True)
    gru_sorted  = sorted(results, key=lambda r: r["gru_drift"],  reverse=True)
    worst_lstm  = lstm_sorted[:n_worst]
    worst_gru   = gru_sorted[:n_worst]

    # Tighter limits around just the worst trajectories
    def limits_for(subset, pred_key, pad_frac=0.08):
        all_x, all_y = [], []
        for r in subset:
            for key in ("seed_xy", "gt_xy", pred_key):
                all_x.append(r[key][:, 0])
                all_y.append(r[key][:, 1])
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        rng   = max(all_x.max() - all_x.min(), all_y.max() - all_y.min())
        pad   = rng * pad_frac
        return (all_x.min() - pad, all_x.max() + pad,
                all_y.min() - pad, all_y.max() + pad)

    lim_lstm = limits_for(worst_lstm, "lstm_xy")
    lim_gru  = limits_for(worst_gru,  "gru_xy")

    # Use a shared scale: take whichever is wider
    span_lstm = max(lim_lstm[1] - lim_lstm[0], lim_lstm[3] - lim_lstm[2])
    span_gru  = max(lim_gru[1]  - lim_gru[0],  lim_gru[3]  - lim_gru[2])
    span      = max(span_lstm, span_gru)

    def centre_limits(lim, span):
        cx = (lim[0] + lim[1]) / 2
        cy = (lim[2] + lim[3]) / 2
        h  = span / 2
        return (cx - h, cx + h, cy - h, cy + h)

    ll = centre_limits(lim_lstm, span)
    lg = centre_limits(lim_gru,  span)
    xlim_l = (ll[0], ll[1]);  ylim_l = (ll[2], ll[3])
    xlim_g = (lg[0], lg[1]);  ylim_g = (lg[2], lg[3])

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#f0f0f0")

    worst_lstm_pids = {r["pid"] for r in worst_lstm}
    worst_gru_pids  = {r["pid"] for r in worst_gru}

    draw_panel(ax_l, worst_lstm, "lstm_drift", "lstm_xy",
               worst_lstm_pids, (xlim_l[0], xlim_l[1]), (ylim_l[0], ylim_l[1]),
               f"LSTM — {n_worst} highest-drift trajectories\n"
               + "  ".join(f"pid {r['pid']} ({r['lstm_drift']:.2f} m)"
                           for r in worst_lstm[:3]) + " …")

    draw_panel(ax_r, worst_gru, "gru_drift", "gru_xy",
               worst_gru_pids, (xlim_g[0], xlim_g[1]), (ylim_g[0], ylim_g[1]),
               f"GRU — {n_worst} highest-drift trajectories\n"
               + "  ".join(f"pid {r['pid']} ({r['gru_drift']:.2f} m)"
                           for r in worst_gru[:3]) + " …")

    fig.legend(handles=make_legend(), loc="lower center", ncol=4,
               fontsize=9, framealpha=0.95, bbox_to_anchor=(0.5, 0.01),
               edgecolor="#cccccc")
    fig.suptitle(
        f"Phase 2B-Final  —  Top-{n_worst} worst-drift trajectories  |  "
        "LSTM vs GRU\n"
        "10-step seed  →  30-step prediction  |  spatial features recomputed "
        "via KDTree",
        fontsize=11, y=1.00,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK]    Top-{n_worst} plan view : {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device : {device}")

    for p in (SRC_CSV, EVAL_CSV):
        if not p.exists():
            sys.exit(f"[ERROR] Not found: {p}")

    eval_df = pd.read_csv(EVAL_CSV)
    print(f"[INFO]  Loaded {len(eval_df)} evaluated trajectories from {EVAL_CSV.name}")

    df_full = pd.read_csv(SRC_CSV)
    if "track_id" in df_full.columns and "person_id" not in df_full.columns:
        df_full = df_full.rename(columns={"track_id": "person_id"})
    if "frame_idx" in df_full.columns and "frame_number" not in df_full.columns:
        df_full = df_full.rename(columns={"frame_idx": "frame_number"})

    print("[INFO]  Building KDTree spatial interpolator ...")
    interp = SpatialInterpolator(df_full, k=IDW_K)

    df = add_deltas(df_full)

    n_feats = len(FEATURE_COLS)
    lstm_model, lstm_fs, lstm_ts = load_model(TrajectoryLSTM, LSTM_DIR, n_feats, device)
    gru_model,  gru_fs,  gru_ts  = load_model(TrajectoryGRU,  GRU_DIR,  n_feats, device)
    print("[INFO]  Models loaded")

    print("[INFO]  Running rollouts ...")
    results = collect_rollouts(
        df, eval_df,
        lstm_model, lstm_fs, lstm_ts,
        gru_model,  gru_fs,  gru_ts,
        interp, device,
    )
    print(f"[INFO]  {len(results)} rollouts collected")

    xlim_min, xlim_max, ylim_min, ylim_max = compute_limits(results)
    xlim = (xlim_min, xlim_max)
    ylim = (ylim_min, ylim_max)

    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    save_all_figure(results, xlim, ylim, N_HIGHLIGHT, FIG_ALL)
    save_worst_figure(results, xlim, ylim, N_WORST, FIG_WORST)

    print(f"\n[DONE]  Figures saved to {PHASE_DIR}")


if __name__ == "__main__":
    main()
