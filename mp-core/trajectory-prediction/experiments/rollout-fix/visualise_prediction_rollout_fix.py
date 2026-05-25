"""
visualise_prediction_rollout_fix.py
-----------------------------------
Standalone experimental visualiser for the rollout-fix branch.

* Lets you pick ANY trajectory ID (or auto-pick diverse types).
* Renders baseline LSTM rollout AND the alpha-adaptive corrected rollout
  side by side, with every rollout step labelled.
* Writes a 4-cell robustness grid (short / turning / straight / noisy)
  containing both rollouts per cell.

Does NOT overwrite mp-visualization/* or any production file.

CLI:
    python visualise_prediction_rollout_fix.py
        --pid 310
        --target-mag 0.04

    python visualise_prediction_rollout_fix.py --grid
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE     = Path(__file__).resolve().parent
MP_ROOT  = HERE.parent.parent.parent.parent
MP_DATA  = MP_ROOT / "mp-data"
OVERFIT  = MP_ROOT / "mp-core" / "trajectory-prediction" / "experiments" / "overfit-10x"
sys.path.insert(0, str(OVERFIT))

from train_phase2b_final import (   # noqa: E402
    TrajectoryLSTM, SpatialInterpolator, add_deltas,
    FEATURE_COLS, WINDOW_SIZE, HIDDEN_SIZE, IDW_K, N_ROLLOUT,
)

SRC_CSV  = MP_DATA / "processed" / "encoded" / "trajectories_encoded.csv"
LSTM_DIR = MP_DATA / "outputs" / "prediction" / "experiments" / "phase-2b-final" / "lstm"
OUT      = MP_DATA / "outputs" / "prediction_rollout_fix"
PLOTS    = OUT / "plots"
DEBUG    = OUT / "debug"

DEFAULT_TARGET_MAG = 0.04   # winning hyperparameter from experiment 03


def load_model(device):
    with open(LSTM_DIR / "scaler.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    model = TrajectoryLSTM(len(FEATURE_COLS), HIDDEN_SIZE).to(device)
    model.load_state_dict(torch.load(LSTM_DIR / "model.pth", map_location=device))
    model.eval()
    return model, bundle["feature_scaler"], bundle["target_scaler"]


def rollout(model, fs, ts, track, interp, device,
            alpha_adaptive_target=None):
    """Single rollout.
    If alpha_adaptive_target is set, every non-zero (du, dv) is rescaled to
    that magnitude.  When None → behaves as the baseline rollout.
    """
    feats = track[FEATURE_COLS].to_numpy(dtype=np.float32)
    pos   = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    feats_s = fs.transform(feats)
    window = feats_s[:WINDOW_SIZE].copy()
    prev_x = float(pos[WINDOW_SIZE - 1, 0])
    prev_y = float(pos[WINDOW_SIZE - 1, 1])

    pred_x, pred_y, pred_du, pred_dv = [], [], [], []
    for _ in range(N_ROLLOUT):
        x_in = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            p_s = model(x_in).cpu().numpy()[0]
        du, dv = ts.inverse_transform(p_s[np.newaxis])[0]
        du, dv = float(du), float(dv)

        if alpha_adaptive_target is not None:
            m = float(np.hypot(du, dv))
            if m > 1e-9:
                s = alpha_adaptive_target / m
                du, dv = s * du, s * dv

        wx, wy = prev_x + du, prev_y + dv
        pred_x.append(wx); pred_y.append(wy)
        pred_du.append(du); pred_dv.append(dv)

        obs, bnd = interp.query(wx, wy)
        new_raw = np.array([wx, wy, du, dv, obs, bnd], dtype=np.float32)
        window = np.vstack([window[1:], fs.transform(new_raw[np.newaxis])[0]])
        prev_x, prev_y = wx, wy

    return (np.array(pred_x), np.array(pred_y),
            np.array(pred_du), np.array(pred_dv))


# ── Trajectory categorisation ──────────────────────────────────────────────
def classify_track(track):
    """Heuristic label: short / straight / turning / noisy."""
    pos = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    if len(pos) < WINDOW_SIZE + N_ROLLOUT:
        return "short"

    # Path length and net displacement → straightness index
    deltas = np.diff(pos[:WINDOW_SIZE + N_ROLLOUT], axis=0)
    path_len = float(np.sum(np.hypot(deltas[:, 0], deltas[:, 1])))
    net      = float(np.hypot(pos[WINDOW_SIZE + N_ROLLOUT - 1, 0] - pos[0, 0],
                              pos[WINDOW_SIZE + N_ROLLOUT - 1, 1] - pos[0, 1]))
    straightness = net / path_len if path_len > 1e-9 else 0.0

    # Direction-change std for noise proxy
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    angle_diff = np.diff(np.unwrap(angles))
    noise = float(np.std(angle_diff))

    if path_len < 0.2:
        return "short"        # very little movement
    if noise > 1.0:
        return "noisy"
    if straightness > 0.85:
        return "straight"
    return "turning"


def pick_diverse_pids(df, eligible_pids, want=("short", "straight", "turning", "noisy")):
    """Return a dict {category: pid} of one trajectory per category."""
    chosen = {}
    seen_hashes = set()

    # Deduplicate against the overfit-10x repeated tracks.
    for pid in sorted(eligible_pids):
        track = (df[df["person_id"] == pid].sort_values("frame_number")
                 .head(WINDOW_SIZE + N_ROLLOUT))
        h = track[["world_x", "world_y"]].round(6).to_numpy().tobytes()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        cat = classify_track(track)
        if cat in want and cat not in chosen:
            chosen[cat] = int(pid)
        if len(chosen) == len(want):
            break
    return chosen


# ── Plotting ───────────────────────────────────────────────────────────────
def draw_one(ax, track, pid, label, model, fs, ts, interp, device,
             alpha_target=None, show_step_labels=True):
    pos = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    seed_xy = pos[:WINDOW_SIZE]
    gt_xy   = pos[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT]

    px, py, pdu, pdv = rollout(model, fs, ts, track, interp, device,
                               alpha_adaptive_target=alpha_target)

    ax.plot(gt_xy[:, 0], gt_xy[:, 1], "-", color="#27ae60", lw=2.0, alpha=0.7,
            label="GT future")
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], "o", color="#27ae60", ms=3, alpha=0.7)
    ax.plot(seed_xy[:, 0], seed_xy[:, 1], "-o", color="#e67e22", lw=2.0,
            ms=4, alpha=0.85, label="Seed (10)")
    ax.plot(px, py, "-s", color="#e74c3c", lw=1.6, ms=4, alpha=0.95,
            label="Rollout")

    if show_step_labels:
        for i, (x, y) in enumerate(zip(px, py), 1):
            ax.annotate(str(i), xy=(x, y), fontsize=6, color="#7f0000",
                        ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points")

    cum_pred = float(np.sum(np.hypot(pdu, pdv)))
    cum_gt   = float(np.sum(np.hypot(np.diff(np.r_[seed_xy[-1, 0], gt_xy[:, 0]]),
                                     np.diff(np.r_[seed_xy[-1, 1], gt_xy[:, 1]]))))
    ratio = cum_pred / cum_gt if cum_gt > 1e-9 else float("nan")
    ade   = float(np.mean(np.hypot(px - gt_xy[:, 0], py - gt_xy[:, 1])))
    fde   = float(np.hypot(px[-1] - gt_xy[-1, 0], py[-1] - gt_xy[-1, 1]))

    ax.set_xlabel("world_x (m)", fontsize=8)
    ax.set_ylabel("world_y (m)", fontsize=8)
    ax.set_title(f"{label}  pid={pid}\n"
                 f"cum-ratio={ratio:.2f}  ADE={ade:.2f}  FDE={fde:.2f}",
                 fontsize=9)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_aspect("equal", adjustable="datalim")
    return {"pid": pid, "label": label, "ratio": ratio,
            "ade": ade, "fde": fde, "cum_pred": cum_pred, "cum_gt": cum_gt}


def render_single(pid, target_mag, model, fs, ts, interp, device, df):
    track = (df[df["person_id"] == pid].reset_index(drop=True)
             .iloc[:WINDOW_SIZE + N_ROLLOUT].copy())
    if len(track) < WINDOW_SIZE + N_ROLLOUT:
        print(f"[ERROR] pid={pid} has only {len(track)} rows; need "
              f"{WINDOW_SIZE + N_ROLLOUT}")
        return

    fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"Rollout-fix visualiser — pid {pid}\n"
                 f"baseline LSTM  vs  alpha-adaptive (target |Δ|={target_mag} m)",
                 fontsize=11, fontweight="bold")
    draw_one(ax_b, track, pid, "Baseline rollout",
             model, fs, ts, interp, device, alpha_target=None)
    draw_one(ax_a, track, pid, "Alpha-adaptive rollout",
             model, fs, ts, interp, device, alpha_target=target_mag)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = PLOTS / f"trajectory_{pid}_baseline_vs_alpha_adaptive.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]    {out}")


def render_grid(target_mag, model, fs, ts, interp, device, df):
    track_lengths = df.groupby("person_id").size()
    eligible = set(track_lengths[track_lengths >= WINDOW_SIZE + N_ROLLOUT].index.tolist())
    picks = pick_diverse_pids(df, eligible,
                              want=("short", "straight", "turning", "noisy"))
    print(f"[INFO]  Auto-picked: {picks}")

    if not picks:
        print("[ERROR] No diverse trajectories found.")
        return

    # If we didn't find all 4 categories, fill remaining cells with any eligible pids
    if len(picks) < 4:
        extras = [p for p in sorted(eligible) if p not in picks.values()]
        for cat in ("short", "straight", "turning", "noisy"):
            if cat not in picks and extras:
                picks[cat] = extras.pop(0)

    order = ["short", "straight", "turning", "noisy"]

    fig, axes = plt.subplots(4, 2, figsize=(14, 22))
    fig.suptitle(
        f"Rollout-fix trajectory robustness grid\n"
        f"alpha-adaptive (target |Δ|={target_mag} m)  vs  baseline LSTM\n"
        f"Auto-picked: short / straight / turning / noisy",
        fontsize=12, fontweight="bold")

    summary = []
    for row, cat in enumerate(order):
        pid = picks.get(cat)
        if pid is None:
            continue
        track = (df[df["person_id"] == pid].reset_index(drop=True)
                 .iloc[:WINDOW_SIZE + N_ROLLOUT].copy())
        ax_b = axes[row, 0]
        ax_a = axes[row, 1]
        rb = draw_one(ax_b, track, pid, f"[{cat}] baseline",
                      model, fs, ts, interp, device, alpha_target=None)
        ra = draw_one(ax_a, track, pid, f"[{cat}] alpha-adaptive",
                      model, fs, ts, interp, device, alpha_target=target_mag)
        summary.append({"category": cat, "pid": pid,
                        "baseline_ratio": rb["ratio"], "baseline_ade": rb["ade"],
                        "alpha_ratio": ra["ratio"],    "alpha_ade":    ra["ade"]})

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = PLOTS / "trajectory_robustness_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]    {out}")

    pd.DataFrame(summary).to_csv(DEBUG / "robustness_grid_metrics.csv", index=False)
    print(f"[OK]    {DEBUG / 'robustness_grid_metrics.csv'}")
    print("\nSummary:")
    for r in summary:
        print(f"  {r['category']:<9} pid={r['pid']:<4}  "
              f"baseline cum={r['baseline_ratio']:.2f} ADE={r['baseline_ade']:.2f}  "
              f"|  alpha cum={r['alpha_ratio']:.2f} ADE={r['alpha_ade']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid",        type=int, default=None,
                    help="Visualise a specific trajectory ID.")
    ap.add_argument("--target-mag", type=float, default=DEFAULT_TARGET_MAG,
                    help="Alpha-adaptive target |delta| (m/step).")
    ap.add_argument("--grid", action="store_true",
                    help="Render the 4-category robustness grid.")
    args = ap.parse_args()

    PLOTS.mkdir(parents=True, exist_ok=True)
    DEBUG.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device : {device}")

    df_full = pd.read_csv(SRC_CSV)
    if "track_id" in df_full.columns and "person_id" not in df_full.columns:
        df_full = df_full.rename(columns={"track_id": "person_id"})
    if "frame_idx" in df_full.columns and "frame_number" not in df_full.columns:
        df_full = df_full.rename(columns={"frame_idx": "frame_number"})

    interp = SpatialInterpolator(df_full, k=IDW_K)
    df = add_deltas(df_full)
    model, fs, ts = load_model(device)

    if args.grid or args.pid is None:
        render_grid(args.target_mag, model, fs, ts, interp, device, df)

    if args.pid is not None:
        render_single(args.pid, args.target_mag, model, fs, ts, interp, device, df)


if __name__ == "__main__":
    main()
