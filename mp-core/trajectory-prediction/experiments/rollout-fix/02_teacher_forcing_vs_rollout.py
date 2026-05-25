"""
02_teacher_forcing_vs_rollout.py
--------------------------------
Compares the trained LSTM under two inference regimes:

  (A) Teacher forcing — at every step the window is filled with GT features.
                        The model still predicts (du, dv) but its own previous
                        prediction is never fed back.

  (B) Autoregressive rollout — current production path.

Question:
    Does the rollout error compound and progressively shrink motion?
    If TF predictions are large but rollout predictions are tiny, the
    collapse is a closed-loop instability, not a representation problem.

Outputs ONLY into mp-data/outputs/prediction_rollout_fix/:
  plots/teacher_forcing_vs_rollout.png
  plots/tf_vs_rollout_per_trajectory_<pid>.png
  debug/tf_vs_rollout_metrics.csv
  reports/tf_vs_rollout_analysis.md
"""

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

# Reuse training-time defs WITHOUT writing to phase-2b-final outputs
from train_phase2b_final import (   # noqa: E402
    TrajectoryLSTM, SpatialInterpolator,
    add_deltas, FEATURE_COLS, WINDOW_SIZE, HIDDEN_SIZE, IDW_K, N_ROLLOUT,
    TARGET_COLS,
)

SRC_CSV   = MP_DATA / "processed" / "encoded" / "trajectories_encoded.csv"
LSTM_DIR  = MP_DATA / "outputs" / "prediction" / "experiments" / "phase-2b-final" / "lstm"
OUT_ROOT  = MP_DATA / "outputs" / "prediction_rollout_fix"
PLOTS     = OUT_ROOT / "plots"
DEBUG     = OUT_ROOT / "debug"
REPORTS   = OUT_ROOT / "reports"

# A varied set of trajectories: includes 310 (the failing case), a few longer
# ones, and several drawn from a fixed seed to cover diverse shapes.
PROBE_PIDS = [310, 50, 100, 200, 300, 400, 500]
MIN_LEN    = WINDOW_SIZE + N_ROLLOUT


def load_model(device):
    with open(LSTM_DIR / "scaler.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    fs = bundle["feature_scaler"]
    ts = bundle["target_scaler"]
    model = TrajectoryLSTM(len(FEATURE_COLS), HIDDEN_SIZE).to(device)
    model.load_state_dict(torch.load(LSTM_DIR / "model.pth", map_location=device))
    model.eval()
    return model, fs, ts


def teacher_forcing_predict(model, fs, ts, track: pd.DataFrame, device):
    """
    For each rollout step, feed the GT window ending at GT frame
    (WINDOW_SIZE + step − 1) and predict the next (du, dv).

    No prediction is ever fed back.
    """
    feats     = track[FEATURE_COLS].to_numpy(dtype=np.float32)
    positions = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    feats_s   = fs.transform(feats)

    pred_du, pred_dv = [], []
    pred_x_abs, pred_y_abs = [], []

    for step in range(N_ROLLOUT):
        i0 = step
        i1 = step + WINDOW_SIZE
        window = feats_s[i0:i1]
        x_in = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            p_s = model(x_in).cpu().numpy()[0]
        du, dv = ts.inverse_transform(p_s[np.newaxis])[0]
        pred_du.append(float(du))
        pred_dv.append(float(dv))

        # Absolute position: anchor on the GT prior step (TF style)
        anchor_x, anchor_y = positions[i1 - 1]
        pred_x_abs.append(float(anchor_x) + float(du))
        pred_y_abs.append(float(anchor_y) + float(dv))

    return (np.array(pred_du), np.array(pred_dv),
            np.array(pred_x_abs), np.array(pred_y_abs))


def autoregressive_rollout(model, fs, ts, track: pd.DataFrame,
                           interp: SpatialInterpolator, device):
    feats     = track[FEATURE_COLS].to_numpy(dtype=np.float32)
    positions = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    feats_s   = fs.transform(feats)

    window = feats_s[:WINDOW_SIZE].copy()
    prev_x, prev_y = float(positions[WINDOW_SIZE - 1, 0]), float(positions[WINDOW_SIZE - 1, 1])

    pred_du, pred_dv = [], []
    pred_x_abs, pred_y_abs = [], []
    for _ in range(N_ROLLOUT):
        x_in = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            p_s = model(x_in).cpu().numpy()[0]
        du, dv = ts.inverse_transform(p_s[np.newaxis])[0]
        du, dv = float(du), float(dv)

        wx, wy = prev_x + du, prev_y + dv
        pred_du.append(du); pred_dv.append(dv)
        pred_x_abs.append(wx); pred_y_abs.append(wy)

        obs, bnd = interp.query(wx, wy)
        new_raw = np.array([wx, wy, du, dv, obs, bnd], dtype=np.float32)
        window = np.vstack([window[1:], fs.transform(new_raw[np.newaxis])[0]])
        prev_x, prev_y = wx, wy

    return (np.array(pred_du), np.array(pred_dv),
            np.array(pred_x_abs), np.array(pred_y_abs))


def gt_deltas(track):
    pos = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    seed_end = pos[WINDOW_SIZE - 1]
    fut_x = pos[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 0]
    fut_y = pos[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 1]
    du = np.r_[fut_x[0] - seed_end[0], np.diff(fut_x)]
    dv = np.r_[fut_y[0] - seed_end[1], np.diff(fut_y)]
    return du, dv, fut_x, fut_y


def metrics_for_run(pred_x, pred_y, gt_x, gt_y, pred_du, pred_dv, gt_du, gt_dv):
    ade  = float(np.mean(np.hypot(pred_x - gt_x, pred_y - gt_y)))
    fde  = float(np.hypot(pred_x[-1] - gt_x[-1], pred_y[-1] - gt_y[-1]))
    cum_pred = float(np.sum(np.hypot(pred_du, pred_dv)))
    cum_gt   = float(np.sum(np.hypot(gt_du,  gt_dv)))
    ratio    = cum_pred / cum_gt if cum_gt > 1e-9 else float("nan")
    mag_mean_pred = float(np.mean(np.hypot(pred_du, pred_dv)))
    mag_mean_gt   = float(np.mean(np.hypot(gt_du,  gt_dv)))
    return {
        "ade": ade, "fde": fde,
        "cum_pred": cum_pred, "cum_gt": cum_gt,
        "cum_ratio": ratio,
        "mag_mean_pred": mag_mean_pred, "mag_mean_gt": mag_mean_gt,
        "mag_ratio": mag_mean_pred / mag_mean_gt if mag_mean_gt > 1e-9 else float("nan"),
    }


def per_traj_plot(pid, gt_du, gt_dv, tf_du, tf_dv, ro_du, ro_dv,
                  gt_x, gt_y, tf_x, tf_y, ro_x, ro_y, out_path):
    steps = np.arange(1, N_ROLLOUT + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Teacher Forcing vs Autoregressive Rollout — pid {pid}",
                 fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(steps, np.hypot(gt_du, gt_dv), "b-o", ms=3, label="GT |delta|")
    ax.plot(steps, np.hypot(tf_du, tf_dv), "g-s", ms=3, label="TF |delta|")
    ax.plot(steps, np.hypot(ro_du, ro_dv), "r-^", ms=3, label="AR |delta|")
    ax.set_xlabel("Rollout step"); ax.set_ylabel("|delta| (m)")
    ax.set_title("Per-step delta magnitude")
    ax.legend(fontsize=8); ax.grid(True, lw=0.3, alpha=0.5)

    ax = axes[0, 1]
    ax.plot(steps, np.cumsum(np.hypot(gt_du, gt_dv)), "b-",  label="GT")
    ax.plot(steps, np.cumsum(np.hypot(tf_du, tf_dv)), "g--", label="TF")
    ax.plot(steps, np.cumsum(np.hypot(ro_du, ro_dv)), "r--", label="AR")
    ax.set_xlabel("Rollout step"); ax.set_ylabel("Cumulative path length (m)")
    ax.set_title("Cumulative displacement")
    ax.legend(fontsize=8); ax.grid(True, lw=0.3, alpha=0.5)

    ax = axes[1, 0]
    ax.plot(gt_x, gt_y, "b-o", ms=3, lw=1.5, alpha=0.7, label="GT future")
    ax.plot(tf_x, tf_y, "g-s", ms=3, lw=1.3, alpha=0.7, label="TF pred")
    ax.plot(ro_x, ro_y, "r-^", ms=3, lw=1.3, alpha=0.7, label="AR pred")
    ax.set_xlabel("world_x (m)"); ax.set_ylabel("world_y (m)")
    ax.set_title("Plan view"); ax.legend(fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.5); ax.set_aspect("equal", adjustable="datalim")

    ax = axes[1, 1]
    tf_drift = np.hypot(tf_x - gt_x, tf_y - gt_y)
    ro_drift = np.hypot(ro_x - gt_x, ro_y - gt_y)
    ax.plot(steps, tf_drift, "g-s", ms=3, label="TF positional error")
    ax.plot(steps, ro_drift, "r-^", ms=3, label="AR positional error")
    ax.set_xlabel("Rollout step"); ax.set_ylabel("L2 error (m)")
    ax.set_title("Drift accumulation"); ax.legend(fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summary_plot(records, out_path):
    """Bar chart of cum_ratio and mag_ratio for TF vs AR across all PIDs."""
    pids = [r["pid"] for r in records]
    x = np.arange(len(pids))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Teacher Forcing vs Autoregressive Rollout — summary across "
                 "probe trajectories", fontsize=12, fontweight="bold")

    ax = axes[0]
    tf_ratio = [r["tf"]["cum_ratio"] for r in records]
    ar_ratio = [r["ar"]["cum_ratio"] for r in records]
    ax.bar(x - w/2, tf_ratio, w, color="#27ae60", alpha=0.85, label="Teacher Forcing")
    ax.bar(x + w/2, ar_ratio, w, color="#e74c3c", alpha=0.85, label="Autoregressive")
    ax.axhline(1.0, color="k", lw=0.6, ls="--", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(pids, rotation=0)
    ax.set_xlabel("trajectory id"); ax.set_ylabel("predicted / GT")
    ax.set_title("Cumulative-displacement ratio  (1.0 = perfect)")
    ax.legend(fontsize=8); ax.grid(True, lw=0.3, alpha=0.5, axis="y")

    ax = axes[1]
    tf_ade = [r["tf"]["ade"] for r in records]
    ar_ade = [r["ar"]["ade"] for r in records]
    ax.bar(x - w/2, tf_ade, w, color="#27ae60", alpha=0.85, label="TF ADE")
    ax.bar(x + w/2, ar_ade, w, color="#e74c3c", alpha=0.85, label="AR ADE")
    ax.set_xticks(x); ax.set_xticklabels(pids, rotation=0)
    ax.set_xlabel("trajectory id"); ax.set_ylabel("ADE (m)")
    ax.set_title("Average Displacement Error (lower is better)")
    ax.legend(fontsize=8); ax.grid(True, lw=0.3, alpha=0.5, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    DEBUG.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device : {device}")

    print("[INFO]  Loading data ...")
    df_full = pd.read_csv(SRC_CSV)
    if "track_id" in df_full.columns and "person_id" not in df_full.columns:
        df_full = df_full.rename(columns={"track_id": "person_id"})
    if "frame_idx" in df_full.columns and "frame_number" not in df_full.columns:
        df_full = df_full.rename(columns={"frame_idx": "frame_number"})
    print(f"[INFO]  {len(df_full):,} rows | {df_full['person_id'].nunique()} persons")

    print("[INFO]  Building KDTree ...")
    interp = SpatialInterpolator(df_full, k=IDW_K)

    df = add_deltas(df_full)
    print("[INFO]  Loading LSTM ...")
    model, fs, ts = load_model(device)

    track_lengths = df.groupby("person_id").size()
    eligible_pids = set(track_lengths[track_lengths >= MIN_LEN].index.tolist())

    probe = [p for p in PROBE_PIDS if p in eligible_pids]
    if not probe:
        sys.exit("[ERROR] No probe PIDs are long enough.")
    print(f"[INFO]  Probing PIDs: {probe}")

    records = []
    for pid in probe:
        track = (df[df["person_id"] == pid]
                 .reset_index(drop=True)
                 .iloc[:MIN_LEN]
                 .copy())

        gt_du, gt_dv, gt_x, gt_y = gt_deltas(track)
        tf_du, tf_dv, tf_x, tf_y = teacher_forcing_predict(model, fs, ts, track, device)
        ro_du, ro_dv, ro_x, ro_y = autoregressive_rollout(model, fs, ts, track, interp, device)

        tf_m = metrics_for_run(tf_x, tf_y, gt_x, gt_y, tf_du, tf_dv, gt_du, gt_dv)
        ar_m = metrics_for_run(ro_x, ro_y, gt_x, gt_y, ro_du, ro_dv, gt_du, gt_dv)

        records.append({"pid": pid, "tf": tf_m, "ar": ar_m})

        per_traj_plot(
            pid, gt_du, gt_dv, tf_du, tf_dv, ro_du, ro_dv,
            gt_x, gt_y, tf_x, tf_y, ro_x, ro_y,
            PLOTS / f"tf_vs_rollout_per_trajectory_{pid}.png",
        )
        print(f"  pid={pid:>4}  TF cum_ratio={tf_m['cum_ratio']:.3f}  "
              f"AR cum_ratio={ar_m['cum_ratio']:.3f}  "
              f"TF ADE={tf_m['ade']:.3f}  AR ADE={ar_m['ade']:.3f}")

    # ── Summary plot ────────────────────────────────────────────────────────
    summary_plot(records, PLOTS / "teacher_forcing_vs_rollout.png")
    print(f"[OK]    Summary plot: teacher_forcing_vs_rollout.png")

    # ── CSV ─────────────────────────────────────────────────────────────────
    rows = []
    for r in records:
        for regime, m in (("teacher_forcing", r["tf"]), ("autoregressive", r["ar"])):
            rows.append({"pid": r["pid"], "regime": regime, **m})
    pd.DataFrame(rows).to_csv(DEBUG / "tf_vs_rollout_metrics.csv", index=False)
    print(f"[OK]    Metrics CSV : tf_vs_rollout_metrics.csv")

    # ── Markdown report ─────────────────────────────────────────────────────
    md = [
        "# Teacher Forcing vs Autoregressive Rollout",
        "",
        "**Question:** does autoregressive feedback compound into motion "
        "collapse, or is the trained LSTM already producing tiny deltas even "
        "when fed pristine GT windows?",
        "",
        "## Setup",
        "",
        f"- Model       : LSTM @ `{LSTM_DIR.relative_to(MP_ROOT).as_posix()}`",
        f"- Probe PIDs  : {probe}",
        f"- Window size : {WINDOW_SIZE}",
        f"- Rollout steps: {N_ROLLOUT}",
        "",
        "## Per-trajectory metrics",
        "",
        "| pid | regime | ADE (m) | FDE (m) | mean &#124;Δ&#124; pred | mean &#124;Δ&#124; GT | cum ratio (pred/GT) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        for regime_name, m in (("TF", r["tf"]), ("AR", r["ar"])):
            md.append(
                f"| {r['pid']} | {regime_name} "
                f"| {m['ade']:.3f} | {m['fde']:.3f} "
                f"| {m['mag_mean_pred']:.4f} | {m['mag_mean_gt']:.4f} "
                f"| {m['cum_ratio']:.3f} |"
            )

    avg_tf_ratio = float(np.mean([r["tf"]["cum_ratio"] for r in records]))
    avg_ar_ratio = float(np.mean([r["ar"]["cum_ratio"] for r in records]))
    avg_tf_ade   = float(np.mean([r["tf"]["ade"] for r in records]))
    avg_ar_ade   = float(np.mean([r["ar"]["ade"] for r in records]))

    md += [
        "",
        "## Aggregate",
        "",
        f"- Mean cumulative-displacement ratio — TF : **{avg_tf_ratio:.3f}**",
        f"- Mean cumulative-displacement ratio — AR : **{avg_ar_ratio:.3f}**",
        f"- Mean ADE — TF : **{avg_tf_ade:.3f} m**",
        f"- Mean ADE — AR : **{avg_ar_ade:.3f} m**",
        "",
        "## Interpretation",
        "",
    ]

    if avg_tf_ratio > 0.7 and avg_ar_ratio < 0.5:
        md.append(
            "**Closed-loop instability confirmed.**  Teacher-forced predictions "
            "produce realistic motion magnitudes, but feeding the prediction back "
            "into the LSTM compounds errors that progressively shrink subsequent "
            "deltas.  Inference-time corrections (alpha amplification, velocity "
            "persistence, motion floor) are the correct fix — they directly "
            "address the closed-loop dynamics without retraining.")
    elif avg_tf_ratio < 0.6 and avg_ar_ratio < 0.6:
        md.append(
            "**Both regimes produce shrunken deltas.**  The model itself has "
            "learned to predict small motions (MSE regression to the mean).  "
            "Inference-time amplification is still effective, but a velocity "
            "persistence / motion-floor hybrid is more reliable since it does "
            "not rely on the model already pointing in the right direction.")
    else:
        md.append(
            "Teacher forcing and autoregressive rollout produce broadly similar "
            "magnitudes.  The collapse is likely model-quality rather than "
            "feedback-driven; lean on alpha amplification + motion floor.")

    md += [
        "",
        f"![summary]({(PLOTS / 'teacher_forcing_vs_rollout.png').as_posix()})",
        "",
        "## Per-trajectory plots",
        "",
    ]
    for r in records:
        path = PLOTS / f"tf_vs_rollout_per_trajectory_{r['pid']}.png"
        md.append(f"- pid {r['pid']} — ![pid {r['pid']}]({path.as_posix()})")

    md += [
        "",
        "---",
        "*Generated by `experiments/rollout-fix/02_teacher_forcing_vs_rollout.py`*",
    ]
    out = REPORTS / "tf_vs_rollout_analysis.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK]    Report : {out}")


if __name__ == "__main__":
    main()
