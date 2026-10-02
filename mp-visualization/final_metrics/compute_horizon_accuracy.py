"""
MOTION PIXELS — HORIZON ACCURACY AUDIT (evaluation-only)

Reads the frozen MODEL_X horizon-sweep per-window evaluation outputs and computes
thesis-friendly, human-readable accuracy scores for each horizon.

Nothing here retrains, regenerates predictions, or touches datasets. It consumes
the already-frozen per_window_metrics.csv files produced by the horizon sweep.

Source (per horizon):
  experiments/MODEL_X_HORIZON_SWEEP/<H>/per_window_metrics.csv
    columns: ade, fde, angular_err_deg, gt_net_disp, gt_path_len, recording_id,
             trajectory_id, ...

Metrics per horizon (computed over the test-window population, the same population
the frozen ADE/FDE in metrics_summary.json was reported on):

  ADE  = mean average-displacement-error over the rollout        [m]
  FDE  = mean final-displacement-error (endpoint error)          [m]
  AngErr = mean heading error of the predicted endpoint vector   [deg]

  Success Rate = fraction of windows whose endpoint error (fde) is within a
                 horizon-specific threshold:
                     success = (fde <= threshold)
                     success_rate = #success / #windows
                 thresholds: H20=1m H60=2m H100=3m H200=5m H400=10m

  Path Accuracy (Normalized Path Accuracy) =
                 mean over windows of clamp(1 - fde / gt_net_disp, 0, 1)
                 where gt_net_disp = actual GT net displacement (start->end).
                 "How much of the future displacement did the model recover?"
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # motion-pixels/
SWEEP = ROOT / "mp-core/trajectory-prediction/experiments/MODEL_X_HORIZON_SWEEP"

# Horizon -> success threshold (m) per the audit spec.
HORIZONS = [
    ("H20", 20, 1.0),
    ("H60", 60, 2.0),
    ("H100", 100, 3.0),
    ("H200", 200, 5.0),
    ("H400", 400, 10.0),
]

rows = []
for name, steps, thresh in HORIZONS:
    df = pd.read_csv(SWEEP / name / "per_window_metrics.csv")
    n = len(df)

    ade = df["ade"].mean()
    fde = df["fde"].mean()
    ang = df["angular_err_deg"].mean()
    ang_med = df["angular_err_deg"].median()

    # Success rate: endpoint within threshold.
    success_rate = float((df["fde"] <= thresh).mean())

    # Normalized path accuracy. Guard zero/near-zero net displacement.
    disp = df["gt_net_disp"].to_numpy()
    fde_arr = df["fde"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = 1.0 - np.where(disp > 1e-9, fde_arr / disp, np.inf)
    acc = np.clip(acc, 0.0, 1.0)
    path_acc = float(np.mean(acc))

    rows.append({
        "horizon": name,
        "steps": steps,
        "approx_dist_m": round(steps * 0.05, 2),
        "threshold_m": thresh,
        "n_windows": n,
        "n_trajectories": int(df["trajectory_id"].nunique()),
        "ADE_m": round(ade, 3),
        "FDE_m": round(fde, 3),
        "AngErr_deg": round(ang, 1),
        "AngErr_median_deg": round(ang_med, 1),
        "success_rate_pct": round(100 * success_rate, 1),
        "path_accuracy_pct": round(100 * path_acc, 1),
    })

res = pd.DataFrame(rows)
res.to_csv(HERE / "horizon_accuracy_metrics.csv", index=False)
print(res.to_string(index=False))

# ----------------------------------------------------------------------------
# Figure: Success Rate & Path Accuracy vs horizon
# ----------------------------------------------------------------------------
x = list(range(len(res)))
labels = res["horizon"].tolist()

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

succ = res["success_rate_pct"].to_numpy()
pacc = res["path_accuracy_pct"].to_numpy()

c_succ = "#1b6ca8"
c_pacc = "#c1502e"

ax.plot(x, succ, "-o", color=c_succ, lw=2.4, ms=8, label="Success Rate")
ax.plot(x, pacc, "-s", color=c_pacc, lw=2.4, ms=8, label="Path Accuracy")

for xi, yi in zip(x, succ):
    ax.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9, color=c_succ)
for xi, yi in zip(x, pacc):
    ax.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points",
                xytext=(0, -16), ha="center", fontsize=9, color=c_pacc)

ax.set_xticks(x)
ax.set_xticklabels([f"{h}\n(~{d:g} m)" for h, d in
                    zip(labels, res["approx_dist_m"])], fontsize=10)
ax.set_ylim(0, 100)
ax.set_ylabel("Percentage (%)", fontsize=11)
ax.set_xlabel("Prediction horizon (frames / approx GT distance)", fontsize=11)
ax.set_title("MODEL_X — Horizon Accuracy (frozen evaluation)", fontsize=13, weight="bold")
ax.grid(True, axis="y", ls="--", alpha=0.4)
ax.legend(loc="upper right", frameon=False, fontsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig(HERE / "horizon_accuracy_summary.png", facecolor="white")
fig.savefig(HERE / "horizon_accuracy_summary.svg", facecolor="white")
print("\nWrote:", HERE / "horizon_accuracy_summary.png")
print("Wrote:", HERE / "horizon_accuracy_summary.svg")
print("Wrote:", HERE / "horizon_accuracy_metrics.csv")
