"""
MOTION PIXELS - PHASE 6 follow-up: GENUINE-TURN evaluation panel.

Read-only. Selects trajectories that are REAL turns (not stationary jitter / micro-
turns):
    GT net displacement over the 20-step horizon  > 2.0 m
    AND GT net heading change                      > 30 deg

Ranks the 20 strongest-turning, rolls out Phase-4 baseline vs Phase-6 angular-loss
(lambda=0.1), and draws GT / Phase-4 / Phase-6 side by side. Reports per trajectory:
angular error (both models), GT displacement magnitude, GT heading change,
trajectory_id, recording_id, split.

Trains nothing; modifies no protected file. Writes only new files under the
Phase-6 output dir.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase6_lib as L
fz = L.fz; P5 = L.P5

DISP_MIN_M = 2.0
HEAD_MIN_DEG = 30.0
TOP_N = 20
# tracking-artifact guard: drop trajectories with an implausible single-step jump.
# 30 fps -> 0.6 m/step ~= 18 m/s instantaneous; anything above is an ID-switch /
# teleport, not a pedestrian. This keeps "genuine turns" genuinely human.
MAX_STEP_M = 0.6

PHASE4_MODEL = L.PHASE4 / "models" / "best_model_C_barcelona.pt"
PHASE4_SCALERS = L.PHASE4 / "01_training_run" / "scalers.pkl"
PHASE6_MODEL = L.D_MODELS / "best_model_lambda_0p1.pth"
PHASE6_SCALERS = L.D_MODELS / "scalers_lambda_0p1.pkl"

OUT_FIG = L.D_FIG / "genuine_turns_panel.png"
OUT_CSV = L.D_METRICS / "genuine_turns_metrics.csv"


def gt_turn_stats(tdf, wb):
    """Return (net_disp_m, net_heading_change_deg, cum_turn_deg) over the horizon,
    or None if the trajectory is too short for a full 20-step rollout."""
    tdf = tdf.sort_values("timestep").reset_index(drop=True)
    if len(tdf) < fz.WINDOW_SIZE + fz.N_ROLLOUT:
        return None
    gt_pos = fz.gt_positions_from_seed(tdf, wb, fz.N_ROLLOUT)
    gt_h = fz.gt_headings_from_df(tdf, fz.N_ROLLOUT)
    if len(gt_h) < 2:
        return None
    net_disp = float(np.linalg.norm(gt_pos[-1] - gt_pos[0]))
    net_head = float(abs(fz.wrap_angle(gt_h[-1] - gt_h[0])))
    cum_turn = float(np.sum([abs(fz.wrap_angle(gt_h[i] - gt_h[i-1])) for i in range(1, len(gt_h))]))
    max_step = float(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1).max())
    return net_disp, np.degrees(net_head), np.degrees(cum_turn), max_step


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = P5.load_dataset(); bounds = P5.load_world_bounds()

    m4 = fz.TrajectoryLSTM(len(L.FEAT_COLS)).to(device)
    m4.load_state_dict(torch.load(PHASE4_MODEL, map_location=device)); m4.eval()
    f4, t4 = P5.load_scalers(PHASE4_SCALERS)
    m6 = fz.TrajectoryLSTM(len(L.FEAT_COLS)).to(device)
    m6.load_state_dict(torch.load(PHASE6_MODEL, map_location=device)); m6.eval()
    f6, t6 = P5.load_scalers(PHASE6_SCALERS)

    # ---- 1. scan all trajectories, keep GENUINE turns (GT-only filter) ----
    cand = []
    for rec in L.SPLIT_MAP:
        rec_df = df[df.recording_id == rec]; wb = bounds[rec]
        for tid in rec_df.trajectory_id.unique():
            st = gt_turn_stats(rec_df[rec_df.trajectory_id == tid], wb)
            if st is None:
                continue
            net_disp, net_head_deg, cum_turn_deg, max_step = st
            if net_disp > DISP_MIN_M and net_head_deg > HEAD_MIN_DEG and max_step < MAX_STEP_M:
                cand.append({"recording_id": rec, "split": L.SPLIT_MAP[rec], "trajectory_id": tid,
                             "gt_disp_m": net_disp, "gt_heading_change_deg": net_head_deg,
                             "gt_cum_turn_deg": cum_turn_deg, "gt_max_step_m": round(max_step, 3)})
    if not cand:
        print("[!] no genuine-turn trajectories matched the filter."); return
    cand = pd.DataFrame(cand).sort_values("gt_heading_change_deg", ascending=False).reset_index(drop=True)
    print(f"[filter] {len(cand)} genuine turns: disp>{DISP_MIN_M}m, heading-change>{HEAD_MIN_DEG}deg, "
          f"max-step<{MAX_STEP_M}m (artifact guard)")
    print("[filter] by recording:\n" + cand.groupby('recording_id').size().to_string())
    bins = pd.cut(cand.gt_heading_change_deg, [30, 60, 90, 120, 150, 180.01],
                  labels=["30-60", "60-90", "90-120", "120-150", "150-180"])
    print("[filter] heading-change distribution:\n" + bins.value_counts().sort_index().to_string())

    # ---- 2. roll out both models for ALL genuine turns (robust aggregate) ----
    kdt_cache = {rec: fz.build_kdt(df[df.recording_id == rec], L.SPATIAL_COLS) for rec in L.SPLIT_MAP}
    all_rolls = []
    for _, row in cand.iterrows():
        rec = row.recording_id; tid = row.trajectory_id; wb = bounds[rec]
        tdf = df[(df.recording_id == rec) & (df.trajectory_id == tid)]
        r4 = P5.rollout_one_trajectory(m4, tdf, wb, f4, t4, kdt_cache[rec], device)
        r6 = P5.rollout_one_trajectory(m6, tdf, wb, f6, t6, kdt_cache[rec], device)
        if r4 is None or r6 is None:
            continue
        all_rolls.append((row, r4, r6))
    rolls = all_rolls[:TOP_N]   # panel: the 20 strongest-turning

    # ---- 3. metrics table (all genuine turns) ----
    rows = []
    for row, r4, r6 in all_rolls:
        rows.append({"rank": len(rows) + 1, "trajectory_id": row.trajectory_id,
                     "recording_id": row.recording_id, "split": row.split,
                     "gt_disp_m": round(row.gt_disp_m, 3),
                     "gt_heading_change_deg": round(row.gt_heading_change_deg, 2),
                     "gt_cum_turn_deg": round(row.gt_cum_turn_deg, 2),
                     "phase4_angular_err_deg": round(np.degrees(r4["angular_error"]), 2),
                     "phase6_angular_err_deg": round(np.degrees(r6["angular_error"]), 2),
                     "d_angular_deg": round(np.degrees(r6["angular_error"] - r4["angular_error"]), 2),
                     "phase4_ade_m": round(r4["ade"], 3), "phase6_ade_m": round(r6["ade"], 3)})
    mt = pd.DataFrame(rows)
    mt.to_csv(OUT_CSV, index=False)

    def agg(sub, label):
        p4a, p6a = sub.phase4_angular_err_deg.mean(), sub.phase6_angular_err_deg.mean()
        p4d, p6d = sub.phase4_ade_m.mean(), sub.phase6_ade_m.mean()
        nb = int((sub.d_angular_deg < 0).sum())
        print(f"  [{label:14s} n={len(sub):2d}] angular Phase4 {p4a:5.2f} -> Phase6 {p6a:5.2f} "
              f"(delta {p6a-p4a:+5.2f}); P6 better {nb}/{len(sub)}; "
              f"ADE {p4d:.3f}->{p6d:.3f} ({p6d-p4d:+.3f})")

    print("\n=== Top-20 strongest-turning trajectories ===")
    print(mt.head(TOP_N)[["rank", "recording_id", "gt_disp_m", "gt_heading_change_deg",
              "phase4_angular_err_deg", "phase6_angular_err_deg", "d_angular_deg"]].to_string(index=False))
    print("\n=== Aggregate: does angular loss help GENUINE turns? ===")
    agg(mt, "all genuine")
    agg(mt.head(TOP_N), "top-20 turns")
    for lo, hi in [(30, 90), (90, 150), (150, 181)]:
        sub = mt[(mt.gt_heading_change_deg >= lo) & (mt.gt_heading_change_deg < hi)]
        if len(sub):
            agg(sub, f"turn {lo}-{hi}deg")

    # ---- 4. panel ----
    n = len(rolls); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15.5, 3.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (row, r4, r6) in zip(axes, rolls):
        seed, gt = r6["seed_pos"], r6["gt_pos"]
        ax.plot(seed[:, 0], seed[:, 1], "-", color="#444", lw=1.4, label="seed")
        ax.plot(gt[:, 0], gt[:, 1], "-o", color="#1f77b4", lw=1.8, ms=3, label="GT")
        ax.plot(r4["pred_pos"][:, 0], r4["pred_pos"][:, 1], "--s", color="#4c8cbf", lw=1.4, ms=2.5,
                label=f"P4 ang{np.degrees(r4['angular_error']):.0f}")
        ax.plot(r6["pred_pos"][:, 0], r6["pred_pos"][:, 1], "--^", color="#e0852a", lw=1.4, ms=2.5,
                label=f"P6 ang{np.degrees(r6['angular_error']):.0f}")
        ax.plot(gt[0, 0], gt[0, 1], "*", color="k", ms=11)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{row.recording_id[:12]} {str(row.trajectory_id).split('__')[-1]}  "
                     f"disp{row.gt_disp_m:.1f}m turn{row.gt_heading_change_deg:.0f}deg", fontsize=8)
        ax.legend(fontsize=6, loc="best")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Genuine turns (disp>2m, GT heading change>30deg): GT vs Phase-4 vs Phase-6(angular)",
                 fontweight="bold", y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(OUT_FIG, dpi=110); plt.close(fig)
    print(f"\n[saved] {OUT_FIG}\n[saved] {OUT_CSV}")


if __name__ == "__main__":
    main()
