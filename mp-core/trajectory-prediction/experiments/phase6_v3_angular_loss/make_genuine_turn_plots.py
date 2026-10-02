"""
MOTION PIXELS - Phase 6 genuine-turn thesis plots (visualization only).

Read-only. Re-uses the already-computed genuine-turn metrics
(03_metrics/genuine_turns_metrics.csv) and the existing Phase-4 / Phase-6
checkpoints (loaded for rollout positions only). Trains nothing, modifies no
model/dataset.

Outputs -> 13_phase6_v3_angular_loss/04_figures/genuine_turns/
  1 genuine_turn_rollout_panel.png
  2 genuine_turn_error_bars.png
  3 genuine_turn_delta_scatter.png
  4 genuine_turn_dataset_map.png
  5 aggregate_vs_genuine_turns.png
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

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 10})

OUTDIR = L.D_FIG / "genuine_turns"
METRICS_CSV = L.D_METRICS / "genuine_turns_metrics.csv"
PHASE4_MODEL = L.PHASE4 / "models" / "best_model_C_barcelona.pt"
PHASE4_SCALERS = L.PHASE4 / "01_training_run" / "scalers.pkl"
PHASE6_MODEL = L.D_MODELS / "best_model_lambda_0p1.pth"
PHASE6_SCALERS = L.D_MODELS / "scalers_lambda_0p1.pkl"

C_GT = "#1f77b4"; C_P4 = "#7f7f7f"; C_P6 = "#e0852a"; C_SEED = "#333333"
C_GOOD = "#2ca02c"; C_BAD = "#d62728"


def turn_band(deg):
    if deg < 90: return "mild (30-90)"
    if deg < 150: return "sharp (90-150)"
    return "U-turn (150-180)"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    mt = pd.read_csv(METRICS_CSV)
    mt["band"] = mt.gt_heading_change_deg.apply(turn_band)
    n_turns = len(mt)
    print(f"[load] {n_turns} genuine turns from {METRICS_CSV.name}")

    # global Phase-6 angular (val/test) for plot 5
    g0 = pd.read_csv(L.D_METRICS / "split_summary_lambda_0.csv").set_index("split")
    g1 = pd.read_csv(L.D_METRICS / "split_summary_lambda_0p1.csv").set_index("split")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = P5.load_dataset(); bounds = P5.load_world_bounds()
    m4 = fz.TrajectoryLSTM(len(L.FEAT_COLS)).to(device); m4.load_state_dict(torch.load(PHASE4_MODEL, map_location=device)); m4.eval()
    f4, t4 = P5.load_scalers(PHASE4_SCALERS)
    m6 = fz.TrajectoryLSTM(len(L.FEAT_COLS)).to(device); m6.load_state_dict(torch.load(PHASE6_MODEL, map_location=device)); m6.eval()
    f6, t6 = P5.load_scalers(PHASE6_SCALERS)
    kdt = {rec: fz.build_kdt(df[df.recording_id == rec], L.SPATIAL_COLS) for rec in L.SPLIT_MAP}

    def roll(model, fsc, tsc, rec, tid):
        tdf = df[(df.recording_id == rec) & (df.trajectory_id == tid)]
        return P5.rollout_one_trajectory(model, tdf, bounds[rec], fsc, tsc, kdt[rec], device)

    # =================================================================== PLOT 1
    # 9 clean illustrative turns: 3 per band, largest displacement (clearest paths)
    picks = []
    for band in ["mild (30-90)", "sharp (90-150)", "U-turn (150-180)"]:
        sub = mt[mt.band == band].sort_values("gt_disp_m", ascending=False)
        picks += list(sub.head(3).itertuples(index=False))
    fig, axes = plt.subplots(3, 3, figsize=(15, 14)); axes = axes.ravel()
    for ax, row in zip(axes, picks):
        r4 = roll(m4, f4, t4, row.recording_id, row.trajectory_id)
        r6 = roll(m6, f6, t6, row.recording_id, row.trajectory_id)
        if r4 is None or r6 is None:
            ax.axis("off"); continue
        seed, gt = r6["seed_pos"], r6["gt_pos"]
        ax.plot(seed[:, 0], seed[:, 1], "-", color=C_SEED, lw=1.6, label="seed (history)")
        ax.plot(gt[:, 0], gt[:, 1], "-o", color=C_GT, lw=2.0, ms=3.5, label="ground truth")
        ax.plot(r4["pred_pos"][:, 0], r4["pred_pos"][:, 1], "--s", color=C_P4, lw=1.6, ms=3,
                label=f"Phase 4 (λ0)  ADE {r4['ade']:.2f}m  ang {np.degrees(r4['angular_error']):.0f}°")
        ax.plot(r6["pred_pos"][:, 0], r6["pred_pos"][:, 1], "--^", color=C_P6, lw=1.6, ms=3,
                label=f"Phase 6 (λ0.1) ADE {r6['ade']:.2f}m  ang {np.degrees(r6['angular_error']):.0f}°")
        ax.plot(gt[0, 0], gt[0, 1], "*", color="k", ms=14, zorder=5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{row.recording_id}  |  GT turn {row.gt_heading_change_deg:.0f}°, "
                     f"disp {row.gt_disp_m:.1f}m\n[{row.band}]", fontsize=9)
        ax.legend(fontsize=7, loc="best"); ax.set_xlabel("world x (m)"); ax.set_ylabel("world y (m)")
    for ax in axes[len(picks):]:
        ax.axis("off")
    fig.suptitle("Genuine pedestrian turns: ground truth vs Phase-4 baseline vs Phase-6 angular-loss",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(OUTDIR / "genuine_turn_rollout_panel.png"); plt.close(fig)
    print("[1] genuine_turn_rollout_panel.png")

    # =================================================================== PLOT 2
    groups = [("all", mt), ("mild (30-90)", mt[mt.band == "mild (30-90)"]),
              ("sharp (90-150)", mt[mt.band == "sharp (90-150)"]),
              ("U-turn (150-180)", mt[mt.band == "U-turn (150-180)"])]
    labels = [f"{g[0]}\n(n={len(g[1])})" for g in groups]
    p4 = [g[1].phase4_angular_err_deg.mean() for g in groups]
    p6 = [g[1].phase6_angular_err_deg.mean() for g in groups]
    x = np.arange(len(groups)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.2))
    b1 = ax.bar(x - w/2, p4, w, label="Phase 4 (λ0)", color=C_P4)
    b2 = ax.bar(x + w/2, p6, w, label="Phase 6 (λ0.1)", color=C_P6)
    ax.axhline(90, color="grey", ls=":", lw=1, label="chance (90°)")
    ax.bar_label(b1, fmt="%.0f", fontsize=8); ax.bar_label(b2, fmt="%.0f", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("mean angular error (deg)")
    ax.set_title("Angular error on genuine turns by turn strength"); ax.legend()
    fig.tight_layout(); fig.savefig(OUTDIR / "genuine_turn_error_bars.png"); plt.close(fig)
    print("[2] genuine_turn_error_bars.png")

    # =================================================================== PLOT 3
    fig, ax = plt.subplots(figsize=(9, 5.6))
    colors = [C_BAD if d > 0 else C_GOOD for d in mt.d_angular_deg]
    ax.scatter(mt.gt_heading_change_deg, mt.d_angular_deg, c=colors, s=42, alpha=0.8, edgecolor="k", lw=0.4)
    ax.axhline(0, color="k", lw=1.2)
    # trend line
    z = np.polyfit(mt.gt_heading_change_deg, mt.d_angular_deg, 1)
    xs = np.linspace(mt.gt_heading_change_deg.min(), mt.gt_heading_change_deg.max(), 50)
    ax.plot(xs, np.polyval(z, xs), "--", color="#444", lw=1.4,
            label=f"trend (slope {z[0]:+.2f}° per deg)")
    ax.text(0.02, 0.96, "above 0 = angular loss WORSE", transform=ax.transAxes, color=C_BAD, fontsize=9, va="top")
    ax.text(0.02, 0.04, "below 0 = angular loss BETTER", transform=ax.transAxes, color=C_GOOD, fontsize=9, va="bottom")
    ax.set_xlabel("GT heading change (deg)"); ax.set_ylabel("Δ angular error  (Phase6 − Phase4, deg)")
    ax.set_title("Does angular loss help stronger turns?  (per genuine turn)"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUTDIR / "genuine_turn_delta_scatter.png"); plt.close(fig)
    print("[3] genuine_turn_delta_scatter.png")

    # =================================================================== PLOT 4
    cnt = mt.recording_id.value_counts()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.barh(cnt.index[::-1], cnt.values[::-1], color="#4c8cbf")
    ax.bar_label(bars, fmt="%d", fontsize=9, padding=3)
    ax.set_xlabel("number of genuine turns (disp>2m, turn>30°, no artifacts)")
    ax.set_title(f"Genuine turns by recording (total {n_turns}) — placa_espanya (roundabout) dominates")
    fig.tight_layout(); fig.savefig(OUTDIR / "genuine_turn_dataset_map.png"); plt.close(fig)
    print("[4] genuine_turn_dataset_map.png")

    # =================================================================== PLOT 5
    bars5 = [
        ("Global val", g1.loc["val", "angular_error_mean_deg"] - g0.loc["val", "angular_error_mean_deg"]),
        ("Global test", g1.loc["test", "angular_error_mean_deg"] - g0.loc["test", "angular_error_mean_deg"]),
        ("Genuine: all", mt.phase6_angular_err_deg.mean() - mt.phase4_angular_err_deg.mean()),
        ("Genuine: sharp", mt[mt.band == "sharp (90-150)"].d_angular_deg.mean()),
        ("Genuine: U-turn", mt[mt.band == "U-turn (150-180)"].d_angular_deg.mean()),
    ]
    names = [b[0] for b in bars5]; vals = [b[1] for b in bars5]
    cols = [C_GOOD if v < 0 else C_BAD for v in vals]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    bb = ax.bar(names, vals, color=cols, edgecolor="k", lw=0.5)
    ax.axhline(0, color="k", lw=1.2)
    ax.bar_label(bb, fmt="%+.1f", fontsize=9)
    ax.set_ylabel("Δ angular error  (Phase6 − Phase4, deg)")
    ax.set_title("KEY FIGURE — angular loss improves GLOBAL heading realism\nbut WORSENS genuine turning decisions")
    ax.text(0.5, 0.95, "negative = improvement", transform=ax.transAxes, color=C_GOOD, ha="center", va="top", fontsize=9)
    fig.tight_layout(); fig.savefig(OUTDIR / "aggregate_vs_genuine_turns.png"); plt.close(fig)
    print("[5] aggregate_vs_genuine_turns.png")

    write_report(mt, n_turns, cnt, g0, g1, groups)


def write_report(mt, n_turns, cnt, g0, g1, groups):
    gv = g1.loc["val", "angular_error_mean_deg"] - g0.loc["val", "angular_error_mean_deg"]
    gt = g1.loc["test", "angular_error_mean_deg"] - g0.loc["test", "angular_error_mean_deg"]
    d_all = mt.phase6_angular_err_deg.mean() - mt.phase4_angular_err_deg.mean()
    R = ["# Genuine Turns — Visual Report (Phase 6)", "",
         "Thesis-quality visualizations of whether the Phase-6 angular loss improves",
         "GENUINE pedestrian turns. Visualization only — no training; existing Phase-4",
         "and Phase-6 (λ=0.1) checkpoints loaded read-only for rollout positions.", "",
         "## Selection criteria (genuine turns)", "",
         "Over the 20-step (~0.66 s) rollout horizon:",
         "- GT net displacement > 2.0 m (excludes stationary jitter)",
         "- GT net heading change > 30° (excludes micro-turns)",
         "- GT max single-step < 0.6 m (artifact guard: excludes ID-switch / teleport)", "",
         f"**Genuine turns: {n_turns}** (the artifact guard alone removed ~140 implausible-jump tracks).",
         "By recording: " + ", ".join(f"{r}={c}" for r, c in cnt.items()) + ".", "",
         "## Plots (`04_figures/genuine_turns/`)", "",
         "1. `genuine_turn_rollout_panel.png` — 9 clean turns (3 mild / 3 sharp / 3 U-turn): "
         "seed, GT, Phase-4, Phase-6, with ADE + angular error in each title.",
         "2. `genuine_turn_error_bars.png` — Phase-4 vs Phase-6 mean angular error for all / mild / sharp / U-turns.",
         "3. `genuine_turn_delta_scatter.png` — Δangular (P6−P4) vs GT heading change, with y=0 and trend line.",
         "4. `genuine_turn_dataset_map.png` — count of genuine turns by recording (placa_espanya dominates).",
         "5. `aggregate_vs_genuine_turns.png` — KEY figure: global angular improvement vs genuine-turn degradation.", "",
         "## Numbers", "",
         "| Subset | n | Phase4 ang | Phase6 ang | Δ |", "|---|---|---|---|---|"]
    for name, g in groups:
        R.append(f"| {name} | {len(g)} | {g.phase4_angular_err_deg.mean():.1f} | "
                 f"{g.phase6_angular_err_deg.mean():.1f} | "
                 f"{g.phase6_angular_err_deg.mean()-g.phase4_angular_err_deg.mean():+.1f} |")
    R += ["",
          f"- Global (sweep): val Δ {gv:+.1f}°, test Δ {gt:+.1f}° (negative = Phase-6 better).",
          f"- Genuine turns: all Δ {d_all:+.1f}° (positive = Phase-6 worse).", "",
          "## Conclusion", "",
          "**Angular loss improves global heading realism but does NOT improve genuine "
          "turning decisions.** Globally (mild / near-straight motion dominates) the "
          "angular metric improves a few degrees; but on genuine turns it is neutral for "
          "mild turns and worse for sharp turns and U-turns. Both models sit at/above "
          "chance (~84–96°) on real U-turns and collapse to short stubby rollouts "
          "that do not follow the turn (see panel). Predicting turn DIRECTION at decision "
          "points remains unsolved at current data scale; the λ=0.1 model should be "
          "reported as a small global-realism gain at no ADE cost, not as a turning "
          "improvement.", ""]
    (L.D_REPORTS / "Genuine_Turns_Visual_Report.md").write_text("\n".join(R), encoding="utf-8")
    print("[report] Genuine_Turns_Visual_Report.md")


if __name__ == "__main__":
    main()
