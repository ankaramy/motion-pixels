"""
build_comparison.py — aggregate all completed horizon runs into
comparison_summary.csv + comparison_report.md, and print the final table.

Scans H20/H60/H100/H200/H400 for metrics_summary.json (works incrementally).
Answers the 6 questions and recommends which model to freeze as Model X_long.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
HORIZONS = [20, 60, 100, 200, 400]

# usability heuristics (documented, not fabricated)
COLLAPSE_OK = 0.35      # collapse rate below this = usable
RATIO_OK = 0.55         # pred/GT length ratio above this = visually useful length


def recommend(row) -> str:
    c = row["collapse_rate"]; r = row["pred_gt_ratio_median"]
    if c <= COLLAPSE_OK and r >= RATIO_OK:
        return "usable — accurate + plausible length"
    if c <= COLLAPSE_OK:
        return "usable but conservative (undershoots length)"
    if c <= 0.55:
        return "marginal — frequent length collapse"
    return "NOT usable — predictions collapse"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = []
    for H in HORIZONS:
        f = HERE / f"H{H}" / "metrics_summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        av = s["data_availability"]
        rows.append({
            "horizon": H, "approx_distance_m": s["approx_distance_m"],
            "train_tracks": av["train_tracks"], "val_tracks": av["val_tracks"],
            "test_tracks": av["test_tracks"], "tracks_dropped_short": av["tracks_dropped_short"],
            "train_windows": s["train_windows"], "val_windows": s["val_rollout_windows"],
            "test_windows": s["test_windows"],
            "ADE": round(s["ADE"], 3), "FDE": round(s["FDE"], 3),
            "angular_err_med_deg": round(s["angular_err_median_deg"], 1),
            "GT_len_med": round(s["gt_len_median"], 2),
            "pred_len_med": round(s["pred_len_median"], 2),
            "pred_len_p75": round(s["pred_len_p75"], 2),
            "pred_gt_ratio_median": round(s["pred_gt_ratio_median"], 3),
            "collapse_rate": round(s["collapse_rate_pred_lt_25pct_gt"], 3),
            "best_epoch": s["best_epoch"],
        })
    if not rows:
        print("No completed horizons yet."); return
    df = pd.DataFrame(rows).sort_values("horizon")
    df["recommended_use"] = df.apply(recommend, axis=1)
    df.to_csv(HERE / "comparison_summary.csv", index=False)

    # answers
    best_ade = df.loc[df.ADE.idxmin()]
    best_fde = df.loc[df.FDE.idxmin()]
    # longest usable = max horizon with collapse<=COLLAPSE_OK; tie-break by pred_len_med
    usable = df[df.collapse_rate <= COLLAPSE_OK]
    longest_usable = usable.loc[usable.horizon.idxmax()] if len(usable) else None
    # collapse onset = first horizon with collapse>COLLAPSE_OK
    collapsed = df[df.collapse_rate > COLLAPSE_OK].sort_values("horizon")
    collapse_h = int(collapsed.iloc[0].horizon) if len(collapsed) else None
    # best for viz = usable with longest pred_len_med (real length) and acceptable ADE
    viz_pool = usable if len(usable) else df
    best_viz = viz_pool.loc[viz_pool.pred_len_med.idxmax()]
    # Model X_long recommendation: longest usable horizon (architectural reach) that is still plausible
    rec_long = longest_usable if longest_usable is not None else best_ade

    lines = ["# MODEL_X Horizon Sweep — Comparison Report", "",
             "Same Model X recipe at every horizon (Model C TrajectoryLSTM, 10 v3 features, mixed",
             "track-level split, single-step next-displacement MSE objective, seed 42). Horizon H sets",
             "the track-length filter (need >= 10+H frames), the autoregressive rollout depth, and the",
             "val-ADE checkpoint. **No multi-step loss, no path scaling, no fabricated length.**", "",
             f"Usability thresholds (documented): *usable* = collapse rate ≤ {COLLAPSE_OK:.0%} "
             f"(share of windows with pred length < 25% of GT); *visually useful length* = median "
             f"pred/GT ratio ≥ {RATIO_OK:.0%}.", "",
             "## Summary table", "",
             "| H | ~dist | train_w | val_w | test_w | ADE | FDE | GT_len_med | pred_len_med | ratio | collapse | recommended_use |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {int(r.horizon)} | {r.approx_distance_m} m | {int(r.train_windows):,} | "
                     f"{int(r.val_windows):,} | {int(r.test_windows):,} | {r.ADE} | {r.FDE} | "
                     f"{r.GT_len_med} | {r.pred_len_med} | {r.pred_gt_ratio_median} | "
                     f"{r.collapse_rate:.0%} | {r.recommended_use} |")

    lines += ["", "## Data availability (tracks dropped for being too short)", "",
              "| H | need frames | train tr | val tr | test tr | dropped |",
              "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {int(r.horizon)} | {10+int(r.horizon)} | {int(r.train_tracks)} | "
                     f"{int(r.val_tracks)} | {int(r.test_tracks)} | {int(r.tracks_dropped_short)} |")

    lines += ["", "## Answers", "",
              f"**1. Best ADE/FDE?** Best ADE = **H{int(best_ade.horizon)}** "
              f"(ADE {best_ade.ADE} m); best FDE = **H{int(best_fde.horizon)}** (FDE {best_fde.FDE} m). "
              "ADE/FDE in metres grow with horizon (more steps to accumulate error) — short horizons win "
              "on raw error, as expected.",
              "",
              f"**2. Longest usable predictions?** "
              + (f"**H{int(longest_usable.horizon)}** (~{longest_usable.approx_distance_m} m) is the "
                 f"longest horizon still under the collapse threshold "
                 f"(collapse {longest_usable.collapse_rate:.0%}, median pred length "
                 f"{longest_usable.pred_len_med} m)."
                 if longest_usable is not None else "No horizon stayed under the collapse threshold."),
              "",
              f"**3. Where does prediction collapse?** "
              + (f"Collapse (>{COLLAPSE_OK:.0%} of windows under 25% of GT length) first appears at "
                 f"**H{collapse_h}**." if collapse_h is not None
                 else f"No horizon exceeded the {COLLAPSE_OK:.0%} collapse threshold in the tested range."),
              "",
              f"**4. Best for architectural visualization?** **H{int(best_viz.horizon)}** "
              f"(~{best_viz.approx_distance_m} m): longest *real* predicted length "
              f"(median {best_viz.pred_len_med} m) while staying usable — the best trade of reach vs "
              "plausibility for plan-scale diagrams.",
              "",
              "**5. Are H200 / H400 realistic or too uncertain?** "
              + h200_400_verdict(df),
              "",
              f"**6. Freeze as Model X_long?** **H{int(rec_long.horizon)}** "
              f"(~{rec_long.approx_distance_m} m, ADE {rec_long.ADE} m, FDE {rec_long.FDE} m, "
              f"collapse {rec_long.collapse_rate:.0%}) — the longest architectural reach that remains "
              "accurate and visually honest. Checkpoint: "
              f"`H{int(rec_long.horizon)}/model_best.pth`.",
              "",
              "## Honesty note",
              "ADE/FDE are reported in metres over the full horizon; predicted paths are the raw model",
              "rollout with **no independent scaling**. Where the model undershoots length (low ratio /",
              "high collapse), that is reported as-is — long horizons are genuinely more uncertain and",
              "the model stays conservative rather than inventing motion."]
    (HERE / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")

    # final printed table
    print("\nhorizon | approx_distance | train_windows | val_windows | test_windows | ADE | FDE | "
          "GT_len_med | pred_len_med | pred/GT_ratio | collapse_rate | recommended_use")
    for _, r in df.iterrows():
        print(f"H{int(r.horizon):<5} | ~{r.approx_distance_m:<6} m | {int(r.train_windows):>12,} | "
              f"{int(r.val_windows):>10,} | {int(r.test_windows):>11,} | {r.ADE:>5} | {r.FDE:>5} | "
              f"{r.GT_len_med:>9} | {r.pred_len_med:>11} | {r.pred_gt_ratio_median:>12} | "
              f"{r.collapse_rate:>11.0%} | {r.recommended_use}")
    print(f"\nReports -> comparison_summary.csv, comparison_report.md")


def h200_400_verdict(df) -> str:
    parts = []
    for H in (200, 400):
        sub = df[df.horizon == H]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        verdict = ("realistic" if r.collapse_rate <= COLLAPSE_OK and r.pred_gt_ratio_median >= RATIO_OK
                   else "usable-but-conservative" if r.collapse_rate <= COLLAPSE_OK
                   else "too uncertain / collapses")
        parts.append(f"H{H} (~{r.approx_distance_m} m): {verdict} "
                     f"(ratio {r.pred_gt_ratio_median}, collapse {r.collapse_rate:.0%}, ADE {r.ADE} m)")
    return "; ".join(parts) if parts else "not yet evaluated."


if __name__ == "__main__":
    main()
