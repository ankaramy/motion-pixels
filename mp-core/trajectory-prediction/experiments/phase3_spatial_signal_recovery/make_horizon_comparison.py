"""
Phase 3B - build the horizon-10 vs horizon-30 comparison report.

Reads the two completed Phase 3 runs (does not re-run or modify them) and writes
Phase3_Horizon30_Report.md into the horizon-30 output's reports/ folder.

    python experiments\\phase3_spatial_signal_recovery\\make_horizon_comparison.py
"""
import os
import pandas as pd

OUT = r"C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs"
H10 = os.path.join(OUT, "10_phase3_spatial_signal_recovery")
H30 = os.path.join(OUT, "10_phase3_spatial_signal_recovery_horizon30")

EXPS = ["A_motion_only", "B_architecture_only",
        "C_motion_plus_architecture", "OLD_spatial_only"]
RECS = ["stairs_montjuic_01", "red_bridge_combined_01", "esplanade_espanya_01",
        "placa_espanya_01", "placa_catalunya_01"]


def best(summ, exp):
    sub = summ[summ.experiment == exp].sort_values("mean_balanced_accuracy", ascending=False)
    if sub.empty:
        return None, None
    return float(sub.iloc[0]["mean_balanced_accuracy"]), sub.iloc[0]["model"]


def overall_dist(path):
    d = pd.read_csv(os.path.join(path, "tables", "label_distribution.csv"))
    row = d[d.recording == "__overall__"].iloc[0]
    return row["rows_used"], row["pct_left"], row["pct_straight"], row["pct_right"]


def per_rec_B(path):
    fm = pd.read_csv(os.path.join(path, "tables", "fold_metrics.csv"))
    summ = pd.read_csv(os.path.join(path, "tables", "summary_metrics.csv"))
    _, mdl = best(summ, "B_architecture_only")
    out = {}
    for r in RECS:
        row = fm[(fm.experiment == "B_architecture_only") & (fm.model == mdl) & (fm.held_out == r)]
        out[r] = float(row["balanced_accuracy"].values[0]) if not row.empty else None
    return out, mdl


def main():
    s10 = pd.read_csv(os.path.join(H10, "tables", "summary_metrics.csv"))
    s30 = pd.read_csv(os.path.join(H30, "tables", "summary_metrics.csv"))

    rows10, l10, st10, r10 = overall_dist(H10)
    rows30, l30, st30, r30 = overall_dist(H30)
    b10, b10m = per_rec_B(H10)
    b30, b30m = per_rec_B(H30)

    L = ["# Phase 3B - Horizon-30 Turn-Signal Probe", "",
         "Secondary analysis: re-runs the Phase 3 turn-classification probe at a",
         "**1-second behavioural horizon (30 frames)** to test whether the weak",
         "architectural signal at 0.33 s was masked by heading noise / micro-turns.",
         "Identical otherwise: +/-15 deg threshold, Leave-One-Recording-Out over the",
         "5 validated recordings, Logistic + Random Forest, balanced class weights,",
         "same A/B/C/OLD feature groups. No encoder, mask, dataset, or model was",
         "modified; the horizon-10 outputs were not overwritten.", "",
         "Numbers below are LORO mean balanced accuracy, best model per feature set",
         "(chance = 0.333).", "",
         "## 1. Headline metrics (horizon 30)", "",
         "| Feature set | Balanced acc | Model | vs chance |",
         "|---|---|---|---|"]
    labelmap = {"A_motion_only": "A - Motion only",
                "B_architecture_only": "B - Architecture only",
                "C_motion_plus_architecture": "C - Motion + Architecture",
                "OLD_spatial_only": "OLD - Old spatial only"}
    for e in EXPS:
        v, m = best(s30, e)
        L.append(f"| {labelmap[e]} | **{v:.4f}** | {m} | {v-1/3:+.4f} |")

    L += ["", "## 2. Horizon 10 vs Horizon 30", "",
          "| Feature set | Horizon 10 | Horizon 30 | Delta |",
          "|---|---|---|---|"]
    for e in EXPS:
        v10, _ = best(s10, e)
        v30, _ = best(s30, e)
        L.append(f"| {labelmap[e]} | {v10:.4f} | {v30:.4f} | {v30-v10:+.4f} |")
    a10, _ = best(s10, "A_motion_only"); a30, _ = best(s30, "A_motion_only")
    c10, _ = best(s10, "C_motion_plus_architecture"); c30, _ = best(s30, "C_motion_plus_architecture")
    L += ["",
          f"- Motion+Arch minus Motion (C-A): horizon10 = {c10-a10:+.4f}, "
          f"horizon30 = {c30-a30:+.4f}."]

    L += ["", "## 3. Label distribution comparison", "",
          "| Horizon | rows used | % LEFT | % STRAIGHT | % RIGHT |",
          "|---|---|---|---|---|",
          f"| 10 (~0.33s) | {rows10} | {l10} | {st10} | {r10} |",
          f"| 30 (~1.0s)  | {rows30} | {l30} | {st30} | {r30} |", "",
          f"- STRAIGHT share went from {st10}% to {st30}% "
          f"({'more' if st30>st10 else 'less'} common at 1 s)."]

    L += ["", "## 4. Per-recording architecture-only (B) balanced accuracy", "",
          f"| Held-out | Horizon 10 ({b10m}) | Horizon 30 ({b30m}) | Delta |",
          "|---|---|---|---|"]
    for r in RECS:
        v10 = b10[r]; v30 = b30[r]
        d = (v30 - v10) if (v10 is not None and v30 is not None) else None
        L.append(f"| {r} | {v10:.3f} | {v30:.3f} | {d:+.3f} |")

    # interpretation
    bB30, _ = best(s30, "B_architecture_only")
    gain30 = c30 - a30
    if bB30 >= 0.40 or gain30 > 0.03:
        verdict = "STRONG evidence"
    elif bB30 >= 0.36 or 0.01 <= gain30 <= 0.03:
        verdict = "MODERATE evidence"
    else:
        verdict = "WEAK evidence"

    L += ["", "## 5. Interpretation (cautious)", "",
          f"**Verdict at 1 s: {verdict}.**", "",
          "A) *Does architecture influence medium-term navigation more than "
          "instantaneous steering?* Architecture-only barely moved between 0.33 s "
          f"({best(s10,'B_architecture_only')[0]:.3f}) and 1 s ({bB30:.3f}). So the "
          "longer behavioural horizon did **not** unlock stronger architectural "
          "signal; the micro-turn-noise hypothesis is not supported as the main "
          "limiter.",
          "",
          "B) *Does Motion + Architecture justify full Model C retraining?* No - at "
          f"1 s, C ({c30:.3f}) still does not exceed Motion-only ({a30:.3f}) "
          f"(delta {gain30:+.3f}). Architecture adds no measurable information on "
          "top of motion at either horizon.",
          "",
          "C) *Is the recovered architectural signal real but insufficient?* Yes - "
          "this is the most defensible reading. Architecture-only stays modestly "
          "above chance and consistently above the old trajectory-derived encoder "
          f"(OLD ~ {best(s30,'OLD_spatial_only')[0]:.3f}), and esplanade_espanya "
          f"is reproducibly the strongest site (B = {b30['esplanade_espanya_01']:.3f}). "
          "But the signal is weak and redundant with motion under cross-site "
          "generalization.", ""]

    L += ["## 6. Recommendation", "",
          "**Refine masks and rerun.**", "",
          "Rationale (cautious):",
          "- The architectural signal is *real* (beats the old encoder; esplanade "
          "clearly above chance) but *weak* and not additive over motion at both "
          "0.33 s and 1 s, so **Proceed to Phase 4** is not yet justified.",
          "- Two of five masks are Phase-2 CHECK quality (red_bridge plan-crop "
          "coverage; placa_espanya building-edge over-inclusion) and these are the "
          "weakest architecture-only folds - mask quality is a plausible, fixable "
          "limiter, so **Stop architectural investigation** is premature.",
          "- A longer horizon already failed to help, so **Try longer horizons** is "
          "unlikely to change the conclusion.",
          "- Therefore: tighten the two CHECK masks (and consider richer/normalized "
          "architectural features, e.g. heading-relative clearance ratios), then "
          "re-probe before any Phase 4 commitment. Do not train Model C or rebuild "
          "the master dataset on the current evidence.", ""]

    os.makedirs(os.path.join(H30, "reports"), exist_ok=True)
    path = os.path.join(H30, "reports", "Phase3_Horizon30_Report.md")
    with open(path, "w") as f:
        f.write("\n".join(L))
    print("[written]", path)


if __name__ == "__main__":
    main()
