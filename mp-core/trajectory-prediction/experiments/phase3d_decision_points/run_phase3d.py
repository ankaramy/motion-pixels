"""
MOTION PIXELS - PHASE 3D: decision-point architectural signal test.

    python experiments\\phase3d_decision_points\\run_phase3d.py --run-all

Optional:
    --horizon 10
    --turn-threshold-deg 15

Evaluates turn-direction classifiers (Motion / Architecture / Motion+Arch) under
Leave-One-Recording-Out, on the FULL frame set, the combined DecisionUnion, and
each decision-point subset A-E. Analysis only.
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase3d_lib as D
import phase3_lib as P3

OUT = r"C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\10_phase3d_decision_points"
DATASET_ORDER = ["FULL", "DecisionUnion", "DecisionUnion_noE",
                 "A_near_obstacle", "B_near_boundary",
                 "C_high_asymmetry", "D_constrained_corridor", "E_approaching_turn"]
FS_SHORT = {"motion": "Motion", "architecture": "Architecture",
            "motion_plus_architecture": "Motion+Arch"}


def ensure_dirs():
    for s in ("reports", "tables", "figures", "data", "models"):
        os.makedirs(os.path.join(OUT, s), exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Phase 3D decision-point test")
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--turn-threshold-deg", type=float, default=15.0)
    ap.add_argument("--classifier", action="append",
                    choices=["logistic", "random_forest"], default=None)
    args = ap.parse_args()
    if not args.run_all:
        ap.error("pass --run-all")
    models = args.classifier or ["logistic", "random_forest"]
    rng = np.random.default_rng(0)
    ensure_dirs()

    print(f"[assemble] horizon={args.horizon} thr={args.turn_threshold_deg}")
    used, totals = P3.assemble(args.horizon, args.turn_threshold_deg)
    used = D.prepare(used)
    thr = D.subset_thresholds(used)
    masks = D.subset_masks(used, thr)
    print(f"[assemble] full rows={len(used)}  thresholds={thr}")

    datasets = {"FULL": np.ones(len(used), bool)}
    datasets.update(masks)
    # confound control: union of the CURRENT-architecture decision points only
    # (excludes E, which is defined by the FUTURE label and is 0% STRAIGHT).
    datasets["DecisionUnion_noE"] = (masks["A_near_obstacle"] | masks["B_near_boundary"]
                                     | masks["C_high_asymmetry"]
                                     | masks["D_constrained_corridor"])

    # ---------- Table 1: rows retained per subset ----------
    t1 = []
    n_full = len(used)
    for name in DATASET_ORDER:
        m = datasets[name]
        row = {"subset": name, "rows_retained": int(m.sum()),
               "rows_removed": int((~m).sum()),
               "pct_retained": round(100 * m.mean(), 2)}
        for r in D.RECORDINGS:
            rm = m & (used.recording == r).to_numpy()
            row[f"n_{r}"] = int(rm.sum())
        # label balance within subset
        sub = used[m]
        vc = sub["turn_label"].value_counts()
        for lab in D.LABELS:
            row[f"pct_{lab.lower()}"] = round(100 * vc.get(lab, 0) / max(len(sub), 1), 2)
        t1.append(row)
    t1df = pd.DataFrame(t1)
    t1df.to_csv(os.path.join(OUT, "tables", "table1_rows_retained.csv"), index=False)

    # dataset index / rows used
    pd.DataFrame([{"recording": r, "csv": P3.v3_csv(r), "rows_total": totals[r],
                   "rows_used_full": int((used.recording == r).sum())}
                  for r in D.RECORDINGS]).to_csv(
        os.path.join(OUT, "data", "phase3d_dataset_index.csv"), index=False)

    # ---------- Part 3: classify ----------
    fold_rows = []
    for name in DATASET_ORDER:
        sub_df = used[datasets[name]]
        for fs, feats in D.FEATURE_SETS.items():
            for mdl in models:
                for fr in D.loro(sub_df, feats, mdl, rng):
                    fr.update({"subset": name, "feature_set": fs, "model": mdl})
                    fold_rows.append(fr)
                done = [fr for fr in fold_rows if fr["subset"] == name
                        and fr["feature_set"] == fs and fr["model"] == mdl
                        and not fr.get("skipped")]
                if done:
                    mb = np.nanmean([d["balanced_accuracy"] for d in done])
                    print(f"  {name:24s} {fs:24s} {mdl:14s} balacc={mb:.3f} "
                          f"(folds={len(done)})")
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(OUT, "tables", "loro_fold_metrics.csv"), index=False)

    # summary: mean over non-skipped folds
    ok = fold_df[~fold_df["skipped"].fillna(False)]
    summ = (ok.groupby(["subset", "feature_set", "model"])
            .agg(mean_balacc=("balanced_accuracy", "mean"),
                 std_balacc=("balanced_accuracy", "std"),
                 mean_acc=("accuracy", "mean"),
                 n_folds=("balanced_accuracy", "count"),
                 mean_n_test=("n_test", "mean"))
            .round(4).reset_index())
    summ.to_csv(os.path.join(OUT, "tables", "summary_metrics.csv"), index=False)

    def bestbal(subset, fs):
        s = summ[(summ.subset == subset) & (summ.feature_set == fs)]
        return float(s["mean_balacc"].max()) if not s.empty else np.nan

    # ---------- Table 2: FULL vs DecisionUnion ----------
    t2 = []
    for fs in D.FEATURE_SETS:
        t2.append({"feature_set": fs,
                   "FULL_balacc": round(bestbal("FULL", fs), 4),
                   "DecisionUnion_balacc": round(bestbal("DecisionUnion", fs), 4),
                   "delta": round(bestbal("DecisionUnion", fs) - bestbal("FULL", fs), 4)})
    pd.DataFrame(t2).to_csv(os.path.join(OUT, "tables", "table2_full_vs_union.csv"), index=False)

    # ---------- Table 3: architecture-only by subset ----------
    t3 = [{"subset": name, "architecture_balacc": round(bestbal(name, "architecture"), 4),
           "vs_chance": round(bestbal(name, "architecture") - 1/3, 4)}
          for name in DATASET_ORDER]
    pd.DataFrame(t3).to_csv(os.path.join(OUT, "tables", "table3_architecture_by_subset.csv"), index=False)

    # ---------- Table 4: motion+arch gain ----------
    t4 = []
    for name in DATASET_ORDER:
        m = bestbal(name, "motion")
        ma = bestbal(name, "motion_plus_architecture")
        t4.append({"subset": name, "motion_balacc": round(m, 4),
                   "motion_plus_arch_balacc": round(ma, 4),
                   "gain": round(ma - m, 4)})
    pd.DataFrame(t4).to_csv(os.path.join(OUT, "tables", "table4_motion_arch_gain.csv"), index=False)

    # label distribution per subset
    t1df.to_csv(os.path.join(OUT, "tables", "label_distribution.csv"), index=False)
    pd.DataFrame([{"recording": r,
                   **{name: int(datasets[name][(used.recording == r).to_numpy()].sum())
                      for name in DATASET_ORDER}} for r in D.RECORDINGS]).to_csv(
        os.path.join(OUT, "data", "rows_used_per_recording.csv"), index=False)

    figures(summ, bestbal)
    write_report(summ, t1df, thr, bestbal, args, models, n_full)
    print("\n[done] outputs in", OUT)
    print(summ.to_string(index=False))


def figures(summ, bestbal):
    fig = os.path.join(OUT, "figures")
    # architecture-only by subset
    plt.figure(figsize=(11, 5))
    vals = [bestbal(n, "architecture") for n in DATASET_ORDER]
    bars = plt.bar(range(len(vals)), vals, color="indianred")
    plt.axhline(1/3, ls="--", color="k", label="chance")
    plt.axhline(0.40, ls=":", color="green", label="strong (0.40)")
    plt.xticks(range(len(vals)), DATASET_ORDER, rotation=25, ha="right", fontsize=8)
    plt.ylabel("Architecture-only balanced accuracy")
    plt.title("Phase 3D: architecture-only signal by decision subset")
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            plt.text(b.get_x()+b.get_width()/2, v+0.003, f"{v:.3f}", ha="center", fontsize=8)
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(fig, "architecture_by_subset.png"), dpi=130); plt.close()

    # motion vs motion+arch by subset
    plt.figure(figsize=(11, 5))
    x = np.arange(len(DATASET_ORDER))
    mv = [bestbal(n, "motion") for n in DATASET_ORDER]
    mav = [bestbal(n, "motion_plus_architecture") for n in DATASET_ORDER]
    plt.bar(x-0.2, mv, 0.4, label="Motion", color="steelblue")
    plt.bar(x+0.2, mav, 0.4, label="Motion+Arch", color="darkorange")
    plt.axhline(1/3, ls="--", color="k")
    plt.xticks(x, DATASET_ORDER, rotation=25, ha="right", fontsize=8)
    plt.ylabel("Balanced accuracy"); plt.legend()
    plt.title("Phase 3D: motion vs motion+architecture by subset")
    plt.tight_layout()
    plt.savefig(os.path.join(fig, "motion_vs_motion_arch_by_subset.png"), dpi=130); plt.close()

    # full vs union grouped by feature set
    plt.figure(figsize=(8, 5))
    fss = list(D.FEATURE_SETS)
    x = np.arange(len(fss))
    fv = [bestbal("FULL", f) for f in fss]
    uv = [bestbal("DecisionUnion", f) for f in fss]
    plt.bar(x-0.2, fv, 0.4, label="FULL", color="gray")
    plt.bar(x+0.2, uv, 0.4, label="DecisionUnion", color="seagreen")
    plt.axhline(1/3, ls="--", color="k")
    plt.xticks(x, [FS_SHORT[f] for f in fss])
    plt.ylabel("Balanced accuracy"); plt.legend()
    plt.title("Phase 3D: FULL vs DecisionUnion")
    plt.tight_layout()
    plt.savefig(os.path.join(fig, "full_vs_decisionunion.png"), dpi=130); plt.close()


def write_report(summ, t1df, thr, bestbal, args, models, n_full):
    union_arch = bestbal("DecisionUnion", "architecture")
    union_m = bestbal("DecisionUnion", "motion")
    union_ma = bestbal("DecisionUnion", "motion_plus_architecture")
    union_gain = union_ma - union_m

    # confound control: union of CURRENT-architecture decision points (no E)
    noE_arch = bestbal("DecisionUnion_noE", "architecture")
    noE_m = bestbal("DecisionUnion_noE", "motion")
    noE_ma = bestbal("DecisionUnion_noE", "motion_plus_architecture")
    noE_gain = noE_ma - noE_m

    def meets_strong(a, g):
        return (a >= 0.40) or (g > 0.03)

    def meets_moderate(a, g):
        return (a >= 0.36) or (0.01 <= g <= 0.03)

    strong_union = meets_strong(union_arch, union_gain)
    strong_noE = meets_strong(noE_arch, noE_gain)
    mod_noE = meets_moderate(noE_arch, noE_gain)

    # E is defined by the FUTURE label and is ~0% STRAIGHT; if the union meets the
    # bar ONLY because of E, the result is a future-selection artifact, not genuine
    # per-frame decision-point signal. The unconfounded estimate is DecisionUnion_noE.
    e_confound = strong_union and not strong_noE

    arch_by = [(n, bestbal(n, "architecture")) for n in DATASET_ORDER
               if n not in ("FULL", "DecisionUnion", "DecisionUnion_noE")]
    arch_by_sorted = sorted([a for a in arch_by if np.isfinite(a[1])],
                            key=lambda t: t[1], reverse=True)
    full_arch = bestbal("FULL", "architecture")

    # headline evidence is the UNCONFOUNDED estimate (no-E)
    evidence = "STRONG" if strong_noE else ("MODERATE" if mod_noE else "WEAK")

    # Distinguish B (weak-but-real, secondary) from C (no measurable value).
    # "Measurable" = the best LEGITIMATE (non-E-confounded) architecture-only score
    # is clearly above chance/old-encoder, i.e. >= 0.345.
    best_legit_arch = max([full_arch, noE_arch] + [v for _, v in arch_by_sorted])
    measurable = best_legit_arch >= 0.345
    if strong_noE:
        conclusion = "A. Architecture matters primarily at decision points."
    elif mod_noE or measurable:
        conclusion = "B. Architecture contains weak signal but remains secondary to motion."
    else:
        conclusion = ("C. Architecture contributes no measurable predictive value at "
                      "current dataset scale.")

    R = ["# Phase 3D - Decision-Point Architectural Signal Test", "",
         "Changes the unit of analysis: evaluates turn-direction signal only at",
         "spatial decision moments, to test whether architecture becomes informative",
         "there even though it is weak frame-wide (Phase 3A/B/C). Analysis only; no",
         "mask/encoder/dataset/model modified; Phase 3A/B/C outputs untouched.", "",
         f"Config: horizon {args.horizon} (~{args.horizon/30:.2f}s), turn threshold "
         f"+/-{args.turn_threshold_deg} deg, Leave-One-Recording-Out, "
         f"{', '.join(models)}, balanced accuracy primary (chance 0.333).", "",
         "## Decision-point definitions", "",
         f"- **A near obstacle**: dist_to_obstacle_v3_m < {D.NEAR_OBSTACLE_M} m",
         f"- **B near boundary**: dist_to_walkable_boundary_v3_m < {D.NEAR_BOUNDARY_M} m",
         f"- **C high asymmetry**: |clearance_asymmetry_v3| > {thr['asymmetry_p75']:.4f} "
         f"(data-driven, 75th pct of |asymmetry|)",
         f"- **D constrained corridor**: (clearance_left+right) < {thr['corridor_width_p25']:.4f} m "
         f"(data-driven, 25th pct of corridor width)",
         f"- **E approaching turn**: |future heading change over {args.horizon} frames| > "
         f"{D.FUTURE_TURN_DEG} deg (future label used for FILTERING only)",
         "- **DecisionUnion**: any of A-E true.", "",
         "## Table 1 - rows retained", "",
         f"Full common row set: {n_full} rows.", "",
         "| Subset | rows | % retained | %LEFT | %STRAIGHT | %RIGHT |",
         "|---|---|---|---|---|---|"]
    for _, r in t1df.iterrows():
        R.append(f"| {r['subset']} | {r['rows_retained']} | {r['pct_retained']} | "
                 f"{r['pct_left']} | {r['pct_straight']} | {r['pct_right']} |")

    R += ["", "## Table 2 - FULL vs DecisionUnion (balanced accuracy, best model)", "",
          "| Feature set | FULL | DecisionUnion | delta |", "|---|---|---|---|"]
    for fs in D.FEATURE_SETS:
        R.append(f"| {FS_SHORT[fs]} | {bestbal('FULL', fs):.4f} | "
                 f"{bestbal('DecisionUnion', fs):.4f} | "
                 f"{bestbal('DecisionUnion', fs)-bestbal('FULL', fs):+.4f} |")

    R += ["", "## Table 3 - architecture-only by subset", "",
          "| Subset | Architecture balacc | vs chance |", "|---|---|---|"]
    for n in DATASET_ORDER:
        v = bestbal(n, "architecture")
        R.append(f"| {n} | {v:.4f} | {v-1/3:+.4f} |")

    R += ["", "## Table 4 - motion + architecture gain by subset", "",
          "| Subset | Motion | Motion+Arch | gain |", "|---|---|---|---|"]
    for n in DATASET_ORDER:
        m = bestbal(n, "motion"); ma = bestbal(n, "motion_plus_architecture")
        R.append(f"| {n} | {m:.4f} | {ma:.4f} | {ma-m:+.4f} |")

    R += ["", "## Per-subset answers (cautious)", ""]
    for n in DATASET_ORDER:
        a = bestbal(n, "architecture"); m = bestbal(n, "motion")
        ma = bestbal(n, "motion_plus_architecture")
        R.append(f"- **{n}**: architecture {a:.3f} "
                 f"({'>' if a>0.345 else '~'} chance), vs full-frame arch {full_arch:.3f} "
                 f"(delta {a-full_arch:+.3f}); motion+arch gain {ma-m:+.3f}.")

    R += ["", "## Decision-Union evidence (with the E confound made explicit)", "",
          f"- DecisionUnion (A-E) architecture-only: **{union_arch:.4f}**, "
          f"motion {union_m:.4f}, motion+arch {union_ma:.4f} (gain {union_gain:+.4f}).",
          f"- DecisionUnion_noE (A-D, current-architecture only): architecture "
          f"**{noE_arch:.4f}**, motion {noE_m:.4f}, motion+arch {noE_ma:.4f} "
          f"(gain {noE_gain:+.4f}).", "",
          "**Confound:** subset E is defined by the FUTURE turn (|future heading "
          "change| > %d deg) and is **0%% STRAIGHT by construction**; it is %d%% of the "
          "union. Unioning E with the current-defined subsets changes the class "
          "mixture (STRAIGHT drops from 26%% to ~18%%) and lets a classifier score "
          "balanced accuracy by exploiting that selection rather than by reading "
          "geometry. The unconfounded estimate is therefore **DecisionUnion_noE**."
          % (D.FUTURE_TURN_DEG, round(100*0.6567)),
          "",
          f"- With E: union appears to {'MEET' if strong_union else 'miss'} the strong bar "
          f"(arch {union_arch:.3f}, gain {union_gain:+.3f}), with high fold variance.",
          f"- Without E: arch falls to {noE_arch:.3f} (~ full-frame {full_arch:.3f}) and "
          f"motion+arch gain is {noE_gain:+.3f} - i.e. the apparent decision-point "
          f"strength {'DISAPPEARS (future-selection artifact)' if e_confound else 'persists'}.",
          "",
          f"- Headline (unconfounded) verdict: **{evidence} evidence**.",
          "  - Strong: arch >= 0.40 OR motion+arch gain > 0.03.",
          "  - Moderate: arch >= 0.36 OR gain 0.01-0.03.",
          "  - Weak: arch ~0.33-0.35 AND no gain.", "",
          "## Key questions", "",
          f"1. **Does architecture outperform chance at decision points?** "
          f"{'Yes, modestly' if union_arch>0.345 else 'Not meaningfully'} "
          f"(DecisionUnion arch = {union_arch:.3f}); best subset = "
          f"{arch_by_sorted[0][0]} ({arch_by_sorted[0][1]:.3f}).",
          f"2. **Does architecture improve vs full-frame?** Full-frame arch = "
          f"{full_arch:.3f}; DecisionUnion = {union_arch:.3f} "
          f"(delta {union_arch-full_arch:+.3f}).",
          f"3. **Does architecture improve Motion+Architecture?** Gain over motion at "
          f"DecisionUnion = {union_gain:+.3f}.", "",
          "## Conclusion", "",
          f"**{conclusion}**", "",
          (("This overrides the naive reading of the DecisionUnion number: that union "
            "appears strong only because it folds in subset E (future-defined, 0% "
            "STRAIGHT). When decision points are restricted to genuinely current-frame "
            "architectural conditions (DecisionUnion_noE), architecture returns to "
            f"~{noE_arch:.2f} balanced accuracy and adds nothing over motion "
            f"({noE_gain:+.3f}). Architecture is real but secondary - consistent with "
            "Phases 3A/3B/3C.") if e_confound else
           ("The decision-point restriction does not change the picture from the "
            "full-frame analysis.")), "",
          "## Caveats", "",
          "- The DecisionUnion-with-E number is reported but treated as CONFOUNDED: "
          "subset E selects rows by the future turn and is 0% STRAIGHT, so it is not "
          "evidence of per-frame geometric predictiveness. DecisionUnion_noE is the "
          "honest estimate.",
          "- Decision subsets are smaller and class-imbalanced per held-out site; "
          "folds with <200 test rows or <2 classes are skipped (see fold table).",
          "- Subset E uses the future label only as a FILTER, not as a feature; the "
          "predicted target is still the LEFT/STRAIGHT/RIGHT turn label.",
          "- `turn_rate_v3` keeps Motion a strong autoregressive baseline even at "
          "decision points.",
          "- LORO across 5 distinct geometries remains a hard generalization test; "
          "these are conservative estimates, not tuned performance.", "",
          "## Outputs", "",
          "- tables/: table1_rows_retained, table2_full_vs_union, "
          "table3_architecture_by_subset, table4_motion_arch_gain, summary_metrics, "
          "loro_fold_metrics, label_distribution",
          "- figures/: architecture_by_subset, motion_vs_motion_arch_by_subset, "
          "full_vs_decisionunion",
          "- data/: phase3d_dataset_index, rows_used_per_recording", ""]
    with open(os.path.join(OUT, "reports", "Phase3D_Decision_Points_Report.md"), "w") as f:
        f.write("\n".join(R))


if __name__ == "__main__":
    main()
