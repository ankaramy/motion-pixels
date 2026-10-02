"""
MOTION PIXELS - PHASE 3C: feature-engineering audit + signal re-test.

    python experiments\\phase3c_feature_engineering_audit\\run_phase3c_feature_audit.py --run-all

Optional:
    --horizons 10 30
    --turn-threshold-deg 15

Part 1: audit V3 architectural feature distributions (+ derived features).
Part 2: add 14 documented derived architectural features.
Part 3: LORO turn classifiers for feature sets A/B/C/D/E/OLD at each horizon.

Diagnostic only: no mask, encoder, encoded output, dataset, or model modified;
Phase 3 / 3B outputs are not touched (separate output directory).
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
import phase3c_lib as L
import phase3_lib as P3

OUT = r"C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\10_phase3c_feature_engineering_audit"
LABELMAP = {
    "A_motion_only": "A motion",
    "B_arch_original": "B arch(orig)",
    "C_motion_plus_arch_original": "C motion+arch",
    "D_enhanced_arch_only": "D enhanced-arch",
    "E_motion_plus_enhanced_arch": "E motion+enhanced",
    "OLD_spatial_only": "OLD spatial",
}


def ensure_dirs():
    for s in ("reports", "tables", "figures", "data", "models"):
        os.makedirs(os.path.join(OUT, s), exist_ok=True)
    for s in ("feature_histograms", "feature_boxplots_by_recording"):
        os.makedirs(os.path.join(OUT, "figures", s), exist_ok=True)


# ----------------------- Part 1: audit -----------------------
def feature_audit(used):
    rows = []
    groups = [("original", L.ARCH_ORIG), ("derived", L.DERIVED_FEATURES)]
    for grp, feats in groups:
        for f in feats:
            # overall
            o = L.audit_feature(used[f])
            o.update({"feature": f, "group": grp, "recording": "__overall__"})
            rows.append(o)
            for r in L.RECORDINGS:
                sub = used[used.recording == r]
                a = L.audit_feature(sub[f])
                a.update({"feature": f, "group": grp, "recording": r})
                rows.append(a)
    df = pd.DataFrame(rows)
    front = ["feature", "group", "recording"]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(os.path.join(OUT, "tables", "feature_distribution_summary.csv"), index=False)
    return df


def audit_figures(used):
    # histograms (overall) for original arch features
    feats = L.ARCH_ORIG
    ncol = 4
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    for ax, f in zip(axes.ravel(), feats):
        a = pd.to_numeric(used[f], errors="coerce").to_numpy()
        a = a[np.isfinite(a)]
        ax.hist(a, bins=50, color="steelblue")
        ax.set_title(f, fontsize=8)
    for ax in axes.ravel()[len(feats):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figures", "feature_histograms", "original_arch_histograms.png"), dpi=120)
    plt.close()

    # histograms for derived
    feats = L.DERIVED_FEATURES
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    for ax, f in zip(axes.ravel(), feats):
        a = pd.to_numeric(used[f], errors="coerce").to_numpy()
        a = a[np.isfinite(a)]
        ax.hist(a, bins=50, color="darkorange")
        ax.set_title(f, fontsize=8)
    for ax in axes.ravel()[len(feats):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figures", "feature_histograms", "derived_arch_histograms.png"), dpi=120)
    plt.close()

    # boxplots by recording for key features
    key = ["dist_to_obstacle_v3_m", "dist_to_walkable_boundary_v3_m",
           "clearance_forward_v3_m", "clearance_left_v3_m",
           "clearance_right_v3_m", "local_corridor_width_v3_m"]
    for f in key:
        data = [pd.to_numeric(used[used.recording == r][f], errors="coerce").dropna().to_numpy()
                for r in L.RECORDINGS]
        plt.figure(figsize=(9, 4.5))
        plt.boxplot(data, labels=[r[:14] for r in L.RECORDINGS], showfliers=False)
        plt.title(f"{f} by recording")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "figures", "feature_boxplots_by_recording", f"{f}.png"), dpi=110)
        plt.close()

    # correlation matrix (orig + derived)
    feats = L.ARCH_ORIG + L.DERIVED_FEATURES
    corr = used[feats].corr().to_numpy()
    plt.figure(figsize=(12, 10))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046)
    plt.xticks(range(len(feats)), feats, rotation=90, fontsize=6)
    plt.yticks(range(len(feats)), feats, fontsize=6)
    plt.title("V3 architectural feature correlation (original + derived)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figures", "correlation_matrix.png"), dpi=130)
    plt.close()


# ----------------------- Part 3: classifiers -----------------------
def run_horizon(horizon, thr, models, rng):
    used, totals = L.assemble(horizon, thr)
    fold_rows, cms = [], {}
    for exp, feats in L.EXPERIMENTS.items():
        if any(f not in used.columns for f in feats):
            continue
        sub = used[np.isfinite(used[feats].to_numpy(float)).all(axis=1)]
        cms[exp] = {}
        for mdl in models:
            cms[exp][mdl] = {}
            pooled = np.zeros((3, 3), int)
            for held in L.RECORDINGS:
                tr = sub[sub.recording != held]
                te = sub[sub.recording == held]
                if len(te) == 0 or te["turn_label"].nunique() < 2:
                    continue
                met, cm = L.run_fold(mdl, feats, tr, te, rng)
                pooled += cm
                cms[exp][mdl][held] = cm.tolist()
                fold_rows.append({
                    "horizon": horizon, "experiment": exp, "model": mdl,
                    "held_out": held, "n_test": int(len(te)),
                    "accuracy": met["accuracy"],
                    "balanced_accuracy": met["balanced_accuracy"],
                    "macro_f1": met["macro_f1"],
                    "majority_baseline_acc": met["majority_baseline_acc"],
                    "f1_LEFT": met["per_class"]["LEFT"]["f1"],
                    "f1_STRAIGHT": met["per_class"]["STRAIGHT"]["f1"],
                    "f1_RIGHT": met["per_class"]["RIGHT"]["f1"],
                })
                print(f"  h{horizon} {exp:28s} {mdl:14s} {held:24s} "
                      f"balacc={met['balanced_accuracy']:.3f}")
            cms[exp][mdl]["pooled"] = pooled.tolist()
    return used, totals, pd.DataFrame(fold_rows), cms


def summarize(fold_df):
    return (fold_df.groupby(["horizon", "experiment", "model"])
            .agg(mean_balanced_accuracy=("balanced_accuracy", "mean"),
                 std_balanced_accuracy=("balanced_accuracy", "std"),
                 mean_accuracy=("accuracy", "mean"),
                 mean_macro_f1=("macro_f1", "mean"),
                 mean_majority_baseline=("majority_baseline_acc", "mean"))
            .round(4).reset_index())


def best(summ, horizon, exp):
    sub = summ[(summ.horizon == horizon) & (summ.experiment == exp)]
    if sub.empty:
        return None, None
    row = sub.sort_values("mean_balanced_accuracy", ascending=False).iloc[0]
    return float(row["mean_balanced_accuracy"]), row["model"]


def label_dist(used_by_h):
    rows = []
    for h, used in used_by_h.items():
        for r in L.RECORDINGS + ["__overall__"]:
            sub = used if r == "__overall__" else used[used.recording == r]
            n = len(sub); vc = sub["turn_label"].value_counts()
            row = {"horizon": h, "recording": r, "rows_used": n}
            for lab in L.LABELS:
                c = int(vc.get(lab, 0))
                row[f"pct_{lab.lower()}"] = round(100 * c / n, 2) if n else 0.0
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "tables", "label_distribution.csv"), index=False)
    return df


# ----------------------- figures for classifiers -----------------------
def classifier_figures(summ, fold_df, horizons):
    exps = list(L.EXPERIMENTS.keys())
    for h in horizons:
        vals, labs = [], []
        for e in exps:
            v, m = best(summ, h, e)
            if v is not None:
                vals.append(v); labs.append(f"{LABELMAP[e]}\n({m})")
        plt.figure(figsize=(11, 5))
        bars = plt.bar(range(len(vals)), vals, color="teal")
        plt.axhline(1/3, ls="--", color="k", label="chance")
        plt.xticks(range(len(vals)), labs, fontsize=8)
        plt.ylabel("Balanced accuracy (LORO mean)")
        plt.title(f"Phase 3C balanced accuracy - horizon {h}")
        for b, v in zip(bars, vals):
            plt.text(b.get_x()+b.get_width()/2, v+0.003, f"{v:.3f}", ha="center", fontsize=8)
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUT, "figures", f"balanced_accuracy_horizon{h}.png"), dpi=130)
        plt.close()

    # enhanced vs original architecture (B vs D, C vs E) across horizons
    plt.figure(figsize=(10, 5))
    groups = [("B_arch_original", "D_enhanced_arch_only"),
              ("C_motion_plus_arch_original", "E_motion_plus_enhanced_arch")]
    xlabels, orig_v, enh_v = [], [], []
    for h in horizons:
        for (o, e) in groups:
            ov, _ = best(summ, h, o); ev, _ = best(summ, h, e)
            xlabels.append(f"h{h}\n{LABELMAP[o]}->{LABELMAP[e]}")
            orig_v.append(ov); enh_v.append(ev)
    x = np.arange(len(xlabels))
    plt.bar(x-0.2, orig_v, 0.4, label="original", color="slategray")
    plt.bar(x+0.2, enh_v, 0.4, label="enhanced", color="darkorange")
    plt.axhline(1/3, ls="--", color="k")
    plt.xticks(x, xlabels, fontsize=7)
    plt.ylabel("Balanced accuracy"); plt.legend()
    plt.title("Enhanced vs original architecture")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figures", "enhanced_vs_original_architecture.png"), dpi=130)
    plt.close()

    # per-recording LORO comparison (B vs D) at smallest horizon
    h0 = horizons[0]
    _, mB = best(summ, h0, "B_arch_original")
    _, mD = best(summ, h0, "D_enhanced_arch_only")
    recs = L.RECORDINGS
    bvals, dvals = [], []
    for r in recs:
        b = fold_df[(fold_df.horizon == h0) & (fold_df.experiment == "B_arch_original") &
                    (fold_df.model == mB) & (fold_df.held_out == r)]["balanced_accuracy"]
        d = fold_df[(fold_df.horizon == h0) & (fold_df.experiment == "D_enhanced_arch_only") &
                    (fold_df.model == mD) & (fold_df.held_out == r)]["balanced_accuracy"]
        bvals.append(b.values[0] if len(b) else 0)
        dvals.append(d.values[0] if len(d) else 0)
    x = np.arange(len(recs))
    plt.figure(figsize=(11, 5))
    plt.bar(x-0.2, bvals, 0.4, label=f"B arch-orig ({mB})", color="slategray")
    plt.bar(x+0.2, dvals, 0.4, label=f"D enhanced ({mD})", color="darkorange")
    plt.axhline(1/3, ls="--", color="k")
    plt.xticks(x, [r[:16] for r in recs], rotation=20, ha="right", fontsize=8)
    plt.ylabel(f"Balanced accuracy (held-out, h{h0})"); plt.legend()
    plt.title("Per-recording LORO: enhanced vs original architecture-only")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figures", "per_recording_loro_comparison.png"), dpi=130)
    plt.close()


# ----------------------- report -----------------------
def write_report(summ, fold_df, audit_df, dist_df, used_by_h, horizons, args):
    near_const = audit_df[(audit_df.recording == "__overall__") &
                          (audit_df.near_constant_warning != "")]["feature"].tolist()
    saturated = audit_df[(audit_df.recording == "__overall__") &
                         (audit_df.saturation_warning != "")][["feature", "saturation_warning"]]
    h0 = horizons[0]

    def line(h):
        out = {}
        for e in L.EXPERIMENTS:
            v, m = best(summ, h, e)
            out[e] = (v, m)
        return out

    def verdict(h):
        D = best(summ, h, "D_enhanced_arch_only")[0]
        A = best(summ, h, "A_motion_only")[0]
        E = best(summ, h, "E_motion_plus_enhanced_arch")[0]
        gain = (E - A) if (E is not None and A is not None) else None
        if (D is not None and D >= 0.40) or (gain is not None and gain > 0.03):
            return "STRONG", D, gain
        if (D is not None and D >= 0.36) or (gain is not None and 0.01 <= gain <= 0.03):
            return "MODERATE", D, gain
        return "WEAK", D, gain

    R = ["# Phase 3C - V3 Feature-Engineering Audit + Signal Re-test", "",
         "Tests whether the V3 architectural *features* (not the masks, which are",
         "accepted) are expressive enough to reveal turn-direction signal. Adds 14",
         "derived architectural features and re-runs the Phase 3 LORO classifiers at",
         "both horizons. Diagnostic only: no mask, encoder, encoded output, dataset,",
         "or model was modified; Phase 3 / 3B outputs untouched.", "",
         f"Config: thr +/-{args.turn_threshold_deg} deg, horizons {horizons}, "
         "Leave-One-Recording-Out, Logistic + Random Forest, balanced weights, "
         "balanced accuracy primary (chance 0.333).", "",
         "## Part 1 - Feature distribution audit (Q1, Q2)", ""]
    R.append("Usable rows: " + ", ".join(f"h{h}={len(used_by_h[h])}" for h in horizons) + ".")
    R.append("")
    R.append("**Near-constant features (overall):** " + (", ".join(near_const) if near_const else "none."))
    if not saturated.empty:
        R.append("")
        R.append("**Saturated features (overall):**")
        for _, s in saturated.iterrows():
            R.append(f"- {s['feature']}: {s['saturation_warning']}")
    R += ["",
          "Selected original-feature spread (overall, h%d):" % h0, "",
          "| Feature | min | median | mean | p95 | max | std | uniq | sat |",
          "|---|---|---|---|---|---|---|---|---|"]
    aov = audit_df[(audit_df.recording == "__overall__") & (audit_df.group == "original")]
    for _, r in aov.iterrows():
        R.append(f"| {r['feature']} | {r.get('min')} | {r.get('median')} | {r.get('mean')} "
                 f"| {r.get('p95')} | {r.get('max')} | {r.get('std')} | "
                 f"{r.get('n_unique_rounded')} | {r.get('saturation_warning','')} |")

    R += ["", "## Part 2 - Derived feature formulas", "",
          "epsilon = 1e-6; ratio/pressure features clipped to [0, %.0f]." % L.CLIP, "",
          "| Feature | Formula | Clip |", "|---|---|---|"]
    for name, formula, clip in L.DERIVED_SPEC:
        R.append(f"| {name} | {formula} | {clip or '-'} |")

    R += ["", "## Part 3 - Classifier results (LORO mean balanced accuracy, best model)", ""]
    for h in horizons:
        R += [f"### Horizon {h}", "",
              "| Feature set | Balanced acc | Model | vs chance |",
              "|---|---|---|---|"]
        ln = line(h)
        for e in L.EXPERIMENTS:
            v, m = ln[e]
            if v is not None:
                R.append(f"| {LABELMAP[e]} | **{v:.4f}** | {m} | {v-1/3:+.4f} |")
        A = ln["A_motion_only"][0]; C = ln["C_motion_plus_arch_original"][0]
        D = ln["D_enhanced_arch_only"][0]; E = ln["E_motion_plus_enhanced_arch"][0]
        B = ln["B_arch_original"][0]
        R += ["",
              f"- Enhanced vs original architecture-only: D {D:.4f} vs B {B:.4f} "
              f"(delta {D-B:+.4f}).",
              f"- Motion+Enhanced vs Motion: E {E:.4f} vs A {A:.4f} (delta {E-A:+.4f}).",
              f"- Motion+Enhanced vs Motion+Original: E {E:.4f} vs C {C:.4f} "
              f"(delta {E-C:+.4f}).", ""]

    R += ["## Part 4 - Label distribution", "",
          "| Horizon | rows | % LEFT | % STRAIGHT | % RIGHT |", "|---|---|---|---|---|"]
    for h in horizons:
        ov = dist_df[(dist_df.horizon == h) & (dist_df.recording == "__overall__")].iloc[0]
        R.append(f"| {h} | {ov['rows_used']} | {ov['pct_left']} | {ov['pct_straight']} | {ov['pct_right']} |")

    # per-recording D vs B at h0
    _, mB = best(summ, h0, "B_arch_original")
    _, mD = best(summ, h0, "D_enhanced_arch_only")
    R += ["", f"## Part 5 - Per-recording architecture-only, h{h0} (Q6)", "",
          f"| Held-out | B arch-orig ({mB}) | D enhanced ({mD}) | Delta |",
          "|---|---|---|---|"]
    for r in L.RECORDINGS:
        b = fold_df[(fold_df.horizon == h0) & (fold_df.experiment == "B_arch_original") &
                    (fold_df.model == mB) & (fold_df.held_out == r)]["balanced_accuracy"]
        d = fold_df[(fold_df.horizon == h0) & (fold_df.experiment == "D_enhanced_arch_only") &
                    (fold_df.model == mD) & (fold_df.held_out == r)]["balanced_accuracy"]
        bv = b.values[0] if len(b) else float("nan")
        dv = d.values[0] if len(d) else float("nan")
        R.append(f"| {r} | {bv:.3f} | {dv:.3f} | {dv-bv:+.3f} |")

    # answers
    v0, D0, g0 = verdict(h0)
    R += ["", "## Findings (cautious)", "",
          f"**Headline verdict (horizon {h0}): {v0} evidence.**", ""]
    aud_meaning = "have meaningful variance" if len(near_const) <= 2 else "include several low-variance columns"
    R += [f"1. **Do current V3 features have meaningful variance?** Mostly yes - most "
          f"features {aud_meaning}; near-constant: {near_const or 'none'}.",
          f"2. **Saturated / near-constant / missing?** "
          f"{'Saturation noted in clearance features (capped at the 25 m search radius); ' if not saturated.empty else ''}"
          f"missing values are ~0% on the common row set (out-of-bounds rows already dropped).",
          ]
    Bh = best(summ, h0, "B_arch_original")[0]
    R.append(f"3. **Do derived features improve Architecture-Only?** D {D0:.3f} vs B {Bh:.3f} "
             f"(delta {D0-Bh:+.3f}) at h{h0} - {'a small improvement' if D0-Bh>0.005 else 'no meaningful improvement'}.")
    A0 = best(summ, h0, "A_motion_only")[0]
    E0 = best(summ, h0, "E_motion_plus_enhanced_arch")[0]
    R.append(f"4. **Does Motion+Enhanced beat Motion-Only?** E {E0:.3f} vs A {A0:.3f} "
             f"(delta {E0-A0:+.3f}) - {'yes' if E0-A0>0.01 else 'no meaningful gain'}.")
    R.append(f"5. **Is architecture still redundant with motion?** "
             f"{'Largely yes' if E0-A0<=0.01 else 'Less so'} - enhanced architecture adds "
             f"{E0-A0:+.3f} over motion at h{h0}.")
    # Q6 most improved
    deltas = []
    for r in L.RECORDINGS:
        b = fold_df[(fold_df.horizon == h0) & (fold_df.experiment == "B_arch_original") &
                    (fold_df.model == mB) & (fold_df.held_out == r)]["balanced_accuracy"]
        d = fold_df[(fold_df.horizon == h0) & (fold_df.experiment == "D_enhanced_arch_only") &
                    (fold_df.model == mD) & (fold_df.held_out == r)]["balanced_accuracy"]
        if len(b) and len(d):
            deltas.append((r, d.values[0]-b.values[0]))
    deltas.sort(key=lambda t: t[1], reverse=True)
    if deltas:
        R.append(f"6. **Which recordings improve/degrade most (D vs B)?** "
                 f"Most improved: {deltas[0][0]} ({deltas[0][1]:+.3f}); "
                 f"most degraded: {deltas[-1][0]} ({deltas[-1][1]:+.3f}).")
    R.append(f"7. **Is Phase 4 retraining now justified?** {phase4(v0)}")

    R += ["", "## Decision rubric & recommendation", "",
          "- Proceed to Phase 4 if: enhanced-arch-only >= 0.40 OR motion+enhanced beats motion by > 0.03.",
          "- Moderate: enhanced-arch-only >= 0.36 OR +0.01..0.03 over motion.",
          "- Weak: enhanced-arch-only ~0.33-0.35 AND no gain over motion.",
          f"- Observed (h{h0}): enhanced-arch-only D = {D0:.4f}, motion+enhanced minus motion = {g0:+.4f}.",
          "", f"**Recommendation: {recommendation(v0, g0, D0)}**", "",
          "## Caveats", "",
          "- LORO across 5 geometrically distinct sites is a deliberately hard test.",
          "- `turn_rate_v3` makes Motion-Only a strong autoregressive baseline.",
          "- Clearance features saturate at the 25 m ray-march cap; derived ratios",
          "  using them inherit that ceiling.",
          "- Many derived features are strongly correlated with their parents (see",
          "  correlation_matrix.png), so they add limited independent information.",
          "- Diagnostic probe, not tuned performance.", "",
          "## Outputs", "",
          f"- reports/Phase3C_Feature_Engineering_Audit_Report.md",
          "- tables/: feature_distribution_summary, derived_feature_formulas, "
          "horizon10_metrics, horizon30_metrics, loro_fold_metrics, label_distribution, confusion_matrices",
          "- figures/: histograms, boxplots, correlation_matrix, balanced_accuracy_horizon*, "
          "enhanced_vs_original_architecture, per_recording_loro_comparison",
          "- data/: phase3c_dataset_index, rows_used_per_recording", ""]
    with open(os.path.join(OUT, "reports", "Phase3C_Feature_Engineering_Audit_Report.md"), "w") as f:
        f.write("\n".join(R))


def phase4(v):
    return {"STRONG": "Yes - enhanced architecture shows strong, generalizing signal; Phase 4 is justified.",
            "MODERATE": "Qualified - moderate evidence; consider one more feature/representation pass before committing.",
            "WEAK": "Not yet - enhanced architecture is still near chance and non-additive over motion."}[v]


def recommendation(v, gain, D):
    if v == "STRONG":
        return "Proceed to Phase 4 (full Model C retraining)."
    if v == "MODERATE":
        return "Borderline - one more targeted feature/representation pass, then re-decide on Phase 4."
    return ("Do NOT proceed to Phase 4 on current evidence. The architectural signal "
            "is real but weak and redundant with motion even after feature engineering; "
            "the limitation appears representational/behavioural, not a missing-derived-feature gap.")


def main():
    ap = argparse.ArgumentParser(description="Phase 3C feature audit + re-test")
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--horizons", type=int, nargs="+", default=[10, 30])
    ap.add_argument("--turn-threshold-deg", type=float, default=15.0)
    ap.add_argument("--classifier", action="append",
                    choices=["logistic", "random_forest"], default=None)
    args = ap.parse_args()
    if not args.run_all:
        ap.error("pass --run-all")
    models = args.classifier or ["logistic", "random_forest"]
    rng = np.random.default_rng(L.SEED)
    ensure_dirs()

    # derived formulas table
    pd.DataFrame([{"feature": n, "formula": f, "clip": c or ""}
                  for n, f, c in L.DERIVED_SPEC]).to_csv(
        os.path.join(OUT, "tables", "derived_feature_formulas.csv"), index=False)

    all_folds, all_cms, used_by_h, totals_ref = [], {}, {}, None
    for h in args.horizons:
        print(f"[horizon {h}] assembling + classifying...")
        used, totals, fdf, cms = run_horizon(h, args.turn_threshold_deg, models, rng)
        used_by_h[h] = used
        totals_ref = totals
        all_folds.append(fdf)
        all_cms[str(h)] = cms

    fold_df = pd.concat(all_folds, ignore_index=True)
    fold_df.to_csv(os.path.join(OUT, "tables", "loro_fold_metrics.csv"), index=False)
    summ = summarize(fold_df)
    for h in args.horizons:
        summ[summ.horizon == h].to_csv(
            os.path.join(OUT, "tables", f"horizon{h}_metrics.csv"), index=False)
    with open(os.path.join(OUT, "tables", "confusion_matrices.json"), "w") as f:
        json.dump(all_cms, f, indent=2)

    # audit on the first horizon's row set
    audit_df = feature_audit(used_by_h[args.horizons[0]])
    audit_figures(used_by_h[args.horizons[0]])
    dist_df = label_dist(used_by_h)

    # dataset index / rows used
    idx = [{"recording": r, "csv": P3.v3_csv(r), "rows_total": totals_ref[r]}
           for r in L.RECORDINGS]
    pd.DataFrame(idx).to_csv(os.path.join(OUT, "data", "phase3c_dataset_index.csv"), index=False)
    ru = []
    for h in args.horizons:
        for r in L.RECORDINGS:
            ru.append({"horizon": h, "recording": r,
                       "rows_used": int((used_by_h[h].recording == r).sum())})
    pd.DataFrame(ru).to_csv(os.path.join(OUT, "data", "rows_used_per_recording.csv"), index=False)

    classifier_figures(summ, fold_df, args.horizons)
    write_report(summ, fold_df, audit_df, dist_df, used_by_h, args.horizons, args)

    # save pooled enhanced models per horizon
    for h in args.horizons:
        used = used_by_h[h]
        for exp in ("D_enhanced_arch_only", "E_motion_plus_enhanced_arch"):
            feats = L.EXPERIMENTS[exp]
            sub = used[np.isfinite(used[feats].to_numpy(float)).all(axis=1)]
            for mdl in models:
                try:
                    import joblib
                    m = L.make_model(mdl)
                    X = sub[feats].to_numpy(float); y = sub["turn_label"].to_numpy()
                    if mdl == "random_forest":
                        X, y = L.stratified_cap(X, y, L.RF_CAP, rng)
                    m.fit(X, y)
                    joblib.dump(m, os.path.join(OUT, "models", f"phase3c_h{h}_{exp}_{mdl}.joblib"))
                except Exception as e:
                    print(f"  [warn] save {exp}/{mdl}: {e}")

    print("\n[done] outputs in", OUT)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
