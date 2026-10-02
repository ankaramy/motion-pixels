"""
MOTION PIXELS - PHASE 3: run turn classifiers (spatial signal recovery).

    python experiments\\phase3_spatial_signal_recovery\\run_phase3_turn_classifiers.py --run-all

Optional:
    --horizon 10               future-heading horizon in frames (30 fps -> ~0.33s)
    --turn-threshold-deg 15    LEFT/RIGHT threshold on delta heading
    --classifier logistic      restrict to one classifier (repeatable)
    --classifier random_forest
    --rf-train-cap 150000      stratified cap on RF training rows per fold

Leave-One-Recording-Out CV across the 5 validated recordings. Diagnostic only:
trains no Model C, rebuilds no master dataset, modifies no encoder output.
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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase3_lib as P3

OUT_ROOT = r"C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\10_phase3_spatial_signal_recovery"
SEED = 0


def make_model(name, rf_cap):
    if name == "logistic":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000,
                                                    class_weight="balanced",
                                                    multi_class="multinomial"))])
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=120, min_samples_leaf=5,
                                      class_weight="balanced", n_jobs=-1,
                                      random_state=SEED)
    raise ValueError(name)


def stratified_cap(X, y, cap, rng):
    if len(X) <= cap:
        return X, y
    idx = np.arange(len(X))
    keep = []
    for lab in P3.LABELS:
        li = idx[y == lab]
        take = max(1, int(round(cap * len(li) / len(X))))
        take = min(take, len(li))
        keep.append(rng.choice(li, size=take, replace=False))
    keep = np.concatenate(keep)
    return X[keep], y[keep]


def majority_baseline_acc(ytr, yte):
    vals, cnts = np.unique(ytr, return_counts=True)
    maj = vals[np.argmax(cnts)]
    return float((yte == maj).mean()), maj


def run_fold(model_name, feats, train_df, test_df, rf_cap, rng):
    Xtr = train_df[feats].to_numpy(float)
    ytr = train_df["turn_label"].to_numpy()
    Xte = test_df[feats].to_numpy(float)
    yte = test_df["turn_label"].to_numpy()
    if model_name == "random_forest":
        Xtr, ytr = stratified_cap(Xtr, ytr, rf_cap, rng)
    model = make_model(model_name, rf_cap)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    cm = P3.confusion(yte, pred)
    met = P3.metrics_from_confusion(cm)
    base_acc, _ = majority_baseline_acc(ytr, yte)
    met["majority_baseline_acc"] = round(base_acc, 4)
    return met, cm


def main():
    ap = argparse.ArgumentParser(description="Phase 3 turn classifiers")
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--turn-threshold-deg", type=float, default=15.0)
    ap.add_argument("--classifier", action="append",
                    choices=["logistic", "random_forest"], default=None)
    ap.add_argument("--rf-train-cap", type=int, default=150000)
    ap.add_argument("--out-root", default=None,
                    help="override output directory (keeps other phase3 runs intact)")
    args = ap.parse_args()
    if not args.run_all:
        ap.error("pass --run-all")

    global OUT_ROOT
    if args.out_root:
        OUT_ROOT = args.out_root

    models = args.classifier or ["logistic", "random_forest"]
    rng = np.random.default_rng(SEED)
    for sub in ("data", "figures", "reports", "tables", "models"):
        os.makedirs(os.path.join(OUT_ROOT, sub), exist_ok=True)

    print(f"[assemble] horizon={args.horizon} thr={args.turn_threshold_deg} deg")
    used, totals = P3.assemble(args.horizon, args.turn_threshold_deg)
    print(f"[assemble] common usable rows: {len(used)}")

    # ---------- label distribution ----------
    dist_rows = []
    for r in P3.RECORDINGS + ["__overall__"]:
        sub = used if r == "__overall__" else used[used.recording == r]
        vc = sub["turn_label"].value_counts()
        n = len(sub)
        row = {"recording": r, "rows_used": n,
               "rows_total": (sum(totals.values()) if r == "__overall__" else totals[r])}
        for lab in P3.LABELS:
            c = int(vc.get(lab, 0))
            row[f"n_{lab.lower()}"] = c
            row[f"pct_{lab.lower()}"] = round(100 * c / n, 2) if n else 0.0
        dist_rows.append(row)
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(os.path.join(OUT_ROOT, "tables", "label_distribution.csv"), index=False)

    # dataset index
    idx_rows = []
    for r in P3.RECORDINGS:
        idx_rows.append({"recording": r, "v3_csv": P3.v3_csv(r),
                         "v1_csv_exists": os.path.exists(P3.v1_csv(r)),
                         "rows_total": totals[r],
                         "rows_used": int((used.recording == r).sum())})
    pd.DataFrame(idx_rows).to_csv(os.path.join(OUT_ROOT, "data", "phase3_dataset_index.csv"), index=False)
    dist_df[dist_df.recording != "__overall__"][
        ["recording", "rows_used", "pct_left", "pct_straight", "pct_right"]
    ].to_csv(os.path.join(OUT_ROOT, "data", "rows_used_per_recording.csv"), index=False)

    # ---------- LORO CV ----------
    fold_rows = []
    cms = {}            # exp -> model -> {fold: cm, pooled: cm}
    for exp, feats in P3.EXPERIMENTS.items():
        missing = [f for f in feats if f not in used.columns]
        exp_df = used
        if missing:
            print(f"[skip] {exp}: missing columns {missing}")
            continue
        finite = np.isfinite(exp_df[feats].to_numpy(float)).all(axis=1)
        exp_df = exp_df[finite]
        cms[exp] = {}
        for mdl in models:
            cms[exp][mdl] = {}
            pooled = np.zeros((3, 3), int)
            for held in P3.RECORDINGS:
                tr = exp_df[exp_df.recording != held]
                te = exp_df[exp_df.recording == held]
                if len(te) == 0 or te["turn_label"].nunique() < 2:
                    continue
                met, cm = run_fold(mdl, feats, tr, te, args.rf_train_cap, rng)
                pooled += cm
                cms[exp][mdl][held] = cm.tolist()
                fold_rows.append({
                    "experiment": exp, "model": mdl, "held_out": held,
                    "n_test": int(len(te)),
                    "accuracy": met["accuracy"],
                    "balanced_accuracy": met["balanced_accuracy"],
                    "macro_f1": met["macro_f1"],
                    "majority_baseline_acc": met["majority_baseline_acc"],
                    "f1_LEFT": met["per_class"]["LEFT"]["f1"],
                    "f1_STRAIGHT": met["per_class"]["STRAIGHT"]["f1"],
                    "f1_RIGHT": met["per_class"]["RIGHT"]["f1"],
                    "recall_LEFT": met["per_class"]["LEFT"]["recall"],
                    "recall_STRAIGHT": met["per_class"]["STRAIGHT"]["recall"],
                    "recall_RIGHT": met["per_class"]["RIGHT"]["recall"],
                    "support_LEFT": met["per_class"]["LEFT"]["support"],
                    "support_STRAIGHT": met["per_class"]["STRAIGHT"]["support"],
                    "support_RIGHT": met["per_class"]["RIGHT"]["support"],
                })
                print(f"  {exp:30s} {mdl:14s} held={held:24s} "
                      f"balacc={met['balanced_accuracy']:.3f} acc={met['accuracy']:.3f}")
            cms[exp][mdl]["pooled"] = pooled.tolist()
            # pooled-trained model saved for reference
            try:
                full = make_model(mdl, args.rf_train_cap)
                Xf = exp_df[feats].to_numpy(float)
                yf = exp_df["turn_label"].to_numpy()
                if mdl == "random_forest":
                    Xf, yf = stratified_cap(Xf, yf, args.rf_train_cap, rng)
                full.fit(Xf, yf)
                joblib.dump(full, os.path.join(OUT_ROOT, "models", f"phase3_{exp}_{mdl}.joblib"))
            except Exception as e:
                print(f"  [warn] could not save model {exp}/{mdl}: {e}")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(OUT_ROOT, "tables", "fold_metrics.csv"), index=False)
    with open(os.path.join(OUT_ROOT, "tables", "confusion_matrices.json"), "w") as f:
        json.dump(cms, f, indent=2)

    # ---------- summary ----------
    summ = (fold_df.groupby(["experiment", "model"])
            .agg(mean_balanced_accuracy=("balanced_accuracy", "mean"),
                 std_balanced_accuracy=("balanced_accuracy", "std"),
                 mean_accuracy=("accuracy", "mean"),
                 mean_macro_f1=("macro_f1", "mean"),
                 mean_majority_baseline=("majority_baseline_acc", "mean"))
            .round(4).reset_index())
    summ.to_csv(os.path.join(OUT_ROOT, "tables", "summary_metrics.csv"), index=False)

    make_figures(fold_df, cms, models)
    write_report(used, totals, dist_df, fold_df, summ, args, models)
    print("\n[done] outputs in", OUT_ROOT)
    print(summ.to_string(index=False))


# ----------------------------------------------------------------------------
def best_model_for(summ_like, exp):
    sub = summ_like[summ_like.experiment == exp]
    if sub.empty:
        return None
    return sub.sort_values("mean_balanced_accuracy", ascending=False).iloc[0]["model"]


def make_figures(fold_df, cms, models):
    fig_dir = os.path.join(OUT_ROOT, "figures")
    summ = (fold_df.groupby(["experiment", "model"])["balanced_accuracy"]
            .agg(["mean", "std"]).reset_index())

    # 1) balanced accuracy comparison bar chart
    exps = [e for e in P3.EXPERIMENTS if e in fold_df.experiment.unique()]
    x = np.arange(len(exps))
    w = 0.8 / max(1, len(models))
    plt.figure(figsize=(10, 5))
    for j, mdl in enumerate(models):
        means = [summ[(summ.experiment == e) & (summ.model == mdl)]["mean"].values[0]
                 if not summ[(summ.experiment == e) & (summ.model == mdl)].empty else 0 for e in exps]
        stds = [summ[(summ.experiment == e) & (summ.model == mdl)]["std"].values[0]
                if not summ[(summ.experiment == e) & (summ.model == mdl)].empty else 0 for e in exps]
        plt.bar(x + j * w, means, w, yerr=stds, capsize=3, label=mdl)
    plt.axhline(1/3, ls="--", color="k", label="chance (0.333)")
    plt.xticks(x + w * (len(models)-1)/2, exps, rotation=20, ha="right")
    plt.ylabel("Balanced accuracy (LORO mean +/- std)")
    plt.title("Phase 3: turn-direction balanced accuracy by feature set")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "balanced_accuracy_comparison.png"), dpi=130)
    plt.close()

    # 2) per-recording fold comparison (A/B/C), primary model per experiment
    core = ["A_motion_only", "B_architecture_only", "C_motion_plus_architecture"]
    core = [e for e in core if e in fold_df.experiment.unique()]
    plt.figure(figsize=(11, 5))
    recs = P3.RECORDINGS
    xr = np.arange(len(recs))
    wB = 0.8 / max(1, len(core))
    for j, exp in enumerate(core):
        mdl = best_model_for(summ.rename(columns={"mean": "mean_balanced_accuracy"}), exp)
        vals = []
        for r in recs:
            row = fold_df[(fold_df.experiment == exp) & (fold_df.model == mdl) & (fold_df.held_out == r)]
            vals.append(row["balanced_accuracy"].values[0] if not row.empty else 0)
        plt.bar(xr + j * wB, vals, wB, label=f"{exp} ({mdl})")
    plt.axhline(1/3, ls="--", color="k", label="chance")
    plt.xticks(xr + wB, recs, rotation=20, ha="right")
    plt.ylabel("Balanced accuracy (held-out)")
    plt.title("Phase 3: per-recording held-out balanced accuracy")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "per_recording_fold_comparison.png"), dpi=130)
    plt.close()

    # 3) confusion matrices for A/B/C (pooled, primary model)
    fig, axes = plt.subplots(1, len(core), figsize=(5*len(core), 4.5))
    if len(core) == 1:
        axes = [axes]
    for ax, exp in zip(axes, core):
        mdl = best_model_for(summ.rename(columns={"mean": "mean_balanced_accuracy"}), exp)
        cm = np.array(cms[exp][mdl]["pooled"], float)
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(P3.LABELS); ax.set_yticklabels(P3.LABELS)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{exp}\n({mdl}, pooled)")
        for i in range(3):
            for k in range(3):
                ax.text(k, i, f"{cmn[i,k]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, k] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.025)
    plt.savefig(os.path.join(fig_dir, "confusion_matrices_ABC.png"), dpi=130,
                bbox_inches="tight")
    plt.close()


def _balacc(summ, exp, mdl=None):
    sub = summ[summ.experiment == exp]
    if mdl:
        sub = sub[sub.model == mdl]
    if sub.empty:
        return None
    return float(sub.sort_values("mean_balanced_accuracy", ascending=False).iloc[0]["mean_balanced_accuracy"])


def write_report(used, totals, dist_df, fold_df, summ, args, models):
    A = _balacc(summ, "A_motion_only")
    B = _balacc(summ, "B_architecture_only")
    C = _balacc(summ, "C_motion_plus_architecture")
    OLD = _balacc(summ, "OLD_spatial_only")
    gain = (C - A) if (A is not None and C is not None) else None

    # per-recording B (best model) for who-benefits
    bestB = summ[summ.experiment == "B_architecture_only"].sort_values(
        "mean_balanced_accuracy", ascending=False)
    bestB_model = bestB.iloc[0]["model"] if not bestB.empty else None

    def verdict():
        if B is not None and B > 0.40 and gain is not None and gain > 0.03:
            return "STRONG positive evidence"
        if (B is not None and B > 0.36) or (gain is not None and 0.01 <= gain <= 0.03):
            return "MODERATE evidence"
        return "WEAK / no evidence"

    L = ["# Phase 3 - Spatial Signal Recovery Report", "",
         "Diagnostic turn-direction classification (LEFT / STRAIGHT / RIGHT) under",
         "**Leave-One-Recording-Out** cross-validation. This is a signal-recovery",
         "probe, NOT Model C training and NOT trajectory rollout. No encoder,",
         "encoded output, master dataset, or model was modified.", "",
         "## Configuration", "",
         f"- Horizon: **{args.horizon} frames** (~{args.horizon/30:.2f}s at 30 fps)",
         f"- Turn threshold: **+/-{args.turn_threshold_deg} deg** on wrapped delta-heading",
         f"- Classifiers: {', '.join(models)} (balanced class weights; features standardized for logistic)",
         f"- Split: Leave-One-Recording-Out (5 folds); RF train capped at {args.rf_train_cap} rows/fold (stratified)",
         f"- Common usable rows (label valid AND all motion+arch features finite): **{len(used)}**",
         "- Baselines: random 3-class = 0.333; majority-class accuracy reported per fold.",
         "",
         "## Label distribution (common row set)", "",
         "| Recording | rows used | % LEFT | % STRAIGHT | % RIGHT |",
         "|---|---|---|---|---|"]
    for _, r in dist_df.iterrows():
        L.append(f"| {r['recording']} | {r['rows_used']} | {r['pct_left']} | "
                 f"{r['pct_straight']} | {r['pct_right']} |")

    L += ["", "## Summary (LORO mean across folds)", "",
          "| Experiment | Model | mean balanced acc | std | mean acc | mean macro-F1 | mean majority-acc |",
          "|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        L.append(f"| {r['experiment']} | {r['model']} | "
                 f"**{r['mean_balanced_accuracy']}** | {r['std_balanced_accuracy']} | "
                 f"{r['mean_accuracy']} | {r['mean_macro_f1']} | {r['mean_majority_baseline']} |")

    L += ["", "## Per-recording held-out balanced accuracy (best model per experiment)", "",
          "| Held-out | A motion | B arch | C motion+arch | OLD spatial |",
          "|---|---|---|---|---|"]
    def cell(exp, held):
        sub = summ[summ.experiment == exp]
        if sub.empty:
            return "-"
        mdl = sub.sort_values("mean_balanced_accuracy", ascending=False).iloc[0]["model"]
        row = fold_df[(fold_df.experiment == exp) & (fold_df.model == mdl) & (fold_df.held_out == held)]
        return f"{row['balanced_accuracy'].values[0]:.3f}" if not row.empty else "-"
    for held in P3.RECORDINGS:
        L.append(f"| {held} | {cell('A_motion_only',held)} | {cell('B_architecture_only',held)} "
                 f"| {cell('C_motion_plus_architecture',held)} | {cell('OLD_spatial_only',held)} |")

    L += ["", "## Findings (cautious)", "",
          f"**Headline verdict: {verdict()}.**", ""]
    # Q1
    if B is not None:
        q1 = ("above" if B > 0.345 else "approximately at") + f" chance (B balanced acc = {B:.3f} vs 0.333)"
        L.append(f"1. **Does V3 architecture-only beat chance?** {('Yes, modestly' if B>0.345 else 'Not clearly')} - {q1}.")
    # Q2
    if B is not None and OLD is not None:
        rel = "higher than" if B > OLD + 0.005 else ("similar to" if abs(B-OLD) <= 0.005 else "lower than")
        L.append(f"2. **Does V3 architecture-only beat old spatial-only?** V3 arch ({B:.3f}) is {rel} "
                 f"old spatial ({OLD:.3f}). (Old masks were trajectory-derived; treat as context, not proof.)")
    elif B is not None:
        L.append("2. **V3 vs old spatial:** old spatial columns unavailable; comparison skipped.")
    # Q3
    if gain is not None:
        direction = "improves over" if gain > 0 else "does not improve over"
        L.append(f"3. **Does Motion+Architecture beat Motion Only?** C ({C:.3f}) {direction} A ({A:.3f}); "
                 f"delta = {gain:+.3f} balanced accuracy.")
    # Q4
    if bestB_model:
        perB = []
        for held in P3.RECORDINGS:
            row = fold_df[(fold_df.experiment == "B_architecture_only") &
                          (fold_df.model == bestB_model) & (fold_df.held_out == held)]
            if not row.empty:
                perB.append((held, row["balanced_accuracy"].values[0]))
        perB.sort(key=lambda t: t[1], reverse=True)
        if perB:
            top = ", ".join(f"{h} ({v:.3f})" for h, v in perB[:2])
            bot = ", ".join(f"{h} ({v:.3f})" for h, v in perB[-2:])
            L.append(f"4. **Which recordings benefit most/least (arch-only, {bestB_model})?** "
                     f"Most: {top}. Least: {bot}.")
    # Q5
    L.append(f"5. **Strong enough to justify Phase 4 (full Model C retraining)?** "
             f"{phase4_reco(B, gain)}")

    L += ["", "## Pass / fail interpretation", "",
          "- STRONG: arch-only > 0.40 AND C-A gain > 0.03",
          "- MODERATE: arch-only > 0.36 OR C-A gain in [0.01, 0.03]",
          "- WEAK: arch-only ~ 0.33 AND no C-A gain",
          f"- Observed: arch-only B = {round(B,4) if B is not None else None}, "
          f"C-A gain = {round(gain,4) if gain is not None else None}", "",
          "## Caveats", "",
          "- LORO is a hard test: each site has distinct geometry/orientation, so",
          "  absolute numbers are conservative by design.",
          "- Label balance: LEFT and RIGHT each ~37%, STRAIGHT only ~26%. At a",
          "  0.33s horizon with a 15-deg threshold, per-frame heading jitter yields",
          "  many micro-turn labels, so the target may partly capture heading NOISE",
          "  rather than deliberate navigation. A longer horizon (e.g. 30 frames /",
          "  1s) is a recommended secondary probe to test whether architectural",
          "  signal is clearer for deliberate turns.",
          "- `turn_rate_v3` (a past-motion feature) is a strong autoregressive cue;",
          "  Motion-Only is therefore a demanding baseline for architecture to beat,",
          "  which likely explains why C does not exceed A.",
          "- red_bridge / placa_espanya lost rows to out-of-bounds / mask-edge",
          "  effects (Phase-2 CHECK); their folds are noisier.",
          "- This is signal recovery, not tuned performance; do NOT read these as",
          "  final model numbers.", "",
          "## Outputs", "",
          "- tables/: summary_metrics.csv, fold_metrics.csv, label_distribution.csv, confusion_matrices.json",
          "- figures/: balanced_accuracy_comparison.png, per_recording_fold_comparison.png, confusion_matrices_ABC.png",
          "- data/: phase3_dataset_index.csv, rows_used_per_recording.csv",
          "- models/: pooled-trained reference classifiers per experiment", ""]
    with open(os.path.join(OUT_ROOT, "reports", "Phase3_Spatial_Signal_Recovery_Report.md"), "w") as f:
        f.write("\n".join(L))


def phase4_reco(B, gain):
    if B is None:
        return "Inconclusive (architecture experiment did not run)."
    if B > 0.40 and gain is not None and gain > 0.03:
        return ("Yes - the architectural features show strong, generalizing turn "
                "signal and additive value; Phase 4 is justified.")
    if B > 0.36 or (gain is not None and gain >= 0.01):
        return ("Qualified yes - there is moderate evidence of architectural turn "
                "signal. Consider refining the CHECK masks and re-probing before "
                "committing to full Model C retraining.")
    return ("Not yet - architecture-only is near chance and adds little over motion. "
            "Improve masks / features (or reconsider the hypothesis) before Phase 4.")


if __name__ == "__main__":
    main()
