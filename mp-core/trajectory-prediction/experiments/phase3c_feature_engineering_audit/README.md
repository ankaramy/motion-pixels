# Phase 3C — V3 Feature-Engineering Audit + Signal Re-test

Diagnostic experiment testing whether the V3 architectural **features** (not the
masks — those are accepted) are expressive enough to reveal turn-direction
signal. The mask/alignment stages are fixed; here we audit feature
distributions, add richer derived features, and re-run the Phase 3 classifiers.

Modifies no mask, encoder, encoded output, dataset, or model. Does **not** touch
Phase 3 / 3B outputs (separate output directory).

## Three parts

1. **Feature distribution audit** — per recording + overall stats for the 12
   original V3 architectural features (count, missing %, min/p01/p05/median/mean/
   p95/p99/max, std, unique rounded values, near-constant and saturation
   warnings) + histograms, boxplots-by-recording, correlation matrix.
2. **Derived features** — 14 documented derived architectural features
   (corridor width, side clearances, bias/affordance ratios, pressures,
   normalized distances, openness, ahead-pressures). Formulas are written to
   `tables/derived_feature_formulas.csv` and the report. ε = 1e-6; ratio /
   pressure features clipped to [0, 10].
3. **Classifier re-run** — Leave-One-Recording-Out, Logistic + Random Forest,
   balanced weights, balanced accuracy primary, at horizons 10 and 30:
   - **A** Motion only
   - **B** Original V3 architecture only
   - **C** Motion + original V3 architecture
   - **D** Enhanced architecture only (original + derived)
   - **E** Motion + enhanced architecture
   - **OLD** Old spatial only (baseline)

## Run

```
python experiments\phase3c_feature_engineering_audit\run_phase3c_feature_audit.py --run-all
```
Options: `--horizons 10 30`, `--turn-threshold-deg 15`,
`--classifier logistic` / `--classifier random_forest`. Run from
`mp-core\trajectory-prediction`.

## Outputs → `MotionPixels_Thesis_Outputs\10_phase3c_feature_engineering_audit\`

- `reports/Phase3C_Feature_Engineering_Audit_Report.md`
- `tables/`: `feature_distribution_summary.csv`, `derived_feature_formulas.csv`,
  `horizon10_metrics.csv`, `horizon30_metrics.csv`, `loro_fold_metrics.csv`,
  `label_distribution.csv`, `confusion_matrices.json`
- `figures/`: `feature_histograms/`, `feature_boxplots_by_recording/`,
  `correlation_matrix.png`, `balanced_accuracy_horizon10.png`,
  `balanced_accuracy_horizon30.png`, `enhanced_vs_original_architecture.png`,
  `per_recording_loro_comparison.png`
- `data/`: `phase3c_dataset_index.csv`, `rows_used_per_recording.csv`
- `models/`: pooled enhanced reference classifiers per horizon

## Decision rubric (architecture)

- **Proceed to Phase 4** if enhanced-arch-only ≥ 0.40 OR motion+enhanced beats
  motion-only by > 0.03 balanced accuracy.
- **Moderate**: enhanced-arch-only ≥ 0.36 OR +0.01–0.03 over motion.
- **Weak**: enhanced-arch-only ~0.33–0.35 AND no gain over motion.

The report answers, cautiously: feature variance/saturation, whether derived
features help, whether enhanced architecture adds over motion, which recordings
move most, and whether Phase 4 Model C retraining is justified.
