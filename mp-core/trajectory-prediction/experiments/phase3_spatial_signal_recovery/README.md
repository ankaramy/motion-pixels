# Phase 3 — Spatial Signal Recovery (turn classification)

Lightweight **diagnostic** probe: do Encoder V3 architectural features carry
measurable signal for pedestrian turning direction (LEFT / STRAIGHT / RIGHT)?

This is **not** Model C, **not** trajectory rollout, **not** master-dataset
rebuilding. It only reads the V3 (and old V1) encoded CSVs and trains simple
classifiers. It modifies no encoder, encoded output, or model.

## Experiments

- **A — Motion only**: `speed_v3, heading_sin, heading_cos, turn_rate_v3`
- **B — Architecture only**: the 12 V3 architectural features (distances,
  clearances, asymmetry, obstacle/boundary bearings, inside flags)
- **C — Motion + Architecture**: A ∪ B
- **OLD — Spatial only** (secondary): old V1 `dist_to_obstacle/boundary/entrance`

## Method

- **Label**: future heading change over `--horizon` frames; `LEFT` if
  Δheading > +threshold, `RIGHT` if < −threshold, else `STRAIGHT`
  (defaults: horizon 10 frames ≈ 0.33 s at 30 fps; threshold 15°).
- **Split**: **Leave-One-Recording-Out** (the headline result — tests
  cross-site generalization). Random row splits are intentionally *not* used.
- **Models**: multinomial Logistic Regression (standardized) and Random Forest,
  both with balanced class weights. RF training is capped per fold (stratified)
  for speed — this is signal recovery, not leaderboard tuning.
- **Common row set**: A/B/C are evaluated on the *same* rows (label valid and
  all motion+arch features finite), so comparisons are apples-to-apples.
- **Primary metric**: balanced accuracy (vs random 0.333 and majority baseline).

## Run

```
python experiments\phase3_spatial_signal_recovery\run_phase3_turn_classifiers.py --run-all
```
Options: `--horizon 10`, `--turn-threshold-deg 15`,
`--classifier logistic` / `--classifier random_forest` (repeatable),
`--rf-train-cap 150000`. Run from `mp-core\trajectory-prediction`.

## Outputs → `MotionPixels_Thesis_Outputs\10_phase3_spatial_signal_recovery\`

- `reports/Phase3_Spatial_Signal_Recovery_Report.md`
- `tables/`: `summary_metrics.csv`, `fold_metrics.csv`, `label_distribution.csv`,
  `confusion_matrices.json`
- `figures/`: `balanced_accuracy_comparison.png`,
  `per_recording_fold_comparison.png`, `confusion_matrices_ABC.png`
- `data/`: `phase3_dataset_index.csv`, `rows_used_per_recording.csv`
- `models/`: pooled-trained reference classifiers (not for deployment)

## Reading the result

The report answers, cautiously: does arch-only beat chance? beat old spatial?
does motion+arch beat motion-only? which recordings benefit? and — the decision
gate — is the evidence strong enough to justify **Phase 4** (full Model C
retraining)? Do not proceed to Phase 4 on weak evidence.
