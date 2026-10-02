# Phase 3D — Decision-Point Architectural Signal Test

Phases 3A/3B/3C converged: architecture-only turn signal is weak (~0.34–0.35
balanced accuracy) and non-additive over motion, frame-wide, at both horizons,
even after feature engineering.

This phase changes the **unit of analysis**. Instead of every frame, it asks:
*does architecture become informative specifically at spatial decision moments?*

Analysis only — no mask, encoder, encoded output, dataset, or model is modified;
Phase 3A/B/C outputs are untouched.

## Decision-point subsets (a frame qualifies if ANY is true)

- **A near obstacle** — `dist_to_obstacle_v3_m < 3 m`
- **B near boundary** — `dist_to_walkable_boundary_v3_m < 3 m`
- **C high asymmetry** — `|clearance_asymmetry_v3| >` 75th percentile (data-driven, documented in report)
- **D constrained corridor** — `clearance_left + clearance_right <` 25th percentile (data-driven)
- **E approaching turn** — `|future heading change over horizon| > 30°` (future label used **only to filter**, not as a feature)
- **DecisionUnion** — any of A–E.

## Method

Same as Phase 3: turn label LEFT/STRAIGHT/RIGHT (15° threshold), feature sets
Motion / Architecture / Motion+Architecture, Logistic + Random Forest, balanced
class weights, **Leave-One-Recording-Out**, balanced accuracy primary. Classifiers
run on FULL, DecisionUnion, and each subset A–E. Folds with <200 test rows or <2
classes are skipped and reported.

## Run

```
python experiments\phase3d_decision_points\run_phase3d.py --run-all
```
Options: `--horizon 10`, `--turn-threshold-deg 15`, `--classifier ...`.
Run from `mp-core\trajectory-prediction`.

## Outputs → `MotionPixels_Thesis_Outputs\10_phase3d_decision_points\`

- `reports/Phase3D_Decision_Points_Report.md`
- `tables/`: `table1_rows_retained`, `table2_full_vs_union`,
  `table3_architecture_by_subset`, `table4_motion_arch_gain`, `summary_metrics`,
  `loro_fold_metrics`, `label_distribution`
- `figures/`: `architecture_by_subset`, `motion_vs_motion_arch_by_subset`,
  `full_vs_decisionunion`
- `data/`: `phase3d_dataset_index`, `rows_used_per_recording`

## Conclusion (the report picks exactly one)

- **A** Architecture matters primarily at decision points.
- **B** Architecture contains weak signal but remains secondary to motion.
- **C** Architecture contributes no measurable predictive value at current dataset scale.

Success criteria (within DecisionUnion): STRONG = arch-only ≥ 0.40 OR
motion+arch beats motion by > 0.03; MODERATE = arch-only ≥ 0.36 OR gain 0.01–0.03;
WEAK = arch ~0.33–0.35 and no gain.
