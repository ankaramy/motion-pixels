# Phase 5 — V3 GRU vs LSTM Model C

A single, clean architecture check: **swap the LSTM cell for a GRU cell** and
change nothing else. Tests whether the recurrent cell is responsible for the
~90° autoregressive angular collapse seen in Phase 4, or whether that is an
objective/rollout problem.

This is a model/objective diagnostic, **not** an encoder experiment and **not**
leaderboard tuning.

## What is held identical to the Phase-4 V3 LSTM

Features (10-feature Model C: motion + position + V3 spatial), targets
(`target_du`, `target_dv`), recording-level split (train=esplanade/catalunya/
espanya, val=stairs, test=red_bridge), window 10, hidden 128 × 2 layers, dropout
0.2, AdamW lr 1e-3 wd 1e-4, StepLR(25, 0.35), grad-clip 1.0, batch 512, max 80
epochs, patience 15, MSE loss, seed 42, StandardScaler (train-fit), 20-step
autoregressive rollout, per-recording KDTree spatial refresh, and the full
evaluation suite. **Only the cell changes: `nn.LSTM` → `nn.GRU`.**

The frozen recipe symbols (hyperparameters, `ColumnScaler`, `make_windows`,
`rollout`, etc.) are imported read-only from `run_bridge_ablation.py`; the only
new class is `TrajectoryGRU`.

## Isolation / guardrails

Writes only under `MotionPixels_Thesis_Outputs/12_phase5_v3_gru_comparison/`.
Before and after the run it MD5-fingerprints the protected files (frozen Model C,
Phase-4 LSTM checkpoint + metrics, V3 dataset, build script) and reports that
they are byte-identical. A code snapshot is copied into `00_code_snapshot/`.

## Run

```
python experiments\phase5_v3_gru_comparison\run_phase5_gru_comparison.py --run-all
# or
python ...\run_phase5_gru_comparison.py --train-only
python ...\run_phase5_gru_comparison.py --evaluate-only
```
Run from `mp-core\trajectory-prediction`.

## Outputs → `MotionPixels_Thesis_Outputs\12_phase5_v3_gru_comparison\`

- `01_training_run/`: epoch_log.csv, loss_curve.png, feature_columns.json, training_summary.json
- `02_models/`: best_gru_model.pth, final_gru_model.pth, scalers.pkl, training_config.json
- `03_metrics/`: metrics_summary, per_recording_metrics, per_trajectory_metrics, error_by_horizon,
  teacher_forced_metrics, v3_lstm_vs_gru_by_split, v3_lstm_vs_gru_by_recording,
  teacher_forced_vs_autoregressive, angular_error_comparison
- `04_figures/`: lstm_vs_gru_ade_fde, lstm_vs_gru_angular,
  teacher_forced_vs_autoregressive_angular, rollout_comparison_test,
  rollout_obstacle_rich_catalunya
- `05_reports/Phase5_V3_GRU_Comparison_Report.md`

## Decision rubric (on the test split)

- **Strong**: test ADE improves > 0.03 m OR angular improves > 5°.
- **Moderate**: ADE improves 0.01–0.03 m OR angular improves 2–5°.
- **Weak/none**: ADE change < 0.01 m AND angular change < 2°  → the cell is not
  the bottleneck; recommend an **objective-level** fix (auxiliary heading loss,
  displacement+angle loss, scheduled sampling, multi-step rollout loss) — not
  another recurrent cell. (Those are not implemented in this phase.)
