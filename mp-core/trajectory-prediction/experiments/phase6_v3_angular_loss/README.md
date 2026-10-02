# Phase 6 — V3 Model C + Angular Loss

The final **objective-level** experiment. Phases 4–5 showed the ~90°
autoregressive angular collapse is not fixed by the encoder (V3) or the
recurrent cell (LSTM↔GRU), and that even *teacher-forced* single-step angular
error is already high. That points to the **training objective** (pure
position-MSE regresses to the mean, near-straight step). This phase adds an
explicit angular term and sweeps its weight.

## Loss

```
total = position_MSE  +  lambda_angle * (1 - cos_sim(pred_disp, true_disp))
```
Cosine similarity is computed on **raw** (unscaled) du/dv displacement vectors,
masked to rows whose true |displacement| > 0.01 m/step so stationary/noisy rows
don't dominate the angular term. Everything else is identical to the Phase-4 V3
LSTM Model C (architecture, features, split, window, optimizer, batch, early
stopping, normalization, rollout). Early stopping uses the **total** objective,
so `lambda=0` reproduces Phase 4.

## Sweep & selection

λ ∈ {0.0, 0.05, 0.1, 0.2, 0.5} (use `--lambdas 0.0 0.1 0.2` for a faster run).
**Best model = lowest validation angular error among λ whose validation ADE
worsens by < 0.05 m vs λ=0.** (Documented; not "lowest ADE".)

## Run

```
python experiments\phase6_v3_angular_loss\run_phase6_angular_loss.py --run-all
python experiments\phase6_v3_angular_loss\run_phase6_angular_loss.py --lambdas 0.0 0.1 0.2
```
Run from `mp-core\trajectory-prediction`.

## Isolation / guardrails

Writes only under `MotionPixels_Thesis_Outputs/13_phase6_v3_angular_loss/`.
MD5-fingerprints the protected files (frozen Model C, Phase-4 LSTM, Phase-5 GRU,
V3 dataset) before and after, and reports they are byte-identical. Code snapshot
saved to `00_code_snapshot/`.

## Outputs → `MotionPixels_Thesis_Outputs\13_phase6_v3_angular_loss\`

- `01_training_runs/` epoch logs per λ
- `02_models/` per-λ checkpoints, scalers, configs
- `03_metrics/` lambda_sweep, per-λ split/per-recording/per-trajectory/teacher-forced,
  A_p4_B_lam0_C_best, teacher_forced_vs_autoregressive_best, teacher_forced_sweep
- `04_figures/` lambda_sweep, teacher_forced_vs_autoregressive_angular,
  rollout_comparison_highturn, rollout_obstacle_rich_catalunya
- `05_reports/Phase6_V3_Angular_Loss_Report.md`

## Decision rubric

- **Strong**: val angular improves > 5° AND test angular improves > 5° AND ADE worsens < 0.05 m.
- **Moderate**: val angular improves 2–5° OR rollouts visibly preserve turns better.
- **Weak/none**: angular improves < 2°, or improves but ADE/FDE collapses.
- If angular improves but ADE worsens → reported as a **tradeoff**, not a failure.

Angular metrics are primary for this phase. Not leaderboard tuning.
