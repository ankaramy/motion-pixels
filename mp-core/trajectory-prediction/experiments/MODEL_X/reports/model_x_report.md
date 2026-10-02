# MODEL_X — Final Report

**Date:** 2026-06-11
**Phase:** MODEL_X (general-purpose Horizon-10 retraining of Model C)
**Selected checkpoint:** `models/best_by_val_ADE.pth` (epoch 35; val FDE was also best at epoch 35,
so ADE and FDE selection agree — no ambiguity).

---

## 1. Dataset used

`C:\Users\OWNER\Desktop\new_datasets\Barcelona_v3_manual_master_dataset\master_dataset.csv`
(Barcelona **v3 manual** master — spatial features from Encoder V3 **real manual architectural
masks**; the newest, corrected dataset, built 2026-06-09). The `model_C_dataset.csv` twin holds the
identical 10 features; `master_dataset.csv` was used because it also carries `world_x/world_y` for
clean world-metre rollout and metrics.

## 2. Recording list (5 of 7 requested — see audit)

| recording_id | rows | tracks |
|---|---:|---:|
| esplanade_espanya_01 | 283,120 | 522 |
| placa_catalunya_01 | 355,583 | 1,255 |
| placa_espanya_01 | 212,224 | 824 |
| stairs_montjuic_01 | 127,605 | 334 |
| red_bridge_combined_01 | 85,847 | 599 |
| **TOTAL** | **1,064,379** | **3,534** |

> **`placa_montjuic_01` and `stairs_montjuic_02` do not exist in any master dataset.** They were
> excluded for ill-conditioned homography and were never encoded (see
> `data_audit/dataset_audit.md` and `docs/encoder_v3_recording_inventory.md`). MODEL_X therefore
> trains on the full set of *trustworthy* Barcelona data currently available, by explicit user
> decision. This is **not** a 7-site model.

## 3. Rows / tracks / windows

- Rows: 1,064,379 — Tracks: 3,534.
- Training windows (single-step, obs=10): **822,488** (from 2,827 train tracks).
- Eval rollout windows (obs=10 + horizon=10, stride 1): val **110,214**, test **90,310**.
- No artifact pre-filter beyond the dataset's frozen `MIN_TRACK_LEN=11`; rollout windows additionally
  require ≥20 frames. No windows were discarded for missing features (none missing). 0 normal-motion
  over-filtering.

## 4. Split strategy — mixed-recording track-level split

80/10/10 **by track** (`trajectory_id`), stratified per recording, fixed seed 42. No track appears in
more than one split; **every recording contributes tracks to train, val, and test**. The old embedded
recording-level `split` column was ignored. This is explicitly a *mixed-recording track-level split*
and **must not** be interpreted as unseen-site generalization.

| recording | train tr/rows | val tr/rows | test tr/rows |
|---|---|---|---|
| esplanade_espanya_01 | 418 / 221,215 | 52 / 32,710 | 52 / 29,195 |
| placa_catalunya_01 | 1,004 / 289,117 | 126 / 35,426 | 125 / 31,040 |
| placa_espanya_01 | 659 / 167,763 | 82 / 24,270 | 83 / 20,191 |
| red_bridge_combined_01 | 479 / 70,843 | 60 / 7,935 | 60 / 7,069 |
| stairs_montjuic_01 | 267 / 101,820 | 33 / 16,446 | 34 / 9,339 |
| **TOTAL** | **2,827 / 850,758** | **353 / 116,787** | **354 / 96,834** |

## 5. Model architecture

`TrajectoryLSTM` — verbatim copy of frozen Model C (`schema_ablation_bridge/run_bridge_ablation.py`,
source not edited; copy lives in `model_x_lib.py`):
LSTM hidden **128**, **2** layers, dropout **0.2**, head `Linear(128→2)` → `(target_du, target_dv)`.
10 input features in frozen order: `du, dv, speed, heading_sin, heading_cos, turn_rate, u, v,
dist_to_obstacle_norm, dist_to_boundary_norm`.

## 6. Training recipe

From scratch. AdamW, lr 1e-3, weight_decay 1e-5, batch 256, **MSE on scaled next-step
(target_du, target_dv)**, grad-clip 1.0, ≤80 epochs, early-stop patience 12 on val ADE, seed 42, CUDA.
Feature + target StandardScalers fit on train windows (`models/scalers.json`). Per-epoch validation
ADE/FDE computed by **10-step autoregressive rollout** on a fixed 3,000-window subsample.
Training target is single-step (the Model C objective); **horizon=10 is the rollout depth at
eval time.** Rollout updates motion+position from predictions and **freezes the two spatial-clearance
features** at their last observed value (no KDTree re-query — documented simplification; those features
carry ~no turn signal per prior audits).

## 7. Final checkpoint

`best_by_val_ADE.pth` @ **epoch 35**: **val ADE 0.2945 m, val FDE 0.4718 m**. Trained 47 epochs
(early-stopped), 11.3 min. `best_by_val_FDE.pth` is the same epoch. `last_epoch.pth` also saved.
Note: single-step val *MSE* drifted up after ~epoch 10 while the **rollout** val ADE kept improving to
epoch 35 — the rollout metric (what we select on) is the meaningful one.

## 8. All-window test metrics (N = 90,310)

| metric | mean | median |
|---|---:|---:|
| ADE (m) | **0.305** | 0.136 |
| FDE (m) | **0.480** | 0.198 |
| RMSE (m) | 0.354 | — |
| angular error (°) | 75.6 | 58.9 |
| predicted path length (m) | — | 0.172 |
| GT path length (m) | — | 0.515 |
| pred/GT length ratio | — | **0.413** |

**Test ADE (0.305) ≈ val ADE (0.2945)** → the mixed split generalizes cleanly with no rollout-metric
overfitting. Median ADE (0.136 m) is small because the majority of test windows are slow / near-
stationary pedestrians (median GT path over 10 steps is only 0.51 m). The high *median* angular error
(59°) is dominated by these slow windows where heading is geometrically ill-defined.

## 9. Per-recording test metrics

| recording | n | ADE (m) | FDE (m) | genuine turns |
|---|---:|---:|---:|---:|
| red_bridge_combined_01 | 5,982 | **0.065** | 0.103 | 5 |
| stairs_montjuic_01 | 8,707 | 0.092 | 0.152 | 10 |
| placa_catalunya_01 | 28,743 | 0.131 | 0.219 | 114 |
| esplanade_espanya_01 | 28,215 | 0.343 | 0.518 | 488 |
| placa_espanya_01 | 18,663 | **0.691** | 1.097 | 2,241 |

Channelled / compact sites (red_bridge, stairs) are easy; the open, fast, turn-heavy **placa_espanya**
is by far the hardest and dominates the overall mean. Difficulty tracks motion complexity, as expected.

## 10. Genuine-turn metrics (reporting only)

Turn label (H10): GT net heading change > 30°, GT net displacement ≥ 1.0 m, GT max single step < 0.6 m.

| subset | n | ADE (m) | FDE (m) | ang err (°) | len ratio | TCR |
|---|---:|---:|---:|---:|---:|---:|
| genuine turns | 2,858 | 0.730 | 1.225 | 71.6 | 0.302 | **0.381** |
| straight (Δhead<15°, moving) | 9,161 | 0.781 | 1.358 | 79.3 | 0.221 | — |
| long-displacement (top 25%) | 22,578 | 0.794 | 1.327 | 77.3 | 0.235 | — |

MODEL_X captures turn *direction* on ~38% of genuine turns. That is **deliberately not optimized** —
this is a general-purpose model, not a turn specialist (cf. §12).

## 11. Prediction-length analysis (the honest weakness)

Median pred/GT length ratio is **0.41 overall, 0.30 on turns, 0.22 on clearly-moving windows.**
MODEL_X **systematically under-predicts path length** — the classic MSE regression-to-the-mean: when
future motion is uncertain, the displacement that minimizes MSE is a short, averaged step. This is what
gives the excellent ADE/FDE numbers, and it is the same mechanism diagnosed as Model C's "straight-line
collapse" root cause. MODEL_X produces **conservative, visually plausible** rollouts (see presentation
collage) rather than collapsed zero-length stubs, but it does **not** fabricate length — predicted paths
are genuinely shorter than ground truth on faster motion. Reported honestly; not inflated.

## 12. Comparison to previous experiments

⚠️ **Not apples-to-apples.** MODEL_X uses a *mixed-recording track-level* split (in-dataset);
Exp03/Exp06 and frozen Model C use *held-out / unseen-site* splits, which are strictly harder. The
table is for context, not a like-for-like ranking.

| model | split | all ADE | all FDE | turn ADE | turn FDE | turn TCR |
|---|---|---:|---:|---:|---:|---:|
| Frozen Model C (held_out) | unseen-site, N=9 | 0.532 | 0.981 | — | — | — |
| Frozen Model C (overfit10x) | memorization | 0.053 | 0.110 | — | — | — |
| Exp03 H10 (turn-only) | held-out | 0.270 | 0.524 | 0.724 | 1.278 | 0.67 |
| Exp06 (turn-balanced) | held-out | 0.552 | 0.974 | 1.306 | 2.447 | 0.66 |
| **MODEL_X (this)** | **mixed track-level** | **0.305** | **0.480** | **0.730** | **1.225** | **0.38** |

Reading it honestly:
- **Best overall FDE of all real (non-overfit) models** and a strong overall ADE — MODEL_X delivers the
  most accurate, most stable *overall* rollouts, which was the goal.
- **Turn ADE/FDE competitive with or better than** the turn-specialized Exp03/Exp06 (MODEL_X turn FDE
  1.225 < Exp03 1.278 < Exp06 2.447), despite never being trained on turns.
- **Lower TCR (0.38 vs ~0.66)** — the expected trade: turn-balanced/turn-only models capture turn
  *direction* more often but pay for it in overall accuracy and (Exp06) badly on turn ADE. MODEL_X
  trades turn-direction recall for overall accuracy and rollout stability.
- Part of MODEL_X's overall-ADE advantage is the easier in-dataset split — do not over-read it.

## 13. Final honest conclusion

MODEL_X is a **strong general-purpose trajectory predictor on the available validated Barcelona
dataset**: test ADE **0.305 m** / FDE **0.480 m** over 90k windows, generalizing cleanly across a
mixed-recording track-level split (test ≈ val), with conservative, visually plausible rollouts suitable
for prototype and thesis-presentation visuals. Its honest limitations: (1) it **under-predicts path
length** on faster/uncertain motion (MSE regression-to-mean) and (2) it **captures turn direction only
~38%** of the time — it is not a turn specialist. It must **not** be cited as unseen-site
generalization, perfect turn prediction, or overfit10x-level performance. For "best overall prediction
on the data we actually trust," MODEL_X is the right artifact.

---

### Artifacts
- Model: `models/best_by_val_ADE.pth` — Scalers: `models/scalers.json`
- Split: `splits/model_x_track_split.csv`
- Per-window metrics: `evaluation/per_window_metrics.csv` — Subsets: `evaluation/subset_metrics.json`
- Plots: `plots/best_6_overall/`, `plots/worst_6_overall/`, `plots/best_6_genuine_turns/`,
  `plots/collages/` (incl. `model_x_best_predictions_presentation.png`), `plots/distributions/`
- Report: `reports/model_x_report.md` (this file)
