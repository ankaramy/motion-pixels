# MP_X — Recovering Visible Angularity in Model C

**Created:** 2026-06-09
**Owner:** Motion Pixels thesis (anka.ramy@gmail.com)

## Why this experiment block exists

Global angular loss (Phase 6) slightly improved *global* heading metrics but did
**not** improve genuine pedestrian turns — Model C still collapses toward straight
futures. However, the **OVERFIT10X** capacity checkpoint proved the *same
architecture* can produce curved, spatially-aware rollouts. So the working
hypothesis tonight is:

> The straight-line collapse is caused by **turn scarcity / class imbalance** in
> the training windows, not by a fundamental incapacity of the architecture.

We will **not** accept "straight-line prediction is the conclusion" until Model C
has been trained and evaluated under explicitly **turn-focused** conditions.

## Baseline being challenged

Phase-4 V3 Model C (frozen recipe, no balancing):

| split | ADE (m) | FDE (m) | angular err (deg) | mean turn pred (rad) | mean turn GT (rad) |
|---|---|---|---|---|---|
| val  | 0.274 | 0.491 | 81.0 | 0.116 | 0.919 |
| test | 0.252 | 0.476 | 93.7 | 0.101 | 0.985 |

Predicted per-step turn rate is ~6–9× smaller than ground truth → collapse.

## Core model (unchanged architecture)

`fz.TrajectoryLSTM` from
`experiments/schema_ablation_bridge/run_bridge_ablation.py` — 10-feature, 2-layer
LSTM (hidden 128, dropout 0.2), `Linear(128→2)` predicting `(target_du, target_dv)`
in m/step. Window 10, rollout horizon 20, MSE objective, AdamW lr 1e-3,
StepLR(25, 0.35), batch 512, patience 15, seed 42. We keep this architecture
**stable** and change only *which windows the optimiser sees* (and, for exp04, the
output head/loss).

Feature order (frozen Model C):
```
du, dv, speed, heading_sin, heading_cos, turn_rate, u, v,
dist_to_obstacle_norm, dist_to_boundary_norm
```
`entrance_affinity_norm` is **not** used.

## Dataset

`C:\Users\OWNER\Desktop\new_datasets\Barcelona_v3_manual_master_dataset\model_C_dataset.csv`
(Barcelona_v3_manual_master, recording-level split, per-recording `world_bounds`
in `manifest.json`). Read-only — never modified.

## Turn taxonomy (shared across all experiments)

Computed on the **GT future** over the prediction horizon (default 20 steps):

- **artifact guard:** max single-step displacement `< 0.6 m` (else → straight; an
  ID-switch / teleport is not a pedestrian turn)
- **minimum displacement for a genuine turn:** GT net displacement `> 2 m`
- **genuine turn:** GT net heading change `> 30°`
  - `mild_30_90`  : 30–90°
  - `sharp_90_150`: 90–150°
  - `uturn_150_180`: 150–180°
- **straight:** `< 30°` or failing any genuine-turn criterion

## Turn-specific metrics (every rollout)

ADE, FDE, angular error, GT heading change, predicted heading change, GT path
length, predicted path length, angularity ratio (= pred heading change / GT
heading change), and **Turn Capture Rate** (GT turn if GT heading change > 30°,
predicted turn if predicted heading change > 20°; TCR = captured GT turns / total
GT turns). Every experiment reports **all windows** and **genuine-turn windows**
separately, plus per-bin breakdowns.

## Tonight's four experiments

| folder | idea |
|---|---|
| `exp01_turn_balanced_model_c` | `WeightedRandomSampler` (straight 1 / mild 8 / sharp 12 / U-turn 16). **Implemented tonight.** |
| `exp02_turn_only_diagnostic`  | Train only on genuine-turn windows (disp>2m, head>30°, max-step<0.6m). |
| `exp03_horizon_sweep`         | Turn-balanced sampler, horizons 5 / 10 / 20, everything else fixed. |
| `exp04_direction_output_variant` | Output `du, dv, heading_sin, heading_cos`; loss = pos MSE + 0.2·heading MSE; turn-balanced sampler. |

## Layout

```
MP_X/
├── README.md                ← this file
├── run_log.md               ← exact commands + timestamps + outcomes
├── shared/                  ← reusable utilities (ONLY shared code lives here)
│   ├── mpx_common.py        ← constants, turn taxonomy, labeled-window builder, metrics, windowed eval
│   ├── mpx_train.py         ← turn-balanced Model C trainer (reused by exp01/exp03)
│   └── mpx_plots.py         ← best/worst/collage rollout plots + heading-change comparison
├── exp01_turn_balanced_model_c/
├── exp02_turn_only_diagnostic/
├── exp03_horizon_sweep/
└── exp04_direction_output_variant/
```

## Rules (enforced)

1. Never overwrite old results. 2. Never edit `frozen_model_C`. 3. Architecture
defs are imported read-only from the frozen recipe; experiments stay isolated.
4. Every script is CLI-runnable. 5. Every experiment writes metrics + plots + a
markdown conclusion. 6. The exact command is saved in `run_log.md`. 7. Dataset
size, #tracks, #windows and turn-bin counts are always reported. 8. "all windows"
and "genuine-turn windows" are always evaluated separately. 9. Trajectory plots
are always generated.
