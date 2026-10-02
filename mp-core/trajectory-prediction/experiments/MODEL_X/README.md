# MODEL_X — General-purpose Horizon-10 Model C

**Phase start: 2026-06-11.** Isolated phase folder. Nothing outside `MODEL_X/` is modified.

## What MODEL_X is

> A general-purpose **Horizon-10** retraining of **Model C**, optimised for **best overall
> trajectory prediction** on the full *available* validated Barcelona dataset.

Model X is **NOT** a new architecture. It is the frozen Model C `TrajectoryLSTM`
(LSTM, hidden 128, 2 layers, dropout 0.2, head `Linear(128→2)` → `target_du, target_dv`,
MSE loss, seed 42) re-trained from scratch with:

- the **10 Model C input features** (`du, dv, speed, heading_sin, heading_cos, turn_rate,
  u, v, dist_to_obstacle_norm, dist_to_boundary_norm`),
- **observation window = 10**, **prediction horizon = 10** (autoregressive rollout),
- a **mixed-recording track-level split** (80/10/10, seed 42),
- **position MSE** loss on next-step displacement,
- checkpointing by **validation ADE/FDE** over the 10-step rollout.

It deliberately does **NOT** use: turn-only training, turn-balanced sampler, angular loss,
a heading-output head, overfit10x data, duplicated trajectories, or a recording-level
held-out split.

## Dataset (resolved against disk — see `data_audit/dataset_audit.md`)

- **Used:** `Barcelona_v3_manual_master_dataset/master_dataset.csv` — spatial features from
  **Encoder V3 real manual architectural masks** (newest, corrected; built 2026-06-09).
- **1,064,379 rows / 3,534 tracks / 5 recordings:** esplanade_espanya_01, placa_catalunya_01,
  placa_espanya_01, stairs_montjuic_01, red_bridge_combined_01.
- **⚠️ Only 5 of the 7 requested recordings exist.** `placa_montjuic_01` and `stairs_montjuic_02`
  were excluded for ill-conditioned homography and were never encoded — they cannot be added
  without recalibration + re-encoding (a separate, blocked task). MODEL_X trains on the full set
  of trustworthy data currently available, by explicit user decision.

## Why these choices

- **Horizon 10** — the best-looking useful rollouts in the prior MP_X investigation came from
  Exp03 at horizon 10. Long enough to show real intent, short enough to stay plausible without
  fabricated path scaling.
- **Mixed-recording track-level split** — we want *best in-dataset prediction quality* across all
  available sites, not unseen-site generalization. Every recording contributes tracks to
  train/val/test; no track appears in two splits. This is honestly labelled a *mixed-recording
  track-level split* and must **not** be read as unseen-site generalization.
- **Spatial features frozen during rollout** — the autoregressive rollout updates motion + position
  from predictions and holds the two spatial-clearance features at their last-observed value (no
  KDTree re-query). Prior audits showed those features carry ~no turn signal; this keeps MODEL_X
  self-contained. Documented as the only eval-time simplification.

## What MODEL_X may / may not be used to claim

**May:** best practical trajectory prediction on the available validated Barcelona dataset;
prototype-quality rollout visuals; final thesis presentation outputs.

**May NOT:** unseen-site generalization; perfect turn-direction prediction; overfit10x-level
memorisation performance.

## Layout

```
MODEL_X/
├── README.md                  ← this file
├── run_log.md                 ← chronological log of each step + results
├── model_x_lib.py             ← arch + scaler + windowing + rollout + metrics (Model C copy, source untouched)
├── config/model_x_config.json ← single source of truth for all settings
├── data_audit/                ← dataset_audit.md (what was actually found on disk)
├── splits/                    ← build_split.py + model_x_track_split.csv
├── training/                  ← train_model_x.py + loss/metric logs + curves
├── models/                    ← best_by_val_ADE.pth, best_by_val_FDE.pth, last_epoch.pth, scalers.json
├── evaluation/                ← evaluate_model_x.py + metrics tables
├── plots/                     ← best/worst/turn panels, collages, distributions
└── reports/                   ← model_x_report.md
```

## Run order

1. `python splits/build_split.py`        → track-level split  *(done)*
2. `python training/train_model_x.py`     → train + checkpoints + curves
3. `python evaluation/evaluate_model_x.py`→ test metrics (overall / per-recording / turn / straight)
4. `python plots/make_plots.py`           → diagnostic + presentation visuals
5. report written to `reports/model_x_report.md`
