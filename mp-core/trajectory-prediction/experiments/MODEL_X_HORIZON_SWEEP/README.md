# MODEL_X Horizon Sweep

**Started 2026-06-11.** Goal: train Model X variants for longer, architectural-scale predictions and
find **how far the model can predict before quality collapses**.

## What this is

Five separate Model X variants — **H20, H60, H100, H200, H400** — corresponding to roughly
1 m / 3 m / 5 m / 10 m / 20 m of predicted reach. Every variant uses the *identical* Model X recipe:

- Barcelona **v3 manual** master (`new_datasets\Barcelona_v3_manual_master_dataset\master_dataset.csv`),
  5 validated recordings.
- Model C `TrajectoryLSTM` (hidden 128, 2 layers, dropout 0.2, obs_len 10, output `target_du,target_dv`).
- 10 input features, mixed-recording track-level split (the **same** `MODEL_X/splits/model_x_track_split.csv`,
  80/10/10, seed 42, every recording in every split).
- AdamW lr 1e-3, wd 1e-5, batch 256, grad-clip 1.0, position MSE, ≤80 epochs, early-stop patience 12.
- **No** turn-only training, angular loss, heading-output head, or overfit data.

## How horizon enters (design decision — read this)

The training **objective stays single-step** next-displacement MSE (the Model X / Model C objective);
the architecture is unchanged. A genuine multi-step rollout loss is infeasible at H=400 (400 sequential
steps × backward per sample). So horizon H controls three things instead:

1. **Track filter** — a track must have ≥ `obs+H` frames to contribute, so each variant trains only on
   trajectories that actually persist that long (the relevant *persistent-walker* distribution). Data
   availability and dropped-track counts are reported per horizon.
2. **Rollout depth** — validation/eval roll the model out `H` autoregressive steps in world metres
   (spatial-clearance features frozen, as in Model X).
3. **Checkpoint** — `model_best.pth` is selected by the `H`-step rollout **validation ADE**.

This makes each horizon a distinct model and lets us measure error/length behaviour vs reach honestly.
It does **not** train the model to resist multi-step drift — so the sweep is also a clean test of how far
a single-step Model X rollout stays usable.

## Collapse / usability definitions

- **collapse rate** = fraction of test windows where predicted path length < 25% of GT path length.
- **visual usefulness** = median and p75 of predicted path length (real length, never rescaled).
- A horizon is called *usable* if collapse rate ≤ 35%, and *visually plausible length* if median
  pred/GT length ratio ≥ 55% (thresholds documented in `build_comparison.py`).

## Layout

```
MODEL_X_HORIZON_SWEEP/
├── README.md, comparison_summary.csv, comparison_report.md
├── run_horizon.py            ← python run_horizon.py <H>
├── build_comparison.py       ← aggregates completed horizons + prints final table
└── H20/ H60/ H100/ H200/ H400/
    ├── model_best.pth, scalers.json, config.json
    ├── train_loss.csv, val_metrics_by_epoch.csv, loss_curve.png
    ├── metrics_summary.{md,json}, per_recording_metrics.csv, per_window_metrics.csv
    ├── best_6_overall/  worst_6_overall/  best_6_long_predictions/
    ├── prediction_length_distribution.png, gt_vs_pred_length_scatter.png
    └── presentation_collage.png
```

## Run order

H20 + H60 first; if both stable, then H100, H200, H400; finally `python build_comparison.py`.

## Honesty rules (enforced)

No fabricated or stretched predictions; no independent path scaling. If H200/H400 are unusable, the
comparison report says so plainly.
