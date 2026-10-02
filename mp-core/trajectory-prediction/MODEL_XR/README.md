# MODEL_XR — controlled objective fix for displacement magnitude

**Phase 4. Started 2026-06-11.** A controlled successor to MODEL_X whose only change is the
**training objective**, aimed at fixing the magnitude underprediction diagnosed in Phases 1–3 while
preserving MODEL_X's already-strong direction learning.

> ⚠️ **Do NOT overwrite MODEL_X.** MODEL_X lives at `experiments/MODEL_X/` and its checkpoints are
> frozen. MODEL_XR imports `experiments/MODEL_X/model_x_lib.py` **read-only** and writes everything
> under `MODEL_XR/`. Dataset, encoder, spatial features, split, architecture size, and horizon
> definitions are all UNCHANGED.

## Relation to MODEL_X
- Same architecture: `TrajectoryLSTM` (hidden 128, 2 layers, dropout 0.2), single-step
  next-displacement output `(target_du, target_dv)`, seed 42.
- Same data: `Barcelona_v3_manual_master_dataset/master_dataset.csv`, same mixed-recording
  **track-level** split (`experiments/MODEL_X/splits/model_x_track_split.csv`, 80/10/10, seed 42, no
  track leakage), same per-axis StandardScaler.
- Same recipe: AdamW lr 1e-3, wd 1e-5, batch 256, grad-clip 1.0, ≤80 epochs, early-stop patience 12 on
  validation ADE (H10 rollout).
- **Only the loss differs.**

## Loss definitions (`losses.py`)
```
loss = MSE(pred_scaled, gt_scaled)                          # base — identical to MODEL_X
     + lambda_mag * MSE(||pred_m||/sigma, ||gt_m||/sigma)   # magnitude (metric, isotropic-normed)
     + lambda_dir * mean(1 - cos(pred_m, gt_m))             # direction (true metric cosine)
```
- Base MSE is in the per-axis **scaled** space (so `lambda_mag=lambda_dir=0` reproduces MODEL_X exactly).
- Magnitude & direction use **unscaled metric** displacement (std_du/std_dv ≈ 3.3 makes scaled-space
  cosine angle-distorted). `sigma` = RMS metric step size of the train targets → magnitude term is O(1)
  and comparable to the scaled-MSE base, so the suggested lambdas are meaningful.

## Variants
| variant | lambda_mag | lambda_dir | purpose |
|---|---|---|---|
| MODEL_XR_A_CONTROL | 0.0 | 0.0 | reproduce MODEL_X (expect ADE≈0.305, FDE≈0.480) |
| MODEL_XR_B_MAG | 1.0 | 0.0 | + magnitude loss |
| MODEL_XR_C_MAG_LIGHT_DIR | 1.0 | 0.1 | + magnitude + light direction |
| MODEL_XR_D_MAG_STRONGER | 2.0 | 0.0 | + stronger magnitude loss |

## Horizon evaluation
Each variant is trained **once** (single-step objective) and **evaluated** at rollout depths
H20/H60/H100/H200/H400 with jitter-aware metrics (cumulative / **net-displacement** / smoothed ratios).
This differs deliberately from `MODEL_X_HORIZON_SWEEP` (which trained a separate model per horizon): here
one model per variant is rolled to multiple depths to **isolate the loss effect**. Horizon *definitions*
(track filter len≥obs+H, rollout depth=H) are unchanged.

## Commands
```
python train_model_xr.py    --config configs/A_control.json     # (B/C/D analogously)
python evaluate_model_xr.py --config configs/A_control.json     # base-H10 metrics
python run_horizon_xr.py    --config configs/A_control.json     # H20..H400 sweep
python aggregate_xr.py                                          # CSVs + figures + report
```

## Output structure
```
MODEL_XR/
├── train_model_xr.py evaluate_model_xr.py run_horizon_xr.py aggregate_xr.py losses.py xr_common.py
├── configs/{A_control,B_mag,C_mag_light_dir,D_mag_stronger}.json
├── checkpoints/<variant>/{model_best.pth, scalers.json, sigma.json, train_log.csv, train_summary.json}
├── experiments/<variant>/{eval_base.json, horizon.json}
├── figures/ (variant comparisons, per-recording, focus, rollout examples — equal metric scaling)
└── reports/ (model_xr_*.csv, MODEL_XR_REPORT.md)
```

## Success / failure criteria
- **Primary success:** t=1 smoothed step ratio moves from ~0.60–0.70 toward **0.85–1.05**.
- **Secondary:** net-disp & smoothed-path ratios improve; ADE/FDE not substantially worse; cosine stays high.
- **Failure:** magnitude up but direction collapses; ADE/FDE significantly worse; overshoot (ratio ≫1);
  improvement only on Red Bridge and not the plazas.
