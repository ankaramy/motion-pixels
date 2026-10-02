# MODEL_XC — curvature-aware controlled successor to MODEL_XR_B_MAG

**Phase 5B. Started 2026-06-11.** Goal: recover **curved trajectory geometry** (lost by
MODEL_XR_B_MAG) while keeping its magnitude fix and MODEL_X's direction. A controlled
**loss-function-only** experiment — architecture, data, split, scalers, horizons all unchanged.

> ⚠️ Do NOT overwrite MODEL_X or MODEL_XR. They live at `experiments/MODEL_X/` and
> `mp-core/trajectory-prediction/MODEL_XR/` and are imported **read-only**. MODEL_XC writes only
> under `MODEL_XC/`.

## Lineage
- **MODEL_X**: good direction, short distance, partial curvature (shape_ratio ≈ 0.36).
- **MODEL_XR_B_MAG**: fixed magnitude (step1≈1.04, nd≈0.94), kept direction (cos≈0.96), **flattened
  geometry** (shape_ratio ≈ 0.043, zero macro turns) — Phase 5A.
- **MODEL_XC**: add a curvature term to B_MAG's loss to restore turning, without breaking magnitude.

## Loss (`losses_xc.py`)
```
loss = MSE(pred_scaled,gt_scaled) + λ_mag·MSE(‖pred_m‖/σ,‖gt_m‖/σ)
     + λ_curv·MSE(Δθ_pred, Δθ_gt)[moving mask] + λ_dir·mean(1−cos)
```
Curvature term (single-step): `v_prev` = smoothed incoming displacement (mean of last 5 observed
du,dv); `Δθ = wrap(atan2(step) − atan2(v_prev))` for pred and GT; masked to steps where both move
> 0.02 m. Wrapped angles (`atan2(sin,cos)`), differentiable. **Limitation:** curvature is enforced at
the single-step turn level (the *immediate* cause of straightening per Phase 5A), not as a macro
multi-step rollout target — documented; masking + smoothed `v_prev` keep it stable.

## Variants
| variant | λ_mag | λ_curv | λ_dir |
|---|---|---|---|
| A_BMAG_CONTROL | 1.0 | 0.0 | 0.0 | (reproduce MODEL_XR_B_MAG) |
| B_CURV_LIGHT | 1.0 | 0.1 | 0.0 |
| C_CURV_MED | 1.0 | 0.25 | 0.0 |
| D_CURV_STRONG | 1.0 | 0.5 | 0.0 |
| E_CURV_DIR_LIGHT | 1.0 | 0.25 | 0.05 | (optional) |

## Success criteria
shape_ratio improves over B_MAG **while** step1 smoothed ratio ∈ [0.85,1.10], net-disp ∈ [0.85,1.15],
cosine high (≥0.90), and no consistent overshoot.

## Commands
```
python train_model_xc.py    --config configs/A_bmag_control.json   # B/C/D/E analogously
python evaluate_model_xc.py --config configs/A_bmag_control.json
python run_horizon_xc.py    --config configs/A_bmag_control.json
python aggregate_xc.py
```

## Output structure
```
MODEL_XC/
├── *.py (train/evaluate/run_horizon/aggregate, losses_xc, xc_common)
├── configs/  checkpoints/<variant>/  experiments/<variant>/  figures/  reports/
└── reports/MODEL_XC_REPORT.md, model_xc_*.csv
```

## Safety
Never deletes MODEL_X / MODEL_XR / Phase 1–5A outputs / dataset / encoder. On CUDA error: save logs,
no concurrent jobs, rerun only the failed variant. One model per variant, evaluated at H20–H400.
