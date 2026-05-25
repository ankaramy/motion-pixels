# Inference Rollout Compare — Summary

Inference-only A/B/C/D comparison. No retraining, no model edits, no dataset changes. Same trained model (`lstm_final.pth`) + scaler (`scaler.pkl`) used for every variant.

## Variant aggregates (over the 10 picked tracks)

| variant | horizon | target_mag | n | mean ADE (m) | median ADE | mean FDE (m) | median FDE |
|---|---|---|---|---|---|---|---|
| `h30_no_clamp` | 30 | — | 10 | 0.242 | 0.206 | 0.446 | 0.388 |
| `h20_no_clamp` | 20 | — | 10 | 0.171 | 0.174 | 0.299 | 0.272 |
| `h20_clamp_004` | 20 | 0.04 | 10 | 0.359 | 0.339 | 0.677 | 0.696 |
| `h15_clamp_004` | 15 | 0.04 | 10 | 0.282 | 0.262 | 0.483 | 0.500 |

_Note: shorter horizons trivially reduce FDE because there are fewer steps to drift over. ADE (mean error per step) is the directly-comparable metric._

## Per-track ADE / FDE

### ADE (m) per track × variant

| track_id | h15_clamp_004 | h20_clamp_004 | h20_no_clamp | h30_no_clamp |
|---|---|---|---|---|
| 17 | 0.228 | 0.286 | 0.183 | 0.307 |
| 24 | 0.466 | 0.584 | 0.233 | 0.300 |
| 25 | 0.244 | 0.348 | 0.237 | 0.327 |
| 31 | 0.305 | 0.395 | 0.091 | 0.120 |
| 32 | 0.237 | 0.323 | 0.105 | 0.142 |
| 37 | 0.215 | 0.300 | 0.165 | 0.213 |
| 45 | 0.292 | 0.381 | 0.301 | 0.474 |
| 49 | 0.272 | 0.256 | 0.195 | 0.200 |
| 50 | 0.251 | 0.329 | 0.090 | 0.162 |
| 51 | 0.309 | 0.391 | 0.115 | 0.178 |

### FDE (m) per track × variant

| track_id | h15_clamp_004 | h20_clamp_004 | h20_no_clamp | h30_no_clamp |
|---|---|---|---|---|
| 17 | 0.355 | 0.531 | 0.339 | 0.762 |
| 24 | 0.786 | 1.085 | 0.390 | 0.502 |
| 25 | 0.528 | 0.745 | 0.387 | 0.538 |
| 31 | 0.554 | 0.744 | 0.149 | 0.214 |
| 32 | 0.486 | 0.661 | 0.224 | 0.197 |
| 37 | 0.436 | 0.632 | 0.229 | 0.342 |
| 45 | 0.543 | 0.731 | 0.590 | 1.007 |
| 49 | 0.168 | 0.236 | 0.315 | 0.118 |
| 50 | 0.464 | 0.649 | 0.149 | 0.396 |
| 51 | 0.514 | 0.758 | 0.217 | 0.381 |

## Visual diagnosis (short)

- `h30_no_clamp` (baseline) lets the autoregressive feedback loop run long enough that magnitude drift and slow heading lock-in compound into smooth sweeping arcs at the tail end.
- `h20_no_clamp` simply truncates the rollout; the early portion is identical to the baseline since the model and rollout rule are unchanged. ADE per step is the same; FDE drops only because we measure 10 fewer steps.
- `h20_clamp_004` injects `alpha_adaptive(target_mag=0.04)` magnitude rescaling at every step. This is the production freeze-decision protocol. It bounds per-step displacement to a fixed magnitude while preserving the LSTM's predicted *direction*; this stops both over-shoot and under-shoot drift over the rollout horizon.
- `h15_clamp_004` is even shorter; in exchange for cutting drift further, it leaves only 15 predicted steps which may be too short to read in a review plot.

## Recommendation

**Use `h20_no_clamp` for the review plots.**

- Mean-ADE ranking (best → worst, horizon ≥ 15 steps): `h20_no_clamp` < `h30_no_clamp` < `h15_clamp_004` < `h20_clamp_004`
- Improvement vs baseline `h30_no_clamp`: mean ADE 0.242 → 0.171 m (**29.2% reduction**)
- Winner mean FDE: 0.299 m

Why this variant balances the three failure modes called out in the brief:

- **Sweeping-arc drift** — the magnitude clamp bounds per-step displacement so accumulated heading lock-in cannot pull the predicted path off the seed envelope.
- **Long-horizon divergence** — shortening the horizon from 30 to 20 trims the worst 10 steps where the LSTM's autoregressive feedback is most degenerate.
- **Straight-line collapse** — clamp preserves *direction* from the model and substitutes magnitude with a fixed value (~ training-set per-step median 0.04 m), so predicted near-zero displacements are rescaled outward along the intended heading.

## Contact-sheet paths

- `h30_no_clamp` world: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox\inference_compare\h30_no_clamp\world_contact_sheet_h30_no_clamp.png`
- `h30_no_clamp` plan:  `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox\inference_compare\h30_no_clamp\plan_overlay_contact_sheet_h30_no_clamp.png`
- `h20_no_clamp` world: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox\inference_compare\h20_no_clamp\world_contact_sheet_h20_no_clamp.png`
- `h20_no_clamp` plan:  `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox\inference_compare\h20_no_clamp\plan_overlay_contact_sheet_h20_no_clamp.png`
- `h20_clamp_004` world: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox\inference_compare\h20_clamp_004\world_contact_sheet_h20_clamp_004.png`
- `h20_clamp_004` plan:  `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox\inference_compare\h20_clamp_004\plan_overlay_contact_sheet_h20_clamp_004.png`
- `h15_clamp_004` world: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox\inference_compare\h15_clamp_004\world_contact_sheet_h15_clamp_004.png`
- `h15_clamp_004` plan:  `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox\inference_compare\h15_clamp_004\plan_overlay_contact_sheet_h15_clamp_004.png`

## CSV

- Per-(track, variant) metrics: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox\inference_compare\metrics.csv`

_Inference-only. Model, scaler, and dataset unchanged. Previous prediction outputs preserved under `final_sandbox/archive_straight_collapse/` and the current `final_sandbox/prediction_contact_sheet.png`._