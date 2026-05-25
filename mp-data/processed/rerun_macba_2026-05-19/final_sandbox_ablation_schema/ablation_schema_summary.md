# Final Sandbox Ablation — Schema Summary

Isolated rebuild using the **full Motion Pixels schema** and explicit angular-continuation prioritisation. All four ablation models share architecture (LSTM 256×2, window 10) and training recipe (×10 duplicate, 30 epochs, MSE on next-step `target_du`/`target_dv`). Only the feature set changes between A/B/C/D.

**Rollout recomputation rule.** At every predicted step the rollout updates `world_x`, `world_y`, `u`, `v`, `delta_x`, `delta_y`, `speed`, `heading_angle`, `is_stop`, `is_shift`, `heading_sin`, `heading_cos`, and `turn_rate` from the new predicted state. Spatial features `dist_to_obstacle`, `dist_to_boundary`, `dist_to_entrance` are re-queried via KDTree-IDW (k=5) over the v2.1C encoded CSV at the new `(world_x, world_y)`. No feature is frozen at seed values.

## Feature sets

| model | label | # features | features |
|---|---|---|---|
| `A_motion` | A motion only | 6 | `delta_x`, `delta_y`, `speed`, `heading_angle`, `is_stop`, `is_shift` |
| `B_position` | B + position | 10 | `delta_x`, `delta_y`, `speed`, `heading_angle`, `is_stop`, `is_shift`, `u`, `v`, `world_x`, `world_y` |
| `C_spatial` | C + spatial | 13 | `delta_x`, `delta_y`, `speed`, `heading_angle`, `is_stop`, `is_shift`, `u`, `v`, `world_x`, `world_y`, `dist_to_obstacle`, `dist_to_boundary`, `dist_to_entrance` |
| `D_angular` | D + angular context | 16 | `delta_x`, `delta_y`, `speed`, `heading_angle`, `is_stop`, `is_shift`, `u`, `v`, `world_x`, `world_y`, `dist_to_obstacle`, `dist_to_boundary`, `dist_to_entrance`, `heading_sin`, `heading_cos`, `turn_rate` |

## Per-model metrics (averaged over picked tracks)

| model | ADE (m) | FDE (m) | heading err (°) | cum-heading ratio | path-len ratio | angularity pred/GT | curvature corr | seed-cont err |
|---|---|---|---|---|---|---|---|---|
| `A_motion` | 0.296 | 0.606 | 92.0 | 0.07 | 1.36 | 0.8/8.8 | -0.07 | 0.000 |
| `B_position` | 0.269 | 0.639 | 79.0 | 0.11 | 1.68 | 2.4/8.8 | 0.05 | 0.000 |
| `C_spatial` | 0.234 | 0.466 | 70.6 | 0.05 | 1.71 | 0.9/8.8 | 0.06 | 0.000 |
| `D_angular` | 0.196 | 0.366 | 71.3 | 0.13 | 1.45 | 1.8/8.8 | 0.03 | 0.000 |

Glossary:
- **cum-heading ratio** = predicted Σ|Δheading| ÷ GT Σ|Δheading|  (1.0 = matched, < 1 = under-turning, > 1 = over-turning)
- **angularity** = number of per-step |Δheading| > 15° in the rollout, compared to the GT count
- **curvature corr** = Pearson r between predicted and GT per-step signed heading deltas
- **seed-cont err** = distance from rollout's first point to the seed's last point (anchored, should be 0)

## Per-track metrics

### ADE per track × model (m)

| track | A_motion | B_position | C_spatial | D_angular |
|---|---|---|---|---|
| 17 | 0.284 | 0.210 | 0.082 | 0.103 |
| 24 | 0.460 | 0.197 | 0.211 | 0.202 |
| 25 | 0.386 | 0.249 | 0.119 | 0.101 |
| 31 | 0.173 | 0.261 | 0.200 | 0.164 |
| 32 | 0.352 | 0.283 | 0.125 | 0.121 |
| 37 | 0.357 | 0.278 | 0.144 | 0.122 |
| 45 | 0.490 | 0.516 | 0.686 | 0.652 |
| 49 | 0.173 | 0.373 | 0.263 | 0.186 |
| 50 | 0.131 | 0.201 | 0.057 | 0.076 |
| 51 | 0.154 | 0.117 | 0.456 | 0.234 |

### Curvature correlation per track × model

| track | A_motion | B_position | C_spatial | D_angular |
|---|---|---|---|---|
| 17 | 0.087 | 0.408 | -0.149 | 0.198 |
| 24 | -0.074 | 0.093 | 0.074 | 0.015 |
| 25 | 0.011 | 0.019 | -0.107 | -0.097 |
| 31 | 0.190 | 0.384 | 0.285 | 0.338 |
| 32 | -0.334 | -0.359 | -0.109 | -0.112 |
| 37 | -0.013 | 0.086 | 0.201 | 0.065 |
| 45 | 0.248 | -0.352 | 0.239 | -0.117 |
| 49 | -0.030 | -0.170 | -0.048 | -0.033 |
| 50 | -0.323 | 0.211 | 0.111 | 0.039 |
| 51 | -0.434 | 0.210 | 0.084 | -0.038 |

## Visual diagnosis

Open the per-model contact sheets and the combined plot to judge angular continuation directly. A model passes the thesis bar if:
- the predicted path **starts at the green seed-end marker** (seed-continuity err ≈ 0);
- the predicted curve **bends in the same direction** as the GT future (curvature corr > 0);
- it does **not** decay into a single smooth arc or a vertical/horizontal sweep over the 20-step horizon;
- it produces a comparable number of angular changes (angularity ratio close to 1).

## Recommendation

- **Lowest ADE/FDE** → `D_angular` (D + angular context)
- **Best angular continuation** → `D_angular` (D + angular context). Selected by composite angular score = 0.5·(1 − |cum_heading_ratio − 1|) + 0.3·max(0, curvature_corr) + 0.2·(1 − |path_len_ratio − 1|).
- **Best thesis presentation** → `D_angular` (D + angular context). Balances angular score against normalised ADE.

### Composite scores (higher = better)

| variant | angular_score | thesis_score |
|---|---|---|
| `A_motion` | 0.161 | -0.339 |
| `B_position` | 0.136 | -0.318 |
| `C_spatial` | 0.101 | -0.296 |
| `D_angular` | 0.181 | -0.151 |

## Paths to inspect (in this order)

1. **Combined plan overlay (all 4 models on calibrated `top_view.png`)** — `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox_ablation_schema\plan_overlays\plan_contact_sheet_combined.png`
2. **Best-thesis per-model world contact sheet** — `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox_ablation_schema\plots\contact_sheet_D_angular.png`
3. **Best-angular plan contact sheet** — `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox_ablation_schema\plan_overlays\plan_contact_sheet_D_angular.png`

Combined-per-track plots (normalised u/v axes, all four models on one panel per track) live under `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox_ablation_schema\plots\combined_all_models`.

## Acceptance check

- [x] dataset built from new calibrated Skate 1 data
- [x] 4 trained models present
- [x] 4 per-model loss curves saved
- [x] 4 per-model world contact sheets saved
- [x] combined per-track plots saved
- [x] 4 plan-overlay contact sheets + combined saved
- [x] metrics CSV present

All artefacts under `mp-data\processed\rerun_macba_2026-05-19\final_sandbox_ablation_schema`. No file outside this folder was modified.