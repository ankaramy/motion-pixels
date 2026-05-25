# schema_ablation_bridge — Visual Summary

Bridges the OLD experiments/ablation/ recipe (real-metric du/dv, MSE-only loss, autoregressive rollout, KDTree spatial refresh) onto the v21C calibrated Skate 1 dataset.

Selected trajectories: `[48, 3, 56, 38, 14, 2, 9, 15]`

## Models

| Tag | Label | Features |
|---|---|---|
| `A_motion_only` | A — motion only | `du`, `dv`, `speed`, `heading_sin`, `heading_cos`, `turn_rate` |
| `B_motion_position` | B — + position | `du`, `dv`, `speed`, `heading_sin`, `heading_cos`, `turn_rate`, `u`, `v` |
| `C_motion_position_spatial` | C — + spatial | `du`, `dv`, `speed`, `heading_sin`, `heading_cos`, `turn_rate`, `u`, `v`, `dist_to_obstacle_norm`, `dist_to_boundary_norm` |
| `D_full_relational` | D — + entrance | `du`, `dv`, `speed`, `heading_sin`, `heading_cos`, `turn_rate`, `u`, `v`, `dist_to_obstacle_norm`, `dist_to_boundary_norm`, `entrance_affinity_norm` |

## Quantitative results (from ablation_results.csv)

| Model | ADE (m) | FDE (m) | Ang err (rad) | ang_ratio | curv_corr | path/GT | N |
|---|---|---|---|---|---|---|---|
| `A_motion_only` | 0.4871 | 0.8998 | 0.7620 | 0.000 | 0.018 | 0.424 | 9 |
| `B_motion_position` | 0.5054 | 0.9478 | 1.0367 | 0.034 | -0.048 | 0.583 | 9 |
| `C_motion_position_spatial` | 0.5315 | 0.9809 | 0.9329 | 0.016 | -0.032 | 0.488 | 9 |
| `D_full_relational` | 0.5273 | 1.0131 | 1.0352 | 0.000 | -0.078 | 0.723 | 9 |

## Outputs

| File | Content |
|---|---|
| `overlay/traj_*.png` | All models vs GT on one axis (local bounds) |
| `grid/traj_*.png` | One model per panel vs GT |
| `error_over_time.png` | Per-step displacement drift, population mean |
| `angular_error_over_time.png` | Per-step heading error, population mean |
| `turn_rate_comparison.png` | Mean |turn_rate| vs GT |