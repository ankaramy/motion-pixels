# schema_ablation_bridge_overfit10x — Visual Summary

> **OVERFIT10X — not thesis generalization evidence.**  Train/val/test share trajectory content by construction; numbers below reflect MEMORISATION capacity, not held-out performance.

Same recipe as schema_ablation_bridge (real-metric du/dv, MSE-only loss, autoregressive rollout, KDTree spatial refresh) — trained on the 10× duplicated dataset.

Selected trajectories: `[24004, 1001, 12007, 28007, 8004, 20000, 23002, 48009, 14007, 7003]`

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
| `A_motion_only` | 0.1037 | 0.2594 | 0.8187 | 1.072 | 0.287 | 0.789 | 73 |
| `B_motion_position` | 0.0957 | 0.2491 | 0.7302 | 0.989 | 0.344 | 0.882 | 73 |
| `C_motion_position_spatial` | 0.0535 | 0.1096 | 0.5686 | 1.132 | 0.496 | 0.852 | 73 |
| `D_full_relational` | 0.0611 | 0.1343 | 0.6226 | 0.963 | 0.398 | 0.862 | 73 |

## Outputs

| File | Content |
|---|---|
| `overlay/traj_*.png` | All models vs GT on one axis (local bounds) |
| `grid/traj_*.png` | One model per panel vs GT |
| `error_over_time.png` | Per-step displacement drift, population mean |
| `angular_error_over_time.png` | Per-step heading error, population mean |
| `turn_rate_comparison.png` | Mean |turn_rate| vs GT |