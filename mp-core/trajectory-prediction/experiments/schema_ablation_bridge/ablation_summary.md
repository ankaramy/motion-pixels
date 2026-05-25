# schema_ablation_bridge — Results

Window=10  Horizon=20  Hidden=128 x 2  Seed=42

## Quantitative results

| Model | Val MSE | ADE (m) | FDE (m) | Ang err (rad) | Pred TR (rad) | GT TR (rad) | path/GT | cum_head/GT | ang_ratio | curv_corr | N |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `A_motion_only` | 0.5129 | 0.4871 | 0.8998 | 0.7620 | 0.0406 | 0.7403 | 0.4240 | 0.1053 | 0.0000 | 0.0179 | 9 |
| `B_motion_position` | 0.5090 | 0.5054 | 0.9478 | 1.0367 | 0.0555 | 0.7403 | 0.5826 | 0.0485 | 0.0344 | -0.0483 | 9 |
| `C_motion_position_spatial` | 0.5218 | 0.5315 | 0.9809 | 0.9329 | 0.0495 | 0.7403 | 0.4882 | 0.0747 | 0.0159 | -0.0318 | 9 |
| `D_full_relational` | 0.5242 | 0.5273 | 1.0131 | 1.0352 | 0.0498 | 0.7403 | 0.7230 | 0.0518 | 0.0000 | -0.0778 | 9 |

## Ranking by ADE (best first)

1. `A_motion_only` — ADE=0.4871 m   FDE=0.8998 m   ang_err=43.7°
2. `B_motion_position` — ADE=0.5054 m   FDE=0.9478 m   ang_err=59.4°
3. `D_full_relational` — ADE=0.5273 m   FDE=1.0131 m   ang_err=59.3°
4. `C_motion_position_spatial` — ADE=0.5315 m   FDE=0.9809 m   ang_err=53.5°

GT mean |turn_rate| = **0.7403 rad / step**.

Notes: angular metrics here are *observed*, not optimised. Only MSE on (target_du, target_dv) was the training objective.
