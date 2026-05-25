# schema_ablation_bridge_overfit10x — Results

> **OVERFIT10X — not thesis generalization evidence.** Train/val/test share trajectory content by construction.

Window=10  Horizon=20  Hidden=128 x 2  Seed=42

## Quantitative results

| Model | Val MSE | ADE (m) | FDE (m) | Ang err (rad) | Pred TR (rad) | GT TR (rad) | path/GT | cum_head/GT | ang_ratio | curv_corr | N |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `A_motion_only` | 0.0318 | 0.1037 | 0.2594 | 0.8187 | 0.6613 | 0.8165 | 0.7890 | 0.8118 | 1.0723 | 0.2872 | 73 |
| `B_motion_position` | 0.0285 | 0.0957 | 0.2491 | 0.7302 | 0.7314 | 0.8165 | 0.8823 | 0.8643 | 0.9889 | 0.3438 | 73 |
| `C_motion_position_spatial` | 0.0265 | 0.0535 | 0.1096 | 0.5686 | 0.8786 | 0.8165 | 0.8518 | 0.9959 | 1.1320 | 0.4961 | 73 |
| `D_full_relational` | 0.0241 | 0.0611 | 0.1343 | 0.6226 | 0.6987 | 0.8165 | 0.8617 | 0.8250 | 0.9634 | 0.3982 | 73 |

## Ranking by ADE (best first)

1. `C_motion_position_spatial` — ADE=0.0535 m   FDE=0.1096 m   ang_err=32.6°
2. `D_full_relational` — ADE=0.0611 m   FDE=0.1343 m   ang_err=35.7°
3. `B_motion_position` — ADE=0.0957 m   FDE=0.2491 m   ang_err=41.8°
4. `A_motion_only` — ADE=0.1037 m   FDE=0.2594 m   ang_err=46.9°

GT mean |turn_rate| = **0.8165 rad / step**.

Notes: angular metrics here are *observed*, not optimised. Only MSE on (target_du, target_dv) was the training objective.
