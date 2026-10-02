# Leakage report

- exp01-03 input features: ['du', 'dv', 'speed', 'heading_sin', 'heading_cos', 'turn_rate', 'u', 'v', 'dist_to_obstacle_norm', 'dist_to_boundary_norm']
- targets ['target_du', 'target_dv'] in inputs: NO
- future_heading in inputs: NO (exp04 adds them only as TARGETS at runtime)
- tracks spanning splits: 0; recordings spanning splits: 0 (split is recording-level)
- identical-track duplicates total: 0; across splits: 0
- top feature/target correlations:
    feature  corr_target_du  corr_target_dv  max_abs
         du        0.266569        0.095918 0.266569
         dv        0.093697        0.227339 0.227339
heading_cos        0.174806        0.071792 0.174806
heading_sin        0.031976        0.142282 0.142282
      speed       -0.014159       -0.018742 0.018742
          v        0.015490        0.010572 0.015490
