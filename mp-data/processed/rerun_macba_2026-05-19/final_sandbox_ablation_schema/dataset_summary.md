# Schema Dataset Summary

- Source: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\spatial_v21C\trajectories_encoded.csv`
- Output: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox_ablation_schema\schema_dataset.csv`
- World extent: x ∈ [0.27, 13.19] m, y ∈ [-26.25, 24.41] m
- Min-track-length filter: 35 frames
- Duplication factor: ×10
- Final rows: **179,270**, unique tracks: **480**

## Behaviour thresholds

- `is_stop`  = 1 if per-step |Δ| < 0.005 m
- `is_shift` = 1 if |turn_rate| > 14.9°

## Stats for derived behaviour features

| feature | mean | std | min | max | non-zero frac |
|---|---|---|---|---|---|
| `speed` | 0.0473 | 0.0621 | 0.0000 | 1.8564 | 99.59% |
| `heading_angle` | 0.2348 | 1.6171 | -3.1271 | 3.1416 | 99.53% |
| `turn_rate` | 0.0046 | 1.4076 | -3.1416 | 3.1416 | 99.34% |
| `is_stop` | 0.1108 | 0.3139 | 0.0000 | 1.0000 | 11.08% |
| `is_shift` | 0.4367 | 0.4960 | 0.0000 | 1.0000 | 43.67% |
| `heading_sin` | 0.0736 | 0.9243 | -1.0000 | 1.0000 | 99.44% |
| `heading_cos` | -0.0051 | 0.3745 | -1.0000 | 1.0000 | 98.93% |
| `dist_to_obstacle` | 1.2531 | 0.6855 | 0.0000 | 3.4000 | 99.80% |
| `dist_to_boundary` | 1.2362 | 0.8016 | 0.3162 | 4.2379 | 100.00% |
| `dist_to_entrance` | 15.1244 | 12.4523 | 0.0000 | 43.1057 | 99.97% |
| `target_du` | -0.0008 | 0.0153 | -0.3159 | 0.3416 | 98.78% |
| `target_dv` | 0.0058 | 0.0767 | -1.4198 | 1.8508 | 99.70% |

## Schema (full set)

`frame`, `time`, `track_id`, `conf`, `x1`, `y1`, `x2`, `y2`, `cx`, `cy`, `foot_x`, `foot_y`, `image_x`, `image_y`, `world_x`, `world_y`, `dist_to_obstacle`, `dist_to_boundary`, `dist_to_entrance`, `delta_x`, `delta_y`, `speed`, `heading_angle`, `heading_sin`, `heading_cos`, `turn_rate`, `is_stop`, `is_shift`, `target_du`, `target_dv`, `u`, `v`, `dup_idx`