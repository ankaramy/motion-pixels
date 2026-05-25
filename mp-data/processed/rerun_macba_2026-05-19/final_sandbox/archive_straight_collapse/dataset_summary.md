# Final Sandbox — Dataset Summary

- Source: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\spatial_v21C\trajectories_encoded.csv`
- Output: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox\final_sandbox_dataset.csv`
- Min-track-length filter: 35 frames
- Duplication factor: ×10
- Final rows: **179,750**, unique track_id: **480**
- World extent: x ∈ [0.27, 13.19] m, y ∈ [-26.25, 24.41] m

## Feature stats

| feature | mean | std | min | max |
|---|---|---|---|---|
| `world_x` | 6.6426 | 3.4763 | 0.2709 | 13.1900 |
| `world_y` | -4.0053 | 12.6400 | -26.2501 | 24.4128 |
| `delta_x` | -0.0008 | 0.0152 | -0.3159 | 0.3416 |
| `delta_y` | 0.0058 | 0.0766 | -1.4198 | 1.8508 |
| `speed` | 0.0474 | 0.0623 | 0.0000 | 1.8564 |
| `heading_sin` | 0.0735 | 0.9243 | -1.0000 | 1.0000 |
| `heading_cos` | -0.0049 | 0.3745 | -1.0000 | 1.0000 |
| `turn_rate` | 0.0046 | 1.4085 | -3.1416 | 3.1416 |
| `u_norm` | 0.4932 | 0.2691 | 0.0000 | 1.0000 |
| `v_norm` | 0.4391 | 0.2495 | 0.0000 | 1.0000 |
| `dist_to_obstacle` | 1.2525 | 0.6856 | 0.0000 | 3.4000 |
| `dist_to_boundary` | 1.2358 | 0.8020 | 0.3162 | 4.2379 |
| `dist_to_entrance` | 15.1233 | 12.4571 | 0.0000 | 43.1057 |

## Columns

`frame`, `time_s`, `track_id`, `conf`, `x1`, `y1`, `x2`, `y2`, `cx`, `cy`, `foot_x`, `foot_y`, `image_x`, `image_y`, `world_x`, `world_y`, `dist_to_obstacle`, `dist_to_boundary`, `dist_to_entrance`, `delta_x`, `delta_y`, `speed`, `heading_sin`, `heading_cos`, `turn_rate`, `u_norm`, `v_norm`, `dup_idx`