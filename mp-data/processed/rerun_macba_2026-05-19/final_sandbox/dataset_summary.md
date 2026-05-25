# Final Sandbox — Dataset Summary

- Source: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\spatial_v21C\trajectories_encoded.csv`
- Output: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox\final_sandbox_dataset.csv`
- Min-track-length filter: 45 frames
- Duplication factor: ×10
- Final rows: **178,230**, unique track_id: **440**
- World extent: x ∈ [0.27, 11.08] m, y ∈ [-25.64, 24.41] m

## Feature stats

| feature | mean | std | min | max |
|---|---|---|---|---|
| `world_x` | 6.6493 | 3.4632 | 0.2709 | 11.0830 |
| `world_y` | -3.9767 | 12.5844 | -25.6380 | 24.4128 |
| `delta_x` | -0.0008 | 0.0151 | -0.3159 | 0.3416 |
| `delta_y` | 0.0058 | 0.0765 | -1.4198 | 1.8508 |
| `dist_to_obstacle` | 1.2579 | 0.6845 | 0.4123 | 3.4000 |
| `dist_to_boundary` | 1.2342 | 0.7967 | 0.3162 | 4.2379 |

## Columns

`frame`, `time_s`, `track_id`, `conf`, `x1`, `y1`, `x2`, `y2`, `cx`, `cy`, `foot_x`, `foot_y`, `image_x`, `image_y`, `world_x`, `world_y`, `dist_to_obstacle`, `dist_to_boundary`, `dist_to_entrance`, `delta_x`, `delta_y`, `dup_idx`