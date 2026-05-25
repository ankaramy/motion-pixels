# schema_ablation_bridge_overfit10x — dataset summary

> **OVERFIT10X — not thesis generalization evidence.**  Every base trajectory is duplicated ×10; train/val/test share trajectory content by design.

- Source: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\spatial_v21C\trajectories_encoded.csv`
- Base trajectories: **52**  (18,014 rows)
- Duplication factor: **10**
- ID offset rule: `new_id = orig_id * 1000 + k`
- Final rows: **180,140**
- Final trajectories: **520**  (filter on base ≥ 11 frames)
- World x: [0.271, 13.190] m  (range 12.919 m)
- World y: [-26.250, 24.413] m  (range 50.663 m)

## Column statistics

| column | mean | std | min | max |
|---|---|---|---|---|
| `u` | 0.4936 | 0.2696 | 0.0000 | 1.0000 |
| `v` | 0.4379 | 0.2496 | 0.0000 | 1.0000 |
| `du` | -0.0007 | 0.0174 | -0.3159 | 1.1515 |
| `dv` | 0.0056 | 0.0772 | -1.5832 | 1.8508 |
| `speed` | 0.0474 | 0.0636 | 0.0000 | 1.9577 |
| `heading_sin` | 0.0722 | 0.9241 | -1.0000 | 1.0000 |
| `heading_cos` | -0.0046 | 0.3751 | -1.0000 | 1.0000 |
| `turn_rate` | 0.0045 | 1.4075 | -3.1416 | 3.1416 |
| `dist_to_obstacle_norm` | 0.5004 | 0.2884 | 0.0019 | 1.0000 |
| `dist_to_boundary_norm` | 0.5003 | 0.2884 | 0.0126 | 1.0000 |
| `entrance_affinity_norm` | 0.4999 | 0.2886 | 0.0001 | 0.9998 |
| `target_du` | -0.0007 | 0.0175 | -0.3159 | 1.1515 |
| `target_dv` | 0.0056 | 0.0775 | -1.5832 | 1.8508 |

## Notes

* `du`, `dv`, `target_du`, `target_dv` are in metres / step — no MOTION_SCALE applied. This matches the OLD ablation recipe.
* `openness_lr_asymmetry` is **NOT computed** here (the v21C encoding only emits `dist_to_obstacle`, `dist_to_boundary`, `dist_to_entrance`). The bridge experiment therefore stops at model `D_full_relational` (no model `E`).