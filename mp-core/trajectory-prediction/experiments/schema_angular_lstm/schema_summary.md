# Relational Schema — schema_angular_lstm

- Source: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\spatial_v21C\trajectories_encoded.csv`
- Output: `trajectories_schema_relational.csv`
- World extent: x in [0.27, 13.19] m, y in [-26.25, 24.41] m
- Tracks kept (>= 35 frames): **48**
- Rows: **17,927**

## Behaviour thresholds
- `is_stop`  = 1 if `speed_rel`     < 0.02
- `is_shift` = 1 if |turn rate|     > 15.0°

## Per-feature stats

| feature | mean | std | min | max |
|---|---|---|---|---|
| `u` | 0.4932 | 0.2691 | 0.0000 | 1.0000 |
| `v` | 0.4392 | 0.2493 | 0.0000 | 1.0000 |
| `du` | -0.0001 | 0.0012 | -0.0245 | 0.0264 |
| `dv` | 0.0001 | 0.0015 | -0.0280 | 0.0365 |
| `speed_rel` | 0.4999 | 0.2887 | 0.0021 | 1.0000 |
| `heading_sin` | 0.0550 | 0.7549 | -1.0000 | 1.0000 |
| `heading_cos` | -0.0244 | 0.6531 | -1.0000 | 1.0000 |
| `turn_rate_rel` | 0.0016 | 0.4027 | -1.0000 | 1.0000 |
| `is_stop` | 0.0200 | 0.1401 | 0.0000 | 1.0000 |
| `is_shift` | 0.5516 | 0.4973 | 0.0000 | 1.0000 |
| `obstacle_clearance_pct` | 0.5004 | 0.2884 | 0.0011 | 1.0000 |
| `boundary_clearance_pct` | 0.5003 | 0.2884 | 0.0125 | 1.0000 |
| `entrance_affinity_pct` | 0.4999 | 0.2886 | 0.0001 | 0.9998 |
| `local_space_openness` | 0.4868 | 0.2784 | 0.0011 | 0.9818 |
| `target_du` | -0.0001 | 0.0012 | -0.0245 | 0.0264 |
| `target_dv` | 0.0001 | 0.0015 | -0.0280 | 0.0365 |
| `target_turn_rate` | 0.0015 | 0.4036 | -1.0000 | 1.0000 |

## Schema columns

`track_id`, `frame_idx`, `u`, `v`, `du`, `dv`, `speed_rel`, `heading_sin`, `heading_cos`, `turn_rate_rel`, `is_stop`, `is_shift`, `obstacle_clearance_pct`, `boundary_clearance_pct`, `entrance_affinity_pct`, `local_space_openness`, `target_du`, `target_dv`, `target_turn_rate`