# Barcelona_v3 Model C — Dataset Schema Report

Faithful V3 mirror of the Barcelona_v1 Model C dataset. Identical recipe,
split, and schema; the **only** difference is that the two spatial features
come from Encoder V3 real architectural masks instead of the old
trajectory-coverage masks.

- Rows: **1,064,379**  Tracks: **3,534**
- Min track length filter: 11
- Split: recording-level (train=esplanade/catalunya/espanya, val=stairs, test=red_bridge)

## Model C feature order (10) + targets

```
u, v, du, dv, speed, heading_sin, heading_cos, turn_rate, dist_to_obstacle_norm, dist_to_boundary_norm
targets: target_du, target_dv
```

## Normalization (per recording)

- u, v: MinMax over recording world_x / world_y
- du, dv, speed: raw metres/step (StandardScaler fit at train time)
- turn_rate: radians in (-pi, pi]
- dist_to_obstacle_norm, dist_to_boundary_norm: percentile rank ascending (0 = closest, 1 = farthest) of the V3 manual-mask distances
- OOB V3 distances imputed to recording max finite distance before ranking

## Column statistics

| column | mean | std | min | max |
|---|---|---|---|---|
| `u` | 0.6252 | 0.2401 | 0.0000 | 1.0000 |
| `v` | 0.5251 | 0.3443 | 0.0000 | 1.0000 |
| `du` | -0.0031 | 0.2159 | -17.0939 | 24.7912 |
| `dv` | 0.0003 | 0.0655 | -8.0498 | 9.6802 |
| `speed` | 0.0951 | 0.2046 | 0.0000 | 25.1094 |
| `heading_sin` | 0.0083 | 0.5593 | -1.0000 | 1.0000 |
| `heading_cos` | -0.0032 | 0.8289 | -1.0000 | 1.0000 |
| `turn_rate` | 0.0039 | 1.4624 | -3.1416 | 3.1416 |
| `dist_to_obstacle_norm` | 0.5001 | 0.2883 | 0.0001 | 0.9996 |
| `dist_to_boundary_norm` | 0.4999 | 0.2883 | 0.0004 | 0.9996 |
| `target_du` | -0.0032 | 0.2184 | -17.0939 | 24.7912 |
| `target_dv` | 0.0002 | 0.0665 | -8.0498 | 9.6802 |