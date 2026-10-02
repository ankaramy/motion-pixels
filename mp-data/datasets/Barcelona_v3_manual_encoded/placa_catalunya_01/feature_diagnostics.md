# Feature Diagnostics - placa_catalunya_01 (Encoder V3 manual)

**Status: PASS**

- Rows: 358351
- Tracks: 1537
- In-bounds (mapped inside mask image): 98.009%
- Inside walkable: 89.201% (91.013% of in-bounds)
- Inside obstacle: 4.747%
- Invalid/NaN distance rows: 1.991%
- Invalid/NaN clearance rows: 1.994%
- Mean metres/pixel: 0.047421
- Transform RMS: 0.0 px (anisotropy 1.0)

## Feature ranges (finite values)

| Feature | min | mean | max |
|---|---|---|---|
| dist_to_obstacle_v3_m | 0.0 | 2.6284 | 7.3976 |
| dist_to_walkable_boundary_v3_m | 0.0 | 2.5722 | 8.2399 |
| clearance_forward_v3_m | 0.25 | 12.1682 | 25.0 |
| clearance_left_v3_m | 0.25 | 11.0421 | 25.0 |
| clearance_right_v3_m | 0.25 | 11.3306 | 25.0 |

## Warnings

- 0.0% rows had undefined heading (stationary); directional features NaN there.
