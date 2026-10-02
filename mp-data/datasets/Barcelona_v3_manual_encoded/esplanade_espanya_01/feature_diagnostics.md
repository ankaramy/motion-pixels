# Feature Diagnostics - esplanade_espanya_01 (Encoder V3 manual)

**Status: PASS**

- Rows: 283875
- Tracks: 568
- In-bounds (mapped inside mask image): 99.918%
- Inside walkable: 91.973% (92.048% of in-bounds)
- Inside obstacle: 7.837%
- Invalid/NaN distance rows: 0.082%
- Invalid/NaN clearance rows: 0.083%
- Mean metres/pixel: 0.124811
- Transform RMS: 0.0 px (anisotropy 1.0)

## Feature ranges (finite values)

| Feature | min | mean | max |
|---|---|---|---|
| dist_to_obstacle_v3_m | 0.0 | 3.6228 | 11.2364 |
| dist_to_walkable_boundary_v3_m | 0.0 | 4.2305 | 20.9682 |
| clearance_forward_v3_m | 0.25 | 18.9149 | 25.0 |
| clearance_left_v3_m | 0.25 | 7.1028 | 25.0 |
| clearance_right_v3_m | 0.25 | 7.3461 | 25.0 |

## Warnings

- 0.0% rows had undefined heading (stationary); directional features NaN there.
