# Feature Diagnostics - stairs_montjuic_01 (Encoder V3 manual)

**Status: PASS**

- Rows: 128259
- Tracks: 392
- In-bounds (mapped inside mask image): 96.605%
- Inside walkable: 94.776% (98.107% of in-bounds)
- Inside obstacle: 1.824%
- Invalid/NaN distance rows: 3.395%
- Invalid/NaN clearance rows: 3.397%
- Mean metres/pixel: 0.02171
- Transform RMS: 0.0 px (anisotropy 1.0)

## Feature ranges (finite values)

| Feature | min | mean | max |
|---|---|---|---|
| dist_to_obstacle_v3_m | 0.0 | 2.1994 | 6.7505 |
| dist_to_walkable_boundary_v3_m | 0.0 | 2.0426 | 6.4887 |
| clearance_forward_v3_m | 0.25 | 11.0737 | 25.0 |
| clearance_left_v3_m | 0.25 | 8.8655 | 25.0 |
| clearance_right_v3_m | 0.25 | 8.9063 | 25.0 |

## Warnings

- 0.0% rows had undefined heading (stationary); directional features NaN there.
