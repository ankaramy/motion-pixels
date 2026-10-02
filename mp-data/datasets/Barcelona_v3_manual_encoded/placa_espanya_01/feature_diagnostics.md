# Feature Diagnostics - placa_espanya_01 (Encoder V3 manual)

**Status: PASS**

- Rows: 213618
- Tracks: 923
- In-bounds (mapped inside mask image): 91.698%
- Inside walkable: 85.444% (93.18% of in-bounds)
- Inside obstacle: 6.055%
- Invalid/NaN distance rows: 8.302%
- Invalid/NaN clearance rows: 8.302%
- Mean metres/pixel: 0.099727
- Transform RMS: 0.0 px (anisotropy 1.0)

## Feature ranges (finite values)

| Feature | min | mean | max |
|---|---|---|---|
| dist_to_obstacle_v3_m | 0.0 | 10.2849 | 30.7607 |
| dist_to_walkable_boundary_v3_m | 0.0 | 10.7134 | 30.622 |
| clearance_forward_v3_m | 0.25 | 20.4165 | 25.0 |
| clearance_left_v3_m | 0.25 | 16.7101 | 25.0 |
| clearance_right_v3_m | 0.25 | 16.977 | 25.0 |

## Warnings

- 0.0% rows had undefined heading (stationary); directional features NaN there.
