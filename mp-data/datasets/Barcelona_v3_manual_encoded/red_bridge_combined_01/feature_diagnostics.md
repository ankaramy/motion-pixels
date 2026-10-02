# Feature Diagnostics - red_bridge_combined_01 (Encoder V3 manual)

**Status: CHECK**

- Rows: 87076
- Tracks: 716
- In-bounds (mapped inside mask image): 72.451%
- Inside walkable: 72.424% (99.964% of in-bounds)
- Inside obstacle: 0.023%
- Invalid/NaN distance rows: 27.549%
- Invalid/NaN clearance rows: 27.556%
- Mean metres/pixel: 0.044696
- Transform RMS: 0.0 px (anisotropy 1.0)

## Feature ranges (finite values)

| Feature | min | mean | max |
|---|---|---|---|
| dist_to_obstacle_v3_m | 0.0 | 3.9013 | 5.4232 |
| dist_to_walkable_boundary_v3_m | 0.0 | 3.5746 | 5.36 |
| clearance_forward_v3_m | 0.25 | 10.0676 | 25.0 |
| clearance_left_v3_m | 0.25 | 10.2604 | 25.0 |
| clearance_right_v3_m | 0.25 | 8.3066 | 25.0 |

## Warnings

- 27.5% of points map OUTSIDE the mask image (plan crop does not cover full trajectory extent).
- 0.01% rows had undefined heading (stationary); directional features NaN there.
