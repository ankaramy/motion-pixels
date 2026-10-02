# Encoder V3 (Manual Architectural Masks) - Aggregate Report

Spatial features computed from **manually annotated architectural masks**
(walkable / obstacle), replacing the old trajectory-occupancy masks of
`encode_spatial_auto_v2.py`. Additive prototype: no old output, master
dataset, classifier, or model was touched.

**Coordinate alignment:** `calib.json` uses `plan_scale_plus_correspondences`; world<->plan is an exact isotropic similarity fitted from correspondences (RMS residual ~0 px, fitted scale == 1/meters_per_plan_pixel), so plan-pixel distance x mpp = metres.

## Summary

| Recording | Status | Rows | Tracks | inWalk % | inObst % | mean dist_obst (m) | mean dist_bnd (m) | mean fwd/left/right clr (m) |
|---|---|---|---|---|---|---|---|---|
| stairs_montjuic_01 | **PASS** | 128259 | 392 | 94.776 | 1.824 | 2.1994 | 2.0426 | 11.0737 / 8.8655 / 8.9063 |
| red_bridge_combined_01 | **CHECK** | 87076 | 716 | 72.424 | 0.023 | 3.9013 | 3.5746 | 10.0676 / 10.2604 / 8.3066 |
| esplanade_espanya_01 | **PASS** | 283875 | 568 | 91.973 | 7.837 | 3.6228 | 4.2305 | 18.9149 / 7.1028 / 7.3461 |
| placa_espanya_01 | **PASS** | 213618 | 923 | 85.444 | 6.055 | 10.2849 | 10.7134 | 20.4165 / 16.7101 / 16.977 |
| placa_catalunya_01 | **PASS** | 358351 | 1537 | 89.201 | 4.747 | 2.6284 | 2.5722 | 12.1682 / 11.0421 / 11.3306 |

## Per-recording detail

### stairs_montjuic_01 - PASS

- Rows / tracks: 128259 / 392
- In-bounds: 96.605%
- Inside walkable: 94.776% (98.107% of in-bounds)
- Inside obstacle: 1.824%
- dist_to_obstacle_v3_m: {'min': 0.0, 'mean': 2.1994, 'max': 6.7505}
- dist_to_walkable_boundary_v3_m: {'min': 0.0, 'mean': 2.0426, 'max': 6.4887}
- clearance fwd/left/right (m): {'min': 0.25, 'mean': 11.0737, 'max': 25.0} / {'min': 0.25, 'mean': 8.8655, 'max': 25.0} / {'min': 0.25, 'mean': 8.9063, 'max': 25.0}
- Invalid dist / clearance NaN: 3.395% / 3.397%
- Output: `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v3_manual_encoded\stairs_montjuic_01\spatial_v3_manual\trajectories_encoded_v3.csv`

  **Comparison vs old encoder (trajectory-derived masks):**
  - OLD dist_to_obstacle [min/mean/max]: [0.0, 1.708, 4.2942]
  - NEW dist_to_obstacle_v3_m [min/mean/max]: [0.0, 2.1994, 6.7505]
  - OLD dist_to_boundary [min/mean/max]: [0.3162, 1.6194, 6.7602]
  - NOTE: old obstacle masks were derived from trajectory coverage, not architecture; numeric ranges are NOT directly comparable and this is NOT a model-quality claim.

  Warnings:
  - 0.0% rows had undefined heading (stationary); directional features NaN there.

### red_bridge_combined_01 - CHECK

- Rows / tracks: 87076 / 716
- In-bounds: 72.451%
- Inside walkable: 72.424% (99.964% of in-bounds)
- Inside obstacle: 0.023%
- dist_to_obstacle_v3_m: {'min': 0.0, 'mean': 3.9013, 'max': 5.4232}
- dist_to_walkable_boundary_v3_m: {'min': 0.0, 'mean': 3.5746, 'max': 5.36}
- clearance fwd/left/right (m): {'min': 0.25, 'mean': 10.0676, 'max': 25.0} / {'min': 0.25, 'mean': 10.2604, 'max': 25.0} / {'min': 0.25, 'mean': 8.3066, 'max': 25.0}
- Invalid dist / clearance NaN: 27.549% / 27.556%
- Output: `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v3_manual_encoded\red_bridge_combined_01\spatial_v3_manual\trajectories_encoded_v3.csv`

  **Comparison vs old encoder (trajectory-derived masks):**
  - OLD dist_to_obstacle [min/mean/max]: [0.4123, 0.8859, 2.6173]
  - NEW dist_to_obstacle_v3_m [min/mean/max]: [0.0, 3.9013, 5.4232]
  - OLD dist_to_boundary [min/mean/max]: [0.3162, 0.7981, 2.5179]
  - NOTE: old obstacle masks were derived from trajectory coverage, not architecture; numeric ranges are NOT directly comparable and this is NOT a model-quality claim.

  Warnings:
  - 27.5% of points map OUTSIDE the mask image (plan crop does not cover full trajectory extent).
  - 0.01% rows had undefined heading (stationary); directional features NaN there.

### esplanade_espanya_01 - PASS

- Rows / tracks: 283875 / 568
- In-bounds: 99.918%
- Inside walkable: 91.973% (92.048% of in-bounds)
- Inside obstacle: 7.837%
- dist_to_obstacle_v3_m: {'min': 0.0, 'mean': 3.6228, 'max': 11.2364}
- dist_to_walkable_boundary_v3_m: {'min': 0.0, 'mean': 4.2305, 'max': 20.9682}
- clearance fwd/left/right (m): {'min': 0.25, 'mean': 18.9149, 'max': 25.0} / {'min': 0.25, 'mean': 7.1028, 'max': 25.0} / {'min': 0.25, 'mean': 7.3461, 'max': 25.0}
- Invalid dist / clearance NaN: 0.082% / 0.083%
- Output: `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v3_manual_encoded\esplanade_espanya_01\spatial_v3_manual\trajectories_encoded_v3.csv`

  **Comparison vs old encoder (trajectory-derived masks):**
  - OLD dist_to_obstacle [min/mean/max]: [0.0, 2.664, 15.8751]
  - NEW dist_to_obstacle_v3_m [min/mean/max]: [0.0, 3.6228, 11.2364]
  - OLD dist_to_boundary [min/mean/max]: [0.3162, 2.5794, 19.8595]
  - NOTE: old obstacle masks were derived from trajectory coverage, not architecture; numeric ranges are NOT directly comparable and this is NOT a model-quality claim.

  Warnings:
  - 0.0% rows had undefined heading (stationary); directional features NaN there.

### placa_espanya_01 - PASS

- Rows / tracks: 213618 / 923
- In-bounds: 91.698%
- Inside walkable: 85.444% (93.18% of in-bounds)
- Inside obstacle: 6.055%
- dist_to_obstacle_v3_m: {'min': 0.0, 'mean': 10.2849, 'max': 30.7607}
- dist_to_walkable_boundary_v3_m: {'min': 0.0, 'mean': 10.7134, 'max': 30.622}
- clearance fwd/left/right (m): {'min': 0.25, 'mean': 20.4165, 'max': 25.0} / {'min': 0.25, 'mean': 16.7101, 'max': 25.0} / {'min': 0.25, 'mean': 16.977, 'max': 25.0}
- Invalid dist / clearance NaN: 8.302% / 8.302%
- Output: `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v3_manual_encoded\placa_espanya_01\spatial_v3_manual\trajectories_encoded_v3.csv`

  **Comparison vs old encoder (trajectory-derived masks):**
  - OLD dist_to_obstacle [min/mean/max]: [0.0, 6.1299, 37.8114]
  - NEW dist_to_obstacle_v3_m [min/mean/max]: [0.0, 10.2849, 30.7607]
  - OLD dist_to_boundary [min/mean/max]: [0.3162, 6.5705, 41.7892]
  - NOTE: old obstacle masks were derived from trajectory coverage, not architecture; numeric ranges are NOT directly comparable and this is NOT a model-quality claim.

  Warnings:
  - 0.0% rows had undefined heading (stationary); directional features NaN there.

### placa_catalunya_01 - PASS

- Rows / tracks: 358351 / 1537
- In-bounds: 98.009%
- Inside walkable: 89.201% (91.013% of in-bounds)
- Inside obstacle: 4.747%
- dist_to_obstacle_v3_m: {'min': 0.0, 'mean': 2.6284, 'max': 7.3976}
- dist_to_walkable_boundary_v3_m: {'min': 0.0, 'mean': 2.5722, 'max': 8.2399}
- clearance fwd/left/right (m): {'min': 0.25, 'mean': 12.1682, 'max': 25.0} / {'min': 0.25, 'mean': 11.0421, 'max': 25.0} / {'min': 0.25, 'mean': 11.3306, 'max': 25.0}
- Invalid dist / clearance NaN: 1.991% / 1.994%
- Output: `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v3_manual_encoded\placa_catalunya_01\spatial_v3_manual\trajectories_encoded_v3.csv`

  **Comparison vs old encoder (trajectory-derived masks):**
  - OLD dist_to_obstacle [min/mean/max]: [0.0, 3.1737, 8.8091]
  - NEW dist_to_obstacle_v3_m [min/mean/max]: [0.0, 2.6284, 7.3976]
  - OLD dist_to_boundary [min/mean/max]: [0.3162, 3.1022, 9.626]
  - NOTE: old obstacle masks were derived from trajectory coverage, not architecture; numeric ranges are NOT directly comparable and this is NOT a model-quality claim.

  Warnings:
  - 0.0% rows had undefined heading (stationary); directional features NaN there.

## Verdict

- PASS: 4   CHECK: 1   FAIL: 0

Phase 2 question answered: spatial features CAN be computed from the accepted manual architectural masks and verified to align with trajectories. Do NOT train, rebuild the master dataset, or run the classifier until these diagnostics are reviewed and accepted.
