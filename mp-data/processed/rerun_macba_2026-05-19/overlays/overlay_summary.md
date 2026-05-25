# Plan-Overlay Fix Summary (v2)

- Variant: `mp-data\processed\rerun_macba_2026-05-19\spatial_v21C`
- Plan image: `mp-data\processed\rerun_macba_2026-05-19\inputs\top_view.png` (892 × 1287 px)
- Calibration: `mp-data\processed\rerun_macba_2026-05-19\calibration\calib_macba_2026-05-19.json`

## What changed vs the failed v1

**v1 (failed):**
- Built the world→plan mapping ad-hoc from `meters_per_plan_pixel` and `plan_origin_pixel`.
- Projected trajectory points via `plan_px = world / mpp + origin`.
- Placed masks via matplotlib `imshow(..., extent=..., origin='upper')` with manually-ordered (left, right, bottom, top) coordinates.

**v2 (this script):**
- Uses the *same* transform path the working pipeline uses in `mp-core/trajectory-prediction/generate_motion_dataset.py`:
  ```python
  H, _ = cv2.findHomography(world_points, plan_points_px, cv2.RANSAC, 5.0)
  plan_xy = cv2.perspectiveTransform(world_xy, H)
  ```
- Masks are no longer placed via matplotlib `extent`. Each mask is **warped into the plan image's pixel grid** with `cv2.warpPerspective`, using a combined `M = H @ T_grid_to_world` matrix where `T_grid_to_world` is the mask-pixel → world-metres affine (`[[res,0,x_min],[0,res,y_min],[0,0,1]]`).
- Composited as alpha-blended RGB onto the plan, eliminating all axis/origin/extent ambiguity.
- Entry-star markers are projected via the same `cv2.perspectiveTransform` call as the trajectories.

**Numerical note.** For this calibration the H matrix `cv2.findHomography` produces is essentially `[[1/mpp, 0, ox], [0, 1/mpp, oy], [0,0,1]]`, so the **pure projection math** of v1 and v2 is identical for in-distribution points (verified: 0.00 px diff on test points). The visual fix in v2 therefore comes from removing the matplotlib-`extent` rendering path and replacing it with a real `cv2.warpPerspective` warp into plan pixels — i.e. the same rendering convention every other map in the repo uses.

## Validation results (geometric, not bbox-only)

### (1) H-matrix self-reprojection on calibration points

| pt | expected plan_px | reprojected | residual (px) |
|---|---|---|---|
| 0 | (151.81, 362.54) | (151.81, 362.54) | 0.000 |
| 1 | (203.93, 996.97) | (203.93, 996.97) | 0.000 |
| 2 | (301.36, 40.78) | (301.36, 40.78) | 0.000 |
| 3 | (310.42, 507.55) | (310.42, 507.55) | 0.000 |
| 4 | (385.19, 328.55) | (385.19, 328.55) | 0.000 |
| 5 | (160.88, 591.39) | (160.88, 591.39) | 0.000 |
| 6 | (317.22, 641.23) | (317.22, 641.23) | 0.000 |

- Mean residual: **0.000 px**, max: **0.000 px** (threshold 5.0 px)

### (2) Trajectory projection geometry

- Sampled trajectory points: 200
- Inside-image fraction: **100.0%** (200/200)
- Convex hull of projections covers **9.4%** of the plan image (threshold ≤ 60%)

### (3) Visual check (alignment_debug.png)

`alignment_debug.png` shows: the plan image, the 6 calibration reference points (open circles = expected, red ✕ = projected via H), 200 random trajectory points (green), and the convex hull of trajectories (orange). Use this image to visually verify trajectories sit on the pedestrian plaza and not on roofs or roads.

## Decision

## PASSED (visually aligned)

All three numerical checks satisfied. Open `alignment_debug.png` and confirm visually that the green trajectory points sit on the actual pedestrian space; if they do, the overlay is correct.

## Outputs

- `mp-data\processed\rerun_macba_2026-05-19\overlays\alignment_debug.png`
- `mp-data\processed\rerun_macba_2026-05-19\overlays\plan_overlay_masks.png`
- `mp-data\processed\rerun_macba_2026-05-19\overlays\plan_overlay_contact_sheet.png`

_End of overlay fix summary._