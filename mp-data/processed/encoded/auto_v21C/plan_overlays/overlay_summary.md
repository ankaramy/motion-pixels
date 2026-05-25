# Plan-overlay Summary

- Variant: `mp-data\processed\encoded\auto_v21C`
- Plan image: `mp-data\raw\images\top-down.png` (892 × 1287 pixels)
- Calibration source: `mp-data\raw\calibration\calib_macba.json`

## Transformation source

The plan image ↔ world-coordinates relationship is recorded in the calibration JSON as a simple affine map.

```
meters_per_plan_pixel = 0.076638
plan_origin_pixel     = (579.996, 258.030)
invert_plan_y         = False
```

## Reprojection logic

World → plan pixel (used for trajectories and entry stars):

```
plan_px_x = world_x / meters_per_plan_pixel + origin_x
plan_px_y = world_y / meters_per_plan_pixel + origin_y   # negate world_y first if invert_plan_y
```

Mask bounding box (used for matplotlib `extent`):

```
(left, right) = world_x bounds projected to plan_px_x
(top,  bottom) = world_y bounds projected to plan_px_y (in image-y-down convention)
```

## Alignment validation

- 100 random trajectory points projected to plan pixels
- Inside-image fraction: **100.0%** (100/100)
- Projected x range: [583.6, 745.2] px (image width 892)
- Projected y range: [49.1, 588.3] px (image height 1287)
- Calibration's recorded reprojection error (from the original calibration step): mean 0.048 m, max 0.0972 m

- Threshold for rendering: ≥ 85% of sampled points must land inside the image.
- **Decision: PASSED** (OK)

## Assumptions

- The calibration in `calib_macba.json` matches the plan image at `mp-data/raw/images/top-down.png` (the calibration names it `top-view.png`; the size and origin coordinates match the file we have).
- The mapping is treated as a global affine. No per-region distortion is applied (the calibration's full homography is for the camera-to-world map, not the plan-to-world map).
- Mask resolution is the encoder default (GRID_RES_M = 0.10 m/pixel) and is scaled by `0.10 / mpp` ≈ 1.305× during overlay.

## Outputs

- `mp-data\processed\encoded\auto_v21C\plan_overlays\plan_overlay_masks.png`
- `mp-data\processed\encoded\auto_v21C\plan_overlays\plan_overlay_trajectories.png`
- `mp-data\processed\encoded\auto_v21C\plan_overlays\plan_overlay_distance_obstacle.png`
- `mp-data\processed\encoded\auto_v21C\plan_overlays\plan_overlay_distance_boundary.png`
- `mp-data\processed\encoded\auto_v21C\plan_overlays\plan_overlay_distance_entrance.png`
- `mp-data\processed\encoded\auto_v21C\plan_overlays\plan_overlay_contact_sheet.png`

_End of overlay summary._