# Auto v2 — Visual Check Summary

_Visual previews of the auto v2 spatial encoding. Read-only: nothing in auto_v2/ was modified._

## Input files used

- ✓ `trajectories_encoded_auto_v2.csv`
- ✓ `walkable_mask.png`
- ✓ `obstacle_mask.png`
- ✓ `boundary_mask.png`
- ✓ `entry_exit_points.csv`

Base layer: trajectories and masks drawn on **synthetic world-space grid** (axes in metres). The top-down image was located but not warped onto the world canvas — the homography to world coordinates is unknown in this context, so the masks themselves provide the visual reference.

## Alignment

- World extent: x ∈ [-3.27, 17.03] m, y ∈ [-22.36, 32.04] m
- Grid: 203 × 544 pixels at 0.1 m/pixel
- Mask alignment to world grid: OK

## What each output shows

- `map_mask_overlay.png` — pale green walkable mask, dark red obstacle mask, blue boundary outline, gold ★ markers for DBSCAN entry/exit cluster centres. No trajectories.
- `map_trajectories_on_mask.png` — same mask layer plus every trajectory drawn as a thin dark grey line. The 10 longest tracks are highlighted in `tab10` colours so you can visually trace dominant flows.
- `map_distance_to_obstacle.png` — trajectory points coloured by `dist_to_obstacle` (viridis colormap). Bright = farther from the nearest auto-derived obstacle pixel.
- `map_distance_to_boundary.png` — trajectory points coloured by `dist_to_boundary` (magma).
- `map_distance_to_entrance.png` — trajectory points coloured by `dist_to_entrance` (plasma). Bright = farther from any DBSCAN entry/exit cluster.
- `visual_check_contact_sheet.png` — 1×4 sheet that bundles the mask overlay plus the three distance-coloured plots for one-glance review.

## Feature stats (from `trajectories_encoded_auto_v2.csv`)

| feature | n_valid | mean | median | min | max |
|---|---|---|---|---|---|
| `dist_to_obstacle` | 180,840 | 1.216 | 1.020 | 0.412 | 4.554 |
| `dist_to_boundary` | 180,840 | 1.409 | 1.200 | 0.316 | 4.455 |
| `dist_to_entrance` | 180,840 | 1.604 | 1.315 | 0.000 | 5.659 |

## Generated files

- `mp-data\processed\encoded\auto_v2\visual_checks\map_mask_overlay.png`
- `mp-data\processed\encoded\auto_v2\visual_checks\map_trajectories_on_mask.png`
- `mp-data\processed\encoded\auto_v2\visual_checks\map_distance_to_obstacle.png`
- `mp-data\processed\encoded\auto_v2\visual_checks\map_distance_to_boundary.png`
- `mp-data\processed\encoded\auto_v2\visual_checks\map_distance_to_entrance.png`
- `mp-data\processed\encoded\auto_v2\visual_checks\visual_check_contact_sheet.png`
- `mp-data\processed\encoded\auto_v2\visual_checks\visual_check_summary.md`

_End of visual check summary._