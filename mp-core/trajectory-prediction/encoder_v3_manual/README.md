# Encoder V3 — Manual Architectural Mask Encoding

Computes spatial features for every trajectory point from **manually annotated
architectural masks** (walkable / obstacle), instead of the old
trajectory-occupancy masks.

## Why Encoder V3 exists

The Encoding Truth Audit proved the old encoder
(`mp-core/trajectory-prediction/encode_spatial_auto_v2.py`) derived obstacle
masks from **trajectory coverage**, not architecture:

```
walkable = where people walked
obstacle = fill_holes(dilate(walkable)) & ~walkable
```

That produced *false architectural absence*. The decisive case — Plaça
Catalunya — encoded as **0 interior obstacle components / 0.0 m²**, even though
the plaza is full of trees, planters and islands. So the prior conclusion
"architecture contains no turn signal" was an **encoding failure**, not a fact
about the world.

Encoder V3 replaces the trajectory-derived mask with a **human-approved
architectural mask** (Phases 1 / 1.5) and recomputes spatial features against
real structure.

### How it differs from `encode_spatial_auto_v2.py`

| | Old (`encode_spatial_auto_v2.py`) | Encoder V3 (this) |
|---|---|---|
| Obstacle source | trajectory occupancy | **manual architectural mask** |
| Walkable source | trajectory occupancy | **manual architectural mask** |
| Coordinate basis | implicit | explicit world↔plan similarity, **residual-validated** |
| Directional features | none | forward/left/right clearance, asymmetry, bearings |
| Touches old outputs | — | **No** (additive prototype, separate output tree) |

This package is fully additive: it never imports, modifies, or runs the old
encoder, the old encoded CSVs, the master dataset, or any model.

## Coordinate alignment (the critical issue)

Masks live in **plan-image pixel space**; trajectories are in **world metres**.
`calib.json` uses `method = "plan_scale_plus_correspondences"` and provides
`world_points ↔ plan_points_px` correspondences plus a constant
`meters_per_plan_pixel`.

V3 fits the world↔plan transform by least squares from those correspondences and
**validates it**: across all five recordings the RMS residual is ~0 px, the two
singular values of the linear part are equal (isotropic), and the fitted scale
equals `1 / meters_per_plan_pixel`. So the transform is an exact similarity and
`plan-pixel distance × meters_per_plan_pixel = metres` (constant scale, no local
variation). If a recording's transform were ambiguous (residual > 5 px,
anisotropy > 1.05, or scale mismatch > 5%) it is flagged `ambiguous` and the
recording fails — alignment is never silently assumed.

## Inputs (per recording)

```
new_datasets\<id>\filtered_250m\trajectories_world_filtered_250m.csv   (has world_x, world_y)
new_datasets\<id>\calibration\calib.json
mp-data\annotations\manual_masks_v3\<id>\walkable_mask_v3_manual.png
mp-data\annotations\manual_masks_v3\<id>\obstacle_mask_v3_manual.png
```

## Outputs (per recording)

```
new_datasets\Barcelona_v3_manual_encoded\<id>\spatial_v3_manual\
    trajectories_encoded_v3.csv      original columns + V3 features
    encoding_v3_metadata.json        transform, scale, residuals, paths, params
    feature_diagnostics.json         metrics + warnings + status
    feature_diagnostics.md           human-readable summary
    mask_alignment_overlay.png       masks + sampled trajectory points
```
Aggregate report: `new_datasets\Barcelona_v3_manual_encoded\Encoder_V3_Manual_Report.md`.

## Features added to each row

| Column | Definition |
|---|---|
| `dist_to_obstacle_v3_m` | metres to nearest obstacle (red) pixel |
| `dist_to_walkable_boundary_v3_m` | metres to nearest edge of the walkable (green) region |
| `inside_walkable_v3` | 1 if the point maps inside the walkable mask |
| `inside_obstacle_v3` | 1 if the point maps inside the obstacle mask |
| `clearance_forward_v3_m` | metres along heading until obstacle / leaving walkable / max (25 m) |
| `clearance_left_v3_m` | same, heading + 90° (CCW in world frame) |
| `clearance_right_v3_m` | same, heading − 90° |
| `clearance_asymmetry_v3` | `(right − left) / (right + left + ε)` |
| `obstacle_bearing_sin_v3` / `_cos_v3` | bearing from heading to nearest obstacle (world frame) |
| `boundary_bearing_sin_v3` / `_cos_v3` | bearing from heading to nearest walkable boundary |
| `nearest_obstacle_px_x/y`, `nearest_boundary_px_x/y` | nearest pixel locations (plan space) |
| `plan_px_x_v3`, `plan_px_y_v3` | trajectory point mapped to plan pixels |
| `inbounds_v3` | 1 if mapped point is inside the mask image |
| `heading_v3_rad`, `heading_valid_v3` | per-track heading (radians) and validity flag |

**Heading:** taken from consecutive world positions per track
(`atan2(dy, dx)`), forward/back-filled within a track; rows of a fully
stationary track are flagged `heading_valid_v3 = 0` and their directional
features set to NaN / 0.

Out-of-bounds points (mapped outside the plan crop) get NaN distances /
clearances, `inside_* = 0`, and `inbounds_v3 = 0`.

## Commands

Encode one recording:
```
python encoder_v3_manual\encode_recording_v3_manual.py --recording placa_catalunya_01
```

Encode all five validated recordings (writes the aggregate report):
```
python encoder_v3_manual\batch_encode_v3_manual.py --all
```

Validate outputs:
```
python encoder_v3_manual\validate_v3_outputs.py --recording placa_catalunya_01
python encoder_v3_manual\validate_v3_outputs.py --all
```

Run from `mp-core\trajectory-prediction`.

## Validated recordings only

`stairs_montjuic_01`, `red_bridge_combined_01`, `esplanade_espanya_01`,
`placa_espanya_01`, `placa_catalunya_01`.
Excluded: `placa_montjuic_01`, `stairs_montjuic_02` (homography instability).

## Validation criteria

A recording is **PASS** when the transform is unambiguous, most points are
in-bounds, most in-bounds points are inside walkable, few are inside obstacle,
and distances/clearances are finite for most rows. It is **CHECK** if it maps
largely outside the plan crop, has an elevated inside-obstacle rate, or low
walkable rate (usually a mask-coverage / mask-edge issue, **not** a coordinate
problem). It is **FAIL** only on an ambiguous transform or gross misalignment.
Failures are written and clearly marked, never hidden.

## ⚠ Do not train until V3 diagnostics pass

This phase ends at encoding + diagnostics. **Do not** build a new master
dataset, run the turn classifier, train Model C, or compare model performance
using these features until the diagnostics (especially the CHECK recordings)
are reviewed and the masks accepted. The only question this phase answers is:
*can we compute spatial features from accepted manual architectural masks and
verify they align with trajectories?*
