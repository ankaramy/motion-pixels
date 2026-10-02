# Barcelona_v1 — Spatial Encoding Report (`spatial_v21C`)

**Phase:** Production encoding of the frozen `Barcelona_v1_filtered_250m` dataset.
**Date:** 2026-06-06
**Encoder:** `mp-core/trajectory-prediction/encode_spatial_auto_v2.py` (frozen), run in the
**v2.1C** configuration via the in-process override pattern used by the canonical runner
`rerun_spatial_pipeline.py` (Step 4). The encoder file was **not modified**.

Batch driver (reuses the frozen encoder unchanged):
`mp-core/trajectory-prediction/run_barcelona_v1_encoding.py`

### v2.1C configuration (overrides applied to the frozen encoder)

| constant | value |
|---|---|
| `ENVELOPE_DILATE_M`  | 4.0 |
| `WALKABLE_CLOSE_M`   | 1.5 |
| `DBSCAN_EPS_M`       | 2.75 |
| `DBSCAN_MIN_SAMPLES` | 12 |
| `GRID_RES_M` | 0.10 (default) |
| `WALKABLE_DILATE_M` | 0.40 (default) |
| `OBSTACLE_MIN_AREA_M2` | 0.30 (default) |
| `LONG_TRACK_MIN_LEN` | 10 frames (default) |

**Input (frozen, read-only):** `<rec>/filtered_250m/trajectories_world_filtered_250m.csv`
**Output namespace (new):** `new_datasets/Barcelona_v1_encoded/<rec>/spatial_v21C/`

> Note on feature naming: the v2.1C encoder produces the **raw** spatial features
> `dist_to_obstacle`, `dist_to_boundary`, `dist_to_entrance`. The normalized
> variants (`*_norm`) and `entrance_affinity` referenced in the task brief are
> created **downstream** during dataset assembly (`make_bridge_dataset.py` /
> `build_master_dataset.py`), which is intentionally **not** part of this
> encoding-only phase. They are therefore not expected in these outputs.

---

## Per-recording results

### 1. esplanade_espanya_01 — **SAFE FOR TRAINING: YES**
- Rows encoded: **283,875** · Tracks: **568** · Entrance clusters: **7**
- Artifacts: all 11 present (encoded CSV, walkable/obstacle/boundary masks,
  entry_exit_points.csv, 3 distance maps, 2 debug plots, summary.md)
- Features (no NaN, no Inf):

  | feature | mean | std | min | max | distinct |
  |---|---|---|---|---|---|
  | dist_to_obstacle | 2.664 | 1.433 | 0.000 | 15.875 | 1,020 |
  | dist_to_boundary | 2.579 | 1.461 | 0.316 | 19.860 | 1,224 |
  | dist_to_entrance | 15.229 | 9.117 | 0.000 | 41.292 | 17,647 |
- Walkable mask = coherent elongated esplanade; entrances at both ends. ✔ varying, no degeneracy.

### 2. stairs_montjuic_01 — **SAFE FOR TRAINING: YES (minor note)**
- Rows encoded: **128,259** · Tracks: **392** · Entrance clusters: **1**
- Artifacts: all 11 present
- Features (no NaN, no Inf):

  | feature | mean | std | min | max | distinct |
  |---|---|---|---|---|---|
  | dist_to_obstacle | 1.708 | 0.928 | 0.000 | 4.294 | 518 |
  | dist_to_boundary | 1.619 | 0.928 | 0.316 | 6.760 | 502 |
  | dist_to_entrance | 5.665 | 2.483 | 0.000 | 16.792 | 2,438 |
- **Note:** only **1** entrance cluster detected (single dominant flow corridor on
  the stairs). `dist_to_entrance` is still well-varying (std 2.48 m), so the feature
  is usable — but with a single cluster it effectively encodes radial distance from
  one node. Acceptable; flagged for awareness, not a blocker.

### 3. red_bridge_combined_01 — **SAFE FOR TRAINING: YES**
- Rows encoded: **87,076** · Tracks: **716** · Entrance clusters: **3**
- Artifacts: all 11 present
- Features (no NaN, no Inf):

  | feature | mean | std | min | max | distinct |
  |---|---|---|---|---|---|
  | dist_to_obstacle | 0.886 | 0.396 | 0.412 | 2.617 | 180 |
  | dist_to_boundary | 0.798 | 0.393 | 0.316 | 2.518 | 171 |
  | dist_to_entrance | 2.468 | 1.685 | 0.000 | 8.875 | 1,177 |
- Smallest physical site → smallest distance magnitudes; two-lobe walkable mask
  (bridge span + landing) matches geometry. ✔ varying, no degeneracy.

### 4. placa_catalunya_01 — **SAFE FOR TRAINING: YES**
- Rows encoded: **358,351** · Tracks: **1,537** · Entrance clusters: **7**
- Artifacts: all 11 present
- Features (no NaN, no Inf):

  | feature | mean | std | min | max | distinct |
  |---|---|---|---|---|---|
  | dist_to_obstacle | 3.174 | 1.774 | 0.000 | 8.809 | 1,642 |
  | dist_to_boundary | 3.102 | 1.797 | 0.316 | 9.626 | 1,736 |
  | dist_to_entrance | 9.836 | 5.162 | 0.000 | 19.729 | 6,733 |
- Largest track count; broad open plaza + separate strip in walkable mask. ✔

### 5. placa_espanya_01 — **SAFE FOR TRAINING: YES**
- Rows encoded: **213,618** · Tracks: **923** · Entrance clusters: **12**
- Artifacts: all 11 present
- Features (no NaN, no Inf):

  | feature | mean | std | min | max | distinct |
  |---|---|---|---|---|---|
  | dist_to_obstacle | 6.130 | 5.482 | 0.000 | 37.811 | 6,390 |
  | dist_to_boundary | 6.570 | 5.940 | 0.316 | 41.789 | 6,650 |
  | dist_to_entrance | 12.165 | 8.598 | 0.000 | 130.313 | 16,697 |
- Largest spatial extent (~337 m span) → largest distance magnitudes and richest
  entrance structure (12 clusters). Previously a tracking outlier; tracking is now
  resolved and encoding is clean. ✔

---

## Cross-recording validation summary

| recording | rows | tracks | entr. clusters | NaN | Inf | all features vary | all artifacts |
|---|---|---|---|---|---|---|---|
| esplanade_espanya_01 | 283,875 | 568 | 7 | 0 | 0 | yes | yes |
| stairs_montjuic_01 | 128,259 | 392 | 1 | 0 | 0 | yes | yes |
| red_bridge_combined_01 | 87,076 | 716 | 3 | 0 | 0 | yes | yes |
| placa_catalunya_01 | 358,351 | 1,537 | 7 | 0 | 0 | yes | yes |
| placa_espanya_01 | 213,618 | 923 | 12 | 0 | 0 | yes | yes |
| **TOTAL** | **1,071,179** | **4,136** | **30** | **0** | **0** | — | — |

- **Are features varying?** Yes — every feature on every recording has std ≫ 0 and
  hundreds–to–tens-of-thousands of distinct values.
- **Near-constant?** None.
- **NaNs?** None.
- **Infinities?** None.
- **Do masks visually make sense?** Yes — walkable masks trace the trajectory clouds,
  obstacles fill the interior holes, boundaries are the outer contours, distance
  fields are smooth radial gradients, and entrance clusters sit at flow extremities
  (see `_contact_sheets/`).

> Note: `dist_to_boundary` has a hard floor of **0.316 m** on every recording. This is
> expected: it is the smallest off-contour grid distance (no trajectory point lands
> exactly on the 1-px boundary rim). Not a degeneracy.

---

## Output locations

```
new_datasets/Barcelona_v1_encoded/
├── esplanade_espanya_01/spatial_v21C/
├── stairs_montjuic_01/spatial_v21C/
├── red_bridge_combined_01/spatial_v21C/
├── placa_catalunya_01/spatial_v21C/
├── placa_espanya_01/spatial_v21C/
│       trajectories_encoded.csv
│       walkable_mask.png  obstacle_mask.png  boundary_mask.png
│       distance_to_obstacle_m.png  distance_to_boundary_m.png  distance_to_entrance_m.png
│       entry_exit_points.csv
│       debug_stages.png  debug_trajectory_sampling.png
│       summary.md
├── _contact_sheets/
│       contact_walkable.png
│       contact_obstacle.png
│       contact_boundary.png
│       contact_distance_fields.png
│       contact_entrance_points.png
├── _encoding_batch_results.json
├── _encoded_validation_audit.json
└── Barcelona_v1_encoding_report.md   ← this file
```

---

## SAFE FOR TRAINING list

| recording | verdict |
|---|---|
| esplanade_espanya_01 | ✅ YES |
| stairs_montjuic_01 | ✅ YES (1 entrance cluster — awareness note) |
| red_bridge_combined_01 | ✅ YES |
| placa_catalunya_01 | ✅ YES |
| placa_espanya_01 | ✅ YES |

**All 5 approved recordings: SAFE FOR TRAINING.**

## Blockers before dataset assembly

- **None.** The frozen `spatial_v21C` encoder generalized cleanly from the sandbox
  dataset to all 5 validated Barcelona recordings (no NaN/Inf, all features vary,
  all artifacts produced, masks geometrically sensible).
- Single advisory: `stairs_montjuic_01` produced only one entrance/exit cluster.
  Usable as-is; revisit only if entrance-affinity features look weak after the
  downstream `make_bridge_dataset.py` normalization step.

## Constraints honoured

- No retracking, no recalibration, no homography changes.
- Frozen filtered CSVs read-only; not modified.
- Encoder/training/bridge/master-dataset logic not modified; no model trained.
- No master dataset built; phase ends at encoding + encoded-space validation.
```
