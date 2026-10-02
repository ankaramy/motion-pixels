# Barcelona_v1_filtered_250m — Frozen Cleaned Dataset

_Generated 2026-06-05_

## Purpose

Remove **projection singularities** (homography blow-ups) from the world-coordinate trajectories before spatial encoding, **without** removing real far-field pedestrians. A far-field audit showed the farthest *real* pedestrian ≈ 228 m from the scene centroid, while the *nearest singularity* ≈ 471 m and the worst ≈ 539 km. A 250 m cap cleanly separates the two.

## Exact rule

For each recording, per observation:

```
median_x = median(world_x)
median_y = median(world_y)
dist     = sqrt((world_x - median_x)^2 + (world_y - median_y)^2)
keep     if dist <= 250 m   (drop if dist > 250 m)
```

Applied **per observation** (singular frames are trimmed); a track is only fully removed if *all* its observations are singular.

## Per-recording removals

| recording | rows before | rows after | rows removed | % | tracks before | tracks after | tracks removed | max dist before (m) | max dist after (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| esplanade_espanya_01 | 283875 | 283875 | 0 | 0.0% | 568 | 568 | 0 | 202.13 | 202.13 |
| stairs_montjuic_01 | 128259 | 128259 | 0 | 0.0% | 392 | 392 | 0 | 19.4 | 19.4 |
| red_bridge_combined_01 | 87076 | 87076 | 0 | 0.0% | 716 | 716 | 0 | 26.56 | 26.56 |
| placa_catalunya_01 | 358351 | 358351 | 0 | 0.0% | 1537 | 1537 | 0 | 101.8 | 101.8 |
| placa_montjuic_01 | 95700 | 93184 | 2516 | 2.629% | 171 | 165 | 6 | 539393.66 | 249.92 |
| stairs_montjuic_02 | 88600 | 88239 | 361 | 0.4074% | 450 | 439 | 11 | 471.33 | 249.99 |
| placa_espanya_01 | 213618 | 213618 | 0 | 0.0% | 923 | 923 | 0 | 228.27 | 228.27 |
| **TOTAL** | **1255479** | **1252602** | **2877** | **0.229%** | **4757** | **4740** | **17** | — | — |

## Outputs

- Per recording: `<recording>/filtered_250m/trajectories_world_filtered_250m.csv`
- Filter summary: `Barcelona_v1_filtered_250m_filter_summary.csv`

## Originals preserved

The original `tracking/trajectories_world.csv` for every recording was **NOT overwritten**. Cleaned copies are written to a separate `filtered_250m/` subfolder. calib.json, homography, image-space CSVs, overlays, and tracking outputs are untouched.

## Status

This is the **frozen dataset-cleaning policy applied before spatial encoding**. No tracking, YOLO, calibration, model code, or training was changed.
