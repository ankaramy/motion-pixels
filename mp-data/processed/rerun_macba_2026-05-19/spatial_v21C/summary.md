# Auto v2 — Spatial Encoding Summary

**Trajectory-informed automatic spatial encoder.** The space is inferred from how pedestrians actually moved, then enriched with conservative obstacle hints derived from holes in the walkable envelope. No model retraining. No edits to the original `encode_spatial_auto.py`.

- Source trajectory CSV: `mp-data\processed\rerun_macba_2026-05-19\trajectories\trajectories_world.csv`
- Output folder: `mp-data\processed\rerun_macba_2026-05-19\spatial_v21C`

## Method parameters

| parameter | value |
|---|---|
| grid resolution | 0.1 m/px |
| padding around extent | 3.0 m |
| walkable dilation radius | 0.4 m |
| walkable closing radius  | 1.5 m |
| walkable min-CC fraction  | 5% of biggest |
| envelope dilation radius | 4.0 m |
| obstacle min area | 0.3 m² |
| long-track min length | 10 frames |
| DBSCAN eps | 2.75 m |
| DBSCAN min_samples | 12 |

## Grid

- World extent: x ∈ [-2.73, 16.27] m, y ∈ [-29.25, 27.45] m
- Grid size: **190 × 567 pixels** (107,730 cells)

## Derived masks

| mask | true pixels | area (m²) | % of grid |
|---|---|---|---|
| occupancy (raw) | 4,653 | 46.5 | 4.3% |
| walkable        | 28,640 | 286.4 | 26.6% |
| envelope        | 77,173 | 771.7 | 71.6% |
| obstacle        | 48,533 | 485.3 | 45.1% |
| boundary        | 1,114 | — | 1.03% |

- Obstacle connected components retained: **2**

## Entry / exit detection

- Long tracks used (>= 10 frames): **52**
- Clusters: **1**

| cluster | center_x (m) | center_y (m) | n_points |
|---|---|---|---|
| 0 | 9.44 | -18.27 | 44 |

## Feature distributions (auto v2, all rows)

| feature | n_valid | mean (m) | median (m) | std (m) | min (m) | max (m) | %≤0.05 m |
|---|---|---|---|---|---|---|---|
| `dist_to_obstacle` | 18,084 | 1.249 | 1.100 | 0.687 | 0.000 | 3.400 | 0.4% |
| `dist_to_boundary` | 18,084 | 1.238 | 1.020 | 0.806 | 0.316 | 4.238 | 0.0% |
| `dist_to_entrance` | 18,084 | 15.094 | 11.100 | 12.450 | 0.000 | 43.106 | 0.0% |

## Comparison vs manual encoding (benchmark only)

- Joined rows: **18,084**

| feature | pairs | auto mean | manual mean | mean |Δ| | Pearson r |
|---|---|---|---|---|---|
| `dist_to_obstacle` | 18,084 | 1.249 | 3.230 | 2.022 | 0.403 |
| `dist_to_boundary` | 18,084 | 1.238 | 10.592 | 9.355 | 0.057 |
| `dist_to_entrance` | 18,084 | 15.094 | 11.506 | 5.676 | 0.806 |

## Warnings

_No automatic warnings raised._

## Acceptance checklist

- [x] dist_to_obstacle mean is not collapsed near 0 (>0.10 m)
- [x] dist_to_boundary mean is not collapsed near 0 (>0.10 m)
- [x] dist_to_entrance is present and non-empty
- [x] no spatial feature is constant / mostly zero / NaN-heavy
- [x] at least one entry/exit cluster detected

**Overall: ACCEPT as candidate**

## Artifacts

- `trajectories_encoded_auto_v2.csv`
- `walkable_mask.png`
- `obstacle_mask.png`
- `boundary_mask.png`
- `distance_to_obstacle_m.png`
- `distance_to_boundary_m.png`
- `entry_exit_points.csv`
- `distance_to_entrance_m.png`
- `debug_stages.png`
- `debug_trajectory_sampling.png`

## Thesis framing

This v2 encoder does not aim for perfect semantic segmentation. It is a trajectory-informed automatic spatial encoder: the walkable region is inferred from observed pedestrian density, obstacles are conservatively derived from holes inside the walkable envelope, and entries/exits are detected from the start and end of long tracks. The result is intentionally a function of the data rather than of any one image-segmentation choice — which is the property the thesis needs.

_End of summary._