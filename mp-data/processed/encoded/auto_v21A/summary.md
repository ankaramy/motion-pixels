# Auto v2 — Spatial Encoding Summary

**Trajectory-informed automatic spatial encoder.** The space is inferred from how pedestrians actually moved, then enriched with conservative obstacle hints derived from holes in the walkable envelope. No model retraining. No edits to the original `encode_spatial_auto.py`.

- Source trajectory CSV: `mp-data\processed\encoded\trajectories_encoded_auto.csv`
- Output folder: `mp-data\processed\encoded\auto_v21A`

## Method parameters

| parameter | value |
|---|---|
| grid resolution | 0.1 m/px |
| padding around extent | 3.0 m |
| walkable dilation radius | 0.4 m |
| walkable closing radius  | 1.0 m |
| walkable min-CC fraction  | 5% of biggest |
| envelope dilation radius | 2.5 m |
| obstacle min area | 0.3 m² |
| long-track min length | 10 frames |
| DBSCAN eps | 1.5 m |
| DBSCAN min_samples | 5 |

## Grid

- World extent: x ∈ [-3.27, 17.03] m, y ∈ [-22.36, 32.04] m
- Grid size: **203 × 544 pixels** (110,432 cells)

## Derived masks

| mask | true pixels | area (m²) | % of grid |
|---|---|---|---|
| occupancy (raw) | 4,704 | 47.0 | 4.3% |
| walkable        | 33,247 | 332.5 | 30.1% |
| envelope        | 65,835 | 658.4 | 59.6% |
| obstacle        | 32,588 | 325.9 | 29.5% |
| boundary        | 1,119 | — | 1.01% |

- Obstacle connected components retained: **4**

## Entry / exit detection

- Long tracks used (>= 10 frames): **520**
- Clusters: **28**

| cluster | center_x (m) | center_y (m) | n_points |
|---|---|---|---|
| 0 | 5.42 | 23.91 | 20 |
| 1 | 1.84 | 21.85 | 20 |
| 2 | 7.33 | -8.26 | 10 |
| 3 | 3.34 | 27.41 | 60 |
| 4 | 5.20 | -5.84 | 50 |
| 5 | 1.11 | 19.46 | 30 |
| 6 | 9.66 | -13.22 | 440 |
| 7 | -0.27 | 17.54 | 10 |
| 8 | 0.13 | 10.87 | 10 |
| 9 | 8.11 | -17.30 | 20 |
| 10 | 0.87 | 2.26 | 10 |
| 11 | -0.26 | -4.63 | 10 |
| 12 | 1.10 | -1.23 | 70 |
| 13 | 0.72 | 5.20 | 30 |
| 14 | 12.46 | -6.93 | 80 |
| 15 | 5.41 | -8.61 | 10 |
| 16 | 9.15 | 14.79 | 10 |
| 17 | 3.51 | 7.69 | 10 |
| 18 | 12.32 | -4.55 | 10 |
| 19 | 2.87 | 12.32 | 10 |
| 20 | 9.47 | -5.10 | 10 |
| 21 | 6.77 | 21.58 | 10 |
| 22 | 3.70 | -2.75 | 10 |
| 23 | 13.78 | -19.16 | 40 |
| 24 | 6.58 | -2.89 | 10 |
| 25 | 11.34 | -2.82 | 10 |
| 26 | 5.33 | 5.13 | 10 |
| 27 | 1.49 | -6.04 | 20 |

## Feature distributions (auto v2, all rows)

| feature | n_valid | mean (m) | median (m) | std (m) | min (m) | max (m) | %≤0.05 m |
|---|---|---|---|---|---|---|---|
| `dist_to_obstacle` | 180,840 | 1.216 | 1.020 | 0.732 | 0.412 | 4.554 | 0.0% |
| `dist_to_boundary` | 180,840 | 1.409 | 1.200 | 0.942 | 0.316 | 4.455 | 0.0% |
| `dist_to_entrance` | 180,840 | 1.604 | 1.315 | 1.080 | 0.000 | 5.659 | 0.6% |

## Comparison vs manual encoding (benchmark only)

- Joined rows: **180,840**

| feature | pairs | auto mean | manual mean | mean |Δ| | Pearson r |
|---|---|---|---|---|---|
| `dist_to_obstacle` | 180,840 | 1.216 | 3.230 | 2.078 | 0.242 |
| `dist_to_boundary` | 180,840 | 1.409 | 10.592 | 9.184 | 0.058 |
| `dist_to_entrance` | 180,840 | 1.604 | 11.506 | 9.986 | -0.075 |

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