# Auto v2 — Spatial Encoding Summary

**Trajectory-informed automatic spatial encoder.** The space is inferred from how pedestrians actually moved, then enriched with conservative obstacle hints derived from holes in the walkable envelope. No model retraining. No edits to the original `encode_spatial_auto.py`.

- Source trajectory CSV: `mp-data\processed\encoded\trajectories_encoded_auto.csv`
- Output folder: `mp-data\processed\encoded\auto_v21C`

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

- World extent: x ∈ [-3.27, 17.03] m, y ∈ [-22.36, 32.04] m
- Grid size: **203 × 544 pixels** (110,432 cells)

## Derived masks

| mask | true pixels | area (m²) | % of grid |
|---|---|---|---|
| occupancy (raw) | 4,704 | 47.0 | 4.3% |
| walkable        | 36,408 | 364.1 | 33.0% |
| envelope        | 81,957 | 819.6 | 74.2% |
| obstacle        | 45,549 | 455.5 | 41.2% |
| boundary        | 1,034 | — | 0.94% |

- Obstacle connected components retained: **1**

## Entry / exit detection

- Long tracks used (>= 10 frames): **520**
- Clusters: **6**

| cluster | center_x (m) | center_y (m) | n_points |
|---|---|---|---|
| 0 | 5.87 | 23.13 | 30 |
| 1 | 1.12 | 19.94 | 60 |
| 2 | 8.32 | -10.08 | 750 |
| 3 | 3.34 | 27.41 | 60 |
| 4 | 0.75 | 4.46 | 40 |
| 5 | 13.78 | -19.16 | 40 |

## Feature distributions (auto v2, all rows)

| feature | n_valid | mean (m) | median (m) | std (m) | min (m) | max (m) | %≤0.05 m |
|---|---|---|---|---|---|---|---|
| `dist_to_obstacle` | 180,840 | 1.634 | 1.342 | 1.063 | 0.000 | 4.738 | 0.4% |
| `dist_to_boundary` | 180,840 | 1.556 | 1.253 | 1.066 | 0.316 | 4.639 | 0.0% |
| `dist_to_entrance` | 180,840 | 4.143 | 3.890 | 2.033 | 0.000 | 11.261 | 0.1% |

## Comparison vs manual encoding (benchmark only)

- Joined rows: **180,840**

| feature | pairs | auto mean | manual mean | mean |Δ| | Pearson r |
|---|---|---|---|---|---|
| `dist_to_obstacle` | 180,840 | 1.634 | 3.230 | 1.690 | 0.415 |
| `dist_to_boundary` | 180,840 | 1.556 | 10.592 | 9.038 | 0.133 |
| `dist_to_entrance` | 180,840 | 4.143 | 11.506 | 7.509 | 0.231 |

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