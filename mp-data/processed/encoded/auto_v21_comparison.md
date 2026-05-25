# Auto v2.1 — Parameter Sweep Comparison

_Controlled sweep over `encode_spatial_auto_v2.py`. Pipeline logic is bit-identical between variants; only the envelope dilation, walkable closing radius, and DBSCAN parameters differ._

## Variants

| variant | label | envelope dilation | walkable closing | DBSCAN eps | DBSCAN min_samples |
|---|---|---|---|---|---|
| `v2.1A` | Baseline (identical to v2) | 2.5 m | 1.0 m | 1.5 m | 5 |
| `v2.1B` | Wider walkable envelope | 4.0 m | 1.5 m | 1.5 m | 5 |
| `v2.1C` | Wider + merged entrances | 4.0 m | 1.5 m | 2.75 m | 12 |

## Geometry

| variant | walkable area (m²) | obstacle area (m²) | boundary length (m) | entry clusters | trajectory points inside walkable |
|---|---|---|---|---|---|
| `v2.1A` | 332.470 | 325.880 | 111.900 | 28 | 99.624% |
| `v2.1B` | 364.080 | 455.490 | 103.400 | 28 | 99.624% |
| `v2.1C` | 364.080 | 455.490 | 103.400 | 6 | 99.624% |

## Per-feature distributions (over all trajectory rows)

### `dist_to_obstacle`

| variant | mean (m) | std (m) | median (m) | min | max |
|---|---|---|---|---|---|
| `v2.1A` | 1.216 | 0.732 | 1.020 | 0.412 | 4.554 |
| `v2.1B` | 1.634 | 1.063 | 1.342 | 0.000 | 4.738 |
| `v2.1C` | 1.634 | 1.063 | 1.342 | 0.000 | 4.738 |

### `dist_to_boundary`

| variant | mean (m) | std (m) | median (m) | min | max |
|---|---|---|---|---|---|
| `v2.1A` | 1.409 | 0.942 | 1.200 | 0.316 | 4.455 |
| `v2.1B` | 1.556 | 1.066 | 1.253 | 0.316 | 4.639 |
| `v2.1C` | 1.556 | 1.066 | 1.253 | 0.316 | 4.639 |

### `dist_to_entrance`

| variant | mean (m) | std (m) | median (m) | min | max |
|---|---|---|---|---|---|
| `v2.1A` | 1.604 | 1.080 | 1.315 | 0.000 | 5.659 |
| `v2.1B` | 1.604 | 1.080 | 1.315 | 0.000 | 5.659 |
| `v2.1C` | 4.143 | 2.033 | 3.890 | 0.000 | 11.261 |

## Internal ranking scores

| variant | prediction score | visual-realism score | thesis-compromise score |
|---|---|---|---|
| `v2.1A` | 0.000 | 0.167 | 0.208 |
| `v2.1B` | 0.667 | 0.500 | 0.708 |
| `v2.1C` | 1.000 | 0.833 | 1.042 |

# Recommendation

- **BEST FOR PREDICTION** → `v2.1C`
- **BEST FOR VISUAL REALISM** → `v2.1C`
- **BEST THESIS COMPROMISE** → `v2.1C`

## Justification

- **Prediction.** `v2.1C` wins on per-feature variance — its three distance features carry the largest spread (std), which gives the LSTM the strongest gradient signal. Specifically: `dist_to_obstacle` std = 1.063, `dist_to_boundary` std = 1.066, `dist_to_entrance` std = 2.033.

- **Visual realism.** `v2.1C` produces the largest mean `dist_to_boundary` (1.556 m), so the boundary line sits visibly away from the trajectories instead of hugging them; and it has 6 entry/exit clusters — fewer fragmented stars than the baseline.

- **Thesis compromise.** `v2.1C` strikes the best balance: it deploys a wider, more plausible walkable region while keeping discriminative feature variance for prediction. Trajectory points sit inside the walkable mask 99.6% of the time, obstacles remain at 455.490 m² (not collapsed), and entry clusters are 6 (legible without being over-merged).

## Decision-criteria scorecard

| criterion | A | B | C |
|---|---|---|---|
| trajectories inside walkable (%) | 99.6% | 99.6% | 99.6% |
| dist_to_boundary mean (m) | 1.41 | 1.56 | 1.56 |
| obstacle area (m²) | 325.9 | 455.5 | 455.5 |
| entry clusters | 28 | 28 | 6 |
| feature collapse? | no | no | no |

_End of comparison._