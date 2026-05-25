# Motion Dataset v2 — Feature Summary

**13 columns** (was 19 in v1): 2 identifiers, 11 input features, 2 targets.

## Changes from v1

| Change | Columns |
|--------|---------|
| Removed | `delta_dist_to_obstacle`, `delta_dist_to_boundary`, `openness_ahead`, `openness_left`, `openness_right` |
| Added | `openness_lr_asymmetry` = openness_left − openness_right |
| Clipped | `du`, `dv`, `target_du`, `target_dv` capped at ±0.3 m/step |

## Clipping report

- `du`: 40 values clipped (0.02% of rows)
- `dv`: 1,090 values clipped (0.60% of rows)
- `target_du`: 40 values clipped (0.02% of rows)
- `target_dv`: 1,110 values clipped (0.62% of rows)

## Per-feature statistics

| feature | mean | std | min | max | skewness | zero_frac |
|---------|------|-----|-----|-----|----------|-----------|
| `u` | 0.5072 | 0.2890 | 0.0000 | 0.9997 | -0.243 | 0.000 |
| `v` | 0.3609 | 0.2290 | 0.0000 | 1.0000 | 0.842 | 0.000 |
| `du` | -0.0009 | 0.0185 | -0.3000 | 0.3000 | -0.258 | 0.015 |
| `dv` | 0.0044 | 0.0639 | -0.3000 | 0.3000 | 0.116 | 0.007 |
| `speed` | 0.0423 | 0.0515 | 0.0000 | 0.4243 | 2.707 | 0.005 |
| `heading_sin` | 0.0710 | 0.8904 | -1.0000 | 1.0000 | -0.153 | 0.004 |
| `heading_cos` | -0.0098 | 0.4495 | -1.0000 | 1.0000 | 0.103 | 0.000 |
| `turn_rate` | 0.0054 | 1.3329 | -3.1416 | 3.1416 | 0.053 | 0.007 |
| `dist_to_obstacle_norm` | 0.1242 | 0.1168 | 0.0000 | 0.3138 | 0.172 | 0.354 |
| `dist_to_boundary_norm` | 0.5176 | 0.1104 | 0.2379 | 0.8315 | 0.366 | 0.000 |
| `openness_lr_asymmetry` | -0.0059 | 0.0798 | -0.1186 | 0.1186 | 0.107 | 0.260 |

## Feature correlation matrix (input features only)

```
                           u      v     du     dv  speed  heading_sin  heading_cos  turn_rate  dist_to_obstacle_norm  dist_to_boundary_norm  openness_lr_asymmetry
u                      1.000 -0.720 -0.018 -0.020 -0.084       -0.011       -0.082     -0.002                  0.955                  0.474                 -0.010
v                     -0.720  1.000 -0.059  0.008  0.313       -0.050        0.023     -0.000                 -0.735                 -0.067                  0.044
du                    -0.018 -0.059  1.000 -0.343 -0.149       -0.171        0.534      0.003                  0.011                  0.005                  0.134
dv                    -0.020  0.008 -0.343  1.000  0.106        0.660       -0.099      0.012                 -0.039                  0.025                 -0.465
speed                 -0.084  0.313 -0.149  0.106  1.000        0.031       -0.021      0.018                 -0.183                  0.185                 -0.002
heading_sin           -0.011 -0.050 -0.171  0.660  0.031        1.000       -0.185     -0.009                 -0.021                 -0.005                 -0.809
heading_cos           -0.082  0.023  0.534 -0.099 -0.021       -0.185        1.000     -0.009                 -0.074                 -0.009                  0.146
turn_rate             -0.002 -0.000  0.003  0.012  0.018       -0.009       -0.009      1.000                 -0.001                 -0.001                  0.026
dist_to_obstacle_norm  0.955 -0.735  0.011 -0.039 -0.183       -0.021       -0.074     -0.001                  1.000                  0.331                  0.004
dist_to_boundary_norm  0.474 -0.067  0.005  0.025  0.185       -0.005       -0.009     -0.001                  0.331                  1.000                 -0.009
openness_lr_asymmetry -0.010  0.044  0.134 -0.465 -0.002       -0.809        0.146      0.026                  0.004                 -0.009                  1.000
```

## Feature-by-feature analysis: redundant or complementary?

### Position: `u`, `v`

**Complementary to each other** (r=-0.720 — independent axes).
**Partially redundant with `dist_to_obstacle_norm`** (r=0.955 — spatial confound from the floor-plan geometry: the obstacle-dense zone sits along one x-band).
**Independent from motion features** (r(u, du)=-0.018).
Position must be kept: it anchors the LSTM's spatial memory and is the integration target for rollout (`u[t+1] = u[t] + Δu`). Dropping it would make the model spatially amnesic.

### Velocity: `du`, `dv`

r(du, dv)=-0.343 — independent motion axes. r(du, speed)=-0.149, r(dv, speed)=0.106.
**Partially redundant with `speed` and `heading_*`**: mathematically, `du = speed · cos(heading)`, `dv = speed · sin(heading)`. All five features (du, dv, speed, heading_sin, heading_cos) encode the same velocity vector via different decompositions.
**This redundancy is intentional and beneficial.** The LSTM sees:
- `du`/`dv`: raw displacement — direct integration target at rollout time.
- `speed`: magnitude — scalar summary the model can weight independently.
- `heading_*`: direction — normalised unit vector, decoupled from magnitude.
Providing all three representations reduces the implicit computation the model must perform and has been shown to help recurrent models converge faster.

### Direction: `heading_sin`, `heading_cos`

r=-0.185. Correctly near-zero: sine and cosine of the same angle are orthogonal over a uniform distribution of headings. **Complementary** — they jointly encode direction as a unit vector without the ±π discontinuity a raw angle would introduce. Neither can replace the other.

### Angular dynamics: `turn_rate`

r(turn_rate, heading_sin)=-0.009, r(turn_rate, heading_cos)=-0.009.
**Complementary to heading.** Heading encodes the current direction; turn_rate encodes how fast it is changing. A sequence of identical heading values with zero turn_rate looks the same as a straight walk regardless of the heading value, but a non-zero turn_rate immediately signals a curve. An LSTM can infer turn_rate from two consecutive heading values, but providing it explicitly reduces the sequence depth needed to capture turning behaviour.

### Obstacle proximity: `dist_to_obstacle_norm`

r(dist_to_obstacle_norm, dist_to_boundary_norm)=0.331.
r(dist_to_obstacle_norm, openness_lr_asymmetry)=0.004.
**Complementary to `dist_to_boundary_norm`** — they measure clearance from different reference objects (scattered obstacles vs walkable-zone edge). Their low mutual correlation confirms they carry independent spatial information.
**Independent from `openness_lr_asymmetry`** — the new asymmetry feature captures lateral gradient, not magnitude, so low correlation is expected and correct.
**Still correlated with `u` (r≈0.95)**, but this is a dataset property, not a reason to drop either feature — see position analysis above.

### Boundary proximity: `dist_to_boundary_norm`

r(dist_to_boundary_norm, u)=0.474, r(dist_to_boundary_norm, v)=-0.067.
**Complementary to all motion and obstacle features.** zero_frac=0.000 — no pathological zeros. The feature is well-behaved and encodes whether the agent is crossing the floor plan or hugging the edge, which is not captured by obstacle proximity or position alone in a non-rectangular space.

### Lateral spatial affordance: `openness_lr_asymmetry`

r(openness_lr_asymmetry, turn_rate)=0.026.
r(openness_lr_asymmetry, heading_sin)=**-0.809**.

**Caution: this feature is strongly coupled to `heading_sin`.**

This correlation is high enough to warrant specific attention. The mechanism:
`openness_lr_asymmetry` = openness_left − openness_right, where left and right are
defined relative to the current heading. In a floor plan with an asymmetric obstacle
layout, walking in one direction consistently puts more open space on one lateral side.
The r=−0.809 means that for this specific floor plan, heading direction already
predicts which lateral side is open with ~80% of the variance explained.

**Implications for the model:**
1. The feature is not purely a spatial affordance signal in this dataset — it is
   substantially a heading-derived signal that correlates with floor-plan geometry.
2. The model may learn to weight it as a heading proxy rather than as an independent
   obstacle signal.
3. During rollout, if the model heads into a geometrically novel region, the
   asymmetry may predict the wrong side because the floor-plan geometry changes.

**This is not a reason to drop the feature** — the 19% of variance that is independent
of heading likely encodes genuine local obstacle asymmetry. But it does mean:
- Run an ablation with and without `openness_lr_asymmetry` to measure its marginal
  contribution beyond what `heading_sin` already provides.
- If the ablation shows no improvement, the feature can be safely removed.
- If it helps, the model is using both the heading-correlated component (floor-plan
  bias) and the residual (local obstacle configuration) — both are legitimate signals
  for this specific space.

r(openness_lr_asymmetry, turn_rate)=0.026 — low correlation with turn rate means
the feature is not leaking future steering decisions into the inputs.

**Caveat from v1 diagnostics:** zero_frac=0.260 (down from ~35% in individual openness
features, as expected — simultaneous OOB/obstacle hits on both sides are less common).

## Summary: redundancy map

```
REDUNDANT CLUSTER (intentional, beneficial):
  du, dv  <->  speed + heading_sin + heading_cos
  (same velocity vector, three complementary decompositions)

SPATIAL CONFOUND (dataset geometry, not a modelling error):
  u  <->  dist_to_obstacle_norm  (r=0.955, floor-plan specific)
  Keep both: u anchors rollout integration; dist_to_obstacle_norm
  gives the absolute clearance value needed for avoidance planning.

HEADING-COUPLED SPATIAL FEATURE (verify via ablation):
  openness_lr_asymmetry  <->  heading_sin  (r=-0.809)
  The lateral asymmetry is 80% predictable from heading alone in this
  floor plan. The feature may act as a noisy heading proxy rather than
  an independent affordance. Ablation will determine its true contribution.

COMPLEMENTARY (low cross-correlation, independent information):
  dist_to_obstacle_norm  vs  dist_to_boundary_norm  (r=0.331)
  turn_rate              vs  heading_sin/cos         (r<0.01)
  u                      vs  v                       (r=-0.720)
  openness_lr_asymmetry  vs  dist_to_obstacle_norm   (r=0.004)
  openness_lr_asymmetry  vs  turn_rate               (r=0.026)
```

## Recommended normalisation before training

| feature | transform |
|---------|-----------|
| `u`, `v` | already [0, 1] — no further scaling needed |
| `du`, `dv`, `target_du`, `target_dv` | StandardScaler (zero-mean, unit-var) fitted on train split |
| `speed` | log1p then StandardScaler (right-skewed, skew=6.6 in v1; clipping reduces tail but doesn't eliminate it) |
| `heading_sin`, `heading_cos` | already [−1, 1] — no scaling needed |
| `turn_rate` | divide by π → [−1, 1] (bounded, symmetric) |
| `dist_to_obstacle_norm`, `dist_to_boundary_norm` | already [0, 1] — no further scaling needed |
| `openness_lr_asymmetry` | StandardScaler or divide by max observed range |
