# Motion Dataset Diagnostics

Dataset: **180,300 rows**, **540 trajectories**, **17 features**

---

## Per-feature variance table

| feature | mean | std | min | max | range | cv | skewness | zero_frac |
|---------|------|-----|-----|-----|-------|----|----------|-----------|
| `u` | 0.5072 | 0.2890 | 0.0000 | 0.9997 | 0.9997 | 0.570 | -0.243 | 0.000 |
| `v` | 0.3609 | 0.2290 | 0.0000 | 1.0000 | 1.0000 | 0.635 | 0.842 | 0.000 |
| `du` | -0.0009 | 0.0207 | -0.3634 | 1.2312 | 1.5946 | 23.138 | 11.185 | 0.015 |
| `dv` | 0.0044 | 0.0709 | -1.1741 | 1.8598 | 3.0339 | 16.235 | 0.444 | 0.007 |
| `speed` | 0.0432 | 0.0601 | 0.0000 | 1.8666 | 1.8666 | 1.390 | 6.628 | 0.005 |
| `heading_sin` | 0.0710 | 0.8904 | -1.0000 | 1.0000 | 2.0000 | 12.545 | -0.153 | 0.004 |
| `heading_cos` | -0.0098 | 0.4495 | -1.0000 | 1.0000 | 2.0000 | 45.637 | 0.103 | 0.000 |
| `turn_rate` | 0.0054 | 1.3329 | -3.1416 | 3.1416 | 6.2832 | 246.001 | 0.053 | 0.007 |
| `dist_to_obstacle_norm` | 0.1242 | 0.1168 | 0.0000 | 0.3138 | 0.3138 | 0.940 | 0.172 | 0.354 |
| `dist_to_boundary_norm` | 0.5176 | 0.1104 | 0.2379 | 0.8315 | 0.5935 | 0.213 | 0.366 | 0.000 |
| `delta_dist_to_obstacle` | ~0.000 | 0.0009 | -0.0178 | 0.0485 | 0.0663 | 32.010 | 6.226 | 0.772 |
| `delta_dist_to_boundary` | ~0.000 | 0.0018 | -0.0508 | 0.0310 | 0.0818 | 115.057 | -2.413 | 0.645 |
| `openness_ahead` | 0.1238 | 0.1172 | 0.0000 | 0.3707 | 0.3707 | 0.947 | 0.184 | 0.361 |
| `openness_left` | 0.1231 | 0.1222 | 0.0000 | 0.3707 | 0.3707 | 0.993 | 0.436 | 0.350 |
| `openness_right` | 0.1289 | 0.1218 | 0.0000 | 0.3707 | 0.3707 | 0.945 | 0.270 | 0.332 |
| `target_du` | -0.0009 | 0.0208 | -0.3634 | 1.2312 | 1.5946 | 23.203 | 11.108 | 0.012 |
| `target_dv` | 0.0044 | 0.0711 | -1.1741 | 1.8598 | 3.0339 | 16.131 | 0.453 | 0.004 |

---

## Near-constant features

Threshold: CV < 0.05 or range < 0.01

_None detected._

---

## Highly correlated feature pairs

Threshold: |Pearson r| >= 0.90

- `u` <-> `dist_to_obstacle_norm`  (r=0.955)
- `u` <-> `openness_ahead`  (r=0.942)
- `u` <-> `openness_left`  (r=0.901)
- `u` <-> `openness_right`  (r=0.911)
- `dist_to_obstacle_norm` <-> `openness_ahead`  (r=0.984)
- `dist_to_obstacle_norm` <-> `openness_left`  (r=0.944)
- `dist_to_obstacle_norm` <-> `openness_right`  (r=0.944)
- `openness_ahead` <-> `openness_left`  (r=0.944)
- `openness_ahead` <-> `openness_right`  (r=0.914)

---

## Potentially noisy / problematic features

Threshold: |skewness| > 3.0 or zero_frac > 40%

- `du`  (|skew|=11.19)
- `speed`  (|skew|=6.63)
- `delta_dist_to_obstacle`  (|skew|=6.23, zero_frac=77.2%)
- `delta_dist_to_boundary`  (zero_frac=64.5%)
- `target_du`  (|skew|=11.11)

---

## Findings and interpretation

### Finding 1 — Spatial confound: `u` absorbs obstacle and openness information (CRITICAL)

`u` correlates with `dist_to_obstacle_norm` at **r=0.955** and with all three openness
features at r=0.901–0.942. This is not a data quality error — it is a property of this
specific floor plan: the obstacle-dense regions happen to be concentrated along one side
of the world x-axis.

**What this means for training:**
The LSTM can predict obstacle proximity almost perfectly from position alone. This
creates a shortcut: the model may never need to attend to the spatial features directly.
During autoregressive rollout, spatial features will update as position evolves, but if
the model learned to ignore them in favour of `u`, it will not replan around novel
obstacles during inference.

**Recommended action:**
Do not drop `u` or the spatial features. Instead, run an ablation comparing:
- `[u, v, du, dv, heading_sin, heading_cos, turn_rate]` (motion only)
- `[u, v, du, dv, heading_sin, heading_cos, turn_rate, dist_to_obstacle_norm, openness_ahead, openness_left, openness_right]` (full)

If the full model matches the motion-only model on validation trajectories that cross
spatially novel regions, the spatial features are being ignored. If the spatial model
generalises better to edge cases (narrow passages, obstacle avoidance), they are
providing real signal beyond what position already encodes.

---

### Finding 2 — Obstacle cluster: `dist_to_obstacle_norm` and `openness_*` are near-redundant

`dist_to_obstacle_norm` ↔ `openness_ahead`: **r=0.984** — nearly collinear.
`dist_to_obstacle_norm` ↔ `openness_left/right`: r=0.944 each.
`openness_ahead` ↔ `openness_left/right`: r=0.914–0.944.

These four features form a tight cluster. The 1.5 m lookahead does not add much
information beyond what the current-position distance already encodes, because the
distance field is spatially smooth — the value 1.5 m away is strongly predicted by
the current value.

**What this means:**
The four spatial features carry approximately one dimension of independent information.
The model is unlikely to learn meaningful distinctions between them without strong
examples of divergence (e.g., a narrow corridor where ahead is open but left/right are
blocked, while the current position is already mid-corridor).

**Recommended action (two options):**

Option A — Keep all four, accept the redundancy.
The LSTM may learn to use the cluster as a robust obstacle signal, with the redundancy
acting as built-in noise tolerance. Low risk, no information loss.

Option B — Replace with two derived features:
- `obs_current = dist_to_obstacle_norm` (clearance now)
- `obs_asymmetry = openness_left - openness_right` (signed lateral bias)

Drop `openness_ahead` (r=0.984 with current, adds nothing) and `openness_left/right`
individually in favour of their difference. This breaks the collinearity while
preserving the turn-preference signal.

The spatial maps for `openness_ahead - openness_left` and `openness_ahead - openness_right`
confirm that these difference features have meaningful spatial structure (non-uniform
distribution over the floor plan), supporting Option B.

---

### Finding 3 — Delta features are sparse spike signals (MODERATE CONCERN)

`delta_dist_to_obstacle`: zero_frac=**77.2%**, std=0.00094, |skew|=6.23.
`delta_dist_to_boundary`: zero_frac=**64.5%**, std=0.00180.

Most timesteps show zero change in distance. Non-zero values occur when an agent
moves toward or away from an obstacle — a rare event relative to straight-line walking.

**Why this could hurt training:**
The gradient signal from these features will be nearly silent on most timesteps.
If the model learns to weight them near zero (correct on 77% of steps), it will fail
to respond to the non-zero approach events where the features matter most.

**Recommended action:**
These features require rescaling relative to the others. Consider:
- Multiply by a constant factor (e.g. ×20) to bring their effective std into the same
  order as the other normalised features (~0.10).
- Alternatively, drop them and rely on the model to compute approach dynamics from
  the sequence of `dist_to_obstacle_norm` values directly, which an LSTM can do.
- At minimum, monitor their weight magnitudes during training with gradient logging.

---

### Finding 4 — `du` and `target_du` are heavily right-skewed (MODERATE CONCERN)

`du`: |skew|=11.19, max=1.23 m/step (mean=−0.0009).
`target_du`: |skew|=11.11.

The extreme positive tail in `du` corresponds to tracking jumps: occasional
frame-to-frame discontinuities where a detection switches between two people or
a track is re-identified after an occlusion gap. These outliers are real artifacts,
not fast walking.

**Evidence:** Normal pedestrian walking at this frame rate should produce `du` values
in the range −0.10 to +0.10 m/step. Values above ~0.3 m/step are physically
implausible without running.

`dv` (skew=0.44) is well-behaved — the skew is axis-specific, consistent with the
primary direction of travel being x-dominant in this space.

**Recommended actions:**
1. Clip `du`, `dv`, `speed` at a physically plausible threshold before training
   (e.g., 99th percentile or 0.3 m/step = 1.5× the 99th percentile speed of
   normal walking).
2. Alternatively, filter outlier trajectories where any single step > threshold
   before training/validation split.
3. This also affects `target_du`/`target_dv` — clip targets to the same threshold
   and exclude those rows from the loss, or treat them as missing.

---

### Finding 5 — `dist_to_obstacle_norm` effective range is only [0, 0.314]

Max observed = 0.314 (= 8.2 m at 26.18 m map max). Trajectories never reach
the open centre of the floor plan where obstacle clearance is highest.

This is not a problem for the current dataset but means the normalisation underuses
the [0.314, 1.0] range. A model trained here may not generalise to spaces with
higher open-area fractions.

**No action needed** for current experiments. Note for future datasets collected
in more open spaces.

---

### Finding 6 — `turn_rate` is well-distributed and informative

std=1.333 rad, skew=0.053, mean≈0. Spans the full (−π, π] range.
The near-zero skewness and mean confirm no turn-direction bias in the dataset.
This feature is the most informative non-position feature for distinguishing
trajectory shapes.

**Action:** Normalise to [−1, 1] by dividing by π before training (or use
StandardScaler). Do not clip — the full range is physically valid.

---

### Finding 7 — `dist_to_boundary_norm` is the cleanest distance feature

zero_frac=0.000, range=[0.24, 0.83], skew=0.366, cv=0.213.
This feature is well-behaved, never zero, moderately correlated with the obstacle
cluster (r≈0.5 from the matrix), and captures a different affordance (edge of the
walkable zone, not obstacle proximity). **Keep as-is.**

---

## Recommended actions for training (priority order)

| Priority | Action |
|----------|--------|
| **High** | Clip `du`, `dv`, `target_du`, `target_dv` at ±0.30 m/step (tracking outlier filter). |
| **High** | Run motion-only ablation to quantify what the spatial features add beyond `u`. |
| **Medium** | Rescale `delta_dist_*` features by ×20 relative to other features, or drop them in the first training run and add back if ablation shows improvement. |
| **Medium** | Consider replacing `[dist_to_obstacle_norm, openness_ahead, openness_left, openness_right]` with `[dist_to_obstacle_norm, openness_left - openness_right]` (2 features instead of 4) to break the collinearity cluster. |
| **Low** | Normalise `turn_rate` to [−1, 1] by dividing by π. |
| **Low** | Consider log1p-transforming `speed` at training time to reduce skew. |
| **None** | Drop any features — no feature is near-constant or purely redundant after the above actions. |
