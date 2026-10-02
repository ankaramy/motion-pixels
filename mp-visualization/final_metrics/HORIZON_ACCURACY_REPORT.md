# MODEL_X — Horizon Accuracy Report

**Model:** MODEL_X (frozen, general-purpose H10 Model C retrain on Barcelona_v3)
**Task type:** Evaluation-only. No retraining, no re-prediction, no dataset changes.
**Source data:** `mp-core/trajectory-prediction/experiments/MODEL_X_HORIZON_SWEEP/<H>/per_window_metrics.csv`
**Computed by:** `mp-visualization/final_metrics/compute_horizon_accuracy.py`
**Date:** 2026-06-20

All numbers below are computed directly from the frozen per-window evaluation
files. The ADE / FDE / angular-error values reproduce
`metrics_summary.json` for every horizon exactly, confirming the inputs were read
faithfully.

---

## 1. What each horizon means

The horizon is the rollout length in frames. At the dataset's ~0.05 m per-step
motion scale this corresponds to an approximate ground-truth distance:

| Horizon | Frames | Approx GT distance |
|---------|--------|--------------------|
| H20  | 20  | ~1 m  |
| H60  | 60  | ~3 m  |
| H100 | 100 | ~5 m  |
| H200 | 200 | ~10 m |
| H400 | 400 | ~20 m |

Evaluation population = the **test-window** set for each horizon (the same
population the frozen ADE/FDE was reported on). Because a single trajectory
yields many overlapping windows, the unique-trajectory count is much smaller than
the window count; both are reported.

---

## 2. Formulas

For each test window the per-window file already stores `ade`, `fde`,
`angular_err_deg`, and `gt_net_disp` (GT net displacement, start→end).

**Standard metrics** (mean over all test windows):

```
ADE       = mean( ade )                    # average displacement error  [m]
FDE       = mean( fde )                     # final/endpoint error        [m]
Ang Err   = mean( angular_err_deg )         # heading error of endpoint   [deg]
```

**Success Rate** — a prediction succeeds if its final predicted position lies
within a horizon-specific threshold of the GT endpoint:

```
success        = ( fde <= threshold )
success_rate   = #success / #windows        (reported as %)
```

| Horizon | Threshold |
|---------|-----------|
| H20  | 1 m  |
| H60  | 2 m  |
| H100 | 3 m  |
| H200 | 5 m  |
| H400 | 10 m |

**Normalized Path Accuracy** — how much of the real future displacement the model
recovered:

```
horizon_distance = gt_net_disp                    # actual GT displacement over horizon
accuracy         = clamp( 1 - FDE / horizon_distance , 0 , 1 )   # per window
Path Accuracy    = mean( accuracy )               (reported as %)
```

Windows with (near-)zero GT net displacement (a pedestrian who barely moved) are
guarded against division by zero; their accuracy clamps to 0.

---

## 3. Results table

| Horizon | ADE (m) | FDE (m) | Ang Err (deg) | Success Rate | Path Accuracy |
|---------|---------|---------|---------------|--------------|---------------|
| H20  | 0.446 | 0.708  | 74.1 | **82.7 %** | 26.3 % |
| H60  | 1.011 | 1.811  | 79.4 | **72.3 %** | 21.5 % |
| H100 | 1.419 | 2.565  | 79.7 | **71.5 %** | 25.4 % |
| H200 | 2.595 | 5.016  | 83.7 | **64.0 %** | 21.4 % |
| H400 | 5.079 | 10.015 | 83.8 | **55.2 %** | 18.2 % |

Population sizes:

| Horizon | Test windows | Unique trajectories |
|---------|--------------|---------------------|
| H20  | 87,320 | 286 |
| H60  | 77,438 | 211 |
| H100 | 69,718 | 175 |
| H200 | 55,505 | 120 |
| H400 | 38,639 | 63  |

---

## 4. Interpretation

**Success Rate (endpoint within tolerance).**
Endpoint placement is strong at short range — **83 %** of 1-metre predictions land
within 1 m of the true endpoint. It holds in the **72 %** band through H60–H100
(3–5 m), then declines steadily to **64 %** at H200 (10 m) and **55 %** at H400
(20 m). The model keeps roughly half its endpoints inside the (generous, scaled)
tolerance even at the longest horizon.

**Path Accuracy (displacement recovered).**
This is the harder, more honest metric. It stays in a narrow **18–26 %** band
across all horizons, i.e. the endpoint error is consistently a large fraction of
the true displacement. This is the documented MODEL_X behaviour: the one-step MSE
objective **regresses toward the mean and undershoots motion magnitude** (median
predicted/GT length ratio ≈ 0.41), and the heading error is large (~74–84°). Note
this metric is dominated by the very short straight-line steps that make up most
of the data, and by stationary pedestrians whose tiny net displacement makes any
error look proportionally large — so it should be read as a *displacement-recovery
fidelity* number, not as a pass/fail success rate.

**Why the two curves disagree.**
Success Rate is high while Path Accuracy is low because the absolute thresholds
(1–10 m) are forgiving relative to how little pedestrians actually move over these
windows. A prediction can sit inside the threshold (counts as "success") while
still recovering only a small fraction of the true displacement vector. Together
they bracket the model: **good at being roughly in the right place, weak at
reproducing the full magnitude/shape of motion** — consistent with the magnitude-
and curvature-fix successors MODEL_XR and MODEL_XC.

### At what horizon does quality begin to noticeably degrade?

**H200 (~10 m).** Success Rate sits in a tight 72–83 % plateau through H100, then
drops to 64 % at H200 and 55 % at H400; angular error also crosses ~84°. The first
clear, sustained fall-off is the H100 → H200 step.

### Practical operating limit of the current model?

**H100 (~5 m).** Through H100 the model holds ~72 % endpoint success with FDE ≈
2.6 m. That is the last horizon before the noticeable degradation at H200. H100 is
the recommended operating ceiling for MODEL_X; H200–H400 are usable only as
coarse, low-confidence long-range context.

---

## 5. Files produced

- `horizon_accuracy_metrics.csv` — full numeric table
- `horizon_accuracy_summary.png` / `.svg` — Success Rate & Path Accuracy vs horizon
- `compute_horizon_accuracy.py` — the evaluation script (reproducible)
