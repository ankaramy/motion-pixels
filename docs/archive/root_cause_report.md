# Motion Pixels — Model C Angular-Failure Root-Cause Report

**Date:** 2026-06-06 · **Type:** READ-ONLY forensic audit (no retrain, no model/dataset/code changes).
**Checkpoint audited:** `models/best_model_C_barcelona.pt` (best epoch 2).
**Figures:** `root_cause_figures/` · **Numbers:** `root_cause_figures/_forensic_numbers.json`

## TL;DR

The angular collapse is **not a rollout, feature-update, target, or scaler bug**. It is a
**model/objective limitation**: under MSE the network learns to predict an almost-zero
displacement for every input (teacher-forced fit slope ≈ **0.07–0.09**, predicted step
magnitude ≈ 40–80 % of true). A near-zero displacement vector has an ill-defined heading,
so angular error sits near chance (~60–90°). This is already present in **teacher-forced**
mode, before any autoregression. A real but secondary **speed domain shift** (train→bridge
KS = 0.49) makes the held-out site worse still. Turning frequency is *not* the culprit
(turn_rate KS = 0.07).

---

## TASK 1 — Teacher-forced vs autoregressive

`root_cause_figures/teacher_forced_vs_autoregressive.csv`

| split | AR ADE (m) | AR FDE (m) | AR angular | TF step-err (m) | TF angular | pred disp mean | true disp mean | n windows |
|---|---|---|---|---|---|---|---|---|
| train | 0.724 | 1.176 | 77.5° | 0.097 | **57.9°** | 0.0445 | 0.1117 | 824,917 |
| val | 0.291 | 0.529 | 82.0° | 0.028 | **62.7°** | 0.0141 | 0.0282 | 124,265 |
| test | 0.235 | 0.445 | 91.3° | 0.022 | **85.3°** | 0.0139 | 0.0169 | 79,857 |

**Reading:** Teacher-forced (no rollout) angular error is already **57.9–85.3°**. Autoregression
adds only ~+6 to +20° more. So the angular failure exists *before* rollout — this is **not**
primarily rollout drift. Drift is present (AR worse than TF) but it is a minor contributor.

> Note on the "surprisingly low" ADE: true per-step displacement on the bridge is **0.017 m/step**
> (~0.34 m over a 20-step horizon). An ADE of 0.235 m and FDE 0.445 m are therefore *large*
> relative to how far the pedestrian actually travels — the low absolute metres are an artifact
> of small per-step motion, not good prediction.

## TASK 4 — Teacher-forced displacement analysis

`predicted_vs_true_displacement_teacher_forced_{test,val}.png`

Predicted-vs-true displacement is a flat near-zero cloud: least-squares **slope ≈ 0.07 (du)** and
**0.09 (dv)** on the test site (similar on val). The model collapses to ≈0 displacement
regardless of the true value — textbook **MSE regression-to-the-mean**. Because the mean
displacement is ~0, predicted vectors are tiny and their direction is essentially noise →
high angular error. **Collapse happens before rollout.**

## TASK 2 — Rollout feature-update audit

Static read of frozen `rollout()` (`run_bridge_ablation.py:292-353`). Per predicted step:

| Feature | Expected update | Actual update (code) | Verdict |
|---|---|---|---|
| du | model output (inverse-scaled) | `du_m,dv_m = t_sc.inverse(pred)` | ✅ Correct |
| dv | model output (inverse-scaled) | same | ✅ Correct |
| speed | recompute from du,dv | `math.hypot(du_m, dv_m)` | ✅ Correct |
| heading_sin | sin(atan2(dv,du)) | `math.sin(heading)`; heading held = prev if speed<1e-6 | ✅ Correct |
| heading_cos | cos(atan2(dv,du)) | `math.cos(heading)` | ✅ Correct |
| turn_rate | wrap(heading − prev_heading) | `wrap_angle(heading - prev_h)`; `prev_h` updated each step | ✅ Correct |
| u | (wx−xmin)/xrng, clipped [0,1] | exactly that | ✅ Correct |
| v | (wy−ymin)/yrng, clipped [0,1] | exactly that | ✅ Correct |
| dist_to_obstacle_norm | re-query at new (u,v) | `kdt_lookup(kdt,u,v)` (IDW, k=5) | ✅ Correct |
| dist_to_boundary_norm | re-query at new (u,v) | same | ✅ Correct |
| window | drop oldest, append new scaled row | `torch.cat([window[1:], new_t])` | ✅ Correct |

**Heading and turn_rate ARE propagated correctly** (`prev_h` is reassigned every step). No
feature-update bug found. (Minor, non-bug note: when predicted speed is near 0 the heading is
held at the previous value, which is the safe convention; it does not cause the collapse.)

## TASK 3 — Feature-order audit

| context | order |
|---|---|
| training (`make_windows` feat_cols) | du, dv, speed, heading_sin, heading_cos, turn_rate, u, v, dist_to_obstacle_norm, dist_to_boundary_norm |
| inference / rollout (`feat_cols` passed) | identical (same `FEATURE_SETS["C_motion_position_spatial"]`) |
| scaler (`ColumnScaler.cols`) | identical (fit with the same list; `scale_col` indexes by name) |

**No mismatch.** Training, inference, and scaler all use the same 10-feature order; the scaler
is fit on train only and applied consistently. Rules out E.

## TASK 5 — Angular target distribution

`turn_rate_distribution_stats.csv` · `turn_rate_distribution_by_split.png`

| split | mean |turn_rate| | median turn_rate | median |turn_rate| | p90 |turn_rate| |
|---|---|---|---|---|
| train | 0.918 | 0.000 | 0.254 | 3.016 |
| val | 0.852 | 0.000 | 0.310 | 2.912 |
| test | 1.028 | 0.000 | 0.478 | 2.914 |

The distribution is **strongly bimodal**: median turn_rate = 0 (most steps near-straight or
near-stationary) with a heavy tail to ±π (p90 ≈ 2.9–3.0 rad). The bridge has only *slightly*
more turning than train (mean |tr| 1.03 vs 0.92; p90 essentially equal). So the test set is
**not** dramatically "more turny" than training. The large mean is driven by the ±π tail,
much of which is heading noise at near-zero speed (where heading is ill-defined) — which is
exactly the regime the model cannot and arguably should not predict.

## TASK 6 — Cross-site difficulty (train pooled vs red_bridge_combined_01)

`cross_site_distributions.png` · KS statistics (0 = identical, 1 = disjoint):

| quantity | KS statistic | interpretation |
|---|---|---|
| **speed** | **0.493** | large shift — bridge moves at very different (slower) per-step speeds |
| heading | 0.213 | moderate shift — different site orientation/geometry |
| turn_rate | **0.067** | negligible — turning behaviour is essentially the same across sites |

**The dominant domain shift is SPEED, not turning.** The test recording is statistically outside
the training domain in speed (and somewhat in heading), which raises test error vs validation —
but it does not introduce unusually more turning.

---

## Ranked causes (most → least likely)

### 1. (F) True model / objective limitation — **PRIMARY**
MSE regression-to-the-mean: the model predicts ≈0 displacement for everything.
- **Evidence FOR:** teacher-forced fit slope 0.07/0.09; predicted step magnitude 40–80 % of true; teacher-forced angular error already 58–85° (before rollout); best epoch = 2 then val rises (almost nothing learnable beyond the mean); near-zero predicted vectors make heading undefined.
- **Evidence AGAINST:** none material. (Positional ADE looks low, but only because true per-step motion is tiny.)
- **Next action:** change the *learning signal*, not the architecture: predict speed + heading (polar) or add an angular/curvature loss term; weight samples by displacement magnitude so near-stationary noise stops dominating MSE. Establish a velocity-persistence baseline to confirm Model C barely beats "repeat last displacement."

### 2. Ill-posed angular metric at low speed — **STRONG CONTRIBUTOR / measurement caveat**
- **Evidence FOR:** true disp 0.017 m/step on test; median turn_rate 0 with a ±π tail; heading of a ~0-length vector is noise, so ~90° angular error is partly unavoidable regardless of model.
- **Evidence AGAINST:** train teacher-forced angular (58°) is well below 90°, so it is not *purely* metric noise — the model genuinely under-turns too.
- **Next action:** report angular error **only on moving steps** (speed above a threshold, e.g. ≥ 0.05 m/step); report turning metrics conditioned on GT |turn_rate| above a threshold.

### 3. (G) Cross-site domain shift — **SECONDARY, real**
- **Evidence FOR:** speed KS 0.49 (large), heading KS 0.21; test teacher-forced angular (85°) markedly worse than train (58°); val loss bottoms at epoch 2 (poor transfer).
- **Evidence AGAINST:** turn_rate KS 0.07 — the shift is in speed/orientation, not turning; so it explains *degradation*, not the *core* collapse (which is present even on train).
- **Next action:** grow training-site diversity (add the two corrected recordings after calibration review); consider speed-normalized features so per-site speed scale transfers.

### 4. (A) Autoregressive rollout drift — **MINOR**
- **Evidence FOR:** AR angular > TF angular by ~6–20°; AR ADE ≫ TF step-error (compounding).
- **Evidence AGAINST:** TF already fails; the rollout adds only modest extra error.
- **Next action:** none needed for the angular question; revisit only after the objective is fixed.

### 5. (B) Feature-update bug in rollout — **RULED OUT**
- FOR: none. AGAINST: full code audit shows every feature (incl. heading & turn_rate) correctly recomputed; and TF bypasses rollout entirely yet still fails. **No action.**

### 6. (D) Target-construction issue — **RULED OUT**
- FOR: none. AGAINST: `target_du[t]==du[t+1]` verified 0 mismatches (dataset validation); teacher-forced predictions track target *sign/scale partially* (just shrunk). **No action.**

### 7. (E) Scaler / feature-order issue — **RULED OUT**
- FOR: none. AGAINST: training/inference/scaler orders identical; scaler fit on train only; TF scatter is centered at origin (no systematic offset). **No action.**

### Special note on (C) "teacher-forced already failing"
This is **confirmed true** — and that is precisely the finding that *rules out* the rollout-side
bugs (A, B) and *points to* the model/objective cause (F). Teacher-forced failing is the
diagnosis, not a separate defect.

---

## One-line conclusion
Model C, trained with MSE on raw displacement over near-stationary pedestrian steps, collapses to
predicting ≈0 motion (regression-to-mean); its direction output is therefore noise — a learning-
objective limitation amplified by a speed-domain shift to the bridge — not a rollout, feature,
target, or scaler bug. **Recommended first fix: change the objective/target (polar speed+heading
or angular loss, moving-step weighting) and add a velocity-persistence baseline — before any
architecture change or retraining decision.**
