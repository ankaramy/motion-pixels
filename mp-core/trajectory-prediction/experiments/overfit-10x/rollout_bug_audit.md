# Rollout Bug Audit — Trajectory 310

**Model audited:** LSTM
**Trajectory ID:** 310
**Date:** 2026-05-13

---

## 1. Target Scaling Verification

| Property | Value |
|---|---|
| Targets scaled during training | **YES** — `MinMaxScaler` via `ts.fit_transform(y)` |
| `inverse_transform` applied before integration | **YES** — line `dx, dy = ts.inverse_transform(p_s[np.newaxis])[0]` executes BEFORE `wx = prev_x + dx` |
| Target scaler fitted on columns | `['delta_x', 'delta_y']` |
| Target scaler min (du, dv) | `-0.363400`, `-1.174100` |
| Target scaler max (du, dv) | `1.231200`, `1.859800` |
| Scaler range du | `1.594600` m |
| Scaler range dv | `3.033900` m |

**Conclusion:** `inverse_transform` is applied correctly — no scaling bug at the integration step.

---

## 2. Per-Step Rollout Values — Trajectory 310

Format: `step | du_scaled | dv_scaled | du_unscaled(m) | dv_unscaled(m) | pred_x | pred_y`

```
step      du_s      dv_s        du_m        dv_m      pred_x      pred_y        gt_x        gt_y
------------------------------------------------------------------------------------------
   1    0.2289    0.3931     0.00168     0.01855      0.8748      2.6169      0.8861      2.6418
   2    0.2292    0.3919     0.00206     0.01480      0.8768      2.6316      0.9049      2.7188
   3    0.2294    0.3905     0.00236     0.01062      0.8792      2.6423      0.9042      2.7561
   4    0.2295    0.3898     0.00259     0.00855      0.8818      2.6508      0.9173      2.7906
   5    0.2297    0.3893     0.00286     0.00705      0.8847      2.6579      0.9300      2.8315
   6    0.2298    0.3888     0.00299     0.00563      0.8876      2.6635      0.9461      2.8697
   7    0.2300    0.3883     0.00342     0.00383      0.8911      2.6673      0.9482      2.8874
   8    0.2302    0.3877     0.00368     0.00203      0.8947      2.6694      0.9487      2.8926
   9    0.2304    0.3872     0.00395     0.00059      0.8987      2.6699      0.9449      2.8982
  10    0.2305    0.3866     0.00414    -0.00105      0.9028      2.6689      0.9551      2.9621
  11    0.2305    0.3862     0.00419    -0.00234      0.9070      2.6666      0.9608      3.0212
  12    0.2306    0.3859     0.00435    -0.00335      0.9114      2.6632      0.9687      3.0735
  13    0.2307    0.3856     0.00449    -0.00424      0.9159      2.6590      0.9773      3.1231
  14    0.2308    0.3853     0.00462    -0.00501      0.9205      2.6540      0.9843      3.1739
  15    0.2309    0.3851     0.00474    -0.00569      0.9252      2.6483      0.9869      3.2298
  16    0.2309    0.3849     0.00484    -0.00631      0.9301      2.6420      0.9953      3.2779
  17    0.2310    0.3847     0.00494    -0.00686      0.9350      2.6351      1.0002      3.2918
  18    0.2310    0.3846     0.00503    -0.00736      0.9400      2.6277      0.9998      3.2977
  19    0.2311    0.3844     0.00511    -0.00780      0.9451      2.6199      1.0035      3.3321
  20    0.2311    0.3843     0.00518    -0.00820      0.9503      2.6117      0.9972      3.3534
  21    0.2312    0.3842     0.00525    -0.00854      0.9556      2.6032      0.9968      3.3959
  22    0.2312    0.3841     0.00531    -0.00884      0.9609      2.5944      0.9986      3.4462
  23    0.2312    0.3840     0.00535    -0.00916      0.9662      2.5852      1.0104      3.4870
  24    0.2313    0.3839     0.00540    -0.00940      0.9716      2.5758      1.0129      3.5035
  25    0.2313    0.3838     0.00545    -0.00957      0.9771      2.5662      1.0232      3.5340
  26    0.2313    0.3838     0.00550    -0.00970      0.9826      2.5565      1.0401      3.6024
  27    0.2314    0.3838     0.00554    -0.00982      0.9881      2.5467      1.0427      3.6318
  28    0.2314    0.3837     0.00557    -0.00994      0.9937      2.5368      1.0400      3.6840
  29    0.2314    0.3836     0.00557    -0.01016      0.9993      2.5266      1.0463      3.7299
  30    0.2314    0.3836     0.00559    -0.01031      1.0049      2.5163      1.0576      3.7335
```

---

## 3. Autoregressive Verification

The rollout is **truly autoregressive**: at timestep `t+1`, the window is
constructed from the **predicted** `(wx, wy, dx, dy)` values, NOT from
ground-truth features.  Specifically:

```python
# From train_phase2b_final.py rollout():
new_raw = np.array([wx, wy, float(dx), float(dy), obs, bnd], dtype=np.float32)
window  = np.vstack([window[1:], fs.transform(new_raw[np.newaxis])[0]])
prev_x, prev_y = wx, wy
```

- `wx`, `wy`, `dx`, `dy` all originate from the model prediction.
- Ground-truth `feats` array is **never indexed** inside the rollout loop.
- `fed_back_gt` flag was `False` for all steps.

| Property | Value |
|---|---|
| GT features read inside loop | **NO** |
| Predicted position fed back as `prev_x/prev_y` | **YES** |
| Predicted delta fed back as `delta_x/delta_y` feature | **YES** |
| Spatial features recomputed at predicted pos via KDTree | **YES** |

---

## 4. Actual Rollout Steps

- `N_ROLLOUT` constant: **30**
- Actual steps executed for trajectory 310: **30**

The loop runs `for step in range(N_ROLLOUT)` with no early-exit condition.
Every execution produces exactly 30 predictions.

---

## 5. Cumulative Displacement Comparison

| Metric | GT | Predicted | Ratio pred/GT |
|---|---|---|---|
| Total path length (m) | `1.1684` | `0.2699` | `0.231` |
| Mean per-step |delta| (m) | `0.03895` | `0.00900` | `0.231` |
| Endpoint L2 error (m) | — | `1.2183` | — |

---

## 6. Findings & Diagnosis

- **WARNING**: 57% of steps produce |delta| < 0.01 m. Model has converged to a near-zero-motion mode.

### Zero-Motion Scaled Setpoint (key diagnostic)

With `MinMaxScaler`, the scaled value that maps to **exactly zero real-world
delta** is:

```
zero_scaled = (0 − min) / (max − min)

  du:  zero_scaled = (0 − (−0.3634)) / 1.5946  =  0.3634 / 1.5946  =  0.2279
  dv:  zero_scaled = (0 − (−1.1741)) / 3.0339  =  1.1741 / 3.0339  =  0.3869
```

The model's actual output across all 30 steps:

| Channel | Zero-motion setpoint | Model output range | Distance from zero |
|---|---|---|---|
| `du` | `0.2279` | `0.2289 – 0.2314` | `+0.001 – +0.004` |
| `dv` | `0.3869` | `0.3836 – 0.3931` | `−0.003 – +0.006` |

**The model is outputting values within ±0.006 of the zero-motion setpoint in
scaled space.** This is why the real-world deltas are ≈ 0.001–0.006 m/step
instead of the expected ≈ 0.039 m/step.

### Scaler Domain Check

The scaler range covers both negative and positive deltas
(min\_du = −0.363 m, min\_dv = −1.174 m), so no silent truncation is present.
`inverse_transform` is mathematically correct.

### Model Quality Check

The predicted mean delta magnitude is **0.00900 m/step**
vs GT **0.03895 m/step** (ratio = 0.231).

The model has regressed to predicting near-zero motion (MSE mean regression /
mode collapse).  This is **not a rollout code bug** — the rollout
implementation is fully correct.  It is a model underfitting issue specific to
this trajectory.  Common causes:

1. **MSE loss drives towards the training-set mean delta**, which is close to
   zero when the dataset contains many near-stationary frames.
2. **Trajectory 310 is at the edge of the eligible-length window** (40 rows =
   exactly `WINDOW_SIZE + N_ROLLOUT`).  The model may not have seen sufficient
   examples with this trajectory's motion pattern.
3. **30-epoch training** with `N_PERSONS=400` may be insufficient for a 256-unit
   LSTM to capture the full delta distribution.

---

## 7. Code-Path Summary

```
train_phase2b_final.py :: train_one()
  ├─ fs = MinMaxScaler().fit_transform(X.reshape(-1,F))   ← features scaled
  ├─ ts = MinMaxScaler().fit_transform(y)                 ← targets ALSO scaled
  └─ model trained on (Xs, ys)                           ← both in [0,1]

train_phase2b_final.py :: rollout()
  ├─ feats_scaled = fs.transform(feats)                  ← seed window scaled
  ├─ for step in range(N_ROLLOUT):
  │    p_s   = model(window)                             ← output in [0,1]
  │    dx,dy = ts.inverse_transform(p_s)                 ← ✅ unscaled
  │    wx    = prev_x + dx                               ← ✅ uses unscaled
  │    obs,bnd = interp.query(wx, wy)                    ← ✅ KDTree recomp
  │    new_raw = [wx, wy, dx, dy, obs, bnd]              ← ✅ unscaled raw
  │    window  = vstack(window[1:],                      ← ✅ re-scaled
  │               fs.transform(new_raw))
  │    prev_x, prev_y = wx, wy                          ← ✅ predicted pos
  └─ return pred_x, pred_y, positions
```

---

*Generated by `rollout_bug_audit.py`*