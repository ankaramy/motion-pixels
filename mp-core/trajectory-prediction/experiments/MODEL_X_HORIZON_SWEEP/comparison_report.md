# MODEL_X Horizon Sweep — Comparison Report

Same Model X recipe at every horizon (Model C TrajectoryLSTM, 10 v3 features, mixed
track-level split, single-step next-displacement MSE objective, seed 42). Horizon H sets
the track-length filter (need >= 10+H frames), the autoregressive rollout depth, and the
val-ADE checkpoint. **No multi-step loss, no path scaling, no fabricated length.**

Usability thresholds (documented): *usable* = collapse rate ≤ 35% (share of windows with pred length < 25% of GT); *visually useful length* = median pred/GT ratio ≥ 55%.

## Summary table

| H | ~dist | train_w | val_w | test_w | ADE | FDE | GT_len_med | pred_len_med | ratio | collapse | recommended_use |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 1.0 m | 817,830 | 2,500 | 87,320 | 0.446 | 0.708 | 1.03 | 0.44 | 0.45 | 26% | usable but conservative (undershoots length) |
| 60 | 3.0 m | 796,736 | 2,500 | 77,438 | 1.011 | 1.811 | 3.2 | 1.16 | 0.355 | 39% | marginal — frequent length collapse |
| 100 | 5.0 m | 772,805 | 2,500 | 69,718 | 1.419 | 2.565 | 5.31 | 1.88 | 0.348 | 38% | marginal — frequent length collapse |
| 200 | 10.0 m | 711,368 | 2,500 | 55,505 | 2.595 | 5.016 | 10.68 | 3.22 | 0.322 | 43% | marginal — frequent length collapse |
| 400 | 20.0 m | 581,319 | 1,434 | 38,639 | 5.079 | 10.015 | 22.98 | 4.86 | 0.266 | 48% | marginal — frequent length collapse |

## Data availability (tracks dropped for being too short)

| H | need frames | train tr | val tr | test tr | dropped |
|---|---|---|---|---|---|
| 20 | 30 | 2269 | 296 | 286 | 683 |
| 60 | 70 | 1695 | 221 | 211 | 1407 |
| 100 | 110 | 1386 | 184 | 175 | 1789 |
| 200 | 210 | 958 | 122 | 120 | 2334 |
| 400 | 410 | 501 | 62 | 63 | 2908 |

## Answers

**1. Best ADE/FDE?** Best ADE = **H20** (ADE 0.446 m); best FDE = **H20** (FDE 0.708 m). ADE/FDE in metres grow with horizon (more steps to accumulate error) — short horizons win on raw error, as expected.

**2. Longest usable predictions?** **H20** (~1.0 m) is the longest horizon still under the collapse threshold (collapse 26%, median pred length 0.44 m).

**3. Where does prediction collapse?** Collapse (>35% of windows under 25% of GT length) first appears at **H60**.

**4. Best for architectural visualization?** Two honest readings, depending on what the figure must show:
- *Faithful length* (the predicted path must be a believable distance): only **H20** qualifies — and it
  is essentially the Model X baseline (~1 m), too short to be "architectural." By a strict length test,
  **no long horizon is good for visualization.**
- *Directional intent at plan scale* (where is this pedestrian heading over a few metres, accepting that
  length is undershot): **H100 (~5 m)** is the practical ceiling. It is the last horizon whose median
  angular error (63.7°) is still clearly below the ~79–90° chance band, and 5 m is a meaningful
  architectural distance. Beyond H100, direction itself collapses (H200 median angular 79.3°).

**5. Are H200 / H400 realistic or too uncertain?** **Too uncertain — not realistic.**
H200 (~10 m): ratio 0.322, collapse 43%, median angular 79.3° (≈ chance), ADE 2.60 m.
H400 (~20 m): ratio 0.266, collapse 48%, ADE 5.08 m, and the test set thins out badly (only 63 tracks,
red_bridge just 3). At these reaches the single-step rollout predicts < 1/3 of the true distance and the
heading is essentially random — they document the collapse, they are not deployable forecasts.

**6. Freeze as Model X_long?** **Recommendation: freeze H100 (`H100/model_best.pth`) as `Model_X_long`,
with an explicit "directional-intent, ~5 m, length-undershooting" label** — *and keep the H10 MODEL_X
baseline as the accurate short-range model.*
Rationale: the strict-usability winner is H20, but that is just the baseline and gives no extra reach.
H100 is the longest horizon that still carries usable *direction* (angular 63.7° vs ~80°+ beyond it) at a
genuinely architectural distance (~5 m). It must NOT be presented as a faithful-length 5 m forecast — its
median predicted length is 1.9 m of a 5.3 m path. If the thesis needs only honest, accurate predictions,
do **not** freeze any long model and stay on MODEL_X (H10); if it needs a plan-scale directional cue,
H100 is the defensible choice and H200/H400 are not.

## Honesty note
ADE/FDE are reported in metres over the full horizon; predicted paths are the raw model
rollout with **no independent scaling**. Where the model undershoots length (low ratio /
high collapse), that is reported as-is — long horizons are genuinely more uncertain and
the model stays conservative rather than inventing motion.