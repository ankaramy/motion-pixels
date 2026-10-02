# MODEL_X Horizon Sweep — Phase 3 Rollout Dynamics Report

**Date:** 2026-06-11 · **Scope:** diagnosis ONLY. No retraining, no weight changes, no dataset changes,
no architecture/loss proposals beyond the evidence-based pointer in (J). Per-timestep rollout paths were
reconstructed by **deterministic replay of the frozen `H*/model_best.pth`** (`model.eval()`, no dropout) —
**byte-identical to the stored predictions** (ADE replay vs stored Δ = 0.00 at every horizon). The sweep
never saved per-step paths, so replay is the only route to t=1..H granularity; it changes nothing.

## Metrics & thresholds
- step length per t; **smoothed** step (moving-avg w=5, jitter-robust); **net displacement from start**
  (primary magnitude metric, per Phase 1/2); cosine & heading error of the (start→t) displacement vectors.
- Step ratio guarded at GT step > 1e-3 m; net-disp ratio / direction guarded at GT net-disp > 0.2 m
  (so very early net-disp ratios — where one step < 0.2 m — are unreliable and rise as t grows).
- `t_decay` = first t where pred step < 80 % of GT step for ≥ 5 consecutive steps (per window).

CSVs: `step_length_by_timestep.csv`, `net_displacement_growth_by_timestep.csv`,
`direction_metrics_by_timestep.csv`, `decay_onset_by_recording.csv`, `step_ratio_milestones.csv`,
`espanya_focus_metrics.csv`, `stairs_focus_metrics.csv`. Figures in `figures/`.

## The decisive numbers (ALL recordings)

Step-length ratio (pred/GT) at fixed steps — median over windows:

| horizon | t1 | t5 | t10 | t20 | t50 | t100 | t200 |
|---|---|---|---|---|---|---|---|
| H20 | 0.671 | 0.528 | 0.491 | 0.488 | – | – | – |
| H60 | 0.633 | 0.420 | 0.425 | 0.408 | 0.402 | – | – |
| H100 | 0.619 | 0.467 | 0.450 | 0.439 | 0.463 | 0.449 | – |
| H200 | 0.626 | 0.460 | 0.423 | 0.376 | 0.369 | 0.367 | 0.349 |
| H400 | 0.574 | 0.412 | 0.375 | 0.369 | 0.350 | 0.351 | 0.340 |

**Smoothed** (jitter-robust) step ratio, t1 → end: H20 0.71→0.63 · H60 0.60→0.66 · H100 0.67→0.64 ·
H200 0.65→0.45 · H400 0.68→0.39.
**Cosine (ALL)** t1 → end: 0.998→0.97 (H20), 0.999→0.94 (H60), 0.999→0.98 (H100), 0.999→0.91 (H200),
0.999→0.97 (H400).
**Decay onset (ALL):** t_decay median = 4 (H20/H60), 5 (H100), 7 (H200/H400); 67–97 % of windows decay.

## Answers

**A. Does predicted step length start too short, or decay over time?**
**It starts too short.** At t=1 — the pure one-step prediction, before any autoregressive compounding —
the model already produces only **~57–67 %** of GT step length (raw) and **~60–71 %** smoothed
(jitter-robust). The magnitude deficit exists from the very first step.

**B. At what timestep does decay begin?**
Effectively **immediately** — median `t_decay` is **4–7 steps**, i.e. the model is already below 80 % of
GT step length within the first handful of steps and stays there. A *second*, gradual decline appears only
for long horizons (smoothed ratio falls from ~0.65 to ~0.40 between t≈50 and t≈400).

**C. Is magnitude loss gradual or abrupt?**
**Both, but the abrupt/immediate component dominates.** The big single drop is at step 1 (instant ~35–40 %
magnitude loss). On top of that there is mild further decay that is negligible for short horizons (≤H100,
smoothed ratio ~flat) and becomes material only for long horizons (H200/H400). So: *abrupt immediate
under-stepping + slow long-range decay.*

**D. Does direction remain accurate while magnitude decays?**
**Yes — magnitude and direction are separable.** Cosine starts ~0.999 and, for the dataset as a whole and
for open-circulation/corridor recordings, stays high (0.9–0.99) throughout the rollout, while step
magnitude is short from t=1. The model gets *where* right immediately and holds it, independently of the
*how-far* deficit. (Exception: in plazas, direction also drifts at long range — see G.)

**E. Which recordings preserve direction best?**
**esplanade_espanya_01** (cosine ≈ 0.99 from t1 to t200 — essentially perfect) and
**red_bridge_combined_01** (≈ 0.76→0.82). Both are straight/channelled flows.

**F. Which recordings lose magnitude fastest?**
**placa_espanya_01** is worst (net-disp ratio ≈ 0.29 at H200-end), then **stairs_montjuic_01** (0.30) and
**placa_catalunya_01** (0.44). **esplanade** retains the most magnitude (0.69). Open, fast, multi-directional
plazas lose magnitude fastest; straight circulation least.

**G. Is Plaça Espanya difficult because of magnitude, direction, or both?**
**Both.** Magnitude is short *from the first step* and the worst of any recording (net-disp ≈ 0.29).
Direction starts perfect (cosine t1 = 0.999) but **collapses over a long rollout** (cosine → 0.13 by
t=200): in the large roundabout people curve and errors compound, so the start→t vector direction becomes
unreliable at long range. Immediate magnitude problem + emergent long-range direction problem.

**H. Is Stairs difficult because of direction, magnitude, or both?**
**Direction-led** (with magnitude also short). Stairs has the **weakest initial direction** of all
recordings (cosine t1 ≈ 0.90 vs ~0.999 elsewhere) and degrades to ~0.35; magnitude is also low (≈ 0.30).
The constrained multi-directional stair geometry makes *heading* hard from early on — the distinctive
problem here is direction, confirming Phase 2's low stairs cosine.

**I. Strongest evidence for the root cause of MODEL_X behavior.**
**The t=1 single-step prediction already undershoots displacement magnitude by ~30–40 % (smoothed,
jitter-robust) while direction is essentially perfect (cosine ≈ 0.999).** Because t=1 involves *no*
autoregressive feedback, this cannot be rollout error accumulation — it is the **one-step predictor's
magnitude bias**: an MSE objective on next-step displacement regresses toward the conditional mean and
shrinks predicted magnitude (averaging over plausible speeds/futures pulls the step shorter). Rollout
compounding adds only mild further decay (significant only beyond ~100 steps). Direction is learned well
and largely preserved. This is the classic regression-to-the-mean magnitude shrinkage, not a drift/
stability failure.

**J. Based solely on the evidence, what should MODEL_XR attempt to fix?**
Fix **per-step displacement MAGNITUDE underprediction in the one-step predictor** — the immediate ~35–40 %
shrinkage is the dominant, jitter-independent failure and it is a *training-target/objective* issue
(MSE-to-mean), not a direction or rollout-stability issue. Direction is already good and should be
preserved. Secondary, lower-priority targets: long-range direction drift in open plazas (placa_espanya)
and weak initial direction at stairs. **Do not** prioritise cumulative-path-length "collapse" (jitter
artifact) or rollout-stability tricks — the evidence says the step magnitude is wrong from step 1.

## Separability summary (for Phase 4 scoping)
- **Direction:** correct immediately (cosine ~0.999 @ t1), well preserved in straight/corridor flows,
  degrades only at long range in plazas. **Not the primary problem.**
- **Magnitude:** short immediately (~0.6 of GT step at t1, ~0.5–0.6 of GT net displacement once stable),
  worst in fast plazas. **The primary problem, and it is a one-step objective problem.**

## Warnings
- Raw step-length ratios are partly deflated by GT per-frame jitter (smoothed step + net-disp are the
  honest magnitude signals — both still show a clear immediate deficit, so the conclusion holds).
- Net-disp ratio at very small t is unreliable (one-step net-disp often < 0.2 m guard); read it at t ≥ 20.
- Long-horizon per-recording values (esp. red_bridge at H200/H400, few tracks) are small-sample; a couple
  of All-NaN medians occur where a recording has no direction-defined windows at a milestone step.
- Replay only; nothing retrained or modified outside `rollout_dynamics_audit/`.

## Figures (`figures/`)
overall_step_length_vs_time · overall_net_displacement_growth · overall_cosine_similarity_vs_time ·
overall_heading_error_vs_time · per_recording_step_length_curves · per_recording_net_displacement_curves ·
per_recording_cosine_curves · espanya_rollout_dynamics · stairs_rollout_dynamics · decay_onset_histograms
