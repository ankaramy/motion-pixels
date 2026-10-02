# MODEL_XR — Phase 5A Curvature / Trajectory-Shape Audit

**Date:** 2026-06-11 · **Measurement only.** No retraining, no new models, no MODEL_XC. The frozen
checkpoints (MODEL_X + the four MODEL_XR variants) were replayed deterministically (`model.eval()`, no
dropout) — rollouts are byte-identical to those already generated. Per-window paths were never stored,
so replay is the only route to per-step geometry.

## Methodology note (corrected)
- **First curvature pass was patched.** The initial version computed heading on lightly-smoothed (w=5)
  paths. That left GT **noise-dominated**: median GT cumulative heading came out **≈1992° over a 100-step
  (~5 m) window** — physically impossible for a pedestrian. Per-frame tracking jitter (Phases 1–2)
  produces large spurious heading swings at slow speed, so per-step curvature is meaningless.
- **Final audit uses raw paths**, and `curv_metrics` **handles denoising internally**: each rollout is
  adaptively smoothed (moving-average window ≈ one segment length) and **down-sampled to 10 macro
  nodes** (`N_SEG=10`, true endpoints kept). Curvature is then the turning between these macro segments —
  real walking-direction change, not jitter.
- **Macro-scale numbers are now physically sane:** among *curvy* windows (GT macro net > 1 m and macro
  cumulative heading > 15°; 48,788 windows at H100), GT median macro cumulative heading is **≈192°**, and
  models fall well below it. All shape ratios below use this macro definition. GT's absolute curvature
  still carries some residual meander/noise in the most extreme windows, so **shape ratios are read
  relatively (model vs model)** — where the conclusion is robust regardless of GT's absolute value.

## Headline results (H100, 48,788 curvy windows)

| model | shape_ratio (pred/GT cum-heading) | turn_count_ratio | macro cum-heading (deg) |
|---|---|---|---|
| GT | 1.00 | 1.00 | 192.3 |
| MODEL_X (= XR_A_CONTROL) | **0.364** | 0.40 | 80.7 |
| **XR_B_MAG** | **0.043** | 0.00 | 6.7 |
| XR_C_MAG_LIGHT_DIR | 0.188 | 0.00 | 31.1 |
| XR_D_MAG_STRONGER | 0.036 | 0.00 | 5.3 |

Tortuosity ratio is ≈0.97–0.99 for all models (a weak discriminator at macro scale — turning shows up in
cumulative heading, not in path-length/net-displacement ratio).

**Curvature growth over time (H100, median):**
GT 9°→33°→84°→148°→**192°** (t=10/25/50/75/100); MODEL_X 16°→45°→66°→76°→**81°**; B_MAG 3°→5°→6°→6.5°→**6.7°**.
→ MODEL_X tracks GT's turning through ~t25–50 then plateaus; **B_MAG is flat from ~step 5 — a straight line.**

**Shape ratio vs horizon:** MODEL_X 0.667(H20)→0.187(H400); **B_MAG ~0.04–0.06 at every horizon** (straight
even at 1 m); C ~0.16–0.22; D ~0.04. The straightening is **immediate and horizon-independent**, mirroring
how MODEL_X's magnitude deficit was immediate.

**Topology (shape_ratio):** OPEN — MODEL_X 0.355, B_MAG **0.036**; CONSTRAINED — MODEL_X 0.475, B_MAG
**0.245**. **Per recording (MODEL_X→B_MAG):** esplanade 0.36→**0.02**, placa_espanya 0.42→**0.01**,
red_bridge 0.30→0.13, placa_catalunya 0.24→0.32, stairs 0.53→0.32.

## Answers

**A. Does B_MAG reduce trajectory curvature?** **Yes — drastically.** It reproduces only ~4 % of GT's
macro turning (median shape_ratio 0.043; cum-heading 6.7° vs GT 192° at H100) and registers **zero**
macro turns > 15°. It predicts near-straight lines.

**B. How much curvature is preserved?** **~4–6 %** (B_MAG) of GT macro cumulative heading, flat across all
horizons. MODEL_X preserves far more — 19–67 % depending on horizon (more at short range).

**C. Does B_MAG preserve shape better than MODEL_X?** **No — substantially worse.** MODEL_X retains
**~6–9× more** curvature than B_MAG at every horizon (e.g. H100: 0.364 vs 0.043).

**D. Does magnitude improvement come at the cost of shape?** **Yes, definitively.** The magnitude loss
that fixed displacement (Phase 4) straightened the trajectories: to make steps the correct *length* under
MSE + magnitude, the model commits to the dominant straight heading and suppresses the small turning that
gives a path its shape. Correct length was bought with lost curvature.

**E. Which recordings lose the most shape?** The **open plazas** — placa_espanya (shape_ratio 0.01) and
esplanade (0.02). Catalunya (0.32) and stairs (0.32) retain the most.

**F. Do open plazas suffer more than constrained spaces?** **Yes, clearly** — B_MAG shape_ratio is
**0.036 (open) vs 0.245 (constrained)**, ~7× worse in open space. Constrained geometry (corridor/stairs)
channels motion, so a straight prediction is less wrong; open plazas allow free curving that the model
flattens.

**G. Is Plaça Espanya a curvature, magnitude, or both problem?** **Now primarily CURVATURE.** Phase 4
fixed its magnitude (net-disp 0.42→1.21) and direction is acceptable, but its **shape is essentially
destroyed (shape_ratio 0.014, the lowest of any recording)**. The remaining espanya gap is path geometry,
not distance.

**H. Is Stairs a direction, curvature, or both problem?** **Primarily DIRECTION** (Phase 3 cosine ≈0.67,
the weakest site) with a secondary **magnitude** shortfall (Phase 4 net-disp 0.54, lowest). Its **shape is
comparatively retained (0.32)**, so stairs is *not* a curvature problem — heading is its issue.

**I. Strongest remaining limitation after MODEL_XR.** **Trajectory shape / path geometry.** After
MODEL_XR_B_MAG the model has correct **direction** (Phase 3/4) and correct **distance** (Phase 4), but its
paths are **too straight** — ~4–6 % of GT curvature, worst in open plazas. Geometry/curvature is the open
front.

**J. What should a future MODEL_XC attempt to improve (evidence only)?** Restore **curvature / path-shape
fidelity** — a curvature-aware or path/sequence objective that penalises *under-turning*, focused on
**open plazas**, **without regressing** the magnitude (B_MAG) or direction gains. A supporting hint:
**C_MAG_LIGHT_DIR retained ~4× more curvature than B_MAG** (0.19 vs 0.04), suggesting direction/shape-aware
terms can recover bend — relevant for MODEL_XC design (not acted on here). No transformer is warranted;
the evidence points to an objective/term change, consistent with the controlled-experiment approach.

## Verdict
- **B_MAG straightening is real and robust** — median shape_ratio 0.043 across **48,788** curvy windows,
  immediate (already flat at H20), confirmed visually (`worst_shape_examples.png`,
  `best_shape_examples.png`).
- **B_MAG improves magnitude but reduces trajectory-shape fidelity** (the Phase-4 win cost curvature).
- **MODEL_X keeps slightly more bend (shape_ratio 0.36) but remains too short** (Phases 3–4) — neither
  model is right on both length and shape.
- **The remaining limitation is curvature / path geometry, especially in open spaces.** A
  curvature-aware **MODEL_XC is justified** (next phase; not started here).

## Artifacts
CSVs: `curvature_metrics.csv`, `shape_preservation_metrics.csv`, `per_recording_curvature_metrics.csv`,
`curvature_over_time.csv`, `curvature_vs_horizon.csv` (extra), `topology_curvature_metrics.csv`,
`shape_example_inventory.csv`.
Figures (`figures/`, equal metric scaling): curvature_distribution_by_model, shape_ratio_by_model,
tortuosity_ratio_by_model, turn_count_ratio_by_model, curvature_vs_horizon, curvature_growth_over_time,
open_vs_constrained_curvature, espanya_shape_analysis, stairs_shape_analysis, best_shape_examples,
worst_shape_examples.
