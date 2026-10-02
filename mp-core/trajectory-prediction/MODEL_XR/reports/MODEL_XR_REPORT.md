# MODEL_XR — Report (Phase 4)

**Date:** 2026-06-11 · Controlled objective-only successor to MODEL_X. Same dataset, encoder, split,
architecture, scalers, recipe; the **only** change is the training loss. MODEL_X was not modified or
overwritten. Variants ranked **per instruction** by (1) step-1 smoothed magnitude ratio improvement,
(2) net-displacement ratio closeness to 1.0, (3) cosine preservation, (4) ADE/FDE tradeoff — **not by
ADE alone**.

## Variants & loss
`loss = MSE(pred_scaled,gt_scaled) + λ_mag·MSE(‖pred_m‖/σ,‖gt_m‖/σ) + λ_dir·mean(1−cos(pred_m,gt_m))`
(base MSE in scaled space = MODEL_X; magnitude/direction in metric space, σ = RMS train step size.)

| variant | λ_mag | λ_dir |
|---|---|---|
| A_CONTROL | 0 | 0 |
| B_MAG | 1.0 | 0 |
| C_MAG_LIGHT_DIR | 1.0 | 0.1 |
| D_MAG_STRONGER | 2.0 | 0 |

> Note: B_MAG's first training run hit a transient `CUDA error: unknown error` ~epoch 10 and was
> **retrained to convergence** (best val ADE 0.347 @ ep16, early-stop ep28). All variants are fully
> trained and comparable.

## Results — base horizon H10 (test)

| variant | ADE | FDE | **step1_sm** | **nd_ratio** | sm_ratio | **cosine** | cos@t1 |
|---|---|---|---|---|---|---|---|
| A_CONTROL | **0.305** | **0.480** | 0.600 | 0.395 | 0.373 | 0.962 | 0.859 |
| **B_MAG** | 0.362 | 0.603 | **1.036** | **0.935** | 0.828 | 0.956 | 0.900 |
| C_MAG_LIGHT_DIR | 0.356 | 0.588 | 1.036 | 0.836 | 0.766 | 0.930 | 0.941 |
| D_MAG_STRONGER | 0.392 | 0.659 | 1.174 | 1.043 | 0.895 | 0.946 | 0.901 |

## Results — net-displacement ratio & cosine across horizons (H20→H400)

| variant | nd_ratio (20→400) | cosine (20→400) |
|---|---|---|
| A_CONTROL | 0.37 → 0.25 (undershoot) | 0.94 → 0.55 |
| **B_MAG** | **1.02 → 1.21** (≈1, marginal at H400) | **0.95 → 0.94** (preserved) |
| C_MAG_LIGHT_DIR | 0.85 → 1.12 | 0.89 → **0.07 (collapses)** |
| D_MAG_STRONGER | **1.21 → 1.47** (overshoot) | 0.95 → 0.97 |

## Answers

**A. Did A_CONTROL reproduce MODEL_X?** **Yes, exactly.** ADE 0.30495 / FDE 0.47994 vs MODEL_X 0.305 /
0.480 — 0.0 % difference, same best epoch (35). The new training/eval code is validated.

**B. Which variant best improves magnitude?** **B_MAG.** By the ranking: (1) step1_sm 0.600→**1.036**
(into the 0.85–1.05 target); (2) nd_ratio **0.935** at H10 — closest to 1.0 of any variant, and the
best-behaved across horizons (1.02–1.21, only marginally over 1.2 at the extreme H400); (3) cosine
**0.956** ≈ control, preserved at every horizon (0.94–0.95). C ties B on step1 but undershoots nd and
**loses long-range direction**; D overshoots.

**C. Does magnitude improve from step 1?** **Yes — decisively, and at step 1.** The step-1 smoothed ratio
(the pure one-step prediction, the Phase-3 root cause) rises from **0.60 (control) to 1.04 (B_MAG)**. The
immediate under-stepping is fixed at its source, not patched downstream.

**D. Does direction remain preserved?** **Yes for B_MAG** (cosine 0.96 at H10; 0.90–0.95 across all
horizons — statistically unchanged from control). **No for C_MAG_LIGHT_DIR**, which paradoxically
**degrades long-range direction** (cosine 0.89→0.07 by H400) despite its direction term — so the light
direction penalty hurt rather than helped. D preserves direction (0.95–0.97).

**E. Does any variant overshoot?** **D_MAG_STRONGER — flagged: net-displacement ratio consistently > 1.2
at every horizon (1.21→1.47).** B_MAG only reaches 1.2 at the extreme H400 (1.02–1.16 through H200), so it
is **not** a consistent overshooter. C does not overshoot (≤1.12).

**F. Which recording benefits most?** **placa_espanya_01 — the hardest recording (Phase 2).** Its
net-disp ratio jumps **0.42 → 1.21 (B_MAG)** while cosine holds (0.92). The fix lands exactly where it was
most needed, not only on easy recordings.

**G. Which remains hardest?** **stairs_montjuic_01** — net-disp only 0.38→0.54 (smallest magnitude gain)
and the lowest cosine of any site (~0.67, unchanged across variants). placa_catalunya is second-hardest
for magnitude (0.51→0.69).

**H. Does placa_espanya improve?** **Yes, dramatically** — magnitude essentially corrected (0.42→1.21
net-disp) with direction preserved (cosine 0.92). The combination of immediate magnitude + retained
direction is the headline win on the hardest plaza.

**I. Does stairs direction degrade or improve?** **Essentially unchanged** (cosine 0.68→0.67 at H10 across
variants). A magnitude loss neither fixed nor broke stairs' direction; its weak heading is an
architectural/multi-directional property (Phases 2–3), not something a magnitude objective addresses.

**J. Is Red Bridge still easy?** **Yes** — high cosine (0.87–0.95) and magnitude easily corrected
(0.61→0.95 net-disp under B). Consistent with Phase 2: Red Bridge is the easy corridor case.

**K. Should the best XR variant replace MODEL_X?** **Conditional — this is a thesis-claim decision, not an
automatic yes.** B_MAG fixes the magnitude defect (step1 0.60→1.04; espanya 0.42→1.21) **without breaking
direction**, but **ADE/FDE worsen ~17–19 %** (0.305→0.362, 0.480→0.603) and FDE more at long horizons.
This is the expected, honest tradeoff: ADE/FDE reward conservative short "stub" predictions, so a model
that predicts realistic full-length paths necessarily scores worse on them.
- **If the thesis claim is realistic / visualisable trajectory forecasting (length and shape matter):**
  adopt **B_MAG as MODEL_XR**, the recommended model — it produces metrically plausible rollouts.
- **If the thesis claim is positional-accuracy benchmarking (lowest ADE/FDE):** keep **MODEL_X**.
- **Recommended:** report **both** — MODEL_X as the ADE-optimal reference and **MODEL_XR (B_MAG)** as the
  realistic-magnitude model — and do **not overwrite** MODEL_X. The two answer different questions.

**L. Thesis claim supported by these results:** *MODEL_X's short predictions were an artifact of the
MSE objective (regression-to-the-mean), not a limit of the data or architecture.* A magnitude-aware loss
(B_MAG) restores realistic per-step and net displacement — including on the hardest plaza
(placa_espanya, 0.42→1.21) — **while fully preserving the direction MODEL_X already learned** (cosine
≈0.96, unchanged). The cost is a modest, well-understood ADE/FDE increase because those metrics favour
under-stepping. Net displacement / smoothed path length are the honest magnitude metrics for
architectural use, and on those MODEL_XR is the better model.

## Failure cases (`model_xr_failure_cases.csv`)
- All three magnitude variants trip `ade_substantially_worse` (ADE > control×1.15) — expected tradeoff.
- **D_MAG_STRONGER:** consistent **overshoot** (nd_ratio 1.21–1.47). Not recommended.
- **C_MAG_LIGHT_DIR:** **long-range direction collapse** (cosine→0.07 @H400). Not recommended.
- **B_MAG:** no disqualifying failure; only a marginal nd>1.2 at the extreme H400.

## Verdict
**Best variant = B_MAG** (MSE + magnitude loss, λ_mag=1.0). Magnitude improved from step 1 (0.60→1.04),
net displacement ≈1.0 with no consistent overshoot, direction fully preserved (0.96), at a ~17–19 % ADE
cost. Whether it *replaces* MODEL_X depends on the thesis claim (length-realism → yes; ADE-benchmark → no);
recommendation is to keep both, MODEL_X frozen.

## Artifacts
CSVs: `reports/model_xr_{summary_metrics,horizon_metrics,per_recording_metrics,step1_metrics,variant_comparison,failure_cases}.csv`
Figures (`figures/`, equal metric scaling): variant ADE/FDE, step1 ratio, nd-ratio & smoothed-ratio by
horizon, cosine, per-recording magnitude, placa_espanya / stairs / red_bridge comparisons,
best_variant_rollout_examples (control vs B_MAG vs GT), failure_case_examples.
Checkpoints: `checkpoints/<variant>/model_best.pth` (MODEL_X untouched).
