# MODEL_XC — Report (Phase 5B, curvature-aware)

**Date:** 2026-06-12 · Controlled loss-only successor to MODEL_XR_B_MAG. Same dataset/split/arch/
scalers/horizons; only a **curvature term** is added. MODEL_X and MODEL_XR untouched. Variants A–D fully
trained+evaluated+swept; **variant E is OPTIONAL / INCOMPLETE** (interrupted ~epoch 23, still improving,
not converged, not evaluated) and is **excluded from all rankings.**

---

## EXECUTIVE SUMMARY

**1. Which variant won?** **MODEL_XC_B_CURV_LIGHT** (λ_mag=1.0, λ_curv=0.1). It recovers the most shape
with the least cost to magnitude/direction.

**2. Did shape improve over MODEL_XR_B_MAG?** **Yes — dramatically.** Macro shape_ratio rose from
**0.05 → 0.33** at H10 (~6×), and **0.06 → 0.53 at H20** (~9×). B's shape (0.33) is back to ~MODEL_X
level (0.36). The straightening Phase 5A diagnosed is substantially reversed.

**3. Did magnitude remain realistic?** **Mostly yes.** step1 smoothed ratio 0.93 (in the 0.85–1.10 band);
net-displacement ratio is **0.98–1.04 across H60–H400** (excellent) and only marginally short at H10
(0.81). The magnitude fix is largely retained.

**4. Did direction remain strong?** **Mildly degraded.** Cosine 0.956 → **0.884** overall (still high, but
below the strict 0.90 guard). Worst at placa_espanya (0.92 → 0.75). Curvature trades against direction —
the main, expected cost.

**5. Did Plaça Espanya improve?** **Yes — the headline win.** shape 0.05 → **0.30** (6×) *and*
net-displacement 1.21 → 0.92 (overshoot corrected), at a direction cost (0.92 → 0.75). Espanya's
curvature problem (Phase 5A) is largely addressed.

**6. Did open-space curvature improve?** **Yes, most of all.** OPEN shape_ratio 0.052 → **0.333** (6.4×),
now exceeding constrained spaces and matching MODEL_X's open level — the fix lands exactly where B_MAG was
worst (esplanade 0.05→0.41, espanya 0.05→0.30).

**7. Did any variant overshoot?** **No.** Curvature *reduces* overshoot (turning consumes net distance):
overshoot rate 31.9% (control) → 23.2% (B) → 14.5% (D); C/D actually *undershoot* (nd 0.65–0.75).

**8. Should MODEL_XC replace MODEL_XR_B_MAG?** **Yes, conditionally — B_CURV_LIGHT is the better
all-round model.** It beats B_MAG on shape (0.33 vs 0.05) *and* ADE (0.332 vs 0.362), keeps realistic
length, at a modest direction cost (cos 0.88 vs 0.96). It does not cleanly pass every preset guard
(H10 net-disp 0.81 < 0.85; cosine 0.88 < 0.90), so it is presented honestly as the best point on a
three-way **magnitude ↔ curvature ↔ direction** tradeoff, not a strict domination.

**9. Thesis framing:** **Present all three — MODEL_X → MODEL_XR → MODEL_XC as baseline → magnitude fix →
curvature fix. Do NOT stop at MODEL_XR.** The arc is a clean, honest contribution: each stage fixes a
*diagnosed* geometric deficit (first distance, then shape) by a controlled objective change, with the
tradeoffs measured and reported. MODEL_XC completes the trajectory-geometry story.

---

## Results — base H10 (test)

| variant | λ_curv | ADE | step1_sm | nd_ratio | cosine | **shape_ratio** | overshoot% |
|---|---|---|---|---|---|---|---|
| A_BMAG_CONTROL (=MODEL_XR_B_MAG) | 0.0 | 0.362 | 1.036 | 0.935 | 0.956 | 0.053 | 31.9 |
| **B_CURV_LIGHT** | 0.10 | **0.332** | 0.933 | 0.805 | 0.884 | **0.331** | 23.2 |
| C_CURV_MED | 0.25 | 0.327 | 0.862 | 0.750 | 0.793 | 0.296 | 19.3 |
| D_CURV_STRONG | 0.50 | 0.324 | 0.804 | 0.656 | 0.693 | 0.328 | 14.5 |
| E_CURV_DIR_LIGHT | 0.25(+dir) | — | — | — | — | — | **OPTIONAL / INCOMPLETE** |

Control reproduced MODEL_XR_B_MAG exactly (ADE 0.3624, step1_sm 1.036, nd 0.935, cos 0.956).

**Trend with λ_curv:** shape recovers immediately (all variants ≈0.30–0.33), but higher λ_curv
progressively erodes magnitude (nd 0.94→0.81→0.75→0.66) and direction (cos 0.96→0.88→0.79→0.69). **B
(lightest) is the sweet spot** — most shape per unit magnitude/direction lost.

## Net-displacement & shape vs horizon (B_CURV_LIGHT)
- **nd_ratio:** 0.89(H20) → 0.98 → **1.02 → 1.03 → 1.04** (H400) — magnitude well-preserved across the
  rollout; only the first ~20 steps run slightly short. (C/D stay 0.65–0.79 — genuinely undershooting.)
- **shape_ratio:** 0.53(H20) → 0.57 → 0.42 → 0.26 → 0.18(H400) — large recovery at all horizons (control
  flat ≈0.05).

## Per-recording (control → B_CURV_LIGHT, H10)
| recording | shape | nd | cosine |
|---|---|---|---|
| placa_espanya | 0.05 → **0.30** | 1.21 → 0.92 | 0.92 → 0.75 |
| esplanade | 0.05 → **0.41** | 1.03 → 0.73 | 0.99 → 0.92 |
| placa_catalunya | 0.23 → 0.31 | 0.69 → 0.84 | 0.90 → 0.88 |
| stairs_montjuic | 0.18 → 0.29 | 0.54 → 0.62 | 0.67 → **0.73** |
| red_bridge | 0.11 → 0.22 | 0.95 → 0.93 | 0.87 → **0.94** |

Open plazas (espanya, esplanade) gain the most shape. Notably **stairs and red_bridge direction
*improved*** (curvature helped), while open-plaza direction dropped (more turning → more chance to mis-turn).

## Topology shape_ratio (control → B)
OPEN 0.052 → **0.333** (6.4×); CONSTRAINED 0.122 → 0.240 (2×). The fix is strongest in open space — exactly
the Phase-5A failure zone.

## Failure flags (`model_xc_failure_cases.csv`)
No variant passes all guards simultaneously (the magnitude/curvature/direction tension is real):
- **B_CURV_LIGHT:** `nd_ratio_out_of_band` (H10 0.81 < 0.85 — but ≥0.98 at H60–H400), `direction_degraded`
  (0.884 < 0.90). Both marginal. **Best balance.**
- **C_CURV_MED / D_CURV_STRONG:** same flags but worse (nd 0.66–0.75, cos 0.69–0.79); D also
  `magnitude_step1_out_of_band`. Over-weighted curvature.
- No overshoot in any curvature variant.

## Is the curvature loss stable and useful?
**Yes.** Training was stable (smooth convergence, normal early-stop), the single-step turn target +
moving mask behaved as designed (smoke-verified: zero when matching GT turn, penalizes straight
predictions when GT turns), and it produced a large, robust, visually-confirmed shape recovery
(`xc_best_shape_examples.png`). Limitation: curvature is enforced at the single-step level, not as a macro
multi-step rollout target (documented); it nonetheless reverses the *immediate* straightening Phase 5A
identified.

## Final thesis claim (MODEL_X → MODEL_XR → MODEL_XC)
A pedestrian-trajectory LSTM's failures were **objective artifacts, fixable by controlled loss design**,
each isolated and measured:
1. **MODEL_X** (MSE): correct direction, **too short** (regression-to-mean), partial curvature.
2. **MODEL_XR_B_MAG** (+magnitude): **realistic distance**, direction preserved, but **flattened shape**.
3. **MODEL_XC_B_CURV_LIGHT** (+curvature): **recovered shape** (~MODEL_X level, best in open plazas) while
   keeping most of the magnitude fix, at a modest direction cost.
The remaining frontier is the **magnitude ↔ curvature ↔ direction trade** — no single-step objective
maxes all three at once; B_CURV_LIGHT is the best measured balance. (A macro multi-step / shape-sequence
objective is the natural next lever — future work, not done here.)

## Recommendation
Adopt **MODEL_XC_B_CURV_LIGHT** as the realistic-geometry thesis model; keep **MODEL_X** as the
ADE-optimal reference and **MODEL_XR_B_MAG** as the magnitude-only milestone. Present the three-stage
progression. Do not overwrite earlier models.

## Artifacts
CSVs: `reports/model_xc_{summary,horizon,per_recording,curvature,topology,variant_comparison,failure_cases}.csv`
Figures (`figures/`, equal metric scaling): xc_variant_ADE_FDE_comparison, xc_magnitude_preservation,
xc_cosine_similarity_comparison, xc_shape_ratio_comparison, xc_curvature_vs_horizon,
xc_open_vs_constrained_shape, xc_placa_espanya_comparison, xc_stairs_comparison, xc_best_shape_examples,
xc_worst_failure_examples, xc_tradeoff_radar_or_bar.
Checkpoints: `checkpoints/<variant>/model_best.pth` (A–D complete; E incomplete). MODEL_X / MODEL_XR not modified.
