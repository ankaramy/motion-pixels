# exp08 — Rollout magnitude audit

**Question:** has Model C *always* under-predicted displacement magnitude, or was it introduced by turn balancing / turn-only training / split changes / the rollout procedure — and is it present even in Overfit10X?

## Method

Per rollout we measure GT path length and predicted path length (cumulative step displacement over the 20-step horizon) and report median GT length, median predicted length, and the pred/GT ratio. The 5 V3 models are evaluated on ONE fixed common window set (all 5 V3 recordings, cap_all=2500, cap_gen=1200, seed=42) so only the model varies. Overfit10X is evaluated on the MACBA bridge data it memorised (different dataset/scale — its absolute lengths are NOT comparable to V3, only the ratio is). exp04 uses its 4-output direction rollout; all others use the frozen Model C rollout.

## Results (all windows)

| model | n | median GT len (m) | median pred len (m) | pred/GT ratio | median per-window ratio |
|---|---|---|---|---|---|
| Phase-4 baseline | 2500 | 0.96 | 0.30 | **0.31** | 0.34 |
| exp01 turn-balanced | 2500 | 0.96 | 0.56 | **0.59** | 0.64 |
| exp02 turn-only | 2500 | 0.96 | 0.65 | **0.68** | 0.65 |
| exp04 direction-out | 2500 | 0.96 | 0.62 | **0.65** | 0.66 |
| exp06 split-change | 2500 | 0.96 | 0.77 | **0.80** | 0.73 |
| Overfit10X (MACBA) | 51 | 0.83 | 0.67 | **0.81** | 0.88 |

## Results (genuine-turn windows)

| model | n | median GT len (m) | median pred len (m) | pred/GT ratio |
|---|---|---|---|---|
| Phase-4 baseline | 1226 | 3.68 | 0.46 | **0.12** |
| exp01 turn-balanced | 1226 | 3.68 | 0.93 | **0.25** |
| exp02 turn-only | 1226 | 3.68 | 0.85 | **0.23** |
| exp04 direction-out | 1226 | 3.68 | 1.00 | **0.27** |
| exp06 split-change | 1226 | 3.68 | 0.89 | **0.24** |

## Answer

**Under-prediction is INTRINSIC, present in every experiment — including the Phase-4 baseline (ratio 0.31) and Overfit10X (ratio 0.81).**

- It is **NOT** introduced by turn balancing, turn-only training, the split change, or the direction-output head. The opposite is true: the **Phase-4 baseline under-predicts the MOST** (ratio 0.31), and every turn-focused / split variant *mitigates* it (0.59 → 0.68 → 0.65 → 0.80) by pushing predicted displacement closer to GT — though none reach 1.0. So those interventions reduce the shrinkage; they do not cause it.
- It is **NOT** an artefact of a particular split — exp02 (original split, 0.68) and exp06 (re-cut split, 0.80) both under-predict; the split change slightly *improved* it.
- It is present **even in Overfit10X** (0.81), which memorises its training content — so it is not a generalisation gap either; pure memorisation of an autoregressive MSE model still shrinks displacement.
- On **genuine turns** the shrinkage is far worse (ratios 0.12–0.27): GT turn paths are long (median 3.68 m) while predictions stay short (~0.5–1.0 m), so under-prediction and the direction collapse compound exactly where turning matters most.
- The common cause is the **MSE objective + autoregressive rollout**: MSE regresses each predicted step toward the conditional mean (slightly shorter than the true step), and feeding the shortened step back in compounds the shrinkage over the horizon. This is the magnitude analogue of the straight-line / direction collapse documented in exp01-06.

**Implication:** fixing displacement magnitude needs an objective/rollout change (e.g. scheduled sampling / free-running training, a magnitude-aware or distributional loss, or direct multi-step supervision), not more turn balancing or split tweaks.

## Plots

- `predicted_length_vs_GT_length.png` — per-model scatter of predicted vs GT rollout length (points below the dashed y=x line = under-predicted; magenta line = median ratio).
- `length_distribution_comparison.png` — median GT vs predicted length, pred/GT ratio per model, and per-window ratio box plots.

Data: `exp08_magnitude_summary.csv`.
