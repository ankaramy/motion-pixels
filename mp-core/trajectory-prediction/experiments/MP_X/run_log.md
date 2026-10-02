# MP_X run log

Exact commands, timestamps, and outcomes. Newest entries at the bottom.

---

## 2026-06-09 — MP_X scaffold created

- Created `MP_X/` with `README.md`, `run_log.md`, `shared/`, and the four
  experiment folders.
- Confirmed environment: `torch 2.7.0+cu118`, CUDA available.
- Confirmed dataset: `Barcelona_v3_manual_master_dataset/model_C_dataset.csv`
  (10 Model C features + `target_du/dv`, recording-level split).
- Architecture imported read-only from
  `experiments/schema_ablation_bridge/run_bridge_ablation.py` (`TrajectoryLSTM`)
  and `experiments/phase5_v3_gru_comparison/phase5_gru_lib.py` (rollout helpers).

---

## 2026-06-09 — exp01: turn-balanced Model C

Command (run from repo root):
```
python mp-core/trajectory-prediction/experiments/MP_X/exp01_turn_balanced_model_c/run_exp01.py
```
Reuse a trained checkpoint with `--skip-train`; bigger all-windows eval sample with `--cap-all N`.

**Environment:** torch 2.7.0+cu118, CUDA. Train 229.1 s (best epoch 2, early-stopped 17),
val MSE 0.49480 (≈ Phase-4 0.4929 — MSE objective unchanged, only sampling differs).

**Turn-scarcity census (the core evidence):**
- train windows 824,917 → 98.7% straight; only 10,905 genuine turns (mild 3,042 / sharp 2,316 / U-turn 5,547).
- val windows 124,265 → 99.96% straight; only **55** genuine-turn windows (stairs_montjuic is ~turn-free).

**Results — held-out (val+test), sliding windows, horizon 20:**
- all windows (n=4000): ADE 0.337 m, FDE 0.726 m, angular 83.8°.
- genuine turns (n=50): ADE 1.418 m, FDE 2.692 m, angular 89.8°, angularity 0.37, **TCR 0.60**, predicted Δhead 38° (GT 124°).
- per bin: mild ang 79° TCR 0.53 · sharp ang 101° TCR 0.42 · U-turn ang 91° TCR 0.74.
- train capacity ref (n=2052 genuine): ADE 1.241 m, angular 71.3°, TCR 0.53.

**Outcome:** PARTIAL SUCCESS. Turn-balancing breaks the straight-line collapse — rollouts
are visibly curved and the model commits to a turn ~60% of the time (vs near-zero baseline),
confirming scarcity was a real cause and the architecture is not incapable. But it
under-rotates and mis-directs (angular ~90°, angularity ~0.3–0.4, worse ADE on turns):
angular *expression* recovered, angular *accuracy* not. Necessary, not sufficient.
Full write-up: `exp01_turn_balanced_model_c/README.md`.

---

## 2026-06-09 — exp02: turn-only diagnostic Model C

Command (run from repo root):
```
python mp-core/trajectory-prediction/experiments/MP_X/exp02_turn_only_diagnostic/run_exp02.py
```
Train ONLY on the 10,905 genuine-turn windows (no sampler). Added shared
`val_window_filter` so early stopping tracks the 55 genuine-turn val windows (not the
99.96%-straight full val), keeping the checkpoint off the straight-collapse minimum.
Best epoch 26/41, train 7.6 s, val(turn) MSE 1.065.

**Results — held-out (val+test) genuine turns (n=50), exp01 → exp02:**
- ADE 1.418 → 1.300 m, FDE 2.692 → 2.484 m (both slightly REDUCED).
- predicted Δhead 38° → 49°, angularity 0.37 → 0.60, **TCR 0.60 → 0.78** (more angular expression).
- angular error 89.8° → 88.0° (UNCHANGED, still ≈chance ~90°).
- per-bin tell: mild turns now angularity **1.28** (over-produces magnitude, predicts 57° vs GT 50°)
  yet angular err 80° → right magnitude, WRONG direction. sharp/U-turn still under-rotate (0.28–0.37).

**Outcome:** turn-only training INCREASES angular expression further and marginally helps
ADE/FDE, but does NOT fix turn DIRECTION (≈chance). Removing straight dilution is not the
missing ingredient → next bottleneck = displacement-only output head (no heading target to
commit to) and/or turn diversity. Motivates exp04 (explicit heading_sin/cos output + heading
loss). Full write-up: `exp02_turn_only_diagnostic/README.md`.

---

## 2026-06-09 — exp04: direction-output variant Model C

Command (run from repo root):
```
python mp-core/trajectory-prediction/experiments/MP_X/exp04_direction_output_variant/run_exp04.py
```
Model C trunk, head widened 2→4 to also predict `future_heading_sin/cos` (= unit next-step
displacement). Loss = position_mse + 0.2·heading_mse (standardized space). Turn-only training,
genuine-turn val early stopping. At rollout the explicit predicted heading drives the
heading_sin/cos/turn_rate input channels. Shared additions: `target_cols` arg on
make_labeled_windows, `rollout_fn` arg on evaluate_windows, `shared/mpx_dir.py`,
`mpx_plots.final_heading_scatter`. Best epoch 22/37, train 6.1 s.

**Three-way comparison — held-out (val+test) genuine turns (n=50):**

| metric | exp01 balanced | exp02 turn-only | exp04 dir-output |
|---|---|---|---|
| ADE (m) | 1.418 | 1.300 | 1.291 |
| FDE (m) | 2.692 | 2.484 | 2.447 |
| angular err (°) | 89.8 | 88.0 | **89.0** |
| pred Δhead (°) | 38 | 49 | 46 |
| angularity | 0.37 | 0.60 | 0.45 |
| TCR | 0.60 | 0.78 | **0.82** |

**Outcome — DECISIVE NEGATIVE for the output-head hypothesis.** Adding an explicit heading
output + heading loss does NOT reduce angular error (89.0°, still ≈chance); TCR ticks up to
0.82 but direction stays wrong. `final_heading_scatter.png` shows predicted final direction is
uncorrelated with GT (median direction error 102°) and collapses to a near-constant direction
regardless of which way the pedestrian turns. So the bottleneck is **not** scarcity (exp01),
**not** straight dilution (exp02), and **not** the output representation (exp04) — it is
**directional ambiguity in the 10-frame input window**: a mostly-straight approach does not
determine the turn direction at a decision point. Next directions: longer/richer context,
decision-point-aware spatial features (which way is open), or multi-modal (mixture) outputs
instead of a single mean heading. Full write-up: `exp04_direction_output_variant/README.md`.

---

## 2026-06-09 — exp03: horizon sweep (turn-only Model C, H=5/10/20)

Command (run from repo root):
```
python mp-core/trajectory-prediction/experiments/MP_X/exp03_horizon_sweep/run_exp03.py
```
exp02 recipe (turn-only, genuine-turn val early stop, Model C, 2-output, no sampler) at
H=5/10/20. **Displacement-floor design choice:** a fixed 2 m genuine-turn floor empties the
short-horizon sets (probe: held-out genuine turns = 1 / 2 / 50 at H=5/10/20). So heading>30°
and max-step<0.6m stay fixed but the displacement floor scales D(H)=2·H/20 (0.5/1.0/2.0 m),
preserving a constant per-step speed floor and giving comparable populations
(held-out genuine 1073 / 442 / 50). Implemented by setting `C.DISP_MIN_M` per horizon.

**Held-out genuine-turn metrics by horizon:**

| H | disp floor | n_turns | ADE | FDE | angular err | GT Δhead | pred Δhead | angularity | TCR |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.5m | 1073 | 0.418 | 0.674 | **78.0°** | 120.9 | 43.1 | 0.45 | 0.58 |
| 10 | 1.0m | 442 | 0.724 | 1.278 | 83.0° | 122.3 | 52.5 | 0.55 | 0.67 |
| 20 | 2.0m | 50 | 1.300 | 2.484 | **88.0°** | 124.2 | 49.4 | 0.60 | 0.78 |

**Outcome:** angular error improves monotonically as horizon shrinks (88°→83°→78°) — so the
direction failure is PARTLY horizon/observability driven — BUT even at H=5 it is 78°, only
~12° below chance, far from usable. GT turn severity is ~constant (~121–124°) across horizons,
so the comparison is fair (not a turn-magnitude artifact). ADE drops at short H (0.42 vs 1.30)
but that is largely trivial (shorter rollout = less drift + shorter paths). Shortening horizon
is a secondary mitigation, not a cure. Full write-up: `exp03_horizon_sweep/comparison_summary.md`.

---

## 2026-06-10 — schema_audit: schema truth audit (13 sections)

Command (run from repo root):
```
python mp-core/trajectory-prediction/experiments/MP_X/schema_audit/run_schema_audit.py
```
Audits the exp01-04 input (`model_C_dataset.csv`) against the ground-truth world_x/world_y +
raw distances from its provenance superset `master_dataset.csv` (same rows). 13 sections,
each PASS/WARNING/FAIL. Audit-only; no source/model modified.

**Result: 11 PASS · 1 WARNING · 0 FAIL → VERDICT = DATA/SPLIT IMBALANCE LIKELY.**
- Schema is CLEAN: du/dv = world backward-diff to **4e-17 m**; target_du/dv = forward t→t+1 to
  **4e-17 m** (and == du[t+1]); heading = atan2(dv,du), cosine-sim **1.000**, normal convention
  on every recording; turn_rate = circular wrap(Δheading) corr **0.989**; u/v = per-recording
  (world−min)/rng to 1e-17; no leakage (no target/future inputs, recording-level split, 0
  cross-split duplicate tracks, max |feat↔target| corr **0.27**); turn labels reproduce
  exp03 (held-out genuine H5/H10/H20 = 1073/442/50).
- Rollout feedback PASS: teacher-forced feature updater matches the dataset to machine precision
  on moving steps (p95 **2.6e-14**); the only non-zero stat is turn_rate MEAN MAE 6.4e-3, caused
  by a benign idle-step convention (dataset stores heading=0 at zero motion; rollout carries prev
  heading) — does not affect moving/turning steps. (First run FAILED this as a false positive
  from an outlier-sensitive MAE threshold; fixed to robust p95.)
- **The one WARNING — Recording/split balance — is the real finding:** the TEST recording
  (red_bridge) has only **4 genuine turns, all right turns** (L-frac 0.00); val (stairs) 46;
  train 10,633 (balanced 0.52). Held-out genuine turns total **50, 92% from one recording**
  (stairs). So exp01-04's "held-out turn direction ≈ chance" is measured on a tiny, single-
  recording, one-directional sample → statistically under-powered, not a code/schema bug.
  Note: placa_espanya (roundabout, the bulk of genuine turns) is in TRAIN, not held-out.

**Consequence for exp01-04:** schema/rollout/leakage are trustworthy, so the pipeline is sound;
but the turn-DIRECTION conclusion is UNDER-MEASURED on held-out data. Next action (see
`schema_audit/recommended_next_action.md`): rebalance the eval split with turn-rich,
direction-balanced recordings and re-run exp01-04 turn-direction eval BEFORE treating
"direction ≈ chance" as final; only then pursue representational fixes (decision-point context,
multi-modal heading output). Full report: `schema_audit/schema_audit_report.md`.

---

## 2026-06-10 — exp06: turn-balanced evaluation split

Command (run from repo root):
```
python mp-core/trajectory-prediction/experiments/MP_X/exp06_turn_balanced_eval_split/run_exp06.py
```
Acts on the schema_audit WARNING (held-out turn set too small/one-sided). Re-cuts the
recording-level split IN MEMORY (source CSV untouched; reassign df["split"] + override
C.SPLIT_MAP) so the held-out set has many, direction-balanced genuine turns, then re-runs
the exp02 turn-only Model C. Schema/architecture unchanged.

**Held-out choice (data-driven, see split_analysis.csv).** Probe found placa_espanya holds
9,224/10,683 genuine turns BUT is artifact-prone (fastest motion 0.15 m/step, p95 step 0.60 m
on the artifact guard, median genuine turn 150°). So it was REJECTED as primary. Chosen
held-out = **esplanade** (1,232 clean, direction-balanced turns L-frac 0.53, normal walking
speed); train keeps placa_espanya so it stays turn-rich. placa_espanya run as a labelled
robustness check.

**Results — held-out genuine turns:**

| run | held-out | n turns | ADE | angular err | angularity | TCR |
|---|---|---|---|---|---|---|
| exp02 (original) | stairs+red_bridge | 50 | 1.300 | 88.0° | 0.60 | 0.78 |
| **exp06 clean (esplanade)** | esplanade+stairs | **1278** | 1.306 | **93.4°** | 0.40 | 0.66 |
| exp06 robustness (placa_espanya) | placa_espanya+stairs | 2106 | 1.581 | 84.3° | 0.56 | 0.63 |

**Verdict (answers the 3 questions):**
1. Was 88-90° a weak-sample artifact? **NO** — on a 26x-larger, direction-balanced, CLEAN
   held-out the angular error is 93.4° (slightly WORSE), and the artifact-prone robustness set
   agrees (84.3°). Both ≈chance.
2. Does direction improve on a balanced held-out? **NO** — stays at chance; predicted heading
   change clusters at 0-50° while GT clusters at 150-180°, uncorrelated.
3. Enough evidence to claim direction failure? **YES** — now WELL-MEASURED (n=1278 clean +
   n=2106 robustness), convergent. The direction failure is real, not a sampling artifact.

**Consequence:** this UPGRADES the schema_audit caveat. schema_audit said the held-out turn
metric was under-powered ("can't yet claim direction failure"); exp06 supplies the properly-
powered, clean, balanced held-out and CONFIRMS direction ≈ chance. The MP_X conclusion is now
solid: Model C expresses turns but cannot predict turn DIRECTION at decision points, confirmed
on a large clean balanced held-out set. Next is representational (decision-point context /
multi-modal heading output), not more split/data tweaks. Full write-up:
`exp06_turn_balanced_eval_split/README.md` + `comparison_vs_exp02.md`.

---

## 2026-06-10 — exp07: publication turn panels

Command (run from repo root):
```
python mp-core/trajectory-prediction/experiments/MP_X/exp07_publication_turn_panels/make_panels.py
```
Thesis/slide-quality visual panels of curved turn predictions, selected from exp06's clean
esplanade held-out. exp06 saved metrics but not trajectories, so selected windows are
re-rolled with the saved exp06 checkpoint (identical to exp06 eval). Reads only exp06 outputs
+ dataset + checkpoint; modifies nothing.

Selection = two-stage visual-quality score (NOT just ADE): rewards a long, visibly-curved
prediction (the model under-predicts turn displacement, median predicted path ~0.5 m, so long
clean curves are scarce), believable angularity [0.35-1.4], clean GT (max-step<0.5), then
re-roll + path-quality (smoothness, no jumps, visible separation). 25 of 1,232 esplanade
genuine turns selected (deduped, ≤2 windows/track). Hero = track 1906: predicted path 2.95 m
turning 166° (GT 177°, angularity 0.94) — captures the turn magnitude, ADE 2.17 m (curve is
right, final position/direction off): the honest "captures turning tendency, direction
uncertain" message.

Outputs: panels/individual/panel_01..25.png (technical), panels/presentation_mode/ (minimal
slide), collage_3x3_best / collage_5x5_selected / comparison_contact_sheet, hero_turn_prediction.png,
robustness_placa_espanya_3x3.png (noisy set, labelled), selected_25_metrics.csv,
selection_method.md, captions/. README states explicitly these are SELECTED qualitative
examples of visible curvature, NOT proof of directional accuracy (exp06 held-out angular error
is still ≈93°/chance). Purpose = communicate the positive visual finding (Motion Pixels can
express curved futures) without overclaiming direction.

---

## 2026-06-10 — exp07b: architectural communication panels (plan overlays)

Command (run from repo root):
```
python mp-core/trajectory-prediction/experiments/MP_X/exp07b_architectural_panels/make_architectural_panels.py
```
Architectural-communication variant of exp07 — **angularity is explicitly NOT optimised**.
Selects 20 SMOOTH, gentle, accurate predictions and renders them on the actual V3 manual
walkable-space masks (world→plan affine from each `encoding_v3_metadata.json`, ~8 px/m). Re-rolls
the exp06 clean-split checkpoint; reads only exp06 outputs + dataset + masks.

Selection (hard filters): GT turn 30-90°, predicted turn 20-100°, ADE < eligible-pool median
(1.09 m), no U-turns (>120°), clean held-out recordings only (esplanade plaza + stairs;
placa_espanya EXCLUDED as artifact-prone). Architectural score (no angularity) rewards smooth
curvature, obstacle avoidance, corridor following, plaza circulation. Added a walkable-on-path
filter (GT ≥0.6, pred ≥0.5 on walkable) to drop mask-edge artifacts that would look like walking
through a wall — selected set has walk_frac ≥ 0.95. Each panel tagged obstacle avoidance /
corridor following / plaza circulation.

Result: 20 panels (plaza 8 / obstacle 6 / corridor 6), ADE 0.53-0.99 m, GT turns 32-89°. Outputs:
panels/plan_overlays/overlay_01..20.png (trajectory on walkable plan, 5 m scale bar, arrows),
panels/presentation/ (minimal slide), collage_4x5_plan_overlays.png, contact_sheet.png,
hero_architectural_turn.png (corridor-following, GT 58°, ADE 0.67 m), selected_20_metrics.csv,
selection_method.md, captions/. README states scope honestly: communication-oriented qualitative
selection (smooth/accurate/legible architectural circulation within walkable space), NOT a
turn-direction-accuracy claim (exp06 held-out angular error still ≈chance).

---

## 2026-06-10 — exp07c: long-arrow presentation panels

Command (run from repo root):
```
python mp-core/trajectory-prediction/experiments/MP_X/exp07c_long_arrow_presentation/make_long_arrow_panels.py
```
Presentation re-render of the qualitative panels in a clean infographic language: bold dominant
MAGENTA prediction (#D81B7D), black observed, soft-grey GT (#A8A8A8), NO stars (small black pebble
at current position), white walkable / light-grey built / thin pale outline, subtle scale bar,
minimal category title + magenta "turn ≈ XX°", no turn-definition block on image. 10 examples on
the V3 walkable plan, re-rolled from the exp06 clean-split checkpoint.

Selection (NOT angularity): clean held-out recs only (esplanade+stairs; placa_espanya excluded),
GT turn 30-90°, pred turn 20-100°, no U-turns, pred path >0.55 m, walkable, smooth — RANKED BY
LONGEST predicted path. Honesty: prediction geometry never fabricated/rescaled; long-arrow
readability comes from selection + tighter crops + stroke styling + display-only Catmull-Rom
smoothing (raw kept in panels/raw_geometry/ and metrics). Three crops per panel: individual
(~13 m balanced), zoomed_arrow (tight, arrow dominant), context_30m (~30-50 m). Only 2/10 are
readable at 30 m (mild predictions are short), so 8 rely on zoomed_arrow — tradeoff documented.

Selected (longest pred first): stairs:676@23 (pred 1.87 m), stairs:768@136 (1.44), stairs:313@7
(0.92), esplanade:99@273 (0.91), stairs:628@197 (0.82), esplanade:3703@38 (0.80),
esplanade:1597@2207 (0.77), esplanade:59@451 (0.73), esplanade:1@3065 (0.72), +1. Outputs:
panels/{individual,zoomed_arrow,context_30m,raw_geometry}/panel_01..10.png, collage_5x2.png,
collage_10.png, hero_prediction_arrow.png, selected_10_metrics.csv, selection_method.md, captions/.
README keeps scope honest: qualitative communication of curved-future EXPRESSION; geometry not
fabricated; turn-DIRECTION accuracy still limited (exp06 ≈chance).

---

## 2026-06-10 — exp08: rollout magnitude audit

Command (run from repo root):
```
python mp-core/trajectory-prediction/experiments/MP_X/exp08_rollout_magnitude_audit/run_audit.py
```
Asks whether Model C has ALWAYS under-predicted displacement magnitude, or whether turn balancing /
turn-only / split change / rollout introduced it. Computes GT vs predicted rollout PATH LENGTH
(20-step horizon) for 6 models; 5 V3 models on ONE fixed common window set (all 5 V3 recordings,
cap_all=2500/cap_gen=1200/seed=42, identical windows → only model varies), Overfit10X on the MACBA
bridge it memorised (ratio comparable, absolute scale not). exp04 via its 4-output dir rollout.

**Pred/GT median length ratio (all windows):** Phase-4 baseline **0.31** (WORST), exp01 0.59,
exp02 0.68, exp04 0.65, exp06 0.80, Overfit10X **0.81**. Genuine turns far worse (0.12–0.27; GT
turn path median 3.68 m vs predicted ~0.5–1.0 m).

**ANSWER: under-prediction is INTRINSIC — present in EVERY variant including the Phase-4 baseline
AND Overfit10X.** It is NOT introduced by turn balancing / turn-only / split / direction-head — the
baseline is the most severe (0.31) and every turn-focused/split variant MITIGATES it (0.59–0.80,
none reach 1.0). Present even under memorisation (Overfit10X 0.81) → not a generalisation gap.
Cause = MSE + autoregressive rollout (regression-to-mean shrinks each step, compounding over the
horizon) — the magnitude analogue of the straight-line/direction collapse. Fix needs an
objective/rollout change (scheduled sampling / free-running, magnitude-aware or distributional loss,
multi-step supervision), not more balancing/split tweaks. Outputs:
predicted_length_vs_GT_length.png, length_distribution_comparison.png, exp08_magnitude_summary.csv,
README.md.

### MP_X synthesis (exp01 + exp02 + exp03 + exp04 + schema_audit + exp06)
Turn scarcity was a real cause of the straight-line collapse and is now fixable — Model C
produces visibly curved rollouts and commits to a turn 58–82% of the time. But turn DIRECTION
is unsolved and is largely invariant to balancing (exp01), turn-only training (exp02), and an
explicit heading head (exp04, all ≈88–90°). Horizon (exp03) is the only lever that moves it,
and only modestly (down to 78° at H=5 — still ~chance). The schema_audit proved the pipeline is
CLEAN (no schema/target/coordinate/leakage/rollout bug) but flagged that the held-out turn set
was tiny/one-sided. exp06 then rebuilt the split to a large (n=1278), clean, direction-balanced
held-out (esplanade) — and the angular error STAYED at chance (93.4°), with an artifact-prone
robustness set (placa_espanya, n=2106) agreeing (84.3°). So the direction failure is now
WELL-MEASURED and real, not a sampling artifact. The thesis claim is therefore NOT "Model C
cannot turn" — it demonstrably expresses turns — but "predicting turn DIRECTION at decision
points is a genuine input/representation limitation at current data + context scale (confirmed
on a properly-powered held-out set); horizon length is a secondary modulator." Next directions:
richer decision-point context (which way is open / goal bearing), or multi-modal / mixture
heading outputs instead of a single mean heading — NOT more split/data-balance tweaks.
