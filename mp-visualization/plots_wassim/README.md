# plots_wassim — clean MODEL_XC trajectory re-plots

`generate_plots_wassim.py` re-plots selected pedestrian-prediction cases in a clean
scientific / architectural style (white background, light grid, simple axes, breathing space)
for **MODEL_XC_B_CURV_LIGHT only**.

**Visualization-only.** No training, retraining, loss tuning, or dataset / encoder changes.
The frozen `MODEL_XC_B_CURV_LIGHT` checkpoint is *replayed* deterministically using the same
window construction as `presentation_visuals/make_presentation_visuals.py` (OBS = 10,
stride = 2, test split).

## What the script does

For each horizon **H20 / H60 / H100 / H200 / H400**:
1. Builds all test-split evaluation windows (10 observed steps + H future steps) and rolls the
   frozen MODEL_XC_B_CURV_LIGHT forward over every window (GPU-batched).
2. Computes per-window **ADE**, **RMSE**, **R²**.
3. **Best 3** = the user-provided track IDs, matched to their exact window via the presentation
   inventory `window_id` (so they are the same windows shown in `presentation_visuals`).
4. **Worst 3** = auto-selected genuine model failures (see below).
5. Saves 6 individual plots + one 2×3 best/worst collage (top row best, bottom row worst).
6. Writes a cross-horizon accuracy summary (CSV + Markdown).

Each horizon folder is cleared of old `*.png` at the start of a run so stale selections never
linger.

## Input files found / used

- Trajectories (world metres): `…/new_datasets/Barcelona_v3_manual_master_dataset/master_dataset.csv`
  (resolved via `model_x_lib.CONFIG`)
- Test split: `…/experiments/MODEL_X/splits/model_x_track_split.csv`
- Model checkpoint: `…/MODEL_XC/checkpoints/MODEL_XC_B_CURV_LIGHT/model_best.pth`
- Scalers: `…/MODEL_XC/checkpoints/MODEL_XC_B_CURV_LIGHT/scalers.json`
- Best-track → window map: `…/presentation_visuals/selected_visuals_inventory.csv`
- Reused read-only library code: `MODEL_XC/xc_common.py` → `MODEL_XR/xr_common.py`
  → `experiments/MODEL_X/model_x_lib.py` (`build_eval_windows`, `rollout_world_batch`,
  `recording_world_bounds`, `TrajectoryLSTM`, `ColumnScaler`, `enrich_full`).

User-provided best tracks:

| H | best tracks |
|---|---|
| H20  | 1261, 6268, 4394 |
| H60  | 4394, 1846, 6268 |
| H100 | 4394, 4703, 1846 |
| H200 | 4152, 3084, 1178 |
| H400 | 112, 2039, 540 |

## Output folder

`$MP_PLOTS_OUT` (default `mp-visualization/plots_wassim/outputs/`, git-ignored)

```
H20/  H60/  H100/  H200/  H400/        # each: best_0{1..3}_trackXXXX.png,
                                       #       worst_0{1..3}_trackXXXX.png,
                                       #       H{H}_best_worst_collage.png
summary_accuracy_scores.csv
summary_accuracy_scores.md
```

## Style settings

- White background, equal aspect ratio, generous padding (≥30 % of span, ≥1 m), short axis kept
  ≥55 % of the long axis so paths are never razor-thin.
- Very light grid (`#cccccc`, alpha 0.5, lw 0.5); light grey spines; small tick labels in metres.
- **history** — dark grey solid (`#444444`, lw 1.5)
- **GT future** — blue solid with small point markers (`#1f77b4`, lw 1.4; markers thinned on long horizons)
- **prediction (XC)** — orange dashed (`#ff7f0e`, lw 1.6)
- **separation / start point** — black star (`*`)
- Compact title: `site · track · H{H}` then `ADE … RMSE … R²`.
- No plan overlay, no feet icons, no glow, no neon, no dark background, no bold linework.

## Worst-selection method

Among all test windows for the horizon, keep only readable, genuine cases:
- `gt_net ≥ MIN_GT_NET[H]` (`{20:1, 60:2, 100:3, 200:5, 400:8}` m) — excludes near-stationary;
- `gt_tortuosity < 3.5` — excludes jitter artifacts;
- `max GT per-step ≤ 2.0 m` and `mean GT per-step ≤ 0.75 m` — excludes ID-switch *teleports* and
  sustained tracking *drift* (not real model failures; the GT median step is ~0.03 m, 99th-pct
  ~0.85 m, so these caps trim only the non-physical tail);
- finite ADE / RMSE / R²; and the window's track is **not** one of the user's best tracks.

Candidates are then ranked by a **badness** score — min-max-normalised within the candidate set —
`badness = norm(ADE) + norm(RMSE) + (1 − norm(R²))` — and the 3 worst **distinct tracks** are taken.
This surfaces real failures (wrong-direction or large magnitude overshoot) rather than corrupted
tracks.

## Accuracy score formula

Metrics are computed over the H predicted future steps (the shared separation point at step 0 is
excluded):

- `ADE  = mean_t ‖pred_t − gt_t‖`
- `RMSE = sqrt(mean_t ‖pred_t − gt_t‖²)`
- `R²   = 1 − Σ‖pred − gt‖² / Σ‖gt − mean(gt)‖²` (over the future points; can be large-negative
  for catastrophic divergence)

The summary reports, per horizon, the mean ADE / RMSE / R² over the selected 6 tracks (3 best +
3 worst). The cross-horizon `accuracy_score` min-max-normalises those three means **across the
five horizons** (ADE and RMSE inverted so lower = better; R² as-is so higher = better) and
averages them; **1 = best, 0 = worst**. Because absolute metres grow with horizon, this score
naturally orders H20 (best) → H400 (worst).

## Run

```
cd mp-visualization/plots_wassim
python generate_plots_wassim.py
```
