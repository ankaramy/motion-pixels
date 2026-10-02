# exp07 — Publication turn panels

Thesis / presentation-quality visual panels of curved turn predictions, selected from **exp06** (clean, direction-balanced `esplanade` held-out). Re-rolls the saved exp06 checkpoint to draw the trajectories. Source data, frozen_model_C, and previous experiments are untouched.

## What these are (and are not)

1. **Selected qualitative examples from exp06** held-out predictions (not training data).
2. They **demonstrate visible angular rollouts** — the model produces curved futures, not straight lines.
3. They are **not cherry-picked as proof of directional accuracy** — selection rewards clear, smooth, believable curvature and clean ground truth, explicitly *not* correct final direction.
4. **Quantitatively, exp06 still shows direction ≈ chance** (held-out mean angular error ~93°). These panels do not change that result.
5. Their purpose is to **communicate the positive visual finding**: Motion Pixels can express curved pedestrian futures (turn *tendency* and *magnitude*), even while the exact direction remains uncertain.

Selected **25** of 1,232 esplanade genuine turns (target 25; 25 passed the quality bar — the model produces few long, clearly-curved predictions, so we do not pad the set with collapsed stubs).

## Outputs

- `panels/individual/panel_01..25.png` — technical panels (ids, ADE/FDE, GT/pred angle, turn-captured flag).
- `panels/presentation_mode/panel_01..25.png` — minimal slide version (Observed / Ground truth / Prediction + angle).
- `panels/collage_3x3_best.png`, `panels/collage_5x5_selected.png`, `panels/comparison_contact_sheet.png`.
- `panels/robustness_placa_espanya_3x3.png` — noisy robustness set (clearly labelled).
- `hero_turn_prediction.png` — single clearest example for a slide.
- `selected_25_metrics.csv`, `selection_method.md`, `captions/`.

Hero example: esplanade track 1906, GT turn 177°, predicted 166°, ADE 2.17 m.

## Colour key

Observed path = dark grey · Ground truth future = blue · Model prediction = orange/red. ★ = prediction start; ● / ■ = GT / predicted end; arrows = final directions.
