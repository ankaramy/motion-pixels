# exp07c — Long-arrow presentation panels

Presentation-ready re-render of the qualitative prediction panels in a clean infographic visual language, with elegant dominant magenta prediction arrows on the architectural walkable plan. Built from exp06 predictions (re-rolled from the clean-split checkpoint) + V3 manual masks. Source data, frozen_model_C and prior experiments are untouched.

## Statement of scope

1. These are **qualitative presentation panels** (selected, architecturally-readable cases).
2. **Prediction geometry is not fabricated or independently scaled** — it is the model's actual rollout; only crop, stroke styling and display-only smoothing change how it reads.
3. **Long-arrow readability** is achieved through selection (longest predictions), tighter crops and styling — never by altering the data.
4. **Quantitative turn-direction accuracy remains limited** (exp06: held-out angular error ≈ chance).
5. They communicate the positive finding: **the model can express curved future movement**.

## Visual language

observed = black (#111111) · ground truth = soft grey (#A8A8A8) · prediction = magenta (#D81B7D, thicker, dominant) · current position = small black pebble (no stars) · walkable = white · built = light grey · thin pale-grey plan outline · subtle scale bar · minimal category title + magenta `turn ≈ XX°`.

## 'turn' definition

`turn` = the absolute heading change between the last observed direction and the final future direction over the prediction horizon (measured on the ground-truth future). This definition is intentionally kept OFF the panels.

## Outputs

- `panels/individual/panel_01..10.png` — recommended balanced crop (~13 m context).
- `panels/zoomed_arrow/panel_01..10.png` — tight crop, prediction arrow dominant.
- `panels/context_30m/panel_01..10.png` — ~30–50 m architectural context.
- `panels/raw_geometry/panel_01..10.png` — unsmoothed geometry (display smoothing disabled).
- `panels/collage_5x2.png`, `panels/collage_10.png` — 10-panel collages (title + legend).
- `hero_prediction_arrow.png` — single strongest example.
- `selected_10_metrics.csv`, `selection_method.md`, `captions/`.

Selected 10. Categories: {'plaza circulation': 6, 'obstacle avoidance': 3, 'corridor following': 1}. Context-30m readable: 2/10 (the rest rely on `zoomed_arrow/`; see `selection_method.md`).
