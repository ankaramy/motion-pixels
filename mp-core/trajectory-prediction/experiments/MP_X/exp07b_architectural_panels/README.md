# exp07b — Architectural communication panels (plan overlays)

Plan-overlay panels communicating architectural pedestrian motion (obstacle avoidance, corridor following, plaza circulation). **Not** a turn-capture / angularity study — angularity is explicitly not optimised. Trajectories are re-rolled with the exp06 clean-split checkpoint and drawn on the V3 manual walkable masks. Source data, frozen_model_C and prior experiments are untouched.

## Selection (architectural, no angularity)

GT turn 30°–90°, predicted turn 20°–100°, ADE below median, no U-turns (>120°), clean held-out recordings only (esplanade + stairs; placa_espanya excluded as artifact-prone). Prioritises smooth curvature, obstacle avoidance, corridor following, plaza circulation. See `selection_method.md`.

Selected 20. Category mix: {'plaza circulation': 8, 'obstacle avoidance': 6, 'corridor following': 6}.

## Outputs

- `panels/plan_overlays/overlay_01..20.png` — trajectory on the walkable plan (scale bar, arrows, metrics).
- `panels/presentation/panel_01..20.png` — minimal slide version (situation + turn angle).
- `panels/collage_4x5_plan_overlays.png`, `panels/contact_sheet.png`.
- `hero_architectural_turn.png` — large plan overlay of the clearest example.
- `selected_20_metrics.csv`, `selection_method.md`, `captions/`.

## Honest scope

These are communication-oriented qualitative selections (smooth, accurate, legible architectural motion). They are NOT a claim about turn-direction accuracy — exp06 shows held-out angular error is ≈chance. The purpose is to show predictions read as plausible architectural circulation within walkable space.

## Colour key

Observed = dark grey · Ground truth = blue · Prediction = orange · walkable = white · built = grey · ★ = prediction start · arrows = final directions · bar = 5 m.
