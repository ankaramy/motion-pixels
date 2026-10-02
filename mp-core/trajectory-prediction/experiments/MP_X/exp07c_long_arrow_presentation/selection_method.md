# Selection method — exp07c (long-arrow presentation)

Presentation re-render of the qualitative panels. Angularity is NOT optimised; selection ranks the LONGEST available predicted rollouts among architecturally-readable mild turns.

## Filters

- clean held-out recordings only (esplanade plaza + stairs corridor); placa_espanya excluded (artifact-prone)
- GT turn 30°–90°, predicted turn 20°–100°, no U-turns (>120°)
- predicted path length > 0.7 m (avoid tiny predictions), ADE < 3 m
- prediction & GT mostly within walkable space; smooth (≤3 heading reversals, no jumps)

## Ranking

Sorted by predicted path length (DESCENDING) — the longest, most readable prediction arrows first; deduplicated to one window per track.

Selected **10**. Categories: {'plaza circulation': 6, 'obstacle avoidance': 3, 'corridor following': 1}.

## Long-arrow readability (no fabrication)

Predicted geometry is the model's actual rollout — never lengthened or scaled independently of GT. Readability is achieved only by (a) selecting the longest predictions, (b) tighter crops, (c) thicker magenta stroke + an elegant dominant arrowhead, and (d) a light Catmull-Rom smoothing applied for DISPLAY ONLY (raw, unsmoothed panels are saved in `panels/raw_geometry/`; metrics use raw values).

## Crop tradeoff (context vs arrow legibility)

`context_30m/` targets ~30–50 m of spatial context, but mild-turn predictions are short, so at 30 m a prediction < ~1.2 m spans <4% of the frame and reads weakly: **8/10** panels fall in this case. `zoomed_arrow/` crops tightly so the magenta prediction is dominant; `individual/` is the recommended balance (~13 m, both the arrow and the obstacle/corridor/plaza context are legible).

## 'turn' definition (kept off the image)

The magenta `turn ≈ XX°` annotation is the absolute heading change between the last observed direction and the final future direction over the prediction horizon, measured on the ground-truth future.