# Selection method — exp07b (architectural)

This panel set is for **architectural communication, not turn-capture metrics**. Angularity is deliberately NOT part of the score.

## Eligibility (hard filters)

- GT turn 30°–90° (gentle, readable turns; no sharp/U-turns)
- predicted turn 20°–100°
- ADE below the eligible-pool median (accurate, believable)
- GT turn < 120° (no U-turns)
- clean held-out recordings only: esplanade (plaza) + stairs (corridor). placa_espanya is EXCLUDED (identified artifact-prone: fastest motion, p95 step on the 0.6 m guard, median turn ~150°); train-side recordings are excluded so all shown predictions are honest held-out.

## Architectural priority score (no angularity)

- smooth curvature (few predicted heading reversals, no jumps) — weight 1.3
- obstacle avoidance (path passes near an obstacle: low min obstacle-clearance) — weight 1.1
- accuracy (low ADE) — weight 0.8
- readable predicted path length — weight 0.6
- corridor following (sustained low boundary-clearance) — weight 0.5

Each window is tagged **obstacle avoidance** / **corridor following** / **plaza circulation** from its obstacle/boundary clearance, and selection is deduplicated by track and spread across categories.

Selected **20** of the eligible pool. Category mix: {'plaza circulation': 8, 'obstacle avoidance': 6, 'corridor following': 6}.

## Plan overlay

World→plan mapping uses each recording's `encoding_v3_metadata.json` affine (`transform_world_to_plan_3x2`, ~8 px/m). The background is the V3 manual walkable mask (white = walkable, grey = built/obstacle, thin line = walkable edge); a 5 m scale bar is drawn.