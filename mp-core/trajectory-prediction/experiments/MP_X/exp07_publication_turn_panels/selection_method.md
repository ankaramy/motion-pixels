# Selection method — exp07

Source: exp06 clean held-out (`esplanade`, 1,232 genuine turns). Goal: pick the best (target 25; **25 passed the quality bar**) most VISUALLY COMPELLING curved predictions for thesis/slide use — believable curvature, not necessarily correct final direction. The model under-predicts displacement on turns (median predicted path ~0.5 m), so genuinely long, clearly-curved predictions are scarce — the quality filters admit 25 of 1,232.

## Two-stage score

**Stage 1 — metric pre-rank** (from exp06 `metrics.csv`), rewards:
- predicted heading change > 20° (the prediction actually turns) — weight 1.6
- angularity ratio in [0.3, 1.2], peak ~0.75 (not collapsed, not wild) — weight 1.4
- low ADE / FDE — weights 1.2 / 0.8
- predicted/GT path-length ratio in [0.4, 1.4] (not collapsed, not overshooting) — weight 1.0
- clean GT: max single step < 0.55 m (avoids tracking-artifact turns) — weight 1.2
- readable net displacement (~2–6 m) — weight 0.5

Hard pre-filters: gt_max_step < 0.55 m, predicted turn > 18°, angularity ∈ [0.30, 1.25], ADE < 1.8 m.

**Stage 2 — path quality** (after re-rolling the top 70 with the exp06 checkpoint):
- smoothness: few predicted heading reversals (no zig-zag)
- no extreme jumps: max predicted step < 1.0 m and < 6× median step (else heavily penalised)
- visible separation: predicted path length > 0.8 m (a path, not a dot)
- GT cleanliness: GT itself not zig-zaggy

Final rank = metric score + 1.3 × path-quality; deduplicated to one window per track for variety; top 25.

## Honesty notes

- These are **qualitative** selections optimised for clear curvature, NOT a random/representative sample.
- They are **not** evidence of directional accuracy; exp06 angular error on this held-out is ≈93° (chance).
- 'turn captured ✓' in a title means predicted heading change > 20° AND GT turn > 30° — it captures that the model turned, not that it turned the correct way.