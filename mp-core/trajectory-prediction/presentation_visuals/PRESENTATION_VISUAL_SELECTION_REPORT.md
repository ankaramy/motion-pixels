# Presentation Visual Selection — MODEL_X / MODEL_XR_B_MAG / MODEL_XC_B_CURV_LIGHT

**Date:** 2026-06-12 · No training, no model/data changes — frozen checkpoints replayed
deterministically. Goal: the most **architecturally readable** prediction visuals per horizon, chosen for
legibility (not lowest error). 6 per horizon × 5 horizons = **30 cases**, with individual figures + a
2×3 collage each.

## How paths were selected (readability, not metric-best)
All three models are rolled out per horizon on the test split; each window gets a **visual-interest
score** from existing geometry/metrics — explicitly NOT lowest ADE/FDE. Hard readability filters first:
- GT net displacement ≥ {H20:1, H60:2, H100:3, H200:5, H400:8} m (long enough to read)
- XC predicted length ≥ half that; bounding-box span ≥ 0.8× the floor (non-tiny)
- XC net-displacement ratio ∈ [0.7, 1.4] (realistic length, no overshoot)
- XC cosine ≥ 0.6 (prediction lies near/along GT — readable, not divergent)
- GT macro curvature 18–200° and tortuosity < 3.5 (some shape, **no tracking-jitter artifacts**)
- shape_ratio ≤ 1.8 (**rejects over-curved "snake" paths**)

Then a weighted score (alignment-first): cosine 0.28, GT-net 0.14, moderate-curvature 0.16, shape-near-1.0
0.12, length-ratio 0.12, bbox 0.10, XC-net 0.08. Selection enforces **≤2 per recording and ≤1 window per
track** for diversity.

## Why these are presentation visuals, not metric-best examples
Lowest-ADE windows are dominated by near-stationary pedestrians (tiny 1–2 m snippets) — visually empty.
This pass instead surfaces **long, legible, well-aligned** paths where the viewer immediately sees the
future motion and where XC's magenta prediction sits near/along GT. Cases that score well on raw metrics
but look bad (snakes, jitter, divergence) are filtered out by design.

## Plotting style (verified on the collages)
Landscape panels, `ax.set_aspect("equal", adjustable="box")` (true metric geometry), visible grid,
**generous padding** (≥25% of bbox and ≥1.5 m each side; short axis expanded to ≥50% of long so paths are
never razor-thin slivers and never clipped to the curve). Line roles: history = black, GT = blue + markers,
MODEL_X = orange dashed, XR_B_MAG = grey dashed, **XC_CURV = bold magenta + arrow**.

## Counts per horizon
6 selected each (30 total). Recording mix per horizon:
- **H20:** espanya×2, esplanade×2, stairs×1, red_bridge×1
- **H60:** espanya×2, esplanade×2, catalunya×1, red_bridge×1
- **H100:** esplanade×2, espanya×2, catalunya×2
- **H200:** esplanade×2, catalunya×2, espanya×1, red_bridge×1
- **H400:** esplanade×2, espanya×2, red_bridge×1, stairs×1

All five recordings appear across the set; catalunya/espanya/esplanade dominate (they have the longest,
most legible plaza walks).

## Missing / difficult
- **All three models loaded** (MODEL_X, XR_B_MAG, XC_CURV) — no missing model outputs.
- **H20** is inherently short (~1 m): readable but least spatially rich; fewer strongly-curved candidates.
- **H200 / H400** are the hardest: longer GT paths wander and XC's direction is weaker at long range, so
  good *aligned* curved examples are rarer — the alignment + shape≤1.8 filters keep them clean but some
  panels show mild direction divergence (a genuine model property at long horizon, not a plotting issue).

## Top 3 strongest visuals overall (by visual score)
1. **H100 · esplanade_espanya · trk 750** — ndR 0.93, shapeR 0.78 (score 0.92). Clean ~5 m path, XC curves
   along GT, XR straighter — shows the curvature fix clearly.
2. **H60 · placa_espanya · trk 4394** — ndR 1.06, shapeR 1.07 (score 0.89). ~3 m, near-perfect length and
   shape match on the hardest recording.
3. **H20 · placa_espanya · trk 6268** — ndR 1.08 (score 0.87). Short but crisp, well-aligned.

## Recommended visuals for the thesis presentation
- **Lead with H100 (~5 m)** — the best balance of readable length, visible curvature, and alignment; its
  collage is the strongest. The top-scoring cases cluster at H100 and H60.
- **H60 (~3 m)** as the clean secondary (high length + shape fidelity).
- Use **H200/H400** only to illustrate *reach and its limits* (longer paths, occasional divergence), not as
  hero shots.
- Per-recording: **placa_espanya** and **esplanade** give the most compelling curved plaza walks;
  **red_bridge** the cleanest corridor.

## Warnings (artifacts / over-curvature)
- An earlier scoring pass surfaced **over-curved "snake" predictions** (shape_ratio 4–7) at H200/H400 —
  these were removed by the shape_ratio ≤ 1.8 filter and the alignment-first re-score. The current set is
  snake-free.
- GT path-length/curvature is jitter-prone (Phases 1–5A); the tortuosity < 3.5 and macro-curvature caps
  exclude the worst jitter windows, but GT lines still show natural per-frame wiggle.
- At H200/H400, treat divergence between XC and GT as a real long-horizon model limitation, not a framing
  problem.

## Final check (performed)
Inspected H100, H200, H400 collages after regeneration: curves not clipped, clear spatial breathing room,
equal axes, readable grid, predictions have length and sit near GT, panels read like the reference style
(not cramped metric-audit plots). The first pass was too tight/snaky and was **fixed** (landscape padding +
alignment-first selection + snake filter) before finalizing.

## Artifacts
`presentation_visuals/H{20,60,100,200,400}/individual/H*_best_0{1..6}.png`, `H*/H*_collage.png`,
`selected_visuals_inventory.csv` (30 rows), `make_presentation_visuals.py`.
