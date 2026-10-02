# bottleneck_animations

Animation tests for the Plaça Catalunya bottleneck map, built on the frozen
**V2D** visual style (see `../generate_bottleneck_map_test.py`). Visualization
only — no retracking, recalibration, or source-data modification.

## Run
```
python generate_bottleneck_animation_tests.py
```
Reuses the V2D primitives (plan→world warp, desaturated underlay, yellow→red
palette, variable square size + no-overlap clamp, ranked markers) from the parent
module so every animation's final frame matches V2D.

## Three options
- **Option 1 — Construction** (`*_option1_construction.gif`): cells appear
  low→high score, then markers fade in. Spatial pressure revealed.
- **Option 2 — Accumulation** (`*_option2_accumulation.gif`): cells start faint
  and intensify/grow as **real trajectory observations** accumulate over `time_s`
  (the CSV is read for *timing only* — scores are not recomputed). Falls back to a
  simulated reveal if the trajectory CSV is unavailable (documented in the report).
- **Option 5 — Hybrid** (`*_option5_hybrid.gif`): quick faint raster → high cells
  grow/intensify → one very subtle pulse on the top cells → markers fade in.

## Outputs (`outputs/`)
- Three GIFs (24 fps, ~1200 px wide, shared 256-colour palette).
- Three `*_final.png` stills (320 dpi, exact V2D composition).
- `bottleneck_animation_tests_report.md` — sources, V2D params, per-GIF duration
  and size, whether Option 2 used real temporal data, and a recommendation.

## Notes
- GIF frames are rendered at a presentation-friendly resolution for file size; the
  `_final.png` stills are full-resolution V2D.
- Only Plaça Catalunya is generated here (per task scope).
