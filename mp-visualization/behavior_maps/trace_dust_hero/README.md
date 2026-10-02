# Trace + Dust Hero (black, glowing)

A single cinematic **hero image** for Motion Pixels: luminous pedestrian movement
— glowing trace strings + motion dust, density-coloured cyan→magenta — imprinted
on a **black** architectural plan rendered as fine white linework. Movement is the
hero; the plan is secondary. Modelled on the black-background glowing reference.

**Visualization-only.** A rendering of existing tracked pedestrian trajectories.
No retracking, recalibration, model inference, or source-data modification.

## Current test
Plaça Catalunya (one hero per file — no collage, no frames, no animation).

```
cd mp-visualization/behavior_maps/trace_dust_hero
python generate_trace_dust_hero.py
```

## Method
1. **Plan → black linework:** the warped plan is greyed, bilateral-filtered and
   **Canny edge-detected** into thin white/grey lines over black (cool-tinted,
   low opacity — secondary). No filled map, no pale overlay.
2. **Trace strings:** every track (arc-length resampled + lightly smoothed) drawn
   hair-thin, very low opacity, **density-coloured per segment**.
3. **Motion dust:** every 4th tracked position as a tiny glowing particle,
   density-coloured, beneath the strings (atmosphere).
4. **Glow:** the movement is rendered to a float buffer on black, then
   **multi-radius Gaussian bloom** is added back with a soft highlight rolloff —
   cinematic luminosity, not neon. Three presets (base / more / less glow).
5. Reuses `warp_plan_to_world` from `../generate_bottleneck_map_test.py`; clipped
   exactly to the plan extent; no axes/margins/text. **No recalibration.**

## Palette (movement only)
`#00D5FF → #2563EB → #7C3AED → #D946EF` (cyan → blue → violet → magenta).
Sparse routes read cyan/blue; dominant corridors and decision knots saturate to
violet/magenta. The plan is never coloured.

## Outputs (`outputs/`) — 4200 × 1943 px, PNG, black bg
- `placa_catalunya_trace_dust_hero_black.png` — **balanced (primary)**
- `placa_catalunya_trace_dust_hero_black_more_glow.png` — softer, atmospheric
- `placa_catalunya_trace_dust_hero_black_less_glow.png` — crisper, sharper
- `placa_catalunya_trace_dust_hero_report.md` — parameters + provenance

## Key knobs (top of script)
`WIDTH_PX`, `TRACE_W` / `TRACE_ALPHA`, `DUST_*`, `CORE_GAIN` (movement brightness),
`PLAN_OPACITY` / `CANNY_*` (linework), `GLOW` presets (radii/weights/strength),
`DENS_GAMMA` (colour spread).

## Note
Movement stays bound to the tracked footprint (honest — that's where the data is).
Colour saturates to magenta in the dense corridor because density is genuinely
high there; cyan/blue appear on the sparse fringes.

## Background V2 (linework upgrade)
`generate_trace_dust_hero_bg_v2.py` **freezes the movement layer** (imported from
this script) and only swaps the background. V1's Canny-on-satellite read as
"contour noodles"; V2 builds clean CAD-style linework via Hough **long straight
lines** (buildings/roads/plaza edges), dropping curved tree/shadow noise.

OSM note: `osmnx` is not installed and the calibration has **no georeference**
(local metric world frame, no lat/lon), so true OSM can't be aligned — the brief's
**Option C** vector-style extraction is used instead.

```
python generate_trace_dust_hero_bg_v2.py
```
Outputs: `..._hero_black_osm.png` (clean structural), `..._hero_black_hybrid.png`
(structural + faint detail), `..._background_comparison.png` (V1 | OSM-style |
hybrid). **Recommended final: hybrid** (legible city, no noodles); OSM-style for a
more minimal look.

## FINAL — V3 (`generate_trace_dust_hero_v3.py`)
The final hero. Layered **architectural drawing** background (major Hough geometry
+ faint speckle-cleaned secondary detail — trees/objects/paving), stronger strings
(width 0.8, opacity 0.045, 1302 traces), denser dust (every 3rd position, 117k),
stronger layered bloom, and **inverted density gamma (1.4)** so cyan/blue spread is
preserved and only the densest corridor saturates to magenta (fixes V2's collapse).

```
python generate_trace_dust_hero_v3.py
```
Outputs (4400×2036, black): `placa_catalunya_trace_dust_hero_final_v3.png`
(**recommended**), `..._final_v3_more_plan.png` (city more legible),
`..._final_v3_more_glow.png` (cinematic), `..._final_v3_report.md`.

## Status
Static hero, single site. Built on the trace-field + dust families.
V1 → V2 (background) → **V3 (final)**.
