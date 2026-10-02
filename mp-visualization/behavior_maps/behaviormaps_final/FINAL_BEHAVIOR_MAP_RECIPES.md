# Motion Pixels — Final Behavior Map Recipes

Source of truth for the production batch pipeline
(`generate_final_behavior_maps.py`). These recipes were extracted from the
accepted scripts/reports under `mp-visualization/behavior_maps/` and the
**final unified series** (`final_unified_series/`), which is the visual standard.

> All three families are **visualization-only**: built from existing tracked
> trajectories / precomputed bottleneck cells. No retracking, recalibration,
> model inference, or source-data modification. Bottleneck scores are read from
> `bottleneck_cells.csv` and never recomputed.

---

## Shared standard (the unified canvas + HUD)

Defined by `final_unified_series/generate_unified_series.py`, derived from
Flow Fields V5.

| Parameter | Value |
|---|---|
| Map area | **1920 × 1080** (16:9 `window_extent`: full X span, Y padded, no stretch) |
| Footer (HUD) | **1920 × 70**, added **below** the map (vstacked, not overlaid) |
| Total canvas | **1920 × 1150** |
| GIF | **760 × 454**, 20 fps source → **10 fps** GIF (stride 2) |
| MP4 | 1920 × 1150, H.264, yuv420p, 20 fps |
| Still PNG | 1920 × 1150 (final composed frame) |
| Font | **Roboto** (regular; 6.5 pt labels @ y0.71, 10.5 pt values @ y0.33) |
| Footer fields (x) | PLACE 0.020 · MAP 0.250 · legend 0.520–0.700 · TIME ELAPSED 0.980 (right) |
| Legend box | x 0.520–0.700, y 0.40–0.50, outline lw ~0.4–0.5 |
| Timer | real source-video time `MM:SS / MM:SS` (frames / 30 fps source) |
| Outline rect | `(0.004, 0.14)`, w 0.992, h 0.72 |

**HUD theme per family** (identical geometry, inverted colours):
- **Flow Fields → DARK** footer: black bg, white outline α0.26, labels `#7e7e88`,
  values `#ededf2`, end labels `#9a9aa2`, gradient α 0.55 (dim on black).
- **Speed Population / Bottleneck Density → LIGHT** footer: bg `#F7F7F7`, outline
  `#D9D9D9`, labels `#333333`, values `#111111`, end labels `#111111`, gradient
  α 1.0 (true colours on white). Dot swatches keep their colours.

The world window for **all three** is `window_extent(plan_extent, 1920, 1080)`
so the plaza is framed identically in every map.

---

## 1. FLOW FIELDS

- **Final script (accepted):** `flow_currents/generate_flow_currents_v5_info.py`
  (V5 = compact footer + real video timer). Movement pipeline frozen from
  `flow_currents/generate_flow_currents_v3.py`; low-level helpers + plan styling
  from `trace_dust_hero/generate_trace_dust_hero_v3.py`.
- **Animation version picked:** V5 (over V2/V3/V4).
- **Output dimensions:** map 1920×1080, footer 70 → 1920×1150; GIF 760 wide.
- **HUD:** DARK. PLACE *Plaça Catalunya* · MAP *Flow Fields* · INTENSITY
  cyan→magenta gradient (`Low flow` / `High flow`) · TIME ELAPSED.
- **Palette:** `#00E5FF → #2F6BFF → #8B5CFF → #FF3DF2` (cyan→blue→violet→magenta),
  `LinearSegmentedColormap`, density-mapped with `DENS_GAMMA = 1.4`.
- **Background:** black; architectural wireframe from the plan
  (`architectural_layers`): major Hough lines op 0.64 + secondary detail op 0.30,
  tints `MAJOR_TINT [0.86,0.90,1.0]` / `SECONDARY_TINT [0.60,0.70,0.86]`.
- **Animation timing (V5, 17 s @ 20 fps = 340 frames):**
  plan fade 0–1 s · strings draw 1–9 s (arc-length appearance) · dust flows in +
  glow builds 9–17 s, hold near end (`GLOW_FULL_T` ≈ 15.5 s).
- **Strings (traces):** width 0.9, α 0.075, core gain 2.0; per-track arc-length
  reveal (`build_segments`).
- **Dust / particles:** Type A `N_A=4200` (blur 0.8, gain 2.4 — sharp filaments);
  Type B `N_B=6500` (blur 4.0, gain 1.1 — soft current); speed 3.8 m/s,
  `K_SAMPLES=200`, `MIN_LEN_M=6`, off-screen lead-in `EXTRA_M=5`, fade-in 0.12 /
  fade-out 0.06, trail decay 0.74, particle core gain 1.8. Flows oriented
  low→high density (convergence).
- **Glow:** bloom sigmas `[3,9,20,30]` weights `[0.5,0.38,0.26,0.16]`, glow draw
  0.15→0.38, dust 0.5→1.0, rolloff 0.12.
- **Source files:** `plan/*.png`, `calibration/calib.json`,
  `filtered_250m/trajectories_world_filtered_250m.csv` (strings + flow are built
  directly from trajectories; `flow_field_*.csv` not required).
- **Sparse-recording rule:** keep per-path particle density constant
  (Plaça Catalunya reference = **465 flow paths**, N_A 4200 ≈ 9.0/path,
  N_B 6500 ≈ 14.0/path). For fewer paths, particle counts scale **down**
  proportionally (clamped, never faked) and the counts are reported.

## 2. SPEED POPULATION

- **Final script (accepted):** `speed_population/generate_speed_population_v3.py`
  (reuses V2 bg/grid/dot helpers, V1 track/calibration loaders).
- **Palette picked:** **warm** — slow `#E85D4F` coral · medium `#F2B84B` amber ·
  fast `#00AFA6` teal (the V3 recommendation over the cool palette).
- **Output dimensions:** map 1920×1080, footer 70 → 1920×1150; GIF 760 wide.
- **HUD:** LIGHT. PLACE *Plaça Catalunya* · MAP *Speed Population* · SPEED
  slow/medium/fast dot legend (warm palette, dots keep colour, labels black) ·
  TIME ELAPSED.
- **Background:** arctic satellite (`arctic_background`: desaturate, gamma,
  contrast, brighten, light tint + edge AO + soft drop shadow) placed in the
  16:9 window with neutral (white) padding; subtle metric grid (`add_grid`).
- **Dot style:** moving dot per active pedestrian; head radius 3.0 px (α 0.95) +
  soft white halo; **no glow, no strings, no heatmap**. Colour = instantaneous
  speed class (slow <0.8 · medium 0.8–1.6 · fast >1.6 m/s), speed measured from
  real world coords over a ±window with light denoising (positions kept exact).
- **Trace persistence:** short fading trace of the dot's **real** recent
  positions, default **5 s** of source time, sampled every 0.14 s, fade γ 1.9,
  trail radius 1.9 px (α 0.50·age); exit fade 0.8 s. Represents recent movement
  presence, not instantaneous count.
- **Speed timing / remapping:** presentation-time remapping — the full source
  range is stretched over the presentation so dots move at a perceivable rate;
  the HUD timer still shows **real source time**. Unified batch duration = 17 s.
- **Source files:** `plan/*.png`, `calibration/calib.json`,
  `filtered_250m/trajectories_world_filtered_250m.csv` (speed derived internally;
  `metrics/speed_per_observation.csv` not required).
- **Sparse-recording rule:** same persistence logic; if concurrency is low the
  5 s fading trace + exit fade keep the field legible without duplicating
  pedestrians. Persistence window reported.

## 3. BOTTLENECK DENSITY

- **Final script (accepted):** `bottleneck_animations/generate_bottleneck_animation_tests.py`
  — **Option 5 (Hybrid)**, refined. V2D cell primitives from
  `generate_bottleneck_map_test.py`.
- **Option selected:** Option 5 Hybrid (3-phase narrative).
- **Output dimensions:** map 1920×1080, footer 70 → 1920×1150; GIF 760 wide.
- **HUD:** LIGHT. PLACE *Plaça Catalunya* · MAP *Bottleneck Density* · INTENSITY
  yellow→orange→red gradient (`Low density` / `High density`) · TIME ELAPSED.
- **Square-cell style (frozen V2D):** 1 m cells, arctic/architectural underlay
  (`to_architectural_underlay`, bg opacity 0.55); variable square size
  `scale = 0.70 + (score_norm**1.6) · 0.25`, clamped to 0.85× grid spacing
  (no overlap, visible gaps), rounded corners; alpha `0.45 + 0.50·score_norm`.
- **Palette:** `#FFF3B0 → #FDBA3B → #F97316 → #DC2626`
  (yellow→amber→orange→red); `vmax` = 99th percentile of `bottleneck_score`.
- **Ranked markers:** top **10** cells (white disc, dark-red ring + rank number).
- **10-second hybrid timing (@ 20 fps = 200 frames):**
  - Phase A (0–3 s): complete field appears immediately, **faint**, no markers
    (0.5 s fade-in + 2.5 s faint hold).
  - Phase B (3–8 s): high-density cells brighten, warm toward orange/red, grow
    slightly; low-density cells stay subdued (a differentiation, not a reveal).
  - Phase C (8–10 s): ranked markers fade in (0.8 s), then a calm hold (1.2 s).
    No pulsing / bounce / glow.
- **Source files:** `plan/*.png`, `calibration/calib.json`,
  `filtered_250m/bottlenecks/bottleneck_cells.csv` (scores read, never
  recomputed); trajectory CSV used only for the real-duration HUD timer.
- **Sparse-recording rule:** none needed (uses precomputed cells); cells outside
  the plan window are dropped as before.

---

## Source-of-truth files inspected
- `flow_currents/generate_flow_currents_v5_info.py`, `…_v3.py`,
  `…_v5_info_report.md`
- `speed_population/generate_speed_population_v3.py`, `…_v2.py`, `…_v1.py`,
  `…_v3_report.md`
- `bottleneck_animations/generate_bottleneck_animation_tests.py`,
  `…/outputs/bottleneck_animation_tests_report.md`
- `final_unified_series/generate_unified_series.py`,
  `…/outputs/final_unified_series_report.md` (visual standard + light HUD)
- `generate_bottleneck_map_test.py`, `trace_dust_hero/generate_trace_dust_hero_v3.py`
