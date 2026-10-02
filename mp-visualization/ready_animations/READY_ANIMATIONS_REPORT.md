# Ready Animations — Batch Report

**Total animations generated:** 21  ·  **previews:** 21  ·  **missing:** 0
**Each:** square canvas, equal aspect, dotted grid, quiet axes, black history, grey dotted GT,
moving prediction arrowhead, simultaneous GT+prediction reveal, ~5.6s @ 24 fps,
GIF + final-frame preview PNG. Annotation = `Horizon: H{h}` / `Track {id}` only (metrics omitted).

> **No training or model modification occurred.** Trajectories + predictions are a deterministic
> replay of the frozen `MODEL_XC_B_CURV_LIGHT` rollout (the same machinery used by the template,
> `generate_plots_wassim.py`, and `generate_long_horizon_stress_test.py`). No new models, losses,
> datasets, or encoders were touched.

## Colour mapping (prediction line + arrowhead)
| Category | Colour |
|---|---|
| best | `#7C3AED` purple |
| worst | `#84CC16` lime green |
| stress_test | `#2563EB` blue |
History `#111111`, GT dotted `#666666`, seed dot `#111111`, grid `#e4e4e4`, axes `#cfcfcf` — unchanged.

## Best animations (purple) — 9
- **H20 · track 1261** — stairs_montjuic_01 (window: inventory; GT 21 pts, pred 21 pts)
- **H20 · track 6268** — placa_espanya_01 (window: inventory; GT 21 pts, pred 21 pts)
- **H60 · track 4394** — placa_espanya_01 (window: inventory; GT 61 pts, pred 61 pts)
- **H60 · track 6268** — placa_espanya_01 (window: inventory; GT 61 pts, pred 61 pts)
- **H100 · track 4703** — placa_catalunya_01 (window: inventory; GT 101 pts, pred 101 pts)
- **H200 · track 3084** — placa_catalunya_01 (window: inventory; GT 201 pts, pred 201 pts)
- **H200 · track 1178** — red_bridge_combined_01 (window: inventory; GT 201 pts, pred 201 pts)
- **H400 · track 112** — esplanade_espanya_01 (window: inventory; GT 401 pts, pred 401 pts)
- **H400 · track 2039** — red_bridge_combined_01 (window: inventory; GT 401 pts, pred 401 pts)

## Worst animations (lime green) — 5
- **H20 · track 1647** — placa_espanya_01 (window: worst_ade; GT 21 pts, pred 21 pts)
- **H20 · track 3016** — placa_espanya_01 (window: worst_ade; GT 21 pts, pred 21 pts)
- **H60 · track 1822** — esplanade_espanya_01 (window: worst_ade; GT 61 pts, pred 61 pts)
- **H200 · track 6268** — placa_espanya_01 (window: worst_ade; GT 201 pts, pred 201 pts)
- **H400 · track 540** — stairs_montjuic_01 (window: inventory; GT 401 pts, pred 401 pts)

## Stress-test animations (blue) — 7
- **H800 · track 2039** — red_bridge_combined_01 (window: stress_seed0; GT 443 pts, pred 801 pts)
- **H800 · track 1178** — red_bridge_combined_01 (window: stress_seed0; GT 331 pts, pred 801 pts)
- **H800 · track 540** — stairs_montjuic_01 (window: stress_seed0; GT 482 pts, pred 801 pts)
- **H1000 · track 1969** — red_bridge_combined_01 (window: stress_seed0; GT 158 pts, pred 1001 pts)
- **H1000 · track 5350** — placa_catalunya_01 (window: stress_seed0; GT 324 pts, pred 1001 pts)
- **H1200 · track 750** — esplanade_espanya_01 (window: stress_seed0; GT 250 pts, pred 1201 pts)
- **H1200 · track 3** — placa_espanya_01 (window: stress_seed0; GT 1201 pts, pred 1201 pts)

## Source files used
- Trajectory machinery / deterministic rollout: `mp-visualization/motion_pixels_animation_test/generate_d3_style_animation.py`
  (template) → `MODEL_XC/xc_common.py` → `xr_common.py` → `model_x_lib.py`
- Master dataset (via `model_x_lib.CONFIG`): `…/Barcelona_v3_manual_master_dataset/master_dataset.csv`
- Test split: `…/experiments/MODEL_X/splits/model_x_track_split.csv`
- Frozen checkpoint: `…/MODEL_XC/checkpoints/MODEL_XC_B_CURV_LIGHT/model_best.pth`
- Best-window lookup: `…/presentation_visuals/selected_visuals_inventory.csv`

## Window-selection method
- **Presentation (best/worst, H20–H400):** if the (horizon, track) exists in the inventory, its
  published `window_id` is used (matches the Plots_Wassim figures). Otherwise — for worst tracks not
  in the inventory — the **worst-ADE, non-artifact window** of that track is chosen deterministically
  (filters: GT net ≥ horizon minimum, max GT per-step ≤ 2.0 m to exclude tracking teleports).
- **Stress (H800–H1200):** each track is seeded from its first 10 frames (start_idx = 0) and the
  frozen model is rolled forward H steps — one continuous blue prediction line, no validated/
  extrapolated split. GT is drawn only where real future data exists (never invented beyond the
  recorded track). These are **exploratory extrapolations beyond the validated range** (≤ H400);
  the context lives here in the report, not as on-frame text.

## Assumptions
- **Stress horizons:** the brief listed "H100: 1969, 5350" but, per the note and the stress-test
  horizons, these were interpreted as **H1000**. Final stress horizons: H800, H1000, H1200.
- **Recordings** were resolved automatically as the unique TEST-split recording for each track id
  with sufficient length (e.g. worst 1647/3016/6268 → placa_espanya, 1822 → esplanade_espanya;
  stress 2039/1178/1969 → red_bridge, 540 → stairs_montjuic, 5350 → placa_catalunya, 750 →
  esplanade_espanya, 3 → placa_espanya).
- Track 540 appears here under *worst* (H400) per the brief, although it is also a curated *best*
  H400 track; it is rendered in lime green from its inventory window as requested.
- Metrics caption omitted (only the template Horizon/Track annotation is shown), per the brief's
  "no extra text beyond the template annotation".

## Missing / failed tracks
None — all requested tracks rendered.
