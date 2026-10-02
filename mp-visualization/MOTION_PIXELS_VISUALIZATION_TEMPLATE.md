# Motion Pixels — Trajectory Visualization Template

**Status:** official template for all trajectory + prediction visuals (animated GIF and static
preview), to be applied consistently across every track and horizon.
**Reference implementation:** `mp-visualization/motion_pixels_animation_test/generate_d3_style_animation.py`
(all values live in the `CFG` dict; retarget by changing `horizon` / `track` / `recording`).
> **Archival note (2026-10-02):** this reference implementation file was lost and is not in the
> repository; it has not been reconstructed. This specification and the surviving outputs
> (`H100_track4703_*_REFINED*.gif`, `ready_animations/`) are what remains. See `ready_animations/README.md`.
**Reference case:** H100 · track 4703 · `MODEL_XC_B_CURV_LIGHT`.

> Visualization-only. Trajectories and predictions are recovered by deterministic replay of the
> frozen model rollout (no training, no new predictions, no data/encoder changes).

---

## 1. Visual philosophy
A clean, editorial / Observable-style motion chart — not a dashboard, not a research plot.
Lots of white space, quiet chrome, a single accent colour, and a "drawn-live" reveal so the
viewer reads the story in one pass: **what happened → what actually came next → what the model
believes**. Geometry is never distorted; the prediction is the focal element once forecasting
begins; everything else stays secondary.

## 2. Plot geometry & aspect ratio
- **Square canvas**, square data window centred on the trajectory.
- **Equal x/y aspect ratio** in metres (`set_aspect("equal")`) — spatial geometry is true, never
  stretched.
- Limits = combined extents of history + GT + prediction, expanded to a square, with **~14%
  data padding** per side; figure margins are generous (≈ white border around the plotting area).
- Result: trajectories breathe inside the frame; flat walks read as a line across a square field
  of white rather than a cramped strip.

## 3. Grid & axis styling
- **Grid:** very light **dotted** lines, colour `#e4e4e4`, hairline weight — for scale only,
  never dominant.
- **Axes:** quiet. Top and right spines removed; left and bottom spines faded (`#cfcfcf`, 0.8 px).
- **Ticks:** ~5 per axis (`MaxNLocator(5)`), small labels in `#7a7a7a`.
- **Axis labels:** `x (m)` / `y (m)`, small and light (`#7a7a7a`).
- No title, no legend, no boxes.

## 4. Typography
- Font family (in order): **Inter → Helvetica Neue → Arial → system sans** fallback.
- All text **regular weight** (never bold), small and refined.
- Tick labels ~8.5 pt · axis labels ~9.5 pt · annotation ~10.5 pt · metrics ~10.5 pt.

## 5. History trajectory (the past)
- **Solid black** (`#111111`), **2.4 px**, round caps/joins — the strongest, most grounded trace.
- **No markers / no sampled dots** — reads as one clean black stroke.
- A single subtle **black dot at the seed/separation point** (last observed position) marks where
  the future begins.

## 6. Ground-truth future trajectory (what actually happened)
- **Dotted dark grey** (`#666666`), **2.0 px**, round dash caps, **opacity 0.85** —
  intentionally de-emphasised (~10–15% quieter) so it supports, not competes with, the prediction.
- Distinctly lighter than the black history and clearly separate from the coloured prediction.
- Drawn only where real future data exists; never invented beyond the recorded track.

## 7. Prediction trajectory (what the model believes)
- **Solid purple** (`#7C3AED`), **2.5 px** — the **only coloured trajectory** in the chart and
  the focal element once forecasting begins (full opacity, drawn above the other strokes).
- Refined, not neon; no glow.

## 8. Arrowhead behaviour
- A single arrowhead in the **same purple** (`#7C3AED`) sits **only at the moving head** of the
  prediction and travels with it as the line grows.
- No arrowheads on history or ground truth.
- At the final hold it rests at the prediction's end, indicating the model's heading.

## 9. Animation timing & reveal sequence (~5.6 s total, 24 fps)
1. **Empty (0.2 s)** — quiet chart, grid + axes + annotation visible.
2. **History (0.8 s)** — black trace draws first (quick), using arc-length reveal (the D3
   `stroke-dasharray` "0,l → l,l" idiom replicated by progressive prefixes).
3. **Separation (0.15 s)** — black seed dot appears.
4. **Future (2.9 s)** — **GT and prediction reveal simultaneously**, growing together from the
   seed point; the purple arrowhead rides the prediction head.
5. **Hold (1.5 s)** — full composition rests; metrics caption appears.
Motion is smooth and readable throughout; the future reveal is the slowest, most legible phase.

## 10. Annotation rules
- A quiet two-line block **inside the upper-left** of the plot (axes-fraction ≈ 0.025, 0.975),
  left-aligned, regular weight, dark grey `#333333`, no box / no background:
  ```
  Horizon: H{horizon}
  Track {track_id}
  ```
- **Both values auto-derive** from the plotted case: `Horizon` from the prediction horizon
  (`H20 / H60 / H100 / H200 / H400 …`), `Track` from the plotted trajectory id. They update
  automatically when the track or horizon changes. No metric/units text in this block.
- **Optional metrics caption:** a single minimal line **above** the plot at the final hold —
  `ADE … m · FDE … m · θ … rad`, regular weight, `#5a5a5a`. Keep minimal or omit; never a HUD.

## 11. Colour palette
| Role | Colour |
|---|---|
| History (past) | `#111111` solid black |
| Ground-truth future | `#666666` dotted grey @ 0.85 opacity |
| Prediction + arrowhead | `#7C3AED` purple *(only colour in the chart)* |
| Seed / separation dot | `#111111` |
| Grid | `#e4e4e4` dotted |
| Axis spines | `#cfcfcf` |
| Tick / axis labels | `#7a7a7a` |
| Annotation text | `#333333` |
| Metrics text | `#5a5a5a` |
| Background | `#ffffff` |

Purple is reserved exclusively for the prediction; introduce no additional accent colours.

## 12. Export specifications
- **Animated GIF** — square **936 × 936 px** (7.8 in @ 120 dpi), **24 fps**, ~5.6 s, Pillow writer
  (GIF because ffmpeg is not available; the writer merges identical static frames but preserves
  timing). Typical size ~0.2–0.3 MB.
- **Static preview PNG** — the final-frame composition at 170 dpi.
- **Naming:** `H{horizon}_track{track}_d3_style_animation_REFINED_v2.gif` and
  `…_REFINED_v2_preview.png`, written to the `mp-visualization/` root.
- For crisp slides, a vector/SVG `stroke-dasharray` variant of the same logic is available; the
  GIF remains the standard raster deliverable.

---

### Application notes
- To produce any other case: set `CFG["horizon"]`, `CFG["track"]`, `CFG["recording"]` — geometry,
  zoom, annotation, and metrics all recompute automatically from the plotted trajectory.
- Curvier tracks (e.g. Red Bridge, Stairs) fill the square more organically; near-straight walks
  read as a horizontal line in a square field of white (by design — geometry is preserved).
- Keep the palette, weights, timing, and quiet chrome fixed so every Motion Pixels trajectory
  visual is immediately recognisable as part of the same template.
