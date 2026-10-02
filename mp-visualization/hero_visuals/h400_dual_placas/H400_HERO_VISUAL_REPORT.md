# H400 Hero Trajectory Visual — Plaça Catalunya & Plaça Espanya

Cinematic dual-panel hero showing **MODEL_X** long-horizon (**H400 ≈ 20 m**) pedestrian
trajectory predictions over the two Barcelona plazas, rendered on dark satellite/plan
backdrops. Visualization only — **no retraining, no architecture change**.

Generated 2026-06-20.

---

## Outputs

| File | Description |
|---|---|
| `hero_H400_placa_catalunya_placa_espanya_background.png` | Hero with dark plan backdrops. **7079 × 1920 px**, black background. |
| `hero_H400_placa_catalunya_placa_espanya_transparent.png` | Same composition, **alpha-transparent** background (plan omitted) for overlaying on thesis pages. 7079 × 1920 px, RGBA. |
| `H400_HERO_VISUAL_REPORT.md` | This report. |
| `selected_windows.json` | The 6 plotted windows (3 per site) with their metrics. |
| `placa_H400_best_per_track.pkl` | Best-ADE window per distinct test track (obs/gt/pred world coords) — selection pool. |
| `placa_H400_all_window_metrics.csv` | Metrics for all 18,805 scored placa windows. |
| `_infer_placa_H400.py` | Frozen-model inference script (reproducibility). |
| `_render_hero.py` | Figure render script (reproducibility). |

Absolute output folder:
`C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-visualization\hero_visuals\h400_dual_placas\`

---

## Source files used

**Model / predictions (MODEL_X H400, frozen):**
- `mp-core/trajectory-prediction/experiments/MODEL_X_HORIZON_SWEEP/H400/model_best.pth` — frozen checkpoint (best val H400-rollout ADE).
- `.../H400/scalers.json` — frozen feature/target scalers.
- `.../H400/config.json` — 10-feature Model C LSTM (hidden 128 × 2, dropout 0.2), single-step MSE objective, 400-step autoregressive rollout.
- `mp-core/trajectory-prediction/experiments/MODEL_X/model_x_lib.py` — rollout / windowing / metrics library.
- `mp-core/trajectory-prediction/experiments/MODEL_X/splits/model_x_track_split.csv` — track-level split (test tracks only used).
- Dataset: `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v3_manual_master_dataset\master_dataset.csv`.

**Plan backdrops + calibration:**
- `C:\Users\OWNER\Desktop\new_datasets\placa_catalunya_01\plan\placa-catalunya.png` (2122 × 982) + `calibration\calib.json`.
- `C:\Users\OWNER\Desktop\new_datasets\placa_espanya_01\plan\placa-espanya.png` (2311 × 1227) + `calibration\calib.json`.

### Why inference was re-run
The cached H400 `plot_windows.pkl` held only **5 cherry-picked placa windows** (2 near-stationary
Catalunya dots + 3 heavy-undershoot Espanya stubs) — and only H20/H400 contained any placa
windows at all. Per the instruction to *use all the best predictions on the two plazas*, the
**frozen** H400 model was re-run (inference only) over the full placa test populations to obtain
trajectory coordinates for genuinely strong examples. The reproduction is exact: per-recording
test metrics match the original H400 run (Catalunya ADE 3.027 / FDE 6.166; Espanya ADE 8.518 /
FDE 16.022).

---

## Recording IDs found

Scouted `C:\Users\OWNER\Desktop\new_datasets\`. Plaza recordings present:
- **`placa_catalunya_01`** ✔ (used)
- **`placa_espanya_01`** ✔ (used)
- `placa_montjuic_01`, `esplanade_espanya_01` (present, **not** requested — excluded)

---

## Windows plotted per site

3 trajectories per panel (6 total), hand-curated from the per-track best-ADE pool for the
cleanest "predicted vs actual" reading at hero scale.

**H400 placa test population scored:** 18,805 windows total
(Catalunya 11,739 windows / 16 qualifying tracks; Espanya 7,066 windows / 16 qualifying tracks).
A track "qualifies" for H400 when it persists ≥ obs(10) + H(400) = **410 frames**.

### Plaça Catalunya (3 plotted)
| trajectory | ADE (m) | FDE (m) | GT len (m) | pred len (m) | len ratio |
|---|---:|---:|---:|---:|---:|
| `placa_catalunya_01__3941` | 0.94 | 3.22 | 11.5 | 6.2 | 0.54 |
| `placa_catalunya_01__264`  | 2.80 | 6.92 | 18.8 | 11.3 | 0.60 |
| `placa_catalunya_01__5477` | 1.23 | 3.40 | 4.6 | 5.4 | 1.19 |

### Plaça Espanya (3 plotted)
| trajectory | ADE (m) | FDE (m) | GT len (m) | pred len (m) | len ratio |
|---|---:|---:|---:|---:|---:|
| `placa_espanya_01__6268` | 6.23 | 16.00 | 110.3 | 17.8 | 0.16 |
| `placa_espanya_01__4394` | 1.10 | 1.55 | 85.5 | 11.0 | 0.13 |
| `placa_espanya_01__122`  | 2.20 | 1.12 | 45.2 | 18.1 | 0.40 |

> **Honest reading of the visual.** MODEL_X holds **heading** reliably (selected tracks have
> net-direction error ≤ ~17°) but **undershoots distance** at long horizons (regression-to-mean,
> documented in the MODEL_X audits). The magenta prediction therefore confidently traces the
> *start* of each journey along the correct direction, while the dotted ground truth continues
> further — most dramatically on the long Avinguda Maria Cristina walks at Espanya
> (GT 45–110 m vs predicted 11–18 m). On Catalunya, long plaza walks also *curve*, which the
> straight-tending prediction does not fully capture. The `angular_err_deg` field in
> `selected_windows.json` is a *final-segment* heading error (can be large) and is **not** the
> overall direction agreement used for selection.

---

## Styling parameters

- **Canvas:** two side-by-side panels, panel boxes sized to each crop's aspect ratio
  (Catalunya 1.52, Espanya 2.40) so both fill at equal height — no letterboxing / dead gap.
  Panel height 8 in, dpi 200 → **7079 px wide** (> 4000 px requirement).
- **Background:** black (`#000000`); transparent variant uses figure/axes alpha 0.
- **Plan backdrop:** RGB darkened ×0.34, drawn at alpha 0.62 (omitted in transparent variant).
- **Coordinate space:** plotted in plan-pixel space via the calib world→plan affine
  (pure scale + offset, **0 px** fit residual; Catalunya 21.09 px/m, Espanya 10.03 px/m).
  Geometry is preserved exactly within each panel; y-axis inverted to image convention.
- **History (observed 10 frames):** bold white solid, lw 5.2.
- **Prediction start:** white **×** marker, size 190, linewidth 3.2.
- **Ground-truth future:** white dotted `(0,(1,2.6))`, lw 2.4, alpha 0.78.
- **Prediction (H400):** magenta `#ff1ad9` — solid lw 5.6 (alpha 0.98) over a wide lw 13
  alpha 0.18 glow; small magenta endpoint dot. **Only strong color in the image.**
- **Crop:** per panel, tight bounding box over all plotted points (obs + GT + pred) + 14% margin.
- **Labels:** minimal — site name under each panel; a single minimal 4-item key at bottom center.
- No axes, no grid, no metrics, no technical legend.

---

## Reproduce

```
# 1) frozen-model inference over placa test tracks (writes the .pkl + metrics csv)
cd mp-core/trajectory-prediction/experiments/MODEL_X
python ../../../mp-visualization/hero_visuals/h400_dual_placas/_infer_placa_H400.py

# 2) render both hero PNGs + selected_windows.json
python mp-visualization/hero_visuals/h400_dual_placas/_render_hero.py
```
