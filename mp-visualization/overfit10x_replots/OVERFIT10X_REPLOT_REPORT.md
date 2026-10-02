# OVERFIT10X — Model C Highlight Replot Report

**Date:** 2026-06-20
**Type:** visualization-only restyle (no training, no metric changes, no animation, no source-data edits)

> **OVERFIT10X is a capacity / memorisation check — NOT thesis generalization evidence.**
> Train/val/test share trajectory content by construction, so the numbers below reflect
> memorisation capacity, not held-out performance.

---

## What was done

Three selected OVERFIT10X trajectories were replotted in the clean official thesis aesthetic
with **Model C** (motion + position + spatial) highlighted. Predictions are **not stored as
CSVs** anywhere in the experiment — they are deterministic autoregressive rollouts of the frozen
model checkpoints (`<set>/best_model.pth`). Since no plotted prediction CSV exists, the rollouts
were recomputed deterministically (`SEED=42`) from the frozen weights. **No model was retrained.**

## Source files used

| File | Role |
|---|---|
| `…/schema_ablation_bridge_overfit10x/schema_ablation_bridge_overfit10x_dataset.csv` | trajectory data (520 ids) |
| `…/schema_ablation_bridge_overfit10x/schema_summary.json` | `world_bounds_used` for u/v mapping |
| `…/schema_ablation_bridge_overfit10x/scalers.pkl` | per-model feature + target scalers |
| `…/A_motion_only/best_model.pth` | frozen Model A checkpoint |
| `…/B_motion_position/best_model.pth` | frozen Model B checkpoint |
| `…/C_motion_position_spatial/best_model.pth` | frozen Model C checkpoint (focal) |
| `…/D_full_relational/best_model.pth` | frozen Model D checkpoint |
| `run_overfit10x_ablation.py` | imported: `rollout`, `build_kdt`, `resolve_feature_sets`, `split_ids`, `COLORS`, constants |
| `visualize_overfit10x_ablation.py` | imported: `load_models_and_scalers`, `run_detailed_rollout`, `world_to_uv` |

Restyle driver: `mp-visualization/overfit10x_replots/restyle_overfit10x.py`

## Selected tracks

`1001`, `12007`, `14007` — only these three were plotted (per spec). All three were present in
the dataset.

## Predictions found per model (Model C ADE in metres)

| Track | A motion-only | B +position | **C +spatial** | D +entrance |
|---|---|---|---|---|
| 1001  | 0.028 | 0.028 | **0.045** | 0.029 |
| 12007 | 0.112 | 0.099 | **0.038** | 0.094 |
| 14007 | 0.015 | 0.038 | **0.011** | 0.015 |

All four models produced a valid 20-step rollout for every selected track. No track and no model
was missing. (Model E `full_affordance` does not exist in this experiment — the
`openness_lr_asymmetry` column is absent — so the comparison is the intended A/B/C/D quartet.)

## Styling applied

Layered the user spec onto the finalized official thesis aesthetic
(`mp-visualization/MOTION_PIXELS_VISUALIZATION_TEMPLATE.md`):

- **Background:** white. **Grid:** very light dotted `#e4e4e4`, hairline. **Spines:** top/right
  removed, left/bottom faded `#cfcfcf`. **Ticks/labels:** quiet `#7a7a7a`, `MaxNLocator(5)`.
- **Equal aspect**, square data window centred on history + GT + predictions, **~14% padding**
  (generous, never cramped).
- **Ground Truth:** dotted **black** (`#000000`), 2.0 px round dash caps — clearly visible, not heavy.
- **Model C:** solid **red** (`#D7261E`), 2.6 px, top z-order, end dot — visually dominant focal trace.
- **Models A / B / D:** solid **light grey** (`#C8C8C8`), uniform 1.8 px — subdued but visible, not competing.
- **Seed / history:** **dark grey** (`#555555`), 2.2 px, subtle, with a small separation dot at the last observed point.
- Faint full-track context line in `#ededed` behind everything for spatial context.
- **Minimal legend** (4 entries): seed/history · ground truth · models A·B·D · Model C.
- Concise title `OVERFIT10X trajectory <id>` / `Model C highlighted`; annotation block carries
  Model C ADE + `capacity check — not generalization evidence`. No markers beyond endpoint dots; no clutter.

## Outputs

`mp-visualization/overfit10x_replots/`

| File | Content |
|---|---|
| `track_1001_modelC_highlight.png` / `.svg` | track 1001 panel |
| `track_12007_modelC_highlight.png` / `.svg` | track 12007 panel |
| `track_14007_modelC_highlight.png` / `.svg` | track 14007 panel |
| `overfit10x_modelC_highlight_collage.png` / `.svg` | 3-panel collage |
| `restyle_overfit10x.py` | reproducible restyle driver |
| `_found.json` | per-track per-model ADE bookkeeping |

## Missing / caveats

- Nothing missing: all 3 tracks × 4 models rendered successfully.
- The `Inter` / `Helvetica Neue` fonts are not installed on this machine; matplotlib fell back to
  Arial/system sans (cosmetic only — emitted `findfont` warnings, no impact on geometry or metrics).
- PNG (raster, 200 dpi) and SVG (vector) versions were exported; PDF was not (SVG already covers
  the vector use-case — say the word and I'll add PDF).

> Reminder: **OVERFIT10X is a capacity check only, not thesis generalization evidence.**
