# MODEL_X Horizon Sweep — Phase 1 Plotting Audit

**Date:** 2026-06-11 · **Scope:** visualization/plotting audit ONLY. No retraining, no weight
changes, no dataset changes, no Phase 2. All inference re-run is deterministic (`model.eval()`,
no dropout) using the already-trained `H*/model_best.pth`.

> Path note: MODEL_X actually lives at `experiments/MODEL_X/` (not `trajectory-prediction/MODEL_X/`).
> The sweep is `experiments/MODEL_X_HORIZON_SWEEP/`. The shared library `model_x_lib.py` lives in
> `experiments/MODEL_X/`. Audit outputs are under `experiments/MODEL_X_HORIZON_SWEEP/plots_audit/`.

## 1. Plotting scripts found

| Script | Role | Aspect call |
|---|---|---|
| `MODEL_X_HORIZON_SWEEP/run_horizon.py` (`diag_panel` L61, `pres_panel` L75) | the sweep's own rollout plots | `ax.set_aspect("equal", "datalim")` |
| `MODEL_X/plots/make_plots.py` (L33, L80) | MODEL_X baseline (H10) plots | `ax.set_aspect("equal", "datalim")` |
| `MODEL_X/plots/make_plots.py` (L121) | distribution scatter | `ax.set_aspect("equal")` |

(For reference, ~30 other experiment plotters across the repo overwhelmingly use
`set_aspect("equal", adjustable="datalim")` too — equal aspect is the house standard.)

## 2. Was unequal scaling found?

**No.** Every horizon-sweep trajectory plot already calls `ax.set_aspect("equal", ...)`, which forces a
**1:1 metric aspect ratio** (1 m on x = 1 m on y). `"datalim"` vs `adjustable="box"` only changes
*which* dimension is adjusted to satisfy that ratio — both preserve geometry. **The trajectory plots
were never metrically distorted.**

## 3. What code was changed

The existing sweep plotting did **not** need a scaling fix, so `run_horizon.py` was **left unmodified**
(consistent with "do not modify MODEL_X" and to keep the trained artifacts' provenance intact). Instead a
new, self-contained audit plotter was added **inside the sweep folder only**:

- `plots_audit/make_audit_plots.py` — regenerates plots with `ax.set_aspect("equal", adjustable="box")`
  (the exact form requested), full visible metadata, and non-cherry-picked example selection.

No other files were created/modified/deleted outside `plots_audit/`.

## 4. How examples were selected (NOT lowest-ADE)

Selection uses the authoritative `H*/per_window_metrics.csv`, which maps every test `window_idx →
(trajectory_id, start_idx)`. Each selected window is reconstructed directly from the v3 master dataset and
re-rolled with the trained model. Four categories, **9 windows each**:

| Category | Rule |
|---|---|
| `longest_gt` | 9 largest **GT** path length |
| `largest_pred` | 9 largest **predicted** path length |
| `median_ade` | 9 windows nearest the **median** ADE |
| `worst_collapse` | 9 smallest **pred/GT ratio**, restricted to GT length ≥ median (so it is real collapse, not a tiny-GT artifact) |

## 5. Metadata on every plot

Each panel (and each individual PNG) shows: recording, track_id, window_id, horizon (+approx distance),
ADE, FDE, GT path length, predicted path length, pred/GT ratio. Legend: observed (black) / GT future
(grey) / prediction (magenta) / current marker (blue).

## 6. Plots generated

**180 annotated panels + 20 contact-sheet grids** = 200 images. 9 panels per (horizon × category):

| Horizon | longest_gt | largest_pred | median_ade | worst_collapse | total |
|---|---|---|---|---|---|
| H20 | 9 | 9 | 9 | 9 | 36 |
| H60 | 9 | 9 | 9 | 9 | 36 |
| H100 | 9 | 9 | 9 | 9 | 36 |
| H200 | 9 | 9 | 9 | 9 | 36 |
| H400 | 9 | 9 | 9 | 9 | 36 |

Inventory: `plots_audit/plot_inventory.csv` (180 rows). No files were missing; no metadata fields were
unavailable (`window_id` present for all).

## 7. Is the visual "flattening" caused by plotting scale or real model behavior?

**Real model behavior — NOT plotting scale.** With confirmed 1:1 metric aspect, three things are now
visible:

1. **Direction is captured well, length is undershot** — the original hypothesis is supported. In
   `median_ade` and `longest_gt` panels the magenta prediction lies *along* the grey GT heading but stops
   short. So the model has learned heading better than travelled distance.
2. **The model is NOT uniformly collapsed** — the `largest_pred` panels (e.g. H100 placa_espanya
   track 1474) show predictions of ~45–48 m that track long GT paths closely (ratio ≈ 1.0–1.2). When the
   observed motion is fast and persistent, the model *does* extend. The low *median* ratio is a
   distribution effect (most pedestrians are slow/short and the model steps conservatively), not a hard
   ceiling.
3. Panels look "flat/thin" because the underlying motion is genuinely near-1-D (people walking roughly
   straight); equal aspect renders that honestly as a wide, short box.

## 8. ⚠️ New finding — GT path-length inflation by tracking jitter (metric caveat)

The audit surfaced a measurement artifact that the previous summary plots hid: **GT path length is
inflated by per-step tracking jitter**, which contaminates `pred/GT ratio` and `collapse_rate`.

Evidence from `plot_inventory.csv`:
- H20 `longest_gt`: GT path length **~29–31 m over a 20-step window** (expected ~1 m). A pedestrian cannot
  travel 31 m in 20 frames — this is cumulative jitter (the grey GT renders as a fuzzy band, not a line).
- These same windows have huge ADE (12–18 m) and ratio ≈ 0.01–0.11, i.e. the metric blames the model for a
  noisy GT.
- `longest_gt` and `worst_collapse` are therefore **dominated by the noisiest tracks**, biasing
  pred/GT-based collapse statistics downward.

Implication: the headline `pred/GT ratio` (0.27–0.45) and `collapse_rate` numbers in the sweep
comparison **overstate** the model's length deficit, because the denominator (GT length) is partly noise.
The *direction-vs-distance* conclusion stands; the *magnitude* of "collapse" should be treated as a
loose upper bound, not a clean measurement. (Net-displacement-based length, or a jitter-smoothed GT, would
be a fairer denominator — flagged for a later phase, not changed here.)

## 9. Warnings

- **Metric contamination (above):** pred/GT ratio & collapse_rate are inflated by GT tracking jitter;
  interpret `longest_gt`/`worst_collapse` categories as "noisiest GT," not "longest real travel."
- **Thin sampling at long horizons:** at H400 the test pool is small (63 tracks; red_bridge only 3), so
  `longest_gt`/`largest_pred` there draw from very few trajectories — low diversity, treat as
  illustrative.
- **No missing files / no missing metadata** were encountered (`_audit_meta.json` warnings list empty).
- This audit re-ran inference only; it did **not** alter `run_horizon.py`, any checkpoint, scalers, the
  split, or the dataset.

## Output map

```
plots_audit/
├── PHASE_1_PLOTTING_AUDIT_REPORT.md   ← this file
├── plot_inventory.csv                 ← 180 rows (one per panel)
├── make_audit_plots.py                ← audit plotter (equal, adjustable=box)
├── _audit_meta.json                   ← counts + warnings
└── H{20,60,100,200,400}/
    └── {longest_gt,largest_pred,median_ade,worst_collapse}/
        ├── _grid_<category>.png       ← 3×3 contact sheet
        └── <category>_NN_<rec>_t<track>_w<window>.png   ← 9 annotated panels
```
