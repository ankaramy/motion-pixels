# behavior_maps

2D behavioral visualization for Motion Pixels — presentation-quality maps for an
architecture / thesis-jury audience (not data scientists).

These scripts are **visualization only**. They read existing pipeline outputs
(bottleneck CSVs, plan images, calibration) and re-render them cleanly. They do
**not** retrack, recalibrate, modify source trajectories, or rerun the pipeline.

## Scripts

### `generate_bottleneck_map_test.py`
Repair test for the Plaça Catalunya bottleneck map. Fixes the "tiny plan inside a
huge empty coordinate box" problem by treating the **plan image extent as the
authoritative visual frame**: cells outside the plan rectangle are masked out and
axis limits are locked to the plan corners. The bottleneck scores are Gaussian-
smoothed into a soft warm field (weak values fade to transparent), axes are
removed, and a small corner title + minimal colorbar are added.

```
python generate_bottleneck_map_test.py
```

Key options: `--top_k` (labeled zones), `--grid_res` (px/m of the smooth field),
`--sigma_m` (smoothing radius in metres), `--dpi`.

**Inputs (defaults):** `new_datasets/placa_catalunya_01/filtered_250m/bottlenecks/bottleneck_cells.csv`,
`.../plan/placa-catalunya.png`, `.../calibration/calib.json`.

**Outputs:** `outputs/placa_catalunya_bottleneck_test.png` and
`outputs/placa_catalunya_bottleneck_test_report.md`.

## Geometry note
The plan→world homography is rebuilt from the calibration's `plan_points_px` →
`world_points` correspondences, identical to
`mp-core/trajectory-extraction/compute_bottlenecks.py::_render_top_view_bg`, so
these overlays align with the existing pipeline without recalibration.

## Outputs
`outputs/` holds generated figures and per-figure reports. Safe to regenerate.
