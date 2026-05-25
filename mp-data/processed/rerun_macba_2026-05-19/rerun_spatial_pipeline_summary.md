# Spatial Pipeline Rerun — `macba_2026-05-19`

Read-only rerun against a new plan image. No existing outputs deleted; all rerun artefacts live under `mp-data/processed/rerun_macba_2026-05-19/`.

## Folder structure

```
mp-data/processed/rerun_macba_2026-05-19/
├── inputs/
│   ├── top_view.png             ← new plan image (user-supplied)
│   └── source_frame.png         ← optional override camera frame
├── calibration/
│   ├── calib_macba_2026-05-19.json        ← produced interactively
│   └── calibration_preview.png
├── trajectories/
│   ├── trajectories_world.csv   ← produced by calibrate_homography.py
│   └── trajectories_world_plot.png
├── spatial_v21C/
│   ├── trajectories_encoded.csv
│   ├── walkable_mask.png  obstacle_mask.png  boundary_mask.png
│   ├── entry_exit_points.csv
│   ├── distance_to_{obstacle,boundary,entrance}_m.png
│   └── summary.md
├── overlays/
│   ├── alignment_debug.png
│   ├── plan_overlay_masks.png
│   ├── plan_overlay_contact_sheet.png
│   └── overlay_summary.md
├── archive/                     ← reserved (nothing copied yet)
├── run_state.json
└── rerun_spatial_pipeline_summary.md   ← this file
```

## Repo inspection (Step 0)

| asset | path | status |
|---|---|---|
| `tracked_image_space_csv` | `mp-data\outputs\tracking\trajectories_image_space.csv` | ✓ present |
| `source_video` | `mp-data\raw\videos\input_macba.MOV` | ✓ present |
| `source_frame_image` | `mp-data\raw\images\frame.png` | ✓ present |
| `interactive_calibration_tool` | `mp-core\trajectory-extraction\calibrate_homography_interactive.py` | ✓ present |
| `apply_homography_tool` | `mp-core\trajectory-extraction\calibrate_homography.py` | ✓ present |
| `auto_v21C_encoder_script` | `mp-core\trajectory-prediction\encode_spatial_auto_v2.py` | ✓ present |
| `overlay_script` | `mp-core\trajectory-prediction\visualize_auto_v21_on_plan_v2.py` | ✓ present |

## Commands run (or to run)

**Step 1 — copy the new plan image** (user action):

```bash
# Linux/macOS
cp /path/to/your_new_plan.png \
   mp-data\processed\rerun_macba_2026-05-19\inputs\top_view.png

# Windows PowerShell
Copy-Item C:\path\to\your_new_plan.png C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\inputs\top_view.png
```

**Step 2 — fresh interactive calibration** (user action, ~5–10 min):

```bash
python mp-core\trajectory-extraction\calibrate_homography_interactive.py \
  --frame_image "C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\raw\images\frame.png" \
  --top_view_image "C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\inputs\top_view.png" \
  --out_json       "C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\calibration\calib_macba_2026-05-19.json" \
  --out_preview    "C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\calibration\calibration_preview.png"
```

Quality thresholds the tool prints itself: GOOD < 0.30 m · OK < 1.00 m · POOR ≥ 1.00 m.

**Step 3 — apply homography** (auto-run by this script):

```bash
python mp-core\trajectory-extraction\calibrate_homography.py \
  --traj_csv   "C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\outputs\tracking\trajectories_image_space.csv" \
  --calib_json "C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\calibration\calib_macba_2026-05-19.json" \
  --out_csv    "C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\trajectories\trajectories_world.csv" \
  --out_plot   "C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\trajectories/trajectories_world_plot.png"
```

**Step 4 — run v2.1C encoder** (auto-run; constants overridden in-process, original encoder file untouched).

**Step 5 — plan overlay + validation** (auto-run; same.)

## State

- ✓ step1_plan_image
- ✓ step2_calibration
- ✓ step3_apply_homography
- ✓ step4_encoder
- ✓ step5_overlay

## Calibration result

- pairs: **7**, inliers: **7**
- mean reprojection error: **0.9846 m**
- median: 0.7752 m   max: 1.762 m
- quality band (tool's bar): **OK**

## Overlay validation result

- Calibration self-reprojection: mean **0.00 px**, max **0.00 px**
- Trajectories inside plan image: **200/200** (100.0%)
- Convex hull of projected trajectories: **9.4%** of plan area
- Decision: **PASS** (OK)

**Visual verification (the hard part):** open `mp-data\processed\rerun_macba_2026-05-19\overlays\alignment_debug.png` and confirm the green trajectory points sit on the actual pedestrian plaza in the new image — not on roofs, roads, or empty pavement. Numerical pass alone is insufficient.

## Next recommended action

→ **Visually verify** `alignment_debug.png` against the new plan image. If trajectories sit on the actual plaza, the geometry rerun is complete and you can proceed to downstream prediction work using the new world CSV at `mp-data\processed\rerun_macba_2026-05-19\trajectories\trajectories_world.csv`.

## Constraints honoured

- No existing CSVs or outputs were deleted.
- `calib_macba.json` was NOT reused — Step 2 produces a fresh JSON.
- The encoder script `encode_spatial_auto_v2.py` and overlay script `visualize_auto_v21_on_plan_v2.py` were imported in-process with their module-level constants overridden — neither file was edited.
- No prediction model has been (or will be) trained as part of this rerun. Geometry + encoding validation only.

_End of rerun summary._