# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

**MotionPixels Prediction** is a three-step pipeline that takes pedestrian tracking data, enriches it with spatial context, trains an LSTM, and visualises predicted future trajectories on a floor-plan image.

It sits inside a parent directory alongside **trajectory-extraction**, which performs the upstream tracking. This pipeline picks up from trajectory-extraction's output.

---

## Pipeline steps

### Step 1 — Encode (`encode_space.py`) — interactive
Opens `top-down.png` in an OpenCV window. The user clicks three sets of spatial landmarks:
- **Obstacles** (Phase 1): physical barriers. Auto-seeded from trajectory-extraction's `bottleneck_cells.csv` if available.
- **Boundaries** (Phase 2): the spatial extent of the observed area.
- **Entrances** (Phase 3): entry/exit points.

For each trajectory row, three distance columns are computed (nearest-neighbour in world metres): `dist_to_obstacle`, `dist_to_boundary`, `dist_to_entrance`.

Output: `outputs/trajectories_encoded.csv`

**Window controls:**
| Key | Action |
|-----|--------|
| Left-click | Add point |
| Right-click | Remove last point |
| A | Toggle accept auto-seeded bottleneck points as obstacles (Phase 1 only) |
| C | Clear all clicked points in current phase |
| R | Toggle reference overlay (trajectory dots + calibration crosses) |
| N | Confirm phase, advance to next |
| Q | Finish (must be in Phase 3) |

### Step 2 — Train (`train_model.py`)
Trains a single-layer LSTM (PyTorch) on the encoded trajectories.

- Input features (8 per timestep): `world_x`, `world_y`, `dist_to_obstacle`, `dist_to_boundary`, `dist_to_entrance`, `frame_number`, `delta_x`, `delta_y`
- Target: next `(world_x, world_y)`
- Window size: 10 frames → predict frame 11
- Filters out near-stationary windows (`< 0.3 m` total displacement)
- Uses `MinMaxScaler`; both feature and target scalers are saved to `scaler.pkl`

Outputs: `outputs/trajectory_model.pth`, `outputs/scaler.pkl`, `outputs/loss_curve.png`

Hyperparameters to tune are defined as module-level constants at the top of the file: `WINDOW_SIZE`, `HIDDEN_SIZE`, `LEARNING_RATE`, `EPOCHS`, `BATCH_SIZE`, `MIN_DISPLACEMENT_M`.

### Step 3 — Visualise (`visualise_prediction.py`)
Loads the trained model and runs autoregressive prediction for the person with the most tracked frames.

- Draws on `top-down.png`: green = full real path, orange = 10-frame seed, red = 30 predicted steps
- Overlays trajectory-extraction flow field arrows and bottleneck heat circles if those CSVs exist
- Autoregressive rollout: each predicted position is fed back as input for the next step; non-positional features (`dist_*`, `frame_number`) are carried forward from the last seed frame

Output: `outputs/prediction_visual.png` + an OpenCV display window (press any key to close)

---

## Run commands

```bash
# Full pipeline (interactive — Step 1 opens an OpenCV window)
python run_pipeline.py

# Skip encoding (re-use existing trajectories_encoded.csv)
python run_pipeline.py --skip-encode

# Re-train + re-visualise only
python run_pipeline.py --from-step train

# Re-visualise only
python run_pipeline.py --from-step visualise

# Skip the confirmation prompt
python run_pipeline.py --yes

# Run individual steps directly
python encode_space.py
python train_model.py
python visualise_prediction.py
```

---

## Prerequisites

trajectory-extraction must be run first to generate the upstream data:

```bash
cd ../trajectory-extraction
python run_pipeline.py --video input_skate_1.MOV \
    --calib_json calib.json \
    --top_view_image YOUR_TOPVIEW_SKATE.png \
    --run_bottlenecks --run_flow_fields
```

This pipeline looks for these files from trajectory-extraction (falls back to `outputs/` if not found):
- `../trajectory-extraction/outputs/trajectories_world.csv` — required
- `../trajectory-extraction/outputs/behavior/bottlenecks/bottleneck_cells.csv` — optional, enables auto-seeding and bottleneck overlay
- `../trajectory-extraction/outputs/behavior/flow_fields/flow_field_cells.csv` — optional, enables flow field overlay

---

## Calibration (`calib.json`)

`calib.json` is the **skate-video calibration** (`input_skate_1.MOV`). It must match the video that generated the trajectory data.

Key values:
- `meters_per_plan_pixel`: `0.05431190417740396`
- `plan_origin_pixel`: `[160.875, 523.41]`
- `invert_plan_y`: `false`

**Critical**: `calib.json` contains a `homography_matrix` that maps *video-frame pixels* to world metres. This matrix must **NOT** be applied to clicks on `top-down.png`. The plan image uses a simple linear (orthographic) mapping:

```
world_x = (plan_px_x - origin_x) * mpp
world_y = (plan_px_y - origin_y) * mpp   # negated if invert_plan_y=true

# Inverse:
plan_px_x = world_x / mpp + origin_x
plan_px_y = world_y / mpp + origin_y
```

The expected trajectory world range for this dataset is approximately `X=[0, 14] m`, `Y=[-19, 29] m`, placing trajectories in the center-left area of `top-down.png`.

---

## File layout

```
trajectory-prediction/
├── encode_space.py          # Step 1 — interactive spatial encoding
├── train_model.py           # Step 2 — LSTM training
├── visualise_prediction.py  # Step 3 — prediction + visualisation
├── run_pipeline.py          # Orchestrates all three steps
├── calib.json               # Skate-video calibration (mpp + origin)
├── top-down.png             # Floor-plan image (892 × 1287 px)
└── outputs/
    ├── trajectories_world.csv   # Copy from trajectory-extraction (or symlinked)
    ├── trajectories_encoded.csv # After Step 1
    ├── trajectory_model.pth     # After Step 2
    ├── scaler.pkl               # After Step 2 (feat + target scalers bundled)
    ├── loss_curve.png           # After Step 2
    └── prediction_visual.png    # After Step 3
```

---

## Architecture notes

### `TrajectoryLSTM`
Defined identically in both `train_model.py` and `visualise_prediction.py` — if you change the architecture in one file you must update the other. Architecture: `Input(8) → LSTM(128 hidden, 1 layer) → Linear(2)`. Only the final hidden state is used (not the full output sequence).

### Scaler bundle (`scaler.pkl`)
Contains a dict with keys `feature_scaler`, `target_scaler`, `feature_cols`, `target_cols`, `window_size`. Both `train_model.py` and `visualise_prediction.py` must use the same feature column order — this order is frozen in `FEATURE_COLS` in `train_model.py` and must be kept consistent.

### Column name normalisation
Trajectories from trajectory-extraction use `track_id` / `frame`. This pipeline internally uses `person_id` / `frame_number`. Both `encode_space.py` and `train_model.py` silently rename these columns on load if needed.

### Reference overlay in `encode_space.py`
`build_reference_layer()` blends trajectory world positions (faint green dots) and calibration correspondence points (yellow crosses) at 35% opacity onto the plan image. This is displayed as the window background so the user can verify calibration alignment before clicking landmarks.

### Auto-seed separation
Auto-seeded bottleneck locations are kept as two parallel structures: `auto_obs_world` (float64 world metres, used for distance computation) and `auto_obs_px` (integer plan pixels, used only for display). They are never round-tripped through each other to avoid quantisation loss.
