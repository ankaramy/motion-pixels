# Motion Pixels — Mapping Out Spatial Intelligence

Master's thesis by **Ramy Anka**: MaAI, Master in AI for Architecture and the Built Environment, Institute for Advanced Architecture of Catalonia (IAAC), Barcelona, 2025–2026. Advisor: Prof. Wassim Jabi.

Motion Pixels turns ordinary video of public space into calibrated pedestrian trajectories, behavioral maps (speed, flow, dwell, bottlenecks) and near-future trajectory predictions. The aim is to add observed movement to the evidence architects use to understand how people move through and respond to built space. The thesis is complete; this repository is its archival, reproducible codebase. Project context and principles: [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).

## Canonical pipeline

```text
site video
 → YOLOv8m + ByteTrack (high-recall config)            mp-core/trajectory-extraction/
 → image→plan homography calibration (calib.json)      mp-core/trajectory-extraction/
 → world-space trajectories (metres)
 → behavioral analysis: speed, dwell/linger, flow fields, bottlenecks, heatmaps
 → manual architectural masks → Encoder V3 spatial features   mp-core/trajectory-prediction/encoder_v3_manual/
 → Barcelona_v3_manual master dataset (5 recordings)
 → MODEL_X (baseline) → MODEL_XR (magnitude) → MODEL_XC (curvature)
 → FINAL MODEL: MODEL_XC_B_CURV_LIGHT
 → evaluation (H10 … H400), visualizations, thesis prototype   mp-visualization/
```

## Final model: `MODEL_XC_B_CURV_LIGHT`

| | |
|---|---|
| Architecture | 2-layer LSTM (hidden 128, dropout 0.2) → Linear(128→2) |
| Input (10 features, 10 observed steps) | `du, dv, speed, heading_sin, heading_cos, turn_rate, u, v, dist_to_obstacle_norm, dist_to_boundary_norm` |
| Output | next-step displacement (m), rolled out autoregressively |
| Loss | MSE + 1.0 · magnitude term + 0.1 · curvature term |
| Data | Barcelona_v3_manual (5 recordings), track-level 80/10/10 split |
| Test, H10 | ADE 0.332 m, FDE 0.533 m (90,310 windows) |
| Checkpoint | [`mp-core/trajectory-prediction/MODEL_XC/checkpoints/MODEL_XC_B_CURV_LIGHT/`](mp-core/trajectory-prediction/MODEL_XC/checkpoints/MODEL_XC_B_CURV_LIGHT) (`model_best.pth`, `scalers.json`, `sigma.json`) |
| Report | [MODEL_XC_REPORT.md](mp-core/trajectory-prediction/MODEL_XC/reports/MODEL_XC_REPORT.md) |

The model code is layered: MODEL_XC imports `MODEL_XR/xr_common.py`, which imports `experiments/MODEL_X/model_x_lib.py` (model class, scalers, rollout, config). All three are part of the final system.

**Limitations.** These are findings of the thesis, not defects to hide:
- The split is mixed-recording (track-level), so results do not show generalization to unseen sites.
- Prediction is most useful up to about H100 (≈5 m of travel). It degrades visibly from H200, so long horizons are coarse behavioral context, not precise paths.
- The model tends to undershoot magnitude and simplify trajectory shape.
- Spatial features are frozen during rollout.

See [HORIZON_ACCURACY_REPORT.md](mp-visualization/final_metrics/HORIZON_ACCURACY_REPORT.md). That report is computed from the per-horizon `MODEL_X_HORIZON_SWEEP` retrains; the MODEL_XC horizon results are in `MODEL_XC/experiments/MODEL_XC_B_CURV_LIGHT/horizon.json`.

## Repository structure

```text
mp-core/trajectory-extraction/    tracking, calibration, reconstruction, behavior metrics, overlays
mp-core/trajectory-prediction/
  MODEL_XC/                       FINAL model: train / evaluate / horizon / predict_example.py
  MODEL_XR/                       magnitude stage (runtime dependency of XC)
  experiments/MODEL_X/            model library, config, split, world bounds (runtime dependency)
  experiments/                    horizon sweep + research experiments (code; outputs archived)
  encoder_v3_manual/, manual_annotation/, build_barcelona_v3_master.py   dataset construction
  presentation_visuals/           MODEL_X vs XR vs XC comparison visuals
  frozen_model_C/                 historical Model C (sandbox era)
mp-data/
  recordings/<rec>/               calib.json + architectural plan per recording
  annotations/manual_masks_v3/    manual walkable/obstacle masks (Encoder V3)
  datasets/                       dataset manifests, schema, feature ranges, encoder metadata
  sample/                         small test-split sample to run the final model
mp-visualization/
  final_metrics/, plots_wassim/, ready_animations/, behavior_maps/, hero_visuals/
  prototype-official-v1/          official frozen thesis prototype (static web app)
  platform-prototype/             earlier Vite/React demonstrator (mock data)
  presentation_animations/        motion-graphics scripts used in the presentation
THESIS/                           thesis text, chapters, figures, research notes
docs/                             technical documentation (data availability, ingest, briefs)
publication_final/                booklet build script and design system
```

## Installation

Use Python **3.10** (developed with 3.10.6).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
# 1) PyTorch: pick ONE
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu     # CPU
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
# 2) everything else
pip install -r requirements.txt
# optional: thesis/booklet builders
pip install -r requirements-publication.txt && playwright install chromium
```

All model code runs on CPU; CUDA only speeds it up. `requirements-freeze-py310-cu121.txt` records the exact original environment.

## Quick start: run the final model

```bash
cd mp-core/trajectory-prediction/MODEL_XC
python predict_example.py                      # 25 sample tracks, H10 → ADE/FDE
python predict_example.py --horizon 100 --plot pred.png
```

To reproduce the full evaluation, get the master dataset (see [docs/DATA_AVAILABILITY.md](docs/DATA_AVAILABILITY.md)), set `MP_DATA_ROOT`, then run:

```bash
python evaluate_model_xc.py --config configs/B_curv_light.json     # H10 test metrics
python run_horizon_xc.py   --config configs/B_curv_light.json      # H20–H400
python train_model_xc.py   --config configs/B_curv_light.json      # retrain (writes checkpoints/; back up first)
```

## Processing a new site

**1. Track and reconstruct.** Use the locked production configuration:

```bash
cd mp-core/trajectory-extraction
python run_pipeline.py --video site.MOV --calib_json calib.json --top_view_image plan.png \
  --model_path yolov8m.pt --tracker_cfg bytetrack_high_recall.yaml --imgsz 1920 --conf 0.10 \
  --run_flow_fields --run_bottlenecks --run_linger_zones --blur_faces
```

`yolov8m.pt` is downloaded automatically by Ultralytics. Outputs: `tracks`, `trajectories_world.csv`, metrics, and an overlay video.

**2. Calibrate.** Pick plan↔frame correspondences interactively:

```bash
python calibrate_homography_interactive.py --video site.MOV --plan_image plan.png --out_json calib.json
```

Example calibrations are in `mp-data/recordings/<rec>/calibration/`.

**3. Behavioral analysis.** Run on any `trajectories_world.csv`:

```bash
python compute_metrics.py      --traj_csv trajectories_world.csv --out_dir metrics
python compute_flow_fields.py  --traj_csv trajectories_world.csv --out_dir flow   --calib_json calib.json --top_view_image plan.png
python compute_bottlenecks.py  --traj_csv trajectories_world.csv --out_dir bottlenecks --calib_json calib.json --top_view_image plan.png
python compute_linger_zones.py --traj_csv trajectories_world.csv --out_dir linger --calib_json calib.json --top_view_image plan.png
```

**4. Spatial encoding.** Annotate masks with `manual_annotation/annotate_masks_v3.py`, then encode with `encoder_v3_manual/batch_encode_v3_manual.py`, then build the dataset with `build_barcelona_v3_master.py`. Details: [encoder_v3_manual/README.md](mp-core/trajectory-prediction/encoder_v3_manual/README.md).

## Visualization

| Output | Script |
|---|---|
| Horizon accuracy summary | `mp-visualization/final_metrics/compute_horizon_accuracy.py` |
| Final-model accuracy plots, long-horizon stress test | `mp-visualization/plots_wassim/` ([README](mp-visualization/plots_wassim/README.md)) |
| MODEL_X / XR / XC comparison stills | `mp-core/trajectory-prediction/presentation_visuals/make_presentation_visuals.py` |
| Final behavior maps (speed, flow, bottlenecks) | `mp-visualization/behavior_maps/behaviormaps_final/generate_final_behavior_maps.py` |
| Best/worst/stress animations | `mp-visualization/ready_animations/`: outputs + report. Its drawing template helper was lost; see [its README](mp-visualization/ready_animations/README.md) |

The style rules are in [MOTION_PIXELS_VISUALIZATION_TEMPLATE.md](mp-visualization/MOTION_PIXELS_VISUALIZATION_TEMPLATE.md). Most visual generators need the full data via `MP_DATA_ROOT`.

## Prototype

`mp-visualization/prototype-official-v1/` is the **official frozen thesis prototype**: upload → calibrate → encode → process → studio, on real Plaça Espanya assets. It is a static app, so no build step is needed:

```bash
cd mp-visualization/prototype-official-v1 && python -m http.server 5178
# open http://127.0.0.1:5178/?screen=home   (Chromium recommended)
```

The 527 MB tracked demo video is not in Git (see the data availability doc), so the video panel stays empty without it. The prototype replays precomputed outputs; it is a demonstrator, not a connected production system. `mp-visualization/platform-prototype/` is the earlier Vite/React demonstrator (`npm ci && npm run dev`), which uses mock data.

## Data availability and privacy

Large datasets, evaluation tables and **all raw video** are deliberately kept out of Git, for repository-size and privacy reasons: the footage shows identifiable people in public space. The repository contains the calibrations, plans, manual masks, dataset metadata, a small sample, and the persisted world bounds the model needs. See **[docs/DATA_AVAILABILITY.md](docs/DATA_AVAILABILITY.md)** for what exists, how each item was produced, and where to place it (`MP_DATA_ROOT`). The tracking pipeline provides `--blur_faces` for any published video.

## Citation

See [CITATION.cff](CITATION.cff).

> Anka, R. (2026). *Motion Pixels: Mapping Out Spatial Intelligence.* Master's thesis, MaAI, Institute for Advanced Architecture of Catalonia (IAAC), Barcelona.

No open-source license has been selected yet. All rights are reserved by the author until a LICENSE file is added.
