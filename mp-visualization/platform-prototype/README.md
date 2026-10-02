# motion pixels — platform prototype

A front-end only scaffold for the Motion Pixels thesis platform.
White, architectural aesthetic with colorful behavioral flow diagrams.

## What this is

A Vite + React prototype demonstrating the full Motion Pixels analysis pipeline as a user-facing interface. All data is mocked. No backend is connected.

## How to run

```bash
cd mp-visualization/platform-prototype
npm install
npm run dev
```

Opens at http://localhost:5173

## Workflow

```
Home → Upload footage + plan → Calibrate control points → Processing → Prediction canvas
```

Each step is a distinct screen. The processing step auto-advances after simulated pipeline completion.

## Aesthetic intent

- White background / faint gray plan geometry
- Colorful semi-transparent trajectory paths as the primary visual element
- Minimal UI: lowercase labels, monospace metadata, thin borders
- No dark dashboard look
- SVG-based prediction canvas

## Mock data

All behavioral data lives in `src/data/mockPredictionData.js`:

| Data | Description |
|---|---|
| `mockPaths` | Past pedestrian trajectory bezier curves |
| `mockPredictedPaths` | Predicted future paths (dashed) |
| `mockBottlenecks` | Congestion zone ellipses |
| `mockLingerZones` | Dwell area circles |
| `mockFlowVectors` | Directional flow arrows |
| `mockMetrics` | Summary statistics |

## Front-end only

- No network requests
- No real CSV parsing
- No Model C inference
- Processing step timer is cosmetic only

## Future integration points

When the backend pipeline is ready, replace mock data with:

| Mock | Real source |
|---|---|
| `mockPaths` | `mp-data/processed/trajectories/trajectories_world.csv` |
| `mockPredictedPaths` | Model C inference output |
| `mockBottlenecks` | `compute_bottlenecks.py` output |
| `mockLingerZones` | `compute_linger_zones.py` output |
| `mockFlowVectors` | `compute_flow_fields.py` output |
| Upload handlers | `mp-core/trajectory-extraction/track_people.py` |
| Calibration data | `mp-data/raw/calibration/calib_*.json` |
| Spatial reference | `mp-data/processed/rerun_*/inputs/top_view.png` |
| Encoding | `mp-data/processed/encoded/trajectories_encoded.csv` |

Model C lives in `mp-core/trajectory-prediction/frozen_model_C/` — do not retrain.
