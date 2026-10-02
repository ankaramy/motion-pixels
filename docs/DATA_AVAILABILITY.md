# Data availability

The repository contains everything needed to **understand, inspect and run** the final Motion Pixels model on a sample. It does **not** contain the large datasets, the large evaluation tables, or any raw video. Those are kept in the author's local archive, and can be published separately (for example as a GitHub Release or a Zenodo record).

## Why the large data is not in Git

- **Size.** The final master dataset alone is 379 MB, and the per-recording data totals about 2 GB. Git and GitHub handle files this size badly; GitHub hard-limits single files to 100 MB.
- **Privacy.** The source videos show identifiable people in public space in Barcelona. Raw, normalized and tracked-overlay footage is never committed. Published imagery should be blurred (`run_pipeline.py --blur_faces`).
- **Regenerability.** Most large tables are deterministic outputs of scripts that are in this repository.

## What is in Git

| Item | Path |
|---|---|
| Calibration (homography, `plan_scale_plus_correspondences`) for 7 recordings | `mp-data/recordings/<rec>/calibration/calib.json` (path fields are relative to the calib file) |
| Architectural plans | `mp-data/recordings/<rec>/plan/*.png` |
| Manual architectural masks (Encoder V3 input) | `mp-data/annotations/manual_masks_v3/<rec>/` |
| Master dataset manifest, schema, feature ranges, split report, validation | `mp-data/datasets/Barcelona_v3_manual_master_dataset/` |
| Encoder V3 report and per-recording metadata/diagnostics | `mp-data/datasets/Barcelona_v3_manual_encoded/` |
| Barcelona_v1 (v21C, Model C lineage) metadata | `mp-data/datasets/Barcelona_v1_*` |
| Track-level split used by MODEL_X / XR / XC | `mp-core/trajectory-prediction/experiments/MODEL_X/splits/model_x_track_split.csv` |
| Per-recording world bounds of the full dataset (used in rollout) | `mp-core/trajectory-prediction/experiments/MODEL_X/config/recording_world_bounds.json` |
| Sample: 25 test-split tracks (5 per recording), 9,464 rows | `mp-data/sample/barcelona_v3_test_sample.csv` |
| Final and comparison checkpoints with scalers | `mp-core/trajectory-prediction/MODEL_XC/`, `MODEL_XR/`, `experiments/MODEL_X/` |

Recordings: `esplanade_espanya_01`, `placa_catalunya_01`, `placa_espanya_01`, `red_bridge_combined_01`, `stairs_montjuic_01` (the 5 in the final dataset), plus `placa_montjuic_01` and `stairs_montjuic_02`, which were tracked and calibrated but excluded from encoding because their homographies were ill-conditioned.

## Large data kept outside Git

| Dataset (canonical name) | Size | Produced by | Needed for |
|---|---|---|---|
| `Barcelona_v3_manual_master_dataset/master_dataset.csv` | 379 MB (1,064,379 rows, md5 `173ba156f52774ec33f3962e7c99f38e`) | `build_barcelona_v3_master.py` | training/evaluating MODEL_X/XR/XC; most prediction visuals |
| `Barcelona_v3_manual_master_dataset/model_C_dataset.csv` | 311 MB | same | Model C column subset (optional) |
| `Barcelona_v3_manual_encoded/<rec>/spatial_v3_manual/trajectories_encoded_v3.csv` | ~90 MB each | `encoder_v3_manual/batch_encode_v3_manual.py` | input to the master builder |
| `<rec>/tracking/trajectories_world.csv` and `<rec>/filtered_250m/trajectories_world_filtered_250m.csv` | 20–84 MB each | `trajectory-extraction/run_pipeline.py` (+ 250 m filter) | encoding, behavior maps |
| `<rec>/tracking/`, `<rec>/filtered_250m/` behavior tables (speed, dwell, flow, bottleneck CSVs) | small–medium | `compute_*.py` | behavior maps |
| `Barcelona_v1_master_dataset/`, `Barcelona_v1_encoded/` | 668 MB / 277 MB | v21C encoder (Model C lineage) | historical only |
| Horizon-sweep and MODEL_X per-window tables (`per_window_metrics.csv`, `recording_audit_window_inventory.csv`) | 9–93 MB each | `MODEL_X_HORIZON_SWEEP/run_horizon.py`, audit scripts | `mp-visualization/final_metrics/compute_horizon_accuracy.py` |
| `MODEL_XR/curvature_audit/curvature_metrics.csv` | 73 MB | `build_curvature_audit.py` | curvature audit figures |
| Sandbox-era MACBA datasets (`mp-data/processed/…`) | ~750 MB | early pipeline | historical; still present in early Git history |
| Prototype demo video `prototype-official-v1/demo_assets/placa_espanya/video/tracked.mp4` | 527 MB | tracking overlay | prototype video panel |

## Where to put the data: `MP_DATA_ROOT`

Scripts that need large data read the environment variable `MP_DATA_ROOT`. If it is unset, they use `<repo>/mp-data/external/` (git-ignored). Expected layout:

```text
$MP_DATA_ROOT/
├── Barcelona_v3_manual_master_dataset/
│   └── master_dataset.csv                     ← required for full train/eval
├── Barcelona_v3_manual_encoded/<rec>/spatial_v3_manual/trajectories_encoded_v3.csv
└── <rec>/                                      e.g. placa_espanya_01
    ├── calibration/calib.json                  (optional; mp-data/recordings/ is used first)
    ├── plan/<plan>.png                         (optional; mp-data/recordings/ is used first)
    ├── tracking/trajectories_world.csv
    └── filtered_250m/trajectories_world_filtered_250m.csv
```

```bash
export MP_DATA_ROOT=/path/to/motion_pixels_data        # Windows PowerShell: $env:MP_DATA_ROOT="D:\mp_data"
python mp-core/trajectory-prediction/MODEL_XC/evaluate_model_xc.py --config configs/B_curv_light.json
```

Scripts wired to `MP_DATA_ROOT`: `experiments/MODEL_X/model_x_lib.py` (and therefore MODEL_X/XR/XC and their visuals), `encoder_v3_manual/v3_lib.py`, `manual_annotation/annotate_masks_v3.py`, `build_barcelona_v3_master.py`, `mp-visualization/behavior_maps/behaviormaps_final/generate_final_behavior_maps.py`. Historical experiment scripts keep their original hard-coded paths as provenance.

## World bounds

During autoregressive rollout the normalized position `u,v` is recomputed from predicted metres using each recording's world bounds. Those bounds come from the **full** master dataset and are persisted in `recording_world_bounds.json`. `model_x_lib.recording_world_bounds()` uses the persisted values, so predictions on the sample are identical to predictions on the full dataset. This was verified: the maximum difference was 0.0 m. Recomputing the bounds from the sample would have shifted predictions by up to 5.4 m.

## Raw video

The original recordings (iPhone `.MOV` plus one `.mp4`, about 4 GB in total) and every derived video showing pedestrians are excluded from the public repository on purpose. They are preserved in the author's private archive. Access for research purposes is at the author's discretion.
