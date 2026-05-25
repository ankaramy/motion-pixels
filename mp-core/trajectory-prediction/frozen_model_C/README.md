# Frozen Model C — sandbox-exit snapshot

Date frozen: 2026-05-21

**Model C** = `C_motion_position_spatial` from the schema ablation bridge — 10-input-feature LSTM (motion kinematics + normalised world position + obstacle/boundary clearance). Selected at sandbox exit as the production architecture for the next phase.

## Architecture (both checkpoints share it)

| Field | Value |
|---|---|
| Class | `TrajectoryLSTM` (see `experiments/schema_ablation_bridge/run_bridge_ablation.py`, defined at line 128) |
| Input features | 10 |
| Hidden size | 128 |
| Num layers | 2 |
| Dropout | 0.2 |
| Head | `Linear(128 → 2)` predicting `(target_du, target_dv)` in m/step |
| Window | 10 frames |
| Rollout horizon | 20 steps, fully autoregressive, spatial features re-queried via KDTree-IDW |
| Loss | MSE on `(target_du, target_dv)` only |
| Seed | 42 |

Feature order (frozen):
```
du, dv, speed, heading_sin, heading_cos, turn_rate,
u, v,
dist_to_obstacle_norm, dist_to_boundary_norm
```

## What lives in this directory

```
frozen_model_C/
├── README.md                          ← this file
├── held_out/                          ← real generalization checkpoint
│   ├── best_model.pth                 ← weights from schema_ablation_bridge
│   ├── scalers.pkl                    ← per-model feat+target StandardScalers (dict keyed by model name)
│   ├── schema_summary.json            ← v21C MACBA rerun encoding metadata
│   └── ablation_results.csv           ← A/B/C/D row; C row is the one this checkpoint reports
└── overfit10x/                        ← capacity-check checkpoint (memorisation, NOT generalization)
    ├── best_model.pth                 ← weights from schema_ablation_bridge_overfit10x
    ├── scalers.pkl                    ← (same shape as held_out)
    ├── schema_summary.json            ← 10× duplicated MACBA rerun
    └── ablation_results.csv
```

`scalers.pkl` is the original four-model bundle — load it and key by `"C_motion_position_spatial"` to get the matching feature + target scalers:

```python
import pickle
with open("held_out/scalers.pkl", "rb") as f:
    bundle = pickle.load(f)
feat_scaler, tgt_scaler = bundle["C_motion_position_spatial"]
```

## Reported metrics (carried over from the source experiments)

| Checkpoint | Source | ADE (m) | FDE (m) | Mean ang err (rad) | N test trajs | Interpretation |
|---|---|---|---|---|---|---|
| `held_out` | `schema_ablation_bridge/` | 0.532 | 0.981 | 0.933 | 9 | Real generalization on 9 unseen MACBA tracks. Honest but small-N — A/B/C/D are within noise of each other; this is the data-volume bottleneck the exit brief discusses. |
| `overfit10x` | `schema_ablation_bridge_overfit10x/` | 0.053 | 0.110 | 0.569 | 73 | OVERFIT10X capacity check — train/val/test share trajectory content by construction. Memorisation only; do **not** cite as held-out performance. |

## How to load

```python
import torch
from pathlib import Path
import sys, pickle

# Reuse the TrajectoryLSTM class from the source experiment
sys.path.insert(0, str(Path("...") / "mp-core/trajectory-prediction/experiments/schema_ablation_bridge"))
from run_bridge_ablation import TrajectoryLSTM

model = TrajectoryLSTM(n_feat=10)
state = torch.load("held_out/best_model.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()

with open("held_out/scalers.pkl", "rb") as f:
    feat_scaler, tgt_scaler = pickle.load(f)["C_motion_position_spatial"]
```

## Why both checkpoints

The held-out checkpoint is the **truthful** reference: real generalization, real (small) test set. The overfit10x checkpoint is the **capacity** reference: it shows the same architecture *can* fit the data when there are enough samples. Together they bracket where the next phase needs to push — the gap between them is the data-volume gap that real-video dataset expansion is meant to close.

## Provenance — do not edit these checkpoints in place

These are frozen sandbox-exit artefacts. Any further training, finetuning, or retraining should land in a new experiment directory; do not overwrite files in this folder.

- Held-out training run: `mp-core/trajectory-prediction/experiments/schema_ablation_bridge/run_bridge_ablation.py`
- Overfit10x training run: `mp-core/trajectory-prediction/experiments/schema_ablation_bridge_overfit10x/run_overfit10x_ablation.py`
- Source encoding: `mp-data/processed/rerun_macba_2026-05-19/spatial_v21C/trajectories_encoded.csv` (v21C, MACBA rerun, 52 base tracks)
