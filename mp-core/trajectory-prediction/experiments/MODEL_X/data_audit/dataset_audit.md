# MODEL_X — Dataset Audit

Generated: 2026-06-11. No assumptions — these are the actual on-disk facts.

## Candidate master datasets found

Exactly **two** Model C master datasets exist on disk (both 5 recordings, NOT 7):

| Version | Path | Rows | Tracks | Recordings |
|---|---|---:|---:|---:|
| Barcelona **v1** | `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_master_dataset\model_C_dataset.csv` | 1,064,379 | 3,534 | 5 |
| Barcelona **v3** (manual) | `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v3_manual_master_dataset\model_C_dataset.csv` | 1,064,379 | 3,534 | 5 |

Both are byte-for-byte identical in rows/tracks/motion/position/targets. They differ **only** in the two
spatial columns:
- **v1**: `dist_to_obstacle_norm`, `dist_to_boundary_norm` derived from old *trajectory-coverage* masks
  (the Encoding Truth Audit classified these as coverage artifacts, not real architecture).
- **v3**: same two columns derived from **Encoder V3 real manual architectural masks** (plan-assisted).
  This is the more recent, corrected dataset (built 2026-06-09).

## Per-recording breakdown (identical for v1 and v3)

| recording_id | rows | tracks | existing split |
|---|---:|---:|---|
| esplanade_espanya_01 | 283,120 | 522 | train |
| placa_catalunya_01 | 355,583 | 1,255 | train |
| placa_espanya_01 | 212,224 | 824 | train |
| stairs_montjuic_01 | 127,605 | 334 | val |
| red_bridge_combined_01 | 85,847 | 599 | test |
| **TOTAL** | **1,064,379** | **3,534** | recording-level |

> Note: the embedded `split` column is the OLD recording-level (unseen-site) split. MODEL_X will IGNORE it
> and build a fresh mixed-recording track-level split.

## Required columns check

All 12 required Model C feature/target columns + 4 ID columns are present in BOTH datasets. **None missing.**

```
IDs:      recording_id, trajectory_id, timestep, split
features: du, dv, speed, heading_sin, heading_cos, turn_rate,
          u, v, dist_to_obstacle_norm, dist_to_boundary_norm
targets:  target_du, target_dv
```

`target_du`/`target_dv` are present natively (next-step world displacement), so no derivation is needed.

## ⚠️ Recording-count discrepancy (5 available vs 7 requested)

The MODEL_X brief expects **7 recordings** including `placa_montjuic_01` and `stairs_montjuic_02`.
**These two do not exist in any master dataset and were never encoded.** Per
`docs/encoder_v3_recording_inventory.md` they are **EXCLUDED** due to ill-conditioned homography /
distorted world geometry (filtered bounds spanning 400+ m, thin diagonal streaks, detached clusters).
Their calibration must be reviewed/redone before they can be encoded — that is a separate, currently
blocked task. They cannot be added to MODEL_X without re-encoding.

**MODEL_X will therefore train on the 5 validated recordings**, which is the full set of trustworthy
Barcelona data currently available. This will be stated honestly in the report — MODEL_X is "best overall
prediction on the available validated Barcelona dataset," not a 7-site model.

## Model C architecture (confirmed, frozen)

Source: `experiments/schema_ablation_bridge/run_bridge_ablation.py:128` (`TrajectoryLSTM`).
Confirmed constants: `WINDOW_SIZE=10`, `HIDDEN_SIZE=128`, `NUM_LAYERS=2`, `DROPOUT=0.2`,
head `Linear(128→2)` → `(target_du, target_dv)`, MSE loss, seed 42. Matches the MODEL_X spec exactly.
