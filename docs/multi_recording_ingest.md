# Multi-Recording Ingest

## What is a recording?

A **recording** is one continuous video session at one location with its own calibration, plan image, tracking outputs, and spatial encoding. Each recording is an independent unit that can be included or excluded from training independently.

---

## Naming convention

```
<site>_<number>
```

Examples: `macba_01`, `raval_01`, `skate_02`, `corridor_01`

The folder name, `recording_id` in the manifest, and all filenames **must match**.

---

## Folder structure

```
mp-data/recordings/
├── recordings_manifest.csv          ← single source of truth
│
├── macba_01/
│   ├── raw_video/                   ← macba_01.mp4
│   ├── calibration/                 ← macba_01_calib.json
│   ├── plan/                        ← macba_01_plan.png
│   ├── tracking/                    ← tracking CSV output
│   ├── landmarks/                   ← landmark JSON/CSV
│   └── spatial_encoding/            ← trajectories_encoded.csv (v21C schema)
│
├── raval_01/
└── skate_02/
```

> The existing MACBA sandbox data lives at `mp-data/processed/rerun_macba_2026-05-19/` and is referenced via the manifest — it has **not** been moved.

---

## Manifest: `recordings_manifest.csv`

The manifest is the **only source of truth** for which recordings exist and how they are used. Scripts must never hardcode recording IDs; they must read from this file.

### Columns

| Column | Description |
|---|---|
| `recording_id` | Unique ID, must match folder name |
| `location` | Human-readable site name |
| `typology` | Space type (plaza, corridor, etc.) |
| `interior_exterior` | `interior` or `exterior` |
| `daytime` | `day` or `night` |
| `fps` | Frames per second of the video |
| `video_path` | Relative path to raw video |
| `plan_path` | Relative path to plan image |
| `calibration_path` | Relative path to calibration JSON |
| `tracking_csv_path` | Relative path to tracking CSV |
| `encoded_csv_path` | Relative path to v21C encoded CSV |
| `usable` | `yes` or `no` |
| `split` | `train`, `val`, `test`, or `pending` |
| `notes` | Free text |

All paths are **relative to the repo root**.

---

## Adding a new recording

1. Create the folder:
   ```
   mp-data/recordings/<site>_<number>/
   ```
   with subfolders: `raw_video/`, `calibration/`, `plan/`, `tracking/`, `landmarks/`, `spatial_encoding/`

2. Run the spatial encoding pipeline to produce `trajectories_encoded.csv` (v21C schema) and place it in `spatial_encoding/`.

3. Add a row to `recordings_manifest.csv`:
   - Set `usable=yes` once encoding is complete and reviewed.
   - Set `split` to `train`, `val`, or `test` (never `pending` for usable recordings).

4. Validate:
   ```
   python mp-core/trajectory-prediction/validate_recordings_manifest.py
   ```

5. Rebuild master dataset:
   ```
   python mp-core/trajectory-prediction/build_master_dataset.py
   ```

---

## Validation

```
python mp-core/trajectory-prediction/validate_recordings_manifest.py
```

Checks:
- Required columns are present
- `recording_id` values are unique
- `split` is one of `train / val / test / pending`
- `usable` is one of `yes / no`
- For every `usable=yes` row: `encoded_csv_path` exists on disk

Exits non-zero on failure.

---

## Building the master dataset

```
python mp-core/trajectory-prediction/build_master_dataset.py
```

- Reads manifest, selects `usable=yes` rows
- Loads each `encoded_csv_path`
- Adds `recording_id` and `split` columns
- Concatenates into one CSV

Output:
```
mp-data/processed/master_dataset/master_dataset.csv
mp-data/processed/master_dataset/master_dataset_summary.md
```

---

## Why split by recording?

Splitting by trajectory (random 80/20) leaks spatial context: the model sees trajectories from the same location in both train and test. This inflates metrics and hides poor generalization.

Correct approach — assign entire recordings to splits:

```
Good:  train = macba_01 + raval_01
       test  = skate_02

Bad:   random trajectories from macba_01 in both train and test
```

The manifest's `split` column enforces this at the data level, before any training code runs.

---

## What NOT to change

- **Model C** — frozen in `mp-core/trajectory-prediction/frozen_model_C/`
- **v21C feature schema** — 10-feature LSTM input; adding or removing features breaks the checkpoint
- **Training architecture** — no changes to `train_model.py` or model definition
- **Existing sandbox outputs** — `mp-data/processed/rerun_macba_2026-05-19/` is read-only reference data
