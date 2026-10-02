# Barcelona_v1 — Master Dataset Schema Report

**Phase:** Build master dataset for Frozen Model C (assembly + split audit only — no training).
**Date:** 2026-06-06
**Output folder:** `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_master_dataset`

## Source files (canonical encoded inputs)

| recording | source CSV |
|---|---|
| esplanade_espanya_01 | `Barcelona_v1_encoded/esplanade_espanya_01/spatial_v21C/trajectories_encoded.csv` |
| stairs_montjuic_01 | `Barcelona_v1_encoded/stairs_montjuic_01/spatial_v21C/trajectories_encoded.csv` |
| red_bridge_combined_01 | `Barcelona_v1_encoded/red_bridge_combined_01/spatial_v21C/trajectories_encoded.csv` |
| placa_catalunya_01 | `Barcelona_v1_encoded/placa_catalunya_01/spatial_v21C/trajectories_encoded.csv` |
| placa_espanya_01 | `Barcelona_v1_encoded/placa_espanya_01/spatial_v21C/trajectories_encoded.csv` |

Review-required recordings (`placa_montjuic_01`, `stairs_montjuic_02`) were **excluded**.

## Recipe provenance (frozen, not reinvented)

Assembly wrapper `mp-core/trajectory-prediction/build_barcelona_v1_master.py` **imports and
reuses** the frozen bridge recipe functions from
`experiments/schema_ablation_bridge/make_bridge_dataset.py`
(`per_track_features`, `rank_pct`, `wrap`, `MIN_TRACK_LEN = 11`) — the exact recipe that
froze Model C — applied **per recording**, then concatenated with recording-level splits.

**Model C feature set** confirmed from `run_bridge_ablation.py` →
`FEATURE_SETS["C_motion_position_spatial"]` = 10 features
(`du, dv, speed, heading_sin, heading_cos, turn_rate, u, v, dist_to_obstacle_norm,
dist_to_boundary_norm`). `entrance_affinity_norm` is the **Model D** feature and is
**excluded** from `model_C_dataset.csv` (kept diagnostic-only in `master_dataset.csv`).

## Exact schema

**`model_C_dataset.csv`** (16 columns — identifiers + 10 features + 2 targets):
```
recording_id, trajectory_id, timestep, split,
u, v,
du, dv, speed, heading_sin, heading_cos, turn_rate,
dist_to_obstacle_norm, dist_to_boundary_norm,
target_du, target_dv
```

**`master_dataset.csv`** (24 columns): the above **plus** diagnostic columns
`entrance_affinity_norm, src_track_id, frame, world_x, world_y,
dist_to_obstacle, dist_to_boundary, dist_to_entrance`.

## Normalization strategy (per-recording)

| feature | method | scope |
|---|---|---|
| `u`, `v` | MinMax of `world_x` / `world_y` → [0,1] | per recording |
| `du`, `dv`, `speed` | raw metres / step (NOT scaled; StandardScaler at train time) | — |
| `turn_rate` | wrapped heading delta, radians ∈ (−π, π] | per track |
| `dist_to_obstacle_norm` | percentile rank **ascending** (0=closest, 1=farthest) | per recording |
| `dist_to_boundary_norm` | percentile rank **ascending** | per recording |
| `entrance_affinity_norm` (diagnostic) | percentile rank **descending** (1=closest to entrance) | per recording |

> **Difference vs. the brief wording, resolved per the brief's own rule:** the frozen recipe
> normalizes the spatial clearances by **percentile rank**, not min-max. Per the brief
> ("use the same normalization method as the production bridge … if already implemented"),
> percentile rank was used to keep the schema faithful to Frozen Model C. All normalization
> is **per-recording** (the brief's explicit requirement; also matches the single-recording
> bridge applied independently per site).

## Per-recording row / track counts (track filter ≥ 11 frames)

| recording | split | rows in | rows used | tracks in | tracks used |
|---|---|---|---|---|---|
| esplanade_espanya_01 | train | 283,875 | 283,120 | 568 | 522 |
| placa_catalunya_01 | train | 358,351 | 355,583 | 1,537 | 1,255 |
| placa_espanya_01 | train | 213,618 | 212,224 | 923 | 824 |
| stairs_montjuic_01 | val | 128,259 | 127,605 | 392 | 334 |
| red_bridge_combined_01 | test | 87,076 | 85,847 | 716 | 599 |
| **TOTAL** | — | **1,071,179** | **1,064,379** | **4,136** | **3,810** |

## Per-split row / track counts

| split | recordings | rows | tracks |
|---|---|---|---|
| train | esplanade_espanya_01, placa_catalunya_01, placa_espanya_01 | 850,927 | 2,601 |
| val | stairs_montjuic_01 | 127,605 | 334 |
| test | red_bridge_combined_01 | 85,847 | 599 |
| **TOTAL** | 5 | **1,064,379** | **3,810** |

Split is **recording-level**: each recording belongs to exactly one split; no track or
recording crosses splits. `trajectory_id` is namespaced `"<recording_id>__<track_id>"`,
so IDs are globally unique.

## Missing / NaN / Inf checks (`model_C_dataset.csv`)

- NaNs: **0**
- Infinities: **0**
- All 16 required Model C columns present, in exact order.
- No forbidden columns (`entrance_affinity_norm`, `world_*`, raw `dist_*`) in the Model C matrix.

## Feature ranges (`model_C_dataset.csv`)

| column | mean | std | min | max |
|---|---|---|---|---|
| u | 0.6252 | 0.2401 | 0.0000 | 1.0000 |
| v | 0.5251 | 0.3443 | 0.0000 | 1.0000 |
| du | -0.0031 | 0.2159 | -17.0939 | 24.7912 |
| dv | 0.0003 | 0.0655 | -8.0498 | 9.6802 |
| speed | 0.0951 | 0.2046 | 0.0000 | 25.1094 |
| heading_sin | 0.0083 | 0.5593 | -1.0000 | 1.0000 |
| heading_cos | -0.0032 | 0.8289 | -1.0000 | 1.0000 |
| turn_rate | 0.0039 | 1.4624 | -3.1416 | 3.1416 |
| dist_to_obstacle_norm | 0.5003 | 0.2885 | 0.0000 | 1.0000 |
| dist_to_boundary_norm | 0.5003 | 0.2885 | 0.0019 | 1.0000 |
| target_du | -0.0032 | 0.2184 | -17.0939 | 24.7912 |
| target_dv | 0.0002 | 0.0665 | -8.0498 | 9.6802 |

- `u, v ∈ [0, 1]` ✔ · `dist_*_norm ∈ (0, 1]` ✔ (percentile-rank, mean ≈ 0.5 by construction).
- `heading_sin/cos ∈ [−1, 1]` ✔ · `turn_rate ∈ [−π, π]` ✔.
- **Target alignment verified:** `target_du[t] == du[t+1]` and `target_dv[t] == dv[t+1]`
  within every trajectory — **0 mismatches**. Final timestep of each track dropped (no target).

## Warnings

- **Single-step velocity outliers:** a small number of rows have large `|du|`/`|dv|`
  (up to ~25 m/step). These originate in the frozen encoded trajectories (far-field
  ID swaps / projection jitter), not in this assembly step. The frozen bridge recipe
  intentionally keeps `du/dv` in raw metres with **no speed clip** (normalization is
  deferred to a train-time StandardScaler), so this matches Model C's sandbox recipe.
  Flagged for awareness; not a blocker. Source CSVs were not modified.

## Model C schema validity

**VALID.** All 12 validation requirements pass (see `dataset_validation.json`):
all 5 recordings included, no review-required recordings, no NaN/Inf, exact Model C
columns, no forbidden columns, `u/v` and `dist_*_norm` in range, targets aligned to the
next timestep, recording-level split with each recording in exactly one split, and
globally-unique namespaced `trajectory_id`.

**SAFE FOR TRAINING: YES.**
