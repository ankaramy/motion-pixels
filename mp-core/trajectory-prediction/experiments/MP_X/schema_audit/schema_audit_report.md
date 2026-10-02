# Schema Truth Audit — Model C dataset

Audit-only. Determines whether the exp01-04 turn-DIRECTION failure is a schema / coordinate / leakage / rollout bug or a genuine data limitation.

## Column resolution (logged, never silent)

```
track_id       -> 'trajectory_id'
frame_seq      -> 'timestep'
frame_raw      -> 'frame'
recording_id   -> 'recording_id'
world_x        -> 'world_x'
world_y        -> 'world_y'

Primary CSV (exp01-04 input): C:\Users\OWNER\Desktop\new_datasets\Barcelona_v3_manual_master_dataset\model_C_dataset.csv
  loaded 1,064,379 rows x 16 cols
Provenance superset: master_dataset.csv -> attached ['dist_to_obstacle_v3_m', 'dist_to_walkable_boundary_v3_m', 'frame', 'inbounds_v3', 'src_track_id', 'world_x', 'world_y']
  merge 1:1 ok=True, world_x non-null=True
manifest.json bounds for 5 recordings loaded
world_x/world_y: from master_dataset.csv (ground truth)
```

### SECTION 1 — Dataset provenance

- Absolute path: `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v3_manual_master_dataset\model_C_dataset.csv`
- Rows: **1,064,379**  ·  Tracks: **3,534**  ·  world coords: **master**
- Recordings: ['esplanade_espanya_01', 'placa_catalunya_01', 'placa_espanya_01', 'red_bridge_combined_01', 'stairs_montjuic_01']
- Splits: ['test', 'train', 'val']
- Columns: ['recording_id', 'trajectory_id', 'timestep', 'split', 'u', 'v', 'du', 'dv', 'speed', 'heading_sin', 'heading_cos', 'turn_rate', 'dist_to_obstacle_norm', 'dist_to_boundary_norm', 'target_du', 'target_dv', 'src_track_id', 'frame', 'world_x', 'world_y', 'dist_to_obstacle_v3_m', 'dist_to_walkable_boundary_v3_m', 'inbounds_v3', 'world_x_p1', 'world_x_n1', 'world_x_n2', 'world_y_p1', 'world_y_n1', 'world_y_n2', 'du_back', 'dv_back', 'du_fwd', 'dv_fwd', 'du_n2', 'dv_n2']
- Missing values: {'world_x_p1': 3534, 'world_x_n1': 3534, 'world_x_n2': 7068, 'world_y_p1': 3534, 'world_y_n1': 3534, 'world_y_n2': 7068, 'du_back': 3534, 'dv_back': 3534, 'du_fwd': 3534, 'dv_fwd': 3534, 'du_n2': 7068, 'dv_n2': 7068}
- Duplicate full rows: 0  ·  Duplicate keys (rec+track+timestep): 0
- Non-contiguous timestep tracks: 0 (timestep is a per-track 0..n-1 index; raw `frame` may legitimately have gaps)
- trajectory_id spanning >1 recording: 0 (IDs are recording-prefixed)
- Status: **PASS** 

Outputs: `provenance_table.csv`.

### SECTION 2 — Required Model C columns + statistics

| column | dtype | min | max | mean | std | nan | inf | zeros | p1 | p50 | p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| du | object | -17.1 | 24.8 | -0.00307 | 0.216 | 0 | 0 | 34023 | -0.594 | 0 | 0.58 |
| dv | object | -8.05 | 9.68 | 0.000261 | 0.0655 | 0 | 0 | 36129 | -0.152 | 0 | 0.137 |
| speed | object | 0 | 25.1 | 0.0951 | 0.205 | 0 | 0 | 25012 | 0 | 0.0317 | 0.83 |
| heading_sin | object | -1 | 1 | 0.00831 | 0.559 | 0 | 0 | 30251 | -0.997 | 0 | 0.997 |
| heading_cos | object | -1 | 1 | -0.00316 | 0.829 | 0 | 0 | 0 | -1 | 6.12e-17 | 1 |
| turn_rate | object | -3.14 | 3.14 | 0.00391 | 1.46 | 0 | 0 | 21075 | -3.13 | 0 | 3.13 |
| u | object | 0 | 1 | 0.625 | 0.24 | 0 | 0 | 4 | 0.113 | 0.658 | 0.986 |
| v | object | 0 | 1 | 0.525 | 0.344 | 0 | 0 | 3 | 0.0437 | 0.458 | 0.995 |
| dist_to_obstacle_norm | object | 0.00011 | 1 | 0.5 | 0.288 | 0 | 0 | 0 | 0.0235 | 0.499 | 0.986 |
| dist_to_boundary_norm | object | 0.000398 | 1 | 0.5 | 0.288 | 0 | 0 | 0 | 0.0124 | 0.5 | 0.986 |
| target_du | object | -17.1 | 24.8 | -0.00317 | 0.218 | 0 | 0 | 30510 | -0.598 | 0 | 0.583 |
| target_dv | object | -8.05 | 9.68 | 0.000222 | 0.0665 | 0 | 0 | 32613 | -0.153 | 0 | 0.138 |
| world_x | object | -131 | 206 | 3.81 | 49.2 | 0 | 0 | 5 | -73 | -1.54 | 125 |
| world_y | object | -94 | 54.2 | -6.53 | 10.2 | 0 | 0 | 1 | -34.2 | -5.85 | 21.5 |

- Status: **PASS** — all ranges valid

Outputs: `column_statistics.csv`.

### SECTION 3 — Recompute motion from world positions

- du stored vs backward diff: MAE 4.14e-17 m, maxabs 3.55e-15, corr 1.0000, sign-agree 1.000
- dv stored vs backward diff: MAE 4.50e-17 m, corr 1.0000, sign-agree 1.000
- speed stored vs hypot(du,dv): MAE 4.23e-17, corr 1.0000
- heading_sin/cos vs atan2(dv,du): MAE 2.38e-17/2.82e-17; mean cosine similarity **1.0000**; opposite-direction 0.00%
- du shift test MAE: backward(t-t-1)=4.14e-17, forward(t+1-t)=9.70e-02, t+2 step=1.13e-01, sign-flip backward=1.75e-01  -> best **backward(t-t-1)**
- Convention: stored `du/dv` = backward (incoming) displacement; `heading = atan2(dv, du)` (world coords).
- Status: **PASS** — motion columns match one consistent convention

Plots: `plots/heading_arrow_checks/`.

### SECTION 4 — Target alignment audit (critical)

| candidate | MAE du | MAE dv | MAE avg (m) | corr du | sign-agree du |
|---|---|---|---|---|---|
| A world[t+1]-world[t] (forward) | 4.14e-17 | 4.50e-17 | 4.32e-17 | 1.0000 | 1.000 |
| C world[t+2]-world[t+1] | 9.70e-02 | 2.57e-02 | 6.13e-02 | 0.2709 | 0.700 |
| B world[t]-world[t-1] (backward) | 9.74e-02 | 2.59e-02 | 6.16e-02 | 0.2671 | 0.700 |
| D sign-flip forward | 1.75e-01 | 4.81e-02 | 1.12e-01 | -1.0000 | 0.029 |

- Best alignment: **A world[t+1]-world[t] (forward)** (MAE 4.32e-17 m).
- Cross-check target_du[t] == du[t+1]: MAE 0.00e+00 (forward target == next incoming displacement, as expected).
- Convention: a window ending at t predicts t->t+1 displacement. CONFIRMED.
- Status: **PASS** — targets are the correct next-step forward displacement

Outputs: `target_alignment_report.csv`, `plots/target_alignment_checks/`.

### SECTION 5 — Heading & turn_rate audit

| hypothesis | MAE | corr | sign-agree |
|---|---|---|---|
| 1 correct wrap(Δheading) | 7.759e-03 | 0.989 | 0.998 |
| 2 sign-flipped | 1.836e+00 | -0.989 | 0.017 |
| 3 shifted +1 | 1.575e+00 | -0.188 | 0.484 |
| 4 shifted -1 | 1.577e+00 | -0.191 | 0.482 |
| 5 degrees | 5.192e+01 | 0.989 | 0.998 |
| 6 swapped sin/cos (atan2(du,dv)) | 1.823e+00 | -0.968 | 0.021 |

- Best hypothesis: **1 correct wrap(Δheading)**.  Correct-wrap: MAE 7.76e-03 rad, corr 0.9887, sign-agree 0.998.
- Status: **PASS** — turn_rate is the circular wrap(Δheading), correct sign & scale

Plots: `plots/turn_rate_checks/`.

### SECTION 6 — Coordinate convention audit

| convention | mean cosine sim |
|---|---|
| normal atan2(dy,dx) | 1.0000 |
| y-flipped atan2(-dy,dx) | 0.3628 |
| x-flipped atan2(dy,-dx) | -0.3628 |
| swapped atan2(dx,dy) | 0.0560 |
| swapped+yflip atan2(dx,-dy) | 0.0000 |

- Per-recording cosine sim (normal convention): esplanade_espanya_01=1.000, placa_catalunya_01=1.000, placa_espanya_01=1.000, red_bridge_combined_01=1.000, stairs_montjuic_01=1.000
- Best global convention: **normal atan2(dy,dx)**.
- **Key invariance note:** the angular-ERROR metric compares predicted vs GT heading in the SAME space, so a global axis flip/swap is convention-invariant and CANNOT by itself cause near-chance angular error. A flip would corrupt visual left/right but not the ≈90° error seen in exp01-04.
- Status: **PASS** — single consistent normal convention across all recordings

Plots: `plots/coordinate_convention_checks/`.

### SECTION 7 — Normalization audit

- u recomputed = (world_x-xmin)/xrng: MAE 1.52e-17; v: MAE 2.99e-17.
- Per-recording u in [0,1]: True  -> normalization is **per-recording** (each recording min-max scaled by its own world_bounds).
- Spatial norm features: dist_to_obstacle_norm: corr(matching raw)=0.56 vs corr(other raw)=0.46 constant=False; dist_to_boundary_norm: corr(matching raw)=0.56 vs corr(other raw)=0.50 constant=False
- Scaler scope (from exp configs): feature/target StandardScalers are fit on TRAIN rows only; u/v min-max uses each recording's own bounds. No val/test statistics enter the training normalization.
- Status: **PASS** — normalization consistent, per-recording, no split leakage

Plots: `plots/normalization_checks/`.

### SECTION 8 — Data leakage audit

- Input features (exp01-03): `['du', 'dv', 'speed', 'heading_sin', 'heading_cos', 'turn_rate', 'u', 'v', 'dist_to_obstacle_norm', 'dist_to_boundary_norm']` — targets present in input: **NO**; future-heading in input: **NO**.
- Split leakage: tracks across splits **0**, recordings across splits **0** (recording-level split).
- Duplicate identical tracks: total 0, across splits **0** (overfit10x check).
- Highest |feature↔target| corr: du=0.27, dv=0.23, heading_cos=0.17, heading_sin=0.14 (du↔target_du is lag-1 velocity persistence — legitimate signal, not leakage).
- Status: **PASS** — no leakage: clean recording-level split, no target/future inputs

Outputs: `leakage_report.md`, `duplicate_window_report.csv`, `feature_target_correlation.csv`.

### SECTION 9 — Turn label truth audit

| H | disp floor | total wins | genuine | mild | sharp | uturn | %artifact (maxstep>0.6) | GT turn° med |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.5 | 1,015,407 | 50,288 | 13163 | 9635 | 27490 | 6.2% | 157° |
| 10 | 1.0 | 999,179 | 32,391 | 7192 | 6247 | 18952 | 9.5% | 161° |
| 20 | 2.0 | 969,140 | 10,683 | 2940 | 2265 | 5478 | 13.6% | 152° |

- Held-out genuine turns: H5=1073, H10=442, H20=50 (exp03 probe expected H5=1073/H10=442/H20=50).
- Status: **PASS** — recomputed counts match exp03/exp01-04 labels

Plots: `plots/turn_label_checks/`. Output: `turn_label_counts.csv`.

### SECTION 10 — Rollout feedback audit (teacher-forced)

- Teacher-forced over 50 tracks (991 steps). Motion-feature max **p95 = 2.61e-14** (moving steps; du/dv/u/v/heading exact, turn_rate p95 ~3e-14) -> schemas identical; spatial p95 **1.49e-06**.
- The only non-machine-precision number is turn_rate's MEAN MAE (6.4e-03): rare π jumps at the first moving step after an idle step (dataset stores heading=0 at idle). Benign convention, not a schema bug.
- Status: **PASS** — benign convention note: at the first moving step after an idle step the dataset stores heading=0 while the rollout carries the previous heading, inflating turn_rate MEAN MAE to 6.4e-03 (p95 2.6e-14 -> 95%+ of steps are exact); 0.5% steps idle

Outputs: `rollout_feedback_report.md`, `teacher_forced_feature_error.csv`, `plots/rollout_feedback/`.

### SECTION 11 — Recording/split & turn-direction balance

| split | recordings | rows | tracks |
|---|---|---|---|
| train | esplanade_espanya_01;placa_catalunya_01;placa_espanya_01 | 850,927 | 2601 |
| val | stairs_montjuic_01 | 127,605 | 334 |
| test | red_bridge_combined_01 | 85,847 | 599 |

**Turn-direction balance (signed net heading change over genuine H=20 turns; left>0, right<0):**

| split | left | right | total | left-fraction |
|---|---|---|---|---|
| train | 5496 | 5137 | 10633 | 0.52 |
| val | 26 | 20 | 46 | 0.57 |
| test | 0 | 4 | 4 | 0.00 |

- Held-out genuine-turn concentration (top recording share): 0.92.
- Train left-fraction 0.52 vs test 0.00.
- Status: **WARNING** — TEST recording has only 4 genuine turns (held-out total 50) — held-out turn metric is statistically fragile; train left-fraction 0.52 vs test 0.00 (direction prior mismatch); held-out genuine turns 92% from one recording

Plots: `plots/coordinate_convention_checks/` (left_right, train_vs_test heading). Output: `signed_turn_directions.csv`.

### SECTION 12 — Visual truth panels

- Panel 1: 20 random trajectories, stored vs recomputed heading arrows.
- Panel 2: 20 genuine turns with GT heading-change, bin, and left/right sign.
- Panel 3: 12 most-suspicious rows (largest heading / turn_rate / target mismatch).
- Outputs: `plots/visual_truth_panels/`, `suspicious_rows.csv`, `suspicious_tracks.csv`.


## SECTION 13 — FINAL ROOT-CAUSE VERDICT

| Category | Status | Evidence | Consequence | Fix |
|---|---|---|---|---|
| Dataset provenance | **PASS** | 1,064,379 rows/3,534 tracks; dup_keys=0; cross_rec=0; noncontig=0 | keys unique & recording-scoped | none |
| Motion column stats | **PASS** | heading in [-1,1]=True; targets non-constant | value ranges valid | none |
| Motion columns du/dv | **PASS** | du MAE 4.1e-17m vs backward-diff; heading cos-sim 1.000; opp 0.00% | motion schema self-consistent | none |
| Target alignment | **PASS** | best=A world[t+1]-world[t] (forward); MAE 4.3e-17m; target==du[t+1] MAE 0.0e+00 | targets correctly aligned to t+1 | none |
| Turn rate | **PASS** | best='1 correct wrap(Δheading)'; MAE 7.8e-03rad; corr 0.989 | turn_rate circular & correct | none |
| Coordinate convention | **PASS** | best='normal atan2(dy,dx)'; all-rec normal>0.99=True | consistent world convention; angular error is convention-invariant | none |
| Normalization | **PASS** | u MAE 1.5e-17; per-rec [0,1]=True; dist_to_obstacle_norm: corr(matching raw)=0.56 vs corr(other | u/v & spatial norms consistent, no split leakage | none |
| Leakage | **PASS** | tgt_in=False; split_cross=0; dup_cross=0; maxcorr=0.27 | no future/target/split leakage | none |
| Turn labels | **PASS** | held-out genuine H5=1073, H10=442, H20=50; matches exp03 probe=True | labels reproduce exp01-04 exactly | none |
| Rollout feedback | **PASS** | moving-step motion p95 2.6e-14 (exact); spatial p95 1.5e-06; turn_rate mean MAE 6.4e-03 from idle-step heading=0 convention (benign) | train/rollout motion schema identical on moving steps | none |
| Recording/split balance | **WARNING** | TEST genuine turns=4, held-out total=50 (92% one recording); train L-frac 0.52 vs test 0.00 | held-out turn-direction metric is tiny, one-recording, direction-imbalanced -> unreliable | add turn-rich, direction-balanced recordings to the held-out/test set before judging turn direction |
| Visual truth panels | **PASS** | panels generated (random / genuine / suspicious) | visual inspection available | none |

### VERDICT

**DATA/SPLIT IMBALANCE LIKELY**

Every schema check is CLEAN — motion (du/dv match world backward-diff to ~1e-17 m), targets (forward t->t+1 to ~1e-17 m), turn_rate (circular wrap, corr 0.99), coordinate convention (normal atan2(dy,dx), all recordings), normalization (per-recording, no split leakage), leakage (no target/future inputs, recording-level split, no cross-split duplicates), turn labels (reproduce exp01-04), and the rollout motion updater (matches the dataset to machine precision on moving steps; only a benign stationary-step heading convention differs). The angular-error metric is also convention-invariant, so a coordinate flip cannot produce it. The decisive issue is the HELD-OUT TURN SAMPLE itself: TEST genuine turns=4, held-out total=50 (92% one recording); train L-frac 0.52 vs test 0.00. The test recording contributes almost no genuine turns and they are one-directional, so the 'near-chance angular error on held-out turns' is measured on a tiny, single-recording, direction-imbalanced sample and is statistically unreliable. This is a data/split sparsity confound, NOT a code/schema bug. Fix by adding turn-rich, direction-balanced recordings to the held-out/test set (and to training), then re-evaluate turn direction. The representational-limitation hypothesis from exp01-04 remains plausible but is currently UNDER-MEASURED on held-out data.

_Counts: 11 PASS · 1 WARNING · 0 FAIL · 0 ERROR._
