# Automatic Spatial Encoding — Diagnostic Report

_Read-only audit of the current automatic spatial encoding system._
_No retraining, no edits to encoders, no overwrites of CSVs._

## 1. File inventory

**CSVs:**

- ✓ `auto_encoded` — `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\encoded\trajectories_encoded_auto.csv` (44,396,655 bytes)
- ✓ `manual_encoded` — `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\encoded\trajectories_encoded.csv` (45,780,880 bytes)
- ✓ `motion_dataset` — `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\encoded\motion_dataset.csv` (49,513,695 bytes)
- ✓ `motion_dataset_v2` — `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\encoded\motion_dataset_v2.csv` (41,022,647 bytes)

**Other artefacts:**

- ✓ `spatial_maps_dir` — `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\encoded\spatial_feature_maps`
- ✓ `debug_overlay` — `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\encoded\debug_overlay.png`
- ✓ `feature_corr` — `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\encoded\feature_correlation.png`
- ✓ `feature_hist` — `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\encoded\feature_histograms.png`
- ✓ `diagnostics_md` — `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\encoded\diagnostics_summary.md`
- ✓ `feature_summary` — `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\encoded\feature_summary_v2.md`
- ✓ `schema_summary` — `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\encoded\schema_summary.json`

**Spatial feature map files:**

- `mp-data\processed\encoded\spatial_feature_maps\dist_to_boundary_norm.png`
- `mp-data\processed\encoded\spatial_feature_maps\dist_to_obstacle_norm.png`
- `mp-data\processed\encoded\spatial_feature_maps\openness_diff_left.png`
- `mp-data\processed\encoded\spatial_feature_maps\openness_diff_right.png`
- `mp-data\processed\encoded\spatial_feature_maps\speed.png`
- `mp-data\processed\encoded\spatial_feature_maps\turn_rate.png`

**Code files (keyword-matched in trajectory-prediction/):**

- `mp-core\trajectory-prediction\build_dataset_v2.py`
- `mp-core\trajectory-prediction\diagnose_auto_spatial_encoding.py`
- `mp-core\trajectory-prediction\diagnose_motion_dataset.py`
- `mp-core\trajectory-prediction\duplicate_trajectories.py`
- `mp-core\trajectory-prediction\encode_space.py`
- `mp-core\trajectory-prediction\encode_spatial_auto.py`
- `mp-core\trajectory-prediction\experiments\ablation\rollout_horizon_audit.py`
- `mp-core\trajectory-prediction\experiments\ablation\run_ablation.py`
- `mp-core\trajectory-prediction\experiments\ablation\visualize_ablation.py`
- `mp-core\trajectory-prediction\experiments\overfit-10x\compare_results.py`
- `mp-core\trajectory-prediction\experiments\overfit-10x\diagnose_spatial_features.py`
- `mp-core\trajectory-prediction\experiments\overfit-10x\eval_phase2b_final_multi.py`
- `mp-core\trajectory-prediction\experiments\overfit-10x\make_comparison_figure.py`
- `mp-core\trajectory-prediction\experiments\overfit-10x\make_overfit_dataset.py`
- `mp-core\trajectory-prediction\experiments\overfit-10x\rollout_bug_audit.py`
- `mp-core\trajectory-prediction\experiments\overfit-10x\train_gru_overfit.py`
- `mp-core\trajectory-prediction\experiments\overfit-10x\train_lstm_overfit.py`
- `mp-core\trajectory-prediction\experiments\overfit-10x\train_phase2a.py`
- `mp-core\trajectory-prediction\experiments\overfit-10x\train_phase2b.py`
- `mp-core\trajectory-prediction\experiments\overfit-10x\train_phase2b_final.py`
- `mp-core\trajectory-prediction\experiments\overfit-10x\train_phase2b_transformer.py`
- `mp-core\trajectory-prediction\experiments\overfit-10x\train_phase2c_fixed_spatial.py`
- `mp-core\trajectory-prediction\experiments\overfit-10x\vis_phase2b_final_plan_view.py`
- `mp-core\trajectory-prediction\experiments\rollout-fix\01_target_distribution_audit.py`
- `mp-core\trajectory-prediction\experiments\rollout-fix\02_teacher_forcing_vs_rollout.py`
- `mp-core\trajectory-prediction\experiments\rollout-fix\03_rollout_stabilization_abc.py`
- `mp-core\trajectory-prediction\experiments\rollout-fix\04_multistep_training.py`
- `mp-core\trajectory-prediction\experiments\rollout-fix\diagnose_turning.py`
- `mp-core\trajectory-prediction\experiments\rollout-fix\freeze_decision.py`
- `mp-core\trajectory-prediction\experiments\rollout-fix\turn_fix_curvature.py`
- `mp-core\trajectory-prediction\experiments\rollout-fix\visualise_prediction_rollout_fix.py`
- `mp-core\trajectory-prediction\generate_motion_dataset.py`
- `mp-core\trajectory-prediction\run_pipeline.py`
- `mp-core\trajectory-prediction\segment_plan.py`
- `mp-core\trajectory-prediction\train_model.py`
- `mp-core\trajectory-prediction\visualise_prediction.py`

## 2. Encoded CSV inspection

### CSV — `trajectories_encoded_auto.csv`

- Path: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\encoded\trajectories_encoded_auto.csv`
- Rows: **180,840**   ·   Columns: **18**   ·   Duplicates: **0**
- Track column: `track_id`   ·   Frame column: `frame`   ·   Unique tracks: **540**

Columns: `frame`, `time_s`, `track_id`, `conf`, `x1`, `y1`, `x2`, `y2`, `cx`, `cy`, `foot_x`, `foot_y`, `image_x`, `image_y`, `world_x`, `world_y`, `dist_to_obstacle`, `dist_to_boundary`

No missing values in any column.

**Key feature summary:**

| feature | present | min | max | mean | median | std | n_unique |
|---|---|---|---|---|---|---|---|
| `dist_to_obstacle` (`dist_to_obstacle`) | yes | 0.000 | 2.610 | 0.268 | 0.160 | 0.302 | 927 |
| `dist_to_boundary` (`dist_to_boundary`) | yes | 0.000 | 4.520 | 0.362 | 0.120 | 0.679 | 1578 |
| `dist_to_entrance` | no | — | — | — | — | — | — |
| `world_x` (`world_x`) | yes | -0.275 | 13.925 | 6.928 | 7.692 | 4.104 | 15961 |
| `world_y` (`world_y`) | yes | -19.362 | 29.020 | -1.902 | -5.435 | 11.090 | 17258 |
| `delta_x` | no | — | — | — | — | — | — |
| `delta_y` | no | — | — | — | — | — | — |
| `speed` | no | — | — | — | — | — | — |

### CSV — `trajectories_encoded.csv (manual)`

- Path: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\encoded\trajectories_encoded.csv`
- Rows: **180,840**   ·   Columns: **19**   ·   Duplicates: **0**
- Track column: `person_id`   ·   Frame column: `frame_number`   ·   Unique tracks: **540**

Columns: `frame_number`, `time_s`, `person_id`, `conf`, `x1`, `y1`, `x2`, `y2`, `cx`, `cy`, `foot_x`, `foot_y`, `image_x`, `image_y`, `world_x`, `world_y`, `dist_to_obstacle`, `dist_to_boundary`, `dist_to_entrance`

No missing values in any column.

**Key feature summary:**

| feature | present | min | max | mean | median | std | n_unique |
|---|---|---|---|---|---|---|---|
| `dist_to_obstacle` (`dist_to_obstacle`) | yes | 0.109 | 8.858 | 3.230 | 3.061 | 1.698 | 15302 |
| `dist_to_boundary` (`dist_to_boundary`) | yes | 0.024 | 19.466 | 10.592 | 10.178 | 3.753 | 16420 |
| `dist_to_entrance` (`dist_to_entrance`) | yes | 0.625 | 24.866 | 11.506 | 9.689 | 6.126 | 16863 |
| `world_x` (`world_x`) | yes | -0.275 | 13.925 | 6.928 | 7.692 | 4.104 | 15961 |
| `world_y` (`world_y`) | yes | -19.362 | 29.020 | -1.902 | -5.435 | 11.090 | 17258 |
| `delta_x` | no | — | — | — | — | — | — |
| `delta_y` | no | — | — | — | — | — | — |
| `speed` | no | — | — | — | — | — | — |

### CSV — `motion_dataset.csv`

- Path: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\encoded\motion_dataset.csv`
- Rows: **180,300**   ·   Columns: **19**   ·   Duplicates: **0**
- Track column: `None`   ·   Frame column: `None`   ·   Unique tracks: **—**

Columns: `trajectory_id`, `timestep`, `u`, `v`, `du`, `dv`, `speed`, `heading_sin`, `heading_cos`, `turn_rate`, `dist_to_obstacle_norm`, `dist_to_boundary_norm`, `delta_dist_to_obstacle`, `delta_dist_to_boundary`, `openness_ahead`, `openness_left`, `openness_right`, `target_du`, `target_dv`

No missing values in any column.

**Key feature summary:**

| feature | present | min | max | mean | median | std | n_unique |
|---|---|---|---|---|---|---|---|
| `dist_to_obstacle` | no | — | — | — | — | — | — |
| `dist_to_boundary` | no | — | — | — | — | — | — |
| `dist_to_entrance` | no | — | — | — | — | — | — |
| `world_x` | no | — | — | — | — | — | — |
| `world_y` | no | — | — | — | — | — | — |
| `delta_x` | no | — | — | — | — | — | — |
| `delta_y` | no | — | — | — | — | — | — |
| `speed` (`speed`) | yes | 0.000 | 1.867 | 0.043 | 0.026 | 0.060 | 16949 |

### CSV — `motion_dataset_v2.csv`

- Path: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\encoded\motion_dataset_v2.csv`
- Rows: **180,300**   ·   Columns: **15**   ·   Duplicates: **0**
- Track column: `None`   ·   Frame column: `None`   ·   Unique tracks: **—**

Columns: `trajectory_id`, `timestep`, `u`, `v`, `du`, `dv`, `speed`, `heading_sin`, `heading_cos`, `turn_rate`, `dist_to_obstacle_norm`, `dist_to_boundary_norm`, `openness_lr_asymmetry`, `target_du`, `target_dv`

No missing values in any column.

**Key feature summary:**

| feature | present | min | max | mean | median | std | n_unique |
|---|---|---|---|---|---|---|---|
| `dist_to_obstacle` | no | — | — | — | — | — | — |
| `dist_to_boundary` | no | — | — | — | — | — | — |
| `dist_to_entrance` | no | — | — | — | — | — | — |
| `world_x` | no | — | — | — | — | — | — |
| `world_y` | no | — | — | — | — | — | — |
| `delta_x` | no | — | — | — | — | — | — |
| `delta_y` | no | — | — | — | — | — | — |
| `speed` (`speed`) | yes | 0.000 | 0.424 | 0.042 | 0.026 | 0.051 | 16927 |

## 3. Manual vs automatic encoding

- Auto rows: 180,840   ·   Manual rows: 180,840   ·   Joined rows (on track + frame): **180,840**

**Per-feature comparison on joined rows:**

| feature | pairs | auto mean | manual mean | mean |Δ| | median |Δ| | max |Δ| | Pearson r |
|---|---|---|---|---|---|---|---|
| `dist_to_obstacle` | 180,840 | 0.268 | 3.230 | 2.967 | 2.715 | 8.377 | 0.090 |
| `dist_to_boundary` | 180,840 | 0.362 | 10.592 | 10.230 | 10.134 | 17.620 | 0.261 |
| `world_x` | 180,840 | 6.928 | 6.928 | 0.000 | 0.000 | 0.000 | 1.000 |
| `world_y` | 180,840 | -1.902 | -1.902 | 0.000 | 0.000 | 0.000 | 1.000 |

## 4. Spatial feature maps

| file | shape | dtype | min | max | mean | std | unique (sample) | class | mostly flat |
|---|---|---|---|---|---|---|---|---|---|
| `dist_to_boundary_norm.png` | [1035, 1139, 4] | uint8 | 0.000 | 255.000 | 248.564 | 31.650 | 253 | continuous | yes |
| `dist_to_obstacle_norm.png` | [1035, 1139, 4] | uint8 | 0.000 | 255.000 | 248.302 | 32.682 | 254 | continuous | yes |
| `openness_diff_left.png` | [1035, 1155, 4] | uint8 | 0.000 | 255.000 | 250.213 | 27.762 | 251 | continuous | yes |
| `openness_diff_right.png` | [1035, 1155, 4] | uint8 | 0.000 | 255.000 | 249.868 | 28.920 | 251 | continuous | yes |
| `speed.png` | [1035, 1139, 4] | uint8 | 0.000 | 255.000 | 248.652 | 32.306 | 253 | continuous | yes |
| `turn_rate.png` | [1035, 1124, 4] | uint8 | 0.000 | 255.000 | 250.919 | 25.111 | 253 | continuous | yes |

## 5. Debug images

| file | shape | unique colours (sample) | mean intensity | mostly black | mostly white | empty/corrupt |
|---|---|---|---|---|---|---|
| `debug_overlay.png` | [1182, 1001, 4] | 2836 | 209.579 | no | no | no |
| `feature_correlation.png` | [1595, 1837, 4] | 1636 | 236.991 | no | yes | no |
| `feature_histograms.png` | [1068, 2010, 4] | 326 | 247.738 | no | yes | yes |

## 6. Code analysis

**Likely auto-encoder script:** `mp-core\trajectory-prediction\encode_spatial_auto.py`

| script | lines | methods detected | hardcoded paths | coord hints |
|---|---|---|---|---|
| `mp-core\trajectory-prediction\build_dataset_v2.py` | 302 | trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\diagnose_auto_spatial_encoding.py` | 1130 | distance_transform, homography, image_thresholding, kdtree, manual_points, morphology, segmentation, trajectory_use | 0 | uses both image_ and world_ coordinates; applies a homography; flips/rotates an image axis |
| `mp-core\trajectory-prediction\diagnose_motion_dataset.py` | 488 | homography, segmentation, trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\duplicate_trajectories.py` | 79 | trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\encode_space.py` | 559 | trajectory_use | 1 | — |
| `mp-core\trajectory-prediction\encode_spatial_auto.py` | 206 | homography, image_thresholding, segmentation, trajectory_use | 0 | applies a homography |
| `mp-core\trajectory-prediction\experiments\ablation\rollout_horizon_audit.py` | 172 | trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\experiments\ablation\run_ablation.py` | 838 | image_thresholding, kdtree, trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\experiments\ablation\visualize_ablation.py` | 871 | homography, kdtree, segmentation, trajectory_use | 1 | applies a homography |
| `mp-core\trajectory-prediction\experiments\overfit-10x\compare_results.py` | 214 | trajectory_use | 1 | — |
| `mp-core\trajectory-prediction\experiments\overfit-10x\diagnose_spatial_features.py` | 395 | trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\experiments\overfit-10x\eval_phase2b_final_multi.py` | 273 | kdtree, trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\experiments\overfit-10x\make_comparison_figure.py` | 400 | trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\experiments\overfit-10x\make_overfit_dataset.py` | 86 | trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\experiments\overfit-10x\rollout_bug_audit.py` | 615 | kdtree, trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\experiments\overfit-10x\train_gru_overfit.py` | 323 | trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\experiments\overfit-10x\train_lstm_overfit.py` | 333 | trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\experiments\overfit-10x\train_phase2a.py` | 462 | trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\experiments\overfit-10x\train_phase2b.py` | 517 | trajectory_use | 0 | flips/rotates an image axis |
| `mp-core\trajectory-prediction\experiments\overfit-10x\train_phase2b_final.py` | 603 | kdtree, trajectory_use | 0 | flips/rotates an image axis |
| `mp-core\trajectory-prediction\experiments\overfit-10x\train_phase2b_transformer.py` | 632 | trajectory_use | 0 | flips/rotates an image axis |
| `mp-core\trajectory-prediction\experiments\overfit-10x\train_phase2c_fixed_spatial.py` | 605 | kdtree, trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\experiments\overfit-10x\vis_phase2b_final_plan_view.py` | 403 | kdtree, trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\experiments\rollout-fix\01_target_distribution_audit.py` | 268 | trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\experiments\rollout-fix\02_teacher_forcing_vs_rollout.py` | 401 | kdtree, trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\experiments\rollout-fix\03_rollout_stabilization_abc.py` | 479 | trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\experiments\rollout-fix\04_multistep_training.py` | 298 | trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\experiments\rollout-fix\diagnose_turning.py` | 554 | image_thresholding, trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\experiments\rollout-fix\freeze_decision.py` | 579 | trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\experiments\rollout-fix\turn_fix_curvature.py` | 662 | trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\experiments\rollout-fix\visualise_prediction_rollout_fix.py` | 309 | trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\generate_motion_dataset.py` | 569 | homography, image_thresholding, segmentation, trajectory_use | 0 | applies a homography |
| `mp-core\trajectory-prediction\run_pipeline.py` | 299 | trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\segment_plan.py` | 326 | distance_transform, image_thresholding, morphology, segmentation | 0 | — |
| `mp-core\trajectory-prediction\train_model.py` | 419 | trajectory_use | 0 | — |
| `mp-core\trajectory-prediction\visualise_prediction.py` | 603 | homography, image_thresholding, trajectory_use | 0 | — |

### Detail — `mp-core\trajectory-prediction\encode_spatial_auto.py`

**Coordinate hints:** applies a homography

## 7. Likely Failure Modes

- distance map mostly flat (top-value-fraction 0.92): dist_to_boundary_norm.png
- distance map mostly flat (top-value-fraction 0.92): dist_to_obstacle_norm.png
- distance map mostly flat (top-value-fraction 0.92): openness_diff_left.png
- distance map mostly flat (top-value-fraction 0.92): openness_diff_right.png
- distance map mostly flat (top-value-fraction 0.92): speed.png
- distance map mostly flat (top-value-fraction 0.92): turn_rate.png
- debug image empty/flat: feature_histograms.png
- auto vs manual dist_to_obstacle: Pearson r=0.09 (weak correlation)
- auto vs manual dist_to_boundary: Pearson r=0.26 (weak correlation)
- auto vs manual dist_to_boundary: mean-abs-diff (10.23) > 2× manual std (3.75)
- entry/exit detection unreliable: auto CSV has no dist_to_entrance column (manual CSV does)

## 8. Plain-English summary

The automatic encoder produced **180,840 rows** across **540 unique tracks**. Its column set is `frame`, `time_s`, `track_id`, `conf`, `x1`, `y1`, `x2`, `y2`, `cx`, `cy`, `foot_x`, `foot_y`, `image_x`, `image_y`, `world_x`, `world_y`, `dist_to_obstacle`, `dist_to_boundary`.
The manual encoder produced **180,840 rows** with columns `frame_number`, `time_s`, `person_id`, `conf`, `x1`, `y1`, `x2`, `y2`, `cx`, `cy`, `foot_x`, `foot_y`, `image_x`, `image_y`, `world_x`, `world_y`, `dist_to_obstacle`, `dist_to_boundary`, `dist_to_entrance`.

On the rows where the two encoders overlap, the spatial features can be compared directly. The Pearson correlations and mean-absolute-differences in Section 3 indicate how close each automatic feature is to its manual counterpart. Low correlations or large differences are highlighted as failure modes.

A complete machine-readable mirror of this report is at `mp-data\processed\encoded\auto_spatial_encoding_diagnostic_summary.json`.

_End of report._