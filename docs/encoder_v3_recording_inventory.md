# Encoder V3 Recording Inventory

Generated: 2026-06-08

Scope: discovery-only audit for Encoder V3 preparation. No training, dataset rebuild, tracking, calibration, or encoder execution was performed while creating this inventory.

## Project Root

`C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels`

Related external data/output roots found during audit:

- Dataset root: `C:\Users\OWNER\Desktop\new_datasets`
- Thesis output root: `C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs`
- Encoding calibration audit root: `C:\Users\OWNER\Desktop\MotionPixels_Encoding_Calibration_Audit`
- Barcelona training visuals root: `C:\Users\OWNER\Desktop\MotionPixels_Barcelona_Training_Visuals`
- Angular fix output root: `C:\Users\OWNER\Desktop\MotionPixels_AngularFix_V1`

## Encoder Files

Current encoder file path:

`C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\encode_spatial_auto_v2.py`

Successful encoding file path / batch driver:

`C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\run_barcelona_v1_encoding.py`

The Barcelona encoding report states that `run_barcelona_v1_encoding.py` imported the frozen `encode_spatial_auto_v2.py` in-process and applied the `spatial_v21C` overrides, without modifying the encoder file.

All encoder / encoding-related variants found:

- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\encode_spatial_auto_v2.py`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\encode_spatial_auto.py`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\encode_spatial_auto_v21_compare.py`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\encode_space.py`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\rerun_spatial_pipeline.py`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\run_barcelona_v1_encoding.py`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\diagnose_auto_spatial_encoding.py`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\visualize_auto_spatial_encoding_v2.py`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\encoder_v2_dev\anatomy_audit.py`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\encoder_v2_dev\audit_overlays.py`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\encoder_v2_dev\directional_features.py`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\encoder_v2_dev\encoding_truth_audit.py`

Dataset builder / training-adjacent files found:

- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\build_barcelona_v1_master.py`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\build_master_dataset.py`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\build_dataset_v2.py`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\make_barcelona_v1_contact_sheets.py`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\train_model.py`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\experiments\schema_ablation_bridge\make_bridge_dataset.py`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\experiments\schema_ablation_bridge\run_bridge_ablation.py`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\experiments\schema_ablation_bridge\run_schema_ablation_bridge.py`

Frozen Model C references:

- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\frozen_model_C\README.md`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\frozen_model_C\held_out\best_model.pth`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\frozen_model_C\held_out\scalers.pkl`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\frozen_model_C\held_out\schema_summary.json`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\frozen_model_C\overfit10x\best_model.pth`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\frozen_model_C\overfit10x\scalers.pkl`
- `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\frozen_model_C\overfit10x\schema_summary.json`

## Master Dataset

Master dataset path:

`C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_master_dataset\master_dataset.csv`

Model C dataset path:

`C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_master_dataset\model_C_dataset.csv`

Master dataset support files:

- `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_master_dataset\manifest.json`
- `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_master_dataset\dataset_validation.json`
- `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_master_dataset\dataset_schema_report.md`
- `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_master_dataset\Frozen_Model_C_Training_Audit.md`
- `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_master_dataset\_feature_ranges.json`

Creation script:

`C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\build_barcelona_v1_master.py`

Recipe provenance:

`build_barcelona_v1_master.py` imports frozen bridge recipe functions from `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\experiments\schema_ablation_bridge\make_bridge_dataset.py` and applies them per recording.

Readable counts from the CSV files:

| file | rows | unique trajectory_id count |
|---|---:|---:|
| `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_master_dataset\master_dataset.csv` | 1,064,379 | 3,534 |
| `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_master_dataset\model_C_dataset.csv` | 1,064,379 | 3,534 |

Important discrepancy:

- The user-provided known fact and `dataset_schema_report.md` text mention 3,810 trajectories.
- The readable CSVs and `manifest.json` report 3,534 unique namespaced `trajectory_id`s.
- Per-recording readable counts are: esplanade 522, placa_catalunya 1,255, placa_espanya 824, stairs_montjuic_01 334, red_bridge 599; these sum to 3,534.

## Audit Folder

Encoding truth audit output folder:

`C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\07_encoding_truth_audit`

Report path:

`C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\07_encoding_truth_audit\reports\Encoding_Truth_Audit_Report.md`

Figures folder:

`C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\07_encoding_truth_audit\figures`

Tables folder:

`C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\07_encoding_truth_audit\tables`

Key audit table:

`C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\07_encoding_truth_audit\tables\mask_diagnostics.csv`

Older encoding calibration audit:

- `C:\Users\OWNER\Desktop\MotionPixels_Encoding_Calibration_Audit\encoding_calibration_audit_report.md`
- `C:\Users\OWNER\Desktop\MotionPixels_Encoding_Calibration_Audit\figures`

Encoder V2 development outputs:

- `C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\06_encoder_v2_dev\phase1_encoder_anatomy`
- `C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\06_encoder_v2_dev\phase2_directional_features`

## Recording Summary Table

| recording_id | status | video path | trajectory world CSV | calibration path | plan image path | satellite/source image path | encoded CSV path | output folder path |
|---|---|---|---|---|---|---|---|---|
| esplanade_espanya_01 | validated | `C:\Users\OWNER\Desktop\new_datasets\esplanade_espanya_01\raw_video\esplanade-espanya.MOV` | `C:\Users\OWNER\Desktop\new_datasets\esplanade_espanya_01\filtered_250m\trajectories_world_filtered_250m.csv` | `C:\Users\OWNER\Desktop\new_datasets\esplanade_espanya_01\calibration\calib.json` | `C:\Users\OWNER\Desktop\new_datasets\esplanade_espanya_01\plan\esplanade-espanya.png` | `C:\Users\OWNER\Desktop\new_datasets\esplanade-espanya.png` | `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\esplanade_espanya_01\spatial_v21C\trajectories_encoded.csv` | `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\esplanade_espanya_01\spatial_v21C` |
| stairs_montjuic_01 | validated | `C:\Users\OWNER\Desktop\new_datasets\stairs_montjuic_01\raw_video\stairs-montjuic-1.MOV` | `C:\Users\OWNER\Desktop\new_datasets\stairs_montjuic_01\filtered_250m\trajectories_world_filtered_250m.csv` | `C:\Users\OWNER\Desktop\new_datasets\stairs_montjuic_01\calibration\calib.json` | `C:\Users\OWNER\Desktop\new_datasets\stairs_montjuic_01\plan\stairs-montjuic-1.png` | `C:\Users\OWNER\Desktop\new_datasets\stairs-montjuic-1.png` | `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\stairs_montjuic_01\spatial_v21C\trajectories_encoded.csv` | `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\stairs_montjuic_01\spatial_v21C` |
| red_bridge_combined_01 | validated | `C:\Users\OWNER\Desktop\new_datasets\red_bridge_combined_01\raw_video\red-bridge-combined.mp4` | `C:\Users\OWNER\Desktop\new_datasets\red_bridge_combined_01\filtered_250m\trajectories_world_filtered_250m.csv` | `C:\Users\OWNER\Desktop\new_datasets\red_bridge_combined_01\calibration\calib.json` | `C:\Users\OWNER\Desktop\new_datasets\red_bridge_combined_01\plan\red-bridge-combined.png` | `C:\Users\OWNER\Desktop\new_datasets\red-bridge-combined.png` | `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\red_bridge_combined_01\spatial_v21C\trajectories_encoded.csv` | `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\red_bridge_combined_01\spatial_v21C` |
| placa_catalunya_01 | validated | `C:\Users\OWNER\Desktop\new_datasets\placa_catalunya_01\raw_video\placa-catalunya.MOV` | `C:\Users\OWNER\Desktop\new_datasets\placa_catalunya_01\filtered_250m\trajectories_world_filtered_250m.csv` | `C:\Users\OWNER\Desktop\new_datasets\placa_catalunya_01\calibration\calib.json` | `C:\Users\OWNER\Desktop\new_datasets\placa_catalunya_01\plan\placa-catalunya.png` | `C:\Users\OWNER\Desktop\new_datasets\placa-catalunya.png` | `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\placa_catalunya_01\spatial_v21C\trajectories_encoded.csv` | `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\placa_catalunya_01\spatial_v21C` |
| placa_espanya_01 | validated | `C:\Users\OWNER\Desktop\new_datasets\placa_espanya_01\raw_video\placa-espanya.MOV` | `C:\Users\OWNER\Desktop\new_datasets\placa_espanya_01\filtered_250m\trajectories_world_filtered_250m.csv` | `C:\Users\OWNER\Desktop\new_datasets\placa_espanya_01\calibration\calib.json` | `C:\Users\OWNER\Desktop\new_datasets\placa_espanya_01\plan\placa-espanya.png` | `C:\Users\OWNER\Desktop\new_datasets\placa-espanya.png` | `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\placa_espanya_01\spatial_v21C\trajectories_encoded.csv` | `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\placa_espanya_01\spatial_v21C` |
| placa_montjuic_01 | excluded | `C:\Users\OWNER\Desktop\new_datasets\placa_montjuic_01\raw_video\placa-montjuic.MOV` | `C:\Users\OWNER\Desktop\new_datasets\placa_montjuic_01\filtered_250m\trajectories_world_filtered_250m.csv` | `C:\Users\OWNER\Desktop\new_datasets\placa_montjuic_01\calibration\calib.json` | `C:\Users\OWNER\Desktop\new_datasets\placa_montjuic_01\plan\placa-montjuic.png` | `C:\Users\OWNER\Desktop\new_datasets\placa-montjuic.png` | missing / not produced in `Barcelona_v1_encoded` | missing / V3 should not run yet |
| stairs_montjuic_02 | excluded | `C:\Users\OWNER\Desktop\new_datasets\stairs_montjuic_02\raw_video\stairs-montjuic-2.MOV` | `C:\Users\OWNER\Desktop\new_datasets\stairs_montjuic_02\filtered_250m\trajectories_world_filtered_250m.csv` | `C:\Users\OWNER\Desktop\new_datasets\stairs_montjuic_02\calibration\calib.json` | `C:\Users\OWNER\Desktop\new_datasets\stairs_montjuic_02\plan\stairs-montjuic-2.png` | `C:\Users\OWNER\Desktop\new_datasets\stairs-montjuic-2.png` | missing / not produced in `Barcelona_v1_encoded` | missing / V3 should not run yet |

## Detailed Recording Sections

### esplanade_espanya_01

recording_id: esplanade_espanya_01
status: validated

video: `C:\Users\OWNER\Desktop\new_datasets\esplanade_espanya_01\raw_video\esplanade-espanya.MOV`
trajectory_world: `C:\Users\OWNER\Desktop\new_datasets\esplanade_espanya_01\filtered_250m\trajectories_world_filtered_250m.csv`
original_trajectory_world: `C:\Users\OWNER\Desktop\new_datasets\esplanade_espanya_01\tracking\trajectories_world.csv`
calibration: `C:\Users\OWNER\Desktop\new_datasets\esplanade_espanya_01\calibration\calib.json`
plan_image: `C:\Users\OWNER\Desktop\new_datasets\esplanade_espanya_01\plan\esplanade-espanya.png`
source_image: `C:\Users\OWNER\Desktop\new_datasets\esplanade-espanya.png`
encoded_csv: `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\esplanade_espanya_01\spatial_v21C\trajectories_encoded.csv`
output_folder: `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\esplanade_espanya_01\spatial_v21C`

notes:
- rotation issues? No active rotation issue found for this recording. The plotting-orientation audit reports no flip issue after the plotting-layer orientation fix.
- scale issues? Wide but real scene; filtered bounds x[-131,132] y[-13,34]. Marked safe for encoding.
- homography concerns? No current blocker found.
- obstacle-rich? Medium. Audit found 1 interior obstacle component and 19.7 m2 interior obstacle area in the old trajectory-derived mask.
- source imagery available? Yes: plan image and root-level source PNG are present.
- V3 readiness: Good candidate after plan-derived mask design is implemented.

### stairs_montjuic_01

recording_id: stairs_montjuic_01
status: validated

video: `C:\Users\OWNER\Desktop\new_datasets\stairs_montjuic_01\raw_video\stairs-montjuic-1.MOV`
trajectory_world: `C:\Users\OWNER\Desktop\new_datasets\stairs_montjuic_01\filtered_250m\trajectories_world_filtered_250m.csv`
original_trajectory_world: `C:\Users\OWNER\Desktop\new_datasets\stairs_montjuic_01\tracking\trajectories_world.csv`
calibration: `C:\Users\OWNER\Desktop\new_datasets\stairs_montjuic_01\calibration\calib.json`
plan_image: `C:\Users\OWNER\Desktop\new_datasets\stairs_montjuic_01\plan\stairs-montjuic-1.png`
source_image: `C:\Users\OWNER\Desktop\new_datasets\stairs-montjuic-1.png`
encoded_csv: `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\stairs_montjuic_01\spatial_v21C\trajectories_encoded.csv`
output_folder: `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\stairs_montjuic_01\spatial_v21C`

notes:
- rotation issues? Earlier plotting orientation issue was fixed at plotting layer; verified planters render at top. No trajectory/calibration data changed.
- scale issues? No blocker; compact filtered bounds x[-11,8] y[-18,7].
- homography concerns? No current blocker found.
- obstacle-rich? Low to medium; channeled stair site. The old trajectory-derived mask is accidentally meaningful for boundaries but has 0 interior obstacle area.
- source imagery available? Yes: plan image and root-level source PNG are present.
- V3 readiness: Easiest technical V3 run; useful as a boundary/channel sanity check.

### red_bridge_combined_01

recording_id: red_bridge_combined_01
status: validated

video: `C:\Users\OWNER\Desktop\new_datasets\red_bridge_combined_01\raw_video\red-bridge-combined.mp4`
trajectory_world: `C:\Users\OWNER\Desktop\new_datasets\red_bridge_combined_01\filtered_250m\trajectories_world_filtered_250m.csv`
original_trajectory_world: `C:\Users\OWNER\Desktop\new_datasets\red_bridge_combined_01\tracking\trajectories_world.csv`
calibration: `C:\Users\OWNER\Desktop\new_datasets\red_bridge_combined_01\calibration\calib.json`
plan_image: `C:\Users\OWNER\Desktop\new_datasets\red_bridge_combined_01\plan\red-bridge-combined.png`
source_image: `C:\Users\OWNER\Desktop\new_datasets\red-bridge-combined.png`
encoded_csv: `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\red_bridge_combined_01\spatial_v21C\trajectories_encoded.csv`
output_folder: `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\red_bridge_combined_01\spatial_v21C`

notes:
- rotation issues? No active rotation issue found.
- scale issues? No blocker; compact filtered bounds x[-2,13] y[-28,10].
- homography concerns? No current blocker found.
- obstacle-rich? Low; narrow bridge/landing geometry. Old mask has 0 interior obstacle area, but boundary proximity is likely meaningful because flow is channeled.
- source imagery available? Yes: plan image and root-level source PNG are present.
- V3 readiness: Good control case; run early to validate boundary handling.

### placa_catalunya_01

recording_id: placa_catalunya_01
status: validated

video: `C:\Users\OWNER\Desktop\new_datasets\placa_catalunya_01\raw_video\placa-catalunya.MOV`
trajectory_world: `C:\Users\OWNER\Desktop\new_datasets\placa_catalunya_01\filtered_250m\trajectories_world_filtered_250m.csv`
original_trajectory_world: `C:\Users\OWNER\Desktop\new_datasets\placa_catalunya_01\tracking\trajectories_world.csv`
calibration: `C:\Users\OWNER\Desktop\new_datasets\placa_catalunya_01\calibration\calib.json`
plan_image: `C:\Users\OWNER\Desktop\new_datasets\placa_catalunya_01\plan\placa-catalunya.png`
source_image: `C:\Users\OWNER\Desktop\new_datasets\placa-catalunya.png`
encoded_csv: `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\placa_catalunya_01\spatial_v21C\trajectories_encoded.csv`
output_folder: `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\placa_catalunya_01\spatial_v21C`

notes:
- rotation issues? No active rotation issue found.
- scale issues? No blocker; filtered bounds x[-80,-18] y[-95,5].
- homography concerns? No current blocker found.
- obstacle-rich? Yes. Visible trees, benches, planters, vegetation, urban furniture.
- source imagery available? Yes: plan image and root-level source PNG are present.
- critical old-encoder warning: Encoding Truth Audit found 0 interior obstacle components and 0.0 m2 interior obstacle area despite visible architecture. Plan-threshold candidate found about 296 m2 of candidate interior structure inside the walked area.
- V3 readiness: High value but difficult; do not interpret old spatial results as architecture signal failure.

### placa_espanya_01

recording_id: placa_espanya_01
status: validated

video: `C:\Users\OWNER\Desktop\new_datasets\placa_espanya_01\raw_video\placa-espanya.MOV`
normalized_video: `C:\Users\OWNER\Desktop\new_datasets\placa_espanya_01\normalized\placa-espanya-30fps.mp4`
trajectory_world: `C:\Users\OWNER\Desktop\new_datasets\placa_espanya_01\filtered_250m\trajectories_world_filtered_250m.csv`
original_trajectory_world: `C:\Users\OWNER\Desktop\new_datasets\placa_espanya_01\tracking\trajectories_world.csv`
calibration: `C:\Users\OWNER\Desktop\new_datasets\placa_espanya_01\calibration\calib.json`
plan_image: `C:\Users\OWNER\Desktop\new_datasets\placa_espanya_01\plan\placa-espanya.png`
source_image: `C:\Users\OWNER\Desktop\new_datasets\placa-espanya.png`
encoded_csv: `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\placa_espanya_01\spatial_v21C\trajectories_encoded.csv`
output_folder: `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\placa_espanya_01\spatial_v21C`

notes:
- rotation issues? No active rotation issue found.
- scale issues? Wide but marked safe; filtered bounds x[-131,206] y[-38,54].
- homography concerns? No current blocker found. Prior tracking outlier noted as resolved in the encoding report.
- obstacle-rich? Yes / medium-high plaza+avenue scene. Old trajectory-derived mask found 2 interior obstacle components and 46.2 m2 interior obstacle area, likely still incomplete for real architecture.
- source imagery available? Yes: plan image and root-level source PNG are present.
- V3 readiness: Good second plaza after Plaça Catalunya.

### placa_montjuic_01

recording_id: placa_montjuic_01
status: excluded

video: `C:\Users\OWNER\Desktop\new_datasets\placa_montjuic_01\raw_video\placa-montjuic.MOV`
trajectory_world: `C:\Users\OWNER\Desktop\new_datasets\placa_montjuic_01\filtered_250m\trajectories_world_filtered_250m.csv`
original_trajectory_world: `C:\Users\OWNER\Desktop\new_datasets\placa_montjuic_01\tracking\trajectories_world.csv`
calibration: `C:\Users\OWNER\Desktop\new_datasets\placa_montjuic_01\calibration\calib.json`
plan_image: `C:\Users\OWNER\Desktop\new_datasets\placa_montjuic_01\plan\placa-montjuic.png`
source_image: `C:\Users\OWNER\Desktop\new_datasets\placa-montjuic.png`
encoded_csv: missing / not present under `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded`
output_folder: missing / not present under `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded`

notes:
- rotation issues? Not the primary issue found.
- scale issues? Yes. Spatial rebuild report marks filtered bounds x[-222,194] y[-129,166] and describes thin diagonal streaks plus detached secondary cluster.
- homography concerns? Yes. Excluded due to homography instability / ill-conditioned geometry below 250 m. Calibration should be reviewed/redone before encoding.
- obstacle-rich? Unsure from files alone; likely plaza/open-space but geometry is not trustworthy.
- source imagery available? Yes: plan image and root-level source PNG are present.
- V3 warning: Do not run V3 until calibration/homography is fixed and geometry is revalidated.

### stairs_montjuic_02

recording_id: stairs_montjuic_02
status: excluded

video: `C:\Users\OWNER\Desktop\new_datasets\stairs_montjuic_02\raw_video\stairs-montjuic-2.MOV`
trajectory_world: `C:\Users\OWNER\Desktop\new_datasets\stairs_montjuic_02\filtered_250m\trajectories_world_filtered_250m.csv`
original_trajectory_world: `C:\Users\OWNER\Desktop\new_datasets\stairs_montjuic_02\tracking\trajectories_world.csv`
calibration: `C:\Users\OWNER\Desktop\new_datasets\stairs_montjuic_02\calibration\calib.json`
plan_image: `C:\Users\OWNER\Desktop\new_datasets\stairs_montjuic_02\plan\stairs-montjuic-2.png`
source_image: `C:\Users\OWNER\Desktop\new_datasets\stairs-montjuic-2.png`
encoded_csv: missing / not present under `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded`
output_folder: missing / not present under `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded`

notes:
- rotation issues? Not the primary issue found.
- scale issues? Yes. Spatial rebuild report marks filtered bounds x[-233,12] y[-118,13] and describes thin diagonal streaks / implausible width.
- homography concerns? Yes. Excluded due to homography instability / ill-conditioned geometry below 250 m. Calibration should be reviewed/redone before encoding.
- obstacle-rich? Unsure from files alone; stair/channeled context likely medium.
- source imagery available? Yes: plan image and root-level source PNG are present.
- V3 warning: Do not run V3 until calibration/homography is fixed and geometry is revalidated.

## Existing Output Folders

Dataset outputs:

- `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded`
- `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\_contact_sheets`
- `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_master_dataset`
- `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_filtered_250m_plots`

Per-recording encoded output folders:

- `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\esplanade_espanya_01\spatial_v21C`
- `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\stairs_montjuic_01\spatial_v21C`
- `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\red_bridge_combined_01\spatial_v21C`
- `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\placa_catalunya_01\spatial_v21C`
- `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded\placa_espanya_01\spatial_v21C`

Audit / report output folders:

- `C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\07_encoding_truth_audit`
- `C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\07_encoding_truth_audit\figures`
- `C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\07_encoding_truth_audit\reports`
- `C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\07_encoding_truth_audit\tables`
- `C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\06_encoder_v2_dev\phase1_encoder_anatomy`
- `C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\06_encoder_v2_dev\phase2_directional_features`
- `C:\Users\OWNER\Desktop\MotionPixels_Encoding_Calibration_Audit`
- `C:\Users\OWNER\Desktop\MotionPixels_Barcelona_Training_Visuals\00_dataset_audit`
- `C:\Users\OWNER\Desktop\MotionPixels_Barcelona_Training_Visuals\root_cause_figures`

## Missing Assets / Uncertainties

Blocking for Encoder V3:

- `placa_montjuic_01` and `stairs_montjuic_02` should not run V3 yet. They have videos, tracking CSVs, calibration JSONs, and plan/source images, but the spatial rebuild report marks them REVIEW REQUIRED due to ill-conditioned homography / distorted world geometry.
- No `Barcelona_v1_encoded` output folders or encoded CSVs were found for `placa_montjuic_01` or `stairs_montjuic_02`.

Uncertainties to resolve before V3 command generation:

- The actual master CSVs contain 3,534 unique `trajectory_id`s, while existing text/user context mentions 3,810 trajectories. Treat 3,534 as the readable file count unless a different trajectory definition is intended.
- The root-level PNGs and per-recording `plan\*.png` files appear to be the available plan/source imagery. No separate file explicitly named satellite/source was found for these Barcelona recordings beyond the root-level PNG copies.
- V3 needs a new plan-derived obstacle source policy: thresholded plan image, manually corrected mask, or hybrid segmentation. Current v21C masks are trajectory-derived and invalid for obstacle-rich plazas.

## Recommended Encoder V3 Order

Easiest:

1. `stairs_montjuic_01` - compact, validated, channeled geometry; good sanity check for boundary handling.
2. `red_bridge_combined_01` - compact, validated, narrow bridge; good control case where old trajectory-derived boundaries were accidentally meaningful.

Medium:

3. `esplanade_espanya_01` - validated but wide; less interior-obstacle rich than plazas.
4. `placa_espanya_01` - validated and obstacle/plaza rich, but larger spatial extent and likely more visual ambiguity.

Difficult / highest-value:

5. `placa_catalunya_01` - validated and the key obstacle-rich failure case; run after V3 mask logic is stable because it is the critical proof case for plan-assisted obstacles.

Do not run yet:

6. `placa_montjuic_01` - excluded due to homography instability; recalibrate/revalidate first.
7. `stairs_montjuic_02` - excluded due to homography instability; recalibrate/revalidate first.

## Warnings Where V3 Should Not Run Yet

- Do not run Encoder V3 on `placa_montjuic_01` or `stairs_montjuic_02` until calibration/homography is reviewed and world geometry is revalidated.
- Do not use the existing `spatial_v21C` obstacle masks as architectural ground truth. The current encoder derives obstacle masks from trajectory occupancy, not real architecture.
- Do not treat previous "no spatial signal" findings as a refutation of the architecture hypothesis. The Encoding Truth Audit classifies the old encoder as failing to represent real architectural obstacles for obstacle-rich plazas.
- For `placa_catalunya_01`, V3 should use plan-assisted obstacle masks before any classifier/training interpretation; old v21C had 0 interior obstacle components and 0.0 m2 interior obstacle area.
