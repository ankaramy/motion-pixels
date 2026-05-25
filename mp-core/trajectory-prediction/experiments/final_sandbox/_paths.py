"""Shared paths + helpers for the final-sandbox scripts."""

from pathlib import Path

HERE     = Path(__file__).resolve().parent
MP_ROOT  = HERE.parent.parent.parent.parent
MP_DATA  = MP_ROOT / "mp-data"

# Inputs (rerun outputs).
RERUN          = MP_DATA / "processed" / "rerun_macba_2026-05-19"
WORLD_CSV      = RERUN / "trajectories" / "trajectories_world.csv"
ENCODED_CSV    = RERUN / "spatial_v21C"  / "trajectories_encoded.csv"
WALKABLE_PNG   = RERUN / "spatial_v21C"  / "walkable_mask.png"
OBSTACLE_PNG   = RERUN / "spatial_v21C"  / "obstacle_mask.png"
BOUNDARY_PNG   = RERUN / "spatial_v21C"  / "boundary_mask.png"
ENTRIES_CSV    = RERUN / "spatial_v21C"  / "entry_exit_points.csv"
CALIB_JSON     = RERUN / "calibration"   / "calib_skate1_2026-05-19.json"
TOPVIEW_PNG    = RERUN / "inputs"        / "top_view.png"

# Outputs.
SANDBOX        = RERUN / "final_sandbox"
DATASET_CSV    = SANDBOX / "final_sandbox_dataset.csv"
DATASET_MD     = SANDBOX / "dataset_summary.md"
MODEL_PTH      = SANDBOX / "lstm_final.pth"
SCALER_PKL     = SANDBOX / "scaler.pkl"
LOSS_PNG       = SANDBOX / "loss_curve.png"
TRAIN_MD       = SANDBOX / "training_summary.md"
PRED_PLOTS_DIR = SANDBOX / "prediction_plots"
PRED_SHEET     = SANDBOX / "prediction_contact_sheet.png"
PLAN_PLOTS_DIR = SANDBOX / "plan_overlay_plots"
PLAN_SHEET     = SANDBOX / "plan_prediction_overlay_contact_sheet.png"
FINAL_MD       = SANDBOX / "final_sandbox_summary.md"

# Hyperparameters / constants.
# These mirror the previous working sandbox (train_phase2b_final.py) which
# produced angular rollouts. Do not add extra invented features (no
# heading_sin/cos, no turn_rate, no u/v_norm, no dist_to_entrance) — they
# bias the LSTM toward mean motion under autoregressive feedback.
WINDOW_SIZE   = 10
N_ROLLOUT     = 30
DUPLICATE_N   = 10
MIN_TRACK_LEN = 45              # ≥ WINDOW + ROLLOUT + a little headroom
HIDDEN_SIZE   = 256
NUM_LAYERS    = 2
EPOCHS        = 30
BATCH_SIZE    = 1024
LR            = 1e-3
IDW_K         = 5               # KDTree neighbours for SpatialInterpolator

# Production-equivalent 6-feature set. The rollout recomputes
# dist_to_obstacle / dist_to_boundary via KDTree-IDW at every step from
# the predicted (world_x, world_y); seed values come from the v2.1C
# encoded CSV.
FEATURE_COLS = [
    "world_x", "world_y",
    "delta_x", "delta_y",
    "dist_to_obstacle", "dist_to_boundary",
]
TARGET_COLS  = ["delta_x", "delta_y"]
