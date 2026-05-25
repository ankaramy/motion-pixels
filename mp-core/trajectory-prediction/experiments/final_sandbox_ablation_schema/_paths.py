"""Shared paths, constants, and feature lists for the ablation experiment.

Models:
  A — motion only        (6 features)
  B — + position         (10 features)
  C — + spatial          (13 features)
  D — + angular context  (16 features)

Critical rollout rule (implemented in visualize_ablation_grid.rollout):
every feature that depends on predicted motion is RECOMPUTED at each
predicted step, never frozen.
"""

from pathlib import Path

HERE     = Path(__file__).resolve().parent
MP_ROOT  = HERE.parent.parent.parent.parent
MP_DATA  = MP_ROOT / "mp-data"

# Inputs (rerun outputs).
RERUN          = MP_DATA / "processed" / "rerun_macba_2026-05-19"
WORLD_CSV      = RERUN / "trajectories" / "trajectories_world.csv"
ENCODED_CSV    = RERUN / "spatial_v21C" / "trajectories_encoded.csv"
CALIB_JSON     = RERUN / "calibration"  / "calib_skate1_2026-05-19.json"
TOPVIEW_PNG    = RERUN / "inputs"       / "top_view.png"

# Outputs.
EXP             = RERUN / "final_sandbox_ablation_schema"
DATASET_CSV     = EXP / "schema_dataset.csv"
DATASET_MD      = EXP / "dataset_summary.md"
MODELS_DIR      = EXP / "models"
PLOTS_DIR       = EXP / "plots"
PLAN_DIR        = EXP / "plan_overlays"
METRICS_CSV     = EXP / "metrics_ablation_schema.csv"
FINAL_MD        = EXP / "ablation_schema_summary.md"

# Hyperparameters.
WINDOW_SIZE   = 10
HORIZON       = 20
DUPLICATE_N   = 10
MIN_TRACK_LEN = 35
HIDDEN_SIZE   = 256
NUM_LAYERS    = 2
EPOCHS        = 30
BATCH_SIZE    = 1024
LR            = 1e-3
IDW_K         = 5

# Behaviour thresholds (per-step world units).
STOP_THRESH_M       = 0.005       # below this magnitude → is_stop = 1
SHIFT_THRESH_RAD    = 0.26        # ~15 deg per-step heading change → is_shift
ANGULAR_BIN_DEG     = 15.0        # used in angularity score

# Feature lists per model (column order matters — used by both the
# dataset builder and the rollout-time row builder).
FEATURES_A = [
    "delta_x", "delta_y",
    "speed", "heading_angle",
    "is_stop", "is_shift",
]
FEATURES_B = FEATURES_A + ["u", "v", "world_x", "world_y"]
FEATURES_C = FEATURES_B + ["dist_to_obstacle", "dist_to_boundary",
                            "dist_to_entrance"]
FEATURES_D = FEATURES_C + ["heading_sin", "heading_cos", "turn_rate"]

MODEL_DEFS = [
    {"name": "A_motion",      "label": "A motion only",
     "color": "#7f8c8d", "features": FEATURES_A},
    {"name": "B_position",    "label": "B + position",
     "color": "#2980b9", "features": FEATURES_B},
    {"name": "C_spatial",     "label": "C + spatial",
     "color": "#27ae60", "features": FEATURES_C},
    {"name": "D_angular",     "label": "D + angular context",
     "color": "#e74c3c", "features": FEATURES_D},
]

TARGET_COLS = ["target_du", "target_dv"]

# Tracks to plot (same set as the previous sandbox runs for continuity).
PRESET_TRACKS = [50, 24, 32, 17, 37, 45, 25, 31, 49, 51]
