"""
Shared paths, hyperparameters, and feature lists for the
schema_angular_lstm_v2 experiment.

v2 changes (Pass 1 — make rollouts visible again):
  * MOTION_SCALE applied to du, dv, target_du, target_dv so the network
    sees motion at a learnable scale while u/v stay in [0, 1].
  * Shorter training horizon with validation-based early stopping.
  * Freeze-collapse penalty discourages zero-motion predictions.
"""

from pathlib import Path

HERE     = Path(__file__).resolve().parent
MP_ROOT  = HERE.parent.parent.parent.parent          # .../motion-pixels
MP_DATA  = MP_ROOT / "mp-data"

# Input: the newly calibrated Skate 1 encoded trajectories.
RERUN          = MP_DATA / "processed" / "rerun_macba_2026-05-19"
ENCODED_CSV    = RERUN / "spatial_v21C" / "trajectories_encoded.csv"
ENTRY_EXIT_CSV = RERUN / "spatial_v21C" / "entry_exit_points.csv"
TOPVIEW_PNG    = RERUN / "inputs" / "top_view.png"

# Experiment outputs live next to this script.
EXP                  = HERE
SCHEMA_CSV           = EXP / "trajectories_schema_relational.csv"
SCHEMA_MD            = EXP / "schema_summary.md"
WORLD_EXTENTS_CSV    = EXP / "world_extents.csv"
SCALER_PKL           = EXP / "scalers.pkl"
MODEL_PTH            = EXP / "model.pth"
LOSS_PNG             = EXP / "loss_curve.png"
TRAIN_SUMMARY_CSV    = EXP / "training_summary.csv"
ROLLOUT_CSV          = EXP / "rollout_predictions.csv"
CONTACT_PNG          = EXP / "contact_sheet_angular_lstm.png"
PER_TRACK_DIR        = EXP / "per_track_plots"
METRICS_CSV          = EXP / "metrics.csv"
METRICS_MD           = EXP / "summary.md"

# Hyperparameters.
MIN_TRACK_LEN  = 35
WINDOW_SIZE    = 10
HORIZON        = 15          # v3: shorter rollout for tuning
HIDDEN_SIZE    = 256
NUM_LAYERS     = 2
EPOCHS         = 40
PATIENCE       = 8
BATCH_SIZE     = 32
LR             = 1e-3
TRAIN_FRAC     = 0.85

# v3: stronger motion scale so the network sees motion at a more
# learnable amplitude.
MOTION_SCALE   = 250.0

# Behavioural thresholds (in the relative-feature space).
STOP_THRESH_REL    = 0.02
SHIFT_THRESH_DEG   = 15.0

# v3 loss weighting — angular & path-length terms strengthened.
W_POSITION = 1.0
W_TURN     = 1.0
W_HEADING  = 0.8
W_FREEZE   = 0.10
W_PATH     = 0.80            # v3: NEW path-length L1 loss

# Freeze threshold (compared against predicted |du, dv| in scaled space).
FREEZE_MAG_THRESH = 0.05     # v3: stronger anti-collapse threshold

# Tracks to spotlight in the rollout contact sheet.
PRESET_TRACKS = [50, 24, 32, 17, 37, 45, 25, 31, 49, 51]

# Feature / target column lists. ORDER MATTERS.
SCHEMA_COLS = [
    "track_id", "frame_idx",
    "u", "v",
    "du", "dv",
    "speed_rel",
    "heading_sin", "heading_cos",
    "turn_rate_rel",
    "is_stop", "is_shift",
    "obstacle_clearance_pct",
    "boundary_clearance_pct",
    "entrance_affinity_pct",
    "local_space_openness",
    "target_du", "target_dv", "target_turn_rate",
]

FEATURE_COLS = [
    "u", "v",
    "du", "dv",
    "speed_rel",
    "heading_sin", "heading_cos",
    "turn_rate_rel",
    "is_stop", "is_shift",
    "obstacle_clearance_pct",
    "boundary_clearance_pct",
    "entrance_affinity_pct",
    "local_space_openness",
]

TARGET_COLS = ["target_du", "target_dv", "target_turn_rate"]
