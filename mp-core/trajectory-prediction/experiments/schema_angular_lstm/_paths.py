"""
Shared paths, hyperparameters, and feature lists for the
schema_angular_lstm experiment.

The experiment trains an autoregressive LSTM on a relational schema
where every spatial feature is expressed as a percentile / clearance
ratio rather than a raw metric distance. Heading is encoded as
sin/cos and turn rate is normalized so the model can reason about
direction without unit-dependent scales.
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

# Experiment outputs live next to this script (NOT under previous sandboxes).
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
MIN_TRACK_LEN  = 35          # tracks shorter than this are dropped
WINDOW_SIZE    = 10          # LSTM input window length
HORIZON        = 20          # autoregressive prediction horizon
HIDDEN_SIZE    = 256
NUM_LAYERS     = 2
EPOCHS         = 100
BATCH_SIZE     = 32
LR             = 1e-3
TRAIN_FRAC     = 0.85        # train/val split on a track-id basis

# Behavioural thresholds (in the relative-feature space).
STOP_THRESH_REL    = 0.02    # speed_rel below this → is_stop
SHIFT_THRESH_DEG   = 15.0    # |turn_rate| above this (deg) → is_shift

# Loss weighting.
W_POSITION = 1.0
W_TURN     = 0.5
W_HEADING  = 0.3

# Tracks to spotlight in the rollout contact sheet.
PRESET_TRACKS = [50, 24, 32, 17, 37, 45, 25, 31, 49, 51]

# Feature / target column lists. ORDER MATTERS — used both when
# scaling and when assembling rollout-time feature rows.
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
