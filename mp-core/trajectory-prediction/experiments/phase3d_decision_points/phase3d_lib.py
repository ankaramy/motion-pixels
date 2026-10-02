"""
MOTION PIXELS - PHASE 3D: decision-point architectural signal test.

Changes the UNIT OF ANALYSIS: instead of every frame, evaluate architectural
turn signal only at spatial decision moments (near obstacle, near boundary,
high lateral asymmetry, constrained corridor, approaching a sharp future turn).

Reuses Phase 3 / 3C machinery (labels, motion features, classifier helpers,
metrics). Analysis only: modifies no mask, encoder, encoded output, dataset, or
model; does not touch Phase 3A/B/C outputs.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P3_DIR = os.path.join(os.path.dirname(HERE), "phase3_spatial_signal_recovery")
P3C_DIR = os.path.join(os.path.dirname(HERE), "phase3c_feature_engineering_audit")
sys.path.insert(0, P3_DIR)
sys.path.insert(0, P3C_DIR)
import phase3_lib as P3       # noqa: E402  (labels, motion, metrics, assemble)
import phase3c_lib as C3      # noqa: E402  (make_model, run_fold, stratified_cap)

RECORDINGS = P3.RECORDINGS
LABELS = P3.LABELS

# Feature sets (same definitions as Phase 3)
FEATURE_SETS = {
    "motion": P3.MOTION_FEATURES,
    "architecture": P3.ARCH_FEATURES,
    "motion_plus_architecture": P3.MOTION_FEATURES + P3.ARCH_FEATURES,
}

# decision-point thresholds (fixed where given by the brief; data-driven where asked)
NEAR_OBSTACLE_M = 3.0
NEAR_BOUNDARY_M = 3.0
FUTURE_TURN_DEG = 30.0


def prepare(used):
    """Add derived columns needed for decision-point filters."""
    cl = used["clearance_left_v3_m"].to_numpy(float)
    cr = used["clearance_right_v3_m"].to_numpy(float)
    used = used.copy()
    used["corridor_width_v3_m"] = cl + cr
    used["abs_asymmetry_v3"] = used["clearance_asymmetry_v3"].abs()
    used["abs_future_turn_deg"] = used["delta_heading_deg"].abs()
    return used


def subset_thresholds(used):
    """Data-driven thresholds (documented): asymmetry p75, corridor-width p25."""
    return {
        "asymmetry_p75": float(np.percentile(used["abs_asymmetry_v3"].to_numpy(), 75)),
        "corridor_width_p25": float(np.percentile(used["corridor_width_v3_m"].to_numpy(), 25)),
    }


def subset_masks(used, thr):
    """Return dict subset_name -> boolean mask (numpy)."""
    a = used["dist_to_obstacle_v3_m"].to_numpy(float) < NEAR_OBSTACLE_M
    b = used["dist_to_walkable_boundary_v3_m"].to_numpy(float) < NEAR_BOUNDARY_M
    c = used["abs_asymmetry_v3"].to_numpy(float) > thr["asymmetry_p75"]
    d = used["corridor_width_v3_m"].to_numpy(float) < thr["corridor_width_p25"]
    e = used["abs_future_turn_deg"].to_numpy(float) > FUTURE_TURN_DEG
    masks = {
        "A_near_obstacle": a,
        "B_near_boundary": b,
        "C_high_asymmetry": c,
        "D_constrained_corridor": d,
        "E_approaching_turn": e,
    }
    masks["DecisionUnion"] = a | b | c | d | e
    return masks


def loro(sub_df, feats, model_name, rng, min_test=200):
    """Leave-One-Recording-Out on a subset. Returns list of per-fold dicts."""
    rows = []
    for held in RECORDINGS:
        tr = sub_df[sub_df.recording != held]
        te = sub_df[sub_df.recording == held]
        if len(te) < min_test or te["turn_label"].nunique() < 2 or tr["turn_label"].nunique() < 2:
            rows.append({"held_out": held, "n_test": int(len(te)),
                         "balanced_accuracy": np.nan, "accuracy": np.nan,
                         "macro_f1": np.nan, "skipped": True})
            continue
        met, _ = C3.run_fold(model_name, feats, tr, te, rng)
        rows.append({"held_out": held, "n_test": int(len(te)),
                     "balanced_accuracy": met["balanced_accuracy"],
                     "accuracy": met["accuracy"], "macro_f1": met["macro_f1"],
                     "majority_baseline_acc": met["majority_baseline_acc"],
                     "skipped": False})
    return rows
