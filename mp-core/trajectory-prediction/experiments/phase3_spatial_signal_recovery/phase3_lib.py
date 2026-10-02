"""
MOTION PIXELS - PHASE 3: Spatial Signal Recovery (turn classification)
======================================================================

Lightweight DIAGNOSTIC classifier stage. Tests whether Encoder V3 architectural
features carry measurable behavioural signal for pedestrian TURNING direction
(LEFT / STRAIGHT / RIGHT), via Leave-One-Recording-Out cross-validation.

This is NOT Model C, NOT trajectory rollout, NOT master-dataset rebuilding.
It only reads the V3 (and, for comparison, old V1) encoded CSVs and trains
simple classifiers. It never modifies any encoder, encoded output, or model.
"""

import os
import json

import numpy as np
import pandas as pd

DATASETS = r"C:\Users\OWNER\Desktop\new_datasets"
V3_ROOT = os.path.join(DATASETS, "Barcelona_v3_manual_encoded")
V1_ROOT = os.path.join(DATASETS, "Barcelona_v1_encoded")

RECORDINGS = [
    "stairs_montjuic_01",
    "red_bridge_combined_01",
    "esplanade_espanya_01",
    "placa_espanya_01",
    "placa_catalunya_01",
]

LABELS = ["LEFT", "STRAIGHT", "RIGHT"]

# feature sets
MOTION_FEATURES = ["speed_v3", "heading_sin", "heading_cos", "turn_rate_v3"]
ARCH_FEATURES = [
    "dist_to_obstacle_v3_m", "dist_to_walkable_boundary_v3_m",
    "clearance_forward_v3_m", "clearance_left_v3_m", "clearance_right_v3_m",
    "clearance_asymmetry_v3",
    "obstacle_bearing_sin_v3", "obstacle_bearing_cos_v3",
    "boundary_bearing_sin_v3", "boundary_bearing_cos_v3",
    "inside_walkable_v3", "inside_obstacle_v3",
]
COMBINED_FEATURES = MOTION_FEATURES + ARCH_FEATURES
OLD_SPATIAL_FEATURES = ["dist_to_obstacle", "dist_to_boundary", "dist_to_entrance"]

EXPERIMENTS = {
    "A_motion_only": MOTION_FEATURES,
    "B_architecture_only": ARCH_FEATURES,
    "C_motion_plus_architecture": COMBINED_FEATURES,
    "OLD_spatial_only": OLD_SPATIAL_FEATURES,
}


def wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


def v3_csv(rid):
    return os.path.join(V3_ROOT, rid, "spatial_v3_manual", "trajectories_encoded_v3.csv")


def v1_csv(rid):
    return os.path.join(V1_ROOT, rid, "spatial_v21C", "trajectories_encoded.csv")


def build_recording(rid, horizon, thr_deg):
    """Load one recording's V3 CSV, derive motion features + turn label, attach
    old spatial features. Returns a DataFrame with features, label, recording."""
    df = pd.read_csv(v3_csv(rid))
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    n_total = len(df)

    g = df.groupby("track_id")
    dt = g["time_s"].diff()
    dx = g["world_x"].diff()
    dy = g["world_y"].diff()
    dist = np.sqrt(dx**2 + dy**2)
    df["speed_v3"] = dist / dt.replace(0, np.nan)

    head = df["heading_v3_rad"].to_numpy()
    df["heading_sin"] = np.sin(head)
    df["heading_cos"] = np.cos(head)
    # past instantaneous turn rate (t-1 -> t), wrapped
    df["turn_rate_v3"] = wrap(g["heading_v3_rad"].diff().to_numpy())

    # future heading at t+horizon within same track (frame-exact)
    fut_head = g["heading_v3_rad"].shift(-horizon).to_numpy()
    fut_frame = g["frame"].shift(-horizon).to_numpy()
    gap_ok = (fut_frame - df["frame"].to_numpy()) == horizon
    delta = wrap(fut_head - head)
    thr = np.deg2rad(thr_deg)
    label = np.full(len(df), "STRAIGHT", dtype=object)
    label[delta > thr] = "LEFT"
    label[delta < -thr] = "RIGHT"

    df["delta_heading_deg"] = np.rad2deg(delta)
    df["turn_label"] = label

    # label validity
    valid = (df["heading_valid_v3"] == 1).to_numpy()
    valid &= np.isfinite(fut_head) & gap_ok
    df["_label_valid"] = valid

    # attach old spatial (merge by track_id, frame) if available
    if os.path.exists(v1_csv(rid)):
        old = pd.read_csv(v1_csv(rid),
                          usecols=lambda c: c in (["track_id", "frame"] + OLD_SPATIAL_FEATURES))
        df = df.merge(old, on=["track_id", "frame"], how="left")

    df["recording"] = rid
    df.attrs["n_total"] = n_total
    return df


def assemble(horizon, thr_deg):
    """Build the common, comparable row set across all recordings.

    A row is kept if its label is valid AND all motion+architecture features are
    finite (this drops out-of-bounds rows, so A/B/C are evaluated on the SAME
    rows). Old-spatial features are kept where present for the secondary
    comparison on the same row set.
    """
    frames = [build_recording(r, horizon, thr_deg) for r in RECORDINGS]
    totals = {r: f.attrs["n_total"] for r, f in zip(RECORDINGS, frames)}
    df = pd.concat(frames, ignore_index=True)

    need = list(dict.fromkeys(COMBINED_FEATURES))  # motion+arch union
    finite = np.isfinite(df[need].to_numpy()).all(axis=1)
    keep = df["_label_valid"].to_numpy() & finite
    used = df[keep].reset_index(drop=True)
    return used, totals


# -------------------- metrics --------------------
def confusion(y_true, y_pred):
    idx = {l: i for i, l in enumerate(LABELS)}
    m = np.zeros((3, 3), int)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return m


def metrics_from_confusion(m):
    support = m.sum(axis=1)
    total = m.sum()
    acc = np.trace(m) / total if total else 0.0
    recalls, precisions, f1s = [], [], []
    for i in range(3):
        tp = m[i, i]
        fn = support[i] - tp
        fp = m[:, i].sum() - tp
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        recalls.append(rec); precisions.append(prec); f1s.append(f1)
    balacc = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(balacc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {LABELS[i]: {"precision": round(precisions[i], 4),
                                  "recall": round(recalls[i], 4),
                                  "f1": round(f1s[i], 4),
                                  "support": int(support[i])} for i in range(3)},
    }
