"""
MOTION PIXELS - PHASE 3C: V3 feature-engineering audit + signal re-test.

Reuses the Phase 3 label / motion / metric machinery (phase3_lib) and adds:
  - a feature-distribution audit of the 12 original V3 architectural features
  - 14 documented DERIVED architectural features
  - feature sets A/B/C/D/E/OLD for the classifier re-run

Diagnostic only. Modifies no mask, encoder, encoded output, dataset, or model.
"""
import os
import sys

import numpy as np
import pandas as pd

# import the Phase 3 library (labels, motion features, metrics)
P3_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "phase3_spatial_signal_recovery")
sys.path.insert(0, P3_DIR)
import phase3_lib as P3  # noqa: E402

EPS = 1e-6
CLIP = 10.0  # clip bound for ratio / pressure features (documented)

RECORDINGS = P3.RECORDINGS
LABELS = P3.LABELS
MOTION_FEATURES = P3.MOTION_FEATURES
ARCH_ORIG = P3.ARCH_FEATURES
OLD_SPATIAL = P3.OLD_SPATIAL_FEATURES

# (name, formula string, clip) -- single source of truth for compute + report
DERIVED_SPEC = [
    ("local_corridor_width_v3_m", "clearance_left + clearance_right", None),
    ("min_side_clearance_v3_m", "min(clearance_left, clearance_right)", None),
    ("max_side_clearance_v3_m", "max(clearance_left, clearance_right)", None),
    ("side_bias_ratio_v3", "(clearance_right - clearance_left) / (clearance_right + clearance_left + eps)", None),
    ("forward_clearance_ratio_v3", "clearance_forward / (local_corridor_width + eps)", f"[0,{CLIP}]"),
    ("obstacle_pressure_v3", "1 / (dist_to_obstacle + eps)", f"[0,{CLIP}]"),
    ("boundary_pressure_v3", "1 / (dist_to_walkable_boundary + eps)", f"[0,{CLIP}]"),
    ("normalized_obstacle_distance_v3", "dist_to_obstacle / (local_corridor_width + eps)", f"[0,{CLIP}]"),
    ("normalized_boundary_distance_v3", "dist_to_walkable_boundary / (local_corridor_width + eps)", f"[0,{CLIP}]"),
    ("walkable_openness_v3", "mean(clearance_forward, clearance_left, clearance_right)", None),
    ("turn_affordance_left_v3", "clearance_left / (clearance_forward + eps)", f"[0,{CLIP}]"),
    ("turn_affordance_right_v3", "clearance_right / (clearance_forward + eps)", f"[0,{CLIP}]"),
    ("obstacle_ahead_pressure_v3", "obstacle_pressure_v3 * max(0, obstacle_bearing_cos_v3)", None),
    ("boundary_ahead_pressure_v3", "boundary_pressure_v3 * max(0, boundary_bearing_cos_v3)", None),
]
DERIVED_FEATURES = [d[0] for d in DERIVED_SPEC]

ENHANCED_ARCH = ARCH_ORIG + DERIVED_FEATURES

EXPERIMENTS = {
    "A_motion_only": MOTION_FEATURES,
    "B_arch_original": ARCH_ORIG,
    "C_motion_plus_arch_original": MOTION_FEATURES + ARCH_ORIG,
    "D_enhanced_arch_only": ENHANCED_ARCH,
    "E_motion_plus_enhanced_arch": MOTION_FEATURES + ENHANCED_ARCH,
    "OLD_spatial_only": OLD_SPATIAL,
}


def add_derived_features(df):
    """Add the 14 derived architectural features. All inputs are finite on the
    common row set; ratio/pressure features are clipped to [0, CLIP]."""
    cf = df["clearance_forward_v3_m"].to_numpy(float)
    cl = df["clearance_left_v3_m"].to_numpy(float)
    cr = df["clearance_right_v3_m"].to_numpy(float)
    do = df["dist_to_obstacle_v3_m"].to_numpy(float)
    db = df["dist_to_walkable_boundary_v3_m"].to_numpy(float)
    obc = df["obstacle_bearing_cos_v3"].to_numpy(float)
    bbc = df["boundary_bearing_cos_v3"].to_numpy(float)

    width = cl + cr
    df["local_corridor_width_v3_m"] = width
    df["min_side_clearance_v3_m"] = np.minimum(cl, cr)
    df["max_side_clearance_v3_m"] = np.maximum(cl, cr)
    df["side_bias_ratio_v3"] = (cr - cl) / (cr + cl + EPS)
    df["forward_clearance_ratio_v3"] = np.clip(cf / (width + EPS), 0, CLIP)
    op = np.clip(1.0 / (do + EPS), 0, CLIP)
    bp = np.clip(1.0 / (db + EPS), 0, CLIP)
    df["obstacle_pressure_v3"] = op
    df["boundary_pressure_v3"] = bp
    df["normalized_obstacle_distance_v3"] = np.clip(do / (width + EPS), 0, CLIP)
    df["normalized_boundary_distance_v3"] = np.clip(db / (width + EPS), 0, CLIP)
    df["walkable_openness_v3"] = (cf + cl + cr) / 3.0
    df["turn_affordance_left_v3"] = np.clip(cl / (cf + EPS), 0, CLIP)
    df["turn_affordance_right_v3"] = np.clip(cr / (cf + EPS), 0, CLIP)
    df["obstacle_ahead_pressure_v3"] = op * np.maximum(0.0, obc)
    df["boundary_ahead_pressure_v3"] = bp * np.maximum(0.0, bbc)
    return df


def assemble(horizon, thr_deg):
    """Common, comparable row set (label valid AND motion+orig-arch finite),
    with derived features added."""
    used, totals = P3.assemble(horizon, thr_deg)
    used = add_derived_features(used)
    return used, totals


# -------------------- feature distribution audit --------------------
def audit_feature(series):
    a = pd.to_numeric(series, errors="coerce").to_numpy(float)
    n = a.size
    miss = float(np.mean(~np.isfinite(a))) * 100 if n else 100.0
    fa = a[np.isfinite(a)]
    if fa.size == 0:
        return {"count": n, "missing_pct": round(miss, 3)}
    rounded_unique = int(np.unique(np.round(fa, 3)).size)
    lo, hi = float(fa.min()), float(fa.max())
    frac_at_max = float(np.mean(np.isclose(fa, hi))) * 100
    frac_at_min = float(np.mean(np.isclose(fa, lo))) * 100
    std = float(fa.std())
    warn = []
    if std < 1e-3 or rounded_unique <= 2:
        warn_const = True
        warn = "NEAR-CONSTANT"
    else:
        warn = ""
    sat = ""
    if frac_at_max > 20:
        sat = f"SATURATED_MAX({frac_at_max:.0f}%@{hi:.2f})"
    elif frac_at_min > 40:
        sat = f"PILED_MIN({frac_at_min:.0f}%@{lo:.2f})"
    return {
        "count": n,
        "missing_pct": round(miss, 3),
        "min": round(lo, 4),
        "p01": round(float(np.percentile(fa, 1)), 4),
        "p05": round(float(np.percentile(fa, 5)), 4),
        "median": round(float(np.median(fa)), 4),
        "mean": round(float(fa.mean()), 4),
        "p95": round(float(np.percentile(fa, 95)), 4),
        "p99": round(float(np.percentile(fa, 99)), 4),
        "max": round(hi, 4),
        "std": round(std, 4),
        "n_unique_rounded": rounded_unique,
        "frac_at_min_pct": round(frac_at_min, 2),
        "frac_at_max_pct": round(frac_at_max, 2),
        "near_constant_warning": warn,
        "saturation_warning": sat,
    }


# -------------------- classifier helpers --------------------
from sklearn.pipeline import Pipeline               # noqa: E402
from sklearn.preprocessing import StandardScaler    # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

SEED = 0
RF_CAP = 120000


def make_model(name):
    if name == "logistic":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000,
                                                    class_weight="balanced"))])
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                      class_weight="balanced", n_jobs=-1,
                                      random_state=SEED)
    raise ValueError(name)


def stratified_cap(X, y, cap, rng):
    if len(X) <= cap:
        return X, y
    idx = np.arange(len(X))
    keep = []
    for lab in LABELS:
        li = idx[y == lab]
        take = min(len(li), max(1, int(round(cap * len(li) / len(X)))))
        keep.append(rng.choice(li, size=take, replace=False))
    keep = np.concatenate(keep)
    return X[keep], y[keep]


def run_fold(model_name, feats, train_df, test_df, rng):
    Xtr = train_df[feats].to_numpy(float)
    ytr = train_df["turn_label"].to_numpy()
    Xte = test_df[feats].to_numpy(float)
    yte = test_df["turn_label"].to_numpy()
    if model_name == "random_forest":
        Xtr, ytr = stratified_cap(Xtr, ytr, RF_CAP, rng)
    model = make_model(model_name)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    cm = P3.confusion(yte, pred)
    met = P3.metrics_from_confusion(cm)
    vals, cnts = np.unique(ytr, return_counts=True)
    maj = vals[np.argmax(cnts)]
    met["majority_baseline_acc"] = round(float((yte == maj).mean()), 4)
    return met, cm
