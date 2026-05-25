"""
diagnose_auto_spatial_encoding.py
---------------------------------
Read-only diagnostic of the *automatic* spatial encoding system.

Inspects existing artefacts under mp-data/processed/encoded/ and the
trajectory-prediction code folder. Does NOT retrain, edit encoders, or
overwrite CSVs.

Outputs:
  mp-data/processed/encoded/auto_spatial_encoding_diagnostic_report.md
  mp-data/processed/encoded/auto_spatial_encoding_diagnostic_summary.json
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import cv2  # noqa: F401
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

try:
    from PIL import Image
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE     = Path(__file__).resolve().parent
MP_ROOT  = HERE.parent.parent
MP_DATA  = MP_ROOT / "mp-data"
ENCODED  = MP_DATA / "processed" / "encoded"
CODE_DIR = MP_ROOT / "mp-core" / "trajectory-prediction"

REPORT_MD   = ENCODED / "auto_spatial_encoding_diagnostic_report.md"
REPORT_JSON = ENCODED / "auto_spatial_encoding_diagnostic_summary.json"

# Column name aliases (the auto CSV uses {track_id,frame}; the manual uses
# {person_id,frame_number}; visualisation code uses {pid,frame_idx} etc.).
ALIAS = {
    "track_id":    ["track_id", "person_id", "pid", "id", "ped_id"],
    "frame":       ["frame", "frame_number", "frame_idx", "fid"],
    "world_x":     ["world_x", "wx", "x_world"],
    "world_y":     ["world_y", "wy", "y_world"],
    "image_x":     ["image_x", "img_x", "px"],
    "image_y":     ["image_y", "img_y", "py"],
    "foot_x":      ["foot_x", "fx"],
    "foot_y":      ["foot_y", "fy"],
    "delta_x":     ["delta_x", "dx"],
    "delta_y":     ["delta_y", "dy"],
    "speed":       ["speed", "v", "velocity"],
    "dist_to_obstacle": ["dist_to_obstacle", "dist_obstacle", "obs_dist"],
    "dist_to_boundary": ["dist_to_boundary", "dist_boundary", "bnd_dist"],
    "dist_to_entrance": ["dist_to_entrance", "dist_entrance", "ent_dist"],
}


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def find_alias(cols: List[str], canonical: str) -> Optional[str]:
    """Return the actual column name matching a canonical alias, or None."""
    for name in ALIAS.get(canonical, [canonical]):
        if name in cols:
            return name
    return None


def safe(fn, *args, **kwargs):
    """Run fn and capture any error as a string instead of crashing."""
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def jsonable(x):
    """Recursively coerce numpy/pandas types into JSON-safe primitives."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (pd.Timestamp,)):
        return str(x)
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def numeric_summary(s: pd.Series) -> Dict[str, Any]:
    s = pd.to_numeric(s, errors="coerce")
    n = int(s.size)
    n_nan = int(s.isna().sum())
    if n - n_nan == 0:
        return {"n": n, "n_nan": n_nan, "min": None, "max": None,
                "mean": None, "median": None, "std": None,
                "n_unique": int(s.nunique(dropna=True))}
    return {
        "n": n, "n_nan": n_nan,
        "min": float(s.min()), "max": float(s.max()),
        "mean": float(s.mean()), "median": float(s.median()),
        "std": float(s.std()),
        "n_unique": int(s.nunique(dropna=True)),
    }


# --------------------------------------------------------------------------- #
# 1. File location
# --------------------------------------------------------------------------- #
def locate_files() -> Dict[str, Any]:
    info = {}

    csv_candidates = {
        "auto_encoded":    ENCODED / "trajectories_encoded_auto.csv",
        "manual_encoded":  ENCODED / "trajectories_encoded.csv",
        "motion_dataset":  ENCODED / "motion_dataset.csv",
        "motion_dataset_v2": ENCODED / "motion_dataset_v2.csv",
    }
    info["csvs"] = {k: {"path": str(p), "exists": p.exists(),
                        "size_bytes": p.stat().st_size if p.exists() else None}
                    for k, p in csv_candidates.items()}

    misc = {
        "spatial_maps_dir": ENCODED / "spatial_feature_maps",
        "debug_overlay":    ENCODED / "debug_overlay.png",
        "feature_corr":     ENCODED / "feature_correlation.png",
        "feature_hist":     ENCODED / "feature_histograms.png",
        "diagnostics_md":   ENCODED / "diagnostics_summary.md",
        "feature_summary":  ENCODED / "feature_summary_v2.md",
        "schema_summary":   ENCODED / "schema_summary.json",
    }
    info["misc"] = {k: {"path": str(p), "exists": p.exists()}
                    for k, p in misc.items()}

    spatial_dir = misc["spatial_maps_dir"]
    if spatial_dir.exists() and spatial_dir.is_dir():
        files = sorted([p for p in spatial_dir.iterdir() if p.is_file()])
        info["spatial_map_files"] = [str(p.relative_to(MP_ROOT)) for p in files]
    else:
        info["spatial_map_files"] = []

    # Code files relevant to spatial / encoding.
    code_hits = []
    keywords = re.compile(
        r"(auto|encode|obstacle|boundary|entrance|distance_transform|"
        r"\bcv2\b|threshold|morpholog)", re.IGNORECASE)
    for py in sorted(CODE_DIR.rglob("*.py")):
        try:
            txt = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        name_match = keywords.search(py.name)
        body_match = keywords.search(txt)
        if name_match or body_match:
            code_hits.append(str(py.relative_to(MP_ROOT)))
    info["code_hits"] = code_hits

    return info


# --------------------------------------------------------------------------- #
# 2. CSV inspection
# --------------------------------------------------------------------------- #
KEY_NUMERIC_COLS = [
    "dist_to_obstacle", "dist_to_boundary", "dist_to_entrance",
    "world_x", "world_y", "delta_x", "delta_y", "speed",
]


def inspect_csv(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False}

    rep: Dict[str, Any] = {"exists": True, "path": str(path)}
    try:
        df = pd.read_csv(path)
    except Exception as e:
        rep["error"] = f"{type(e).__name__}: {e}"
        return rep

    rep["row_count"]    = int(len(df))
    rep["columns"]      = list(df.columns)
    rep["dup_rows"]     = int(df.duplicated().sum())
    rep["missing_per_col"] = {c: int(df[c].isna().sum()) for c in df.columns}

    tid = find_alias(df.columns, "track_id")
    rep["track_col"]   = tid
    rep["n_unique_tracks"] = int(df[tid].nunique()) if tid else None

    frame_col = find_alias(df.columns, "frame")
    rep["frame_col"]   = frame_col

    # Numeric summaries.
    summaries = {}
    for canon in KEY_NUMERIC_COLS:
        col = find_alias(df.columns, canon)
        if col is None:
            summaries[canon] = {"present": False}
        else:
            summaries[canon] = {"present": True, "actual_col": col,
                                **numeric_summary(df[col])}

    rep["key_features"] = summaries

    # All numeric columns (compact).
    num_df = df.select_dtypes(include="number")
    rep["numeric_columns"] = list(num_df.columns)
    rep["all_numeric_summary"] = {c: numeric_summary(num_df[c])
                                  for c in num_df.columns}

    return rep


# --------------------------------------------------------------------------- #
# 3. Manual vs Auto comparison
# --------------------------------------------------------------------------- #
def compare_manual_vs_auto(auto_path: Path,
                           manual_path: Path) -> Dict[str, Any]:
    rep: Dict[str, Any] = {"compared": False}
    if not (auto_path.exists() and manual_path.exists()):
        rep["reason"] = "one or both CSVs missing"
        return rep

    try:
        df_a = pd.read_csv(auto_path)
        df_m = pd.read_csv(manual_path)
    except Exception as e:
        rep["error"] = f"{type(e).__name__}: {e}"
        return rep

    tid_a   = find_alias(df_a.columns, "track_id")
    tid_m   = find_alias(df_m.columns, "track_id")
    frame_a = find_alias(df_a.columns, "frame")
    frame_m = find_alias(df_m.columns, "frame")
    if not (tid_a and tid_m and frame_a and frame_m):
        rep["reason"] = (f"missing key cols (auto tid/frame={tid_a}/{frame_a}, "
                         f"manual tid/frame={tid_m}/{frame_m})")
        return rep

    # Inner-join on (track, frame) to compare aligned rows.
    a = df_a.rename(columns={tid_a: "_tid", frame_a: "_fr"})
    m = df_m.rename(columns={tid_m: "_tid", frame_m: "_fr"})
    common_features = []
    for canon in ["dist_to_obstacle", "dist_to_boundary",
                  "dist_to_entrance", "world_x", "world_y"]:
        ca = find_alias(a.columns, canon)
        cm = find_alias(m.columns, canon)
        if ca and cm:
            common_features.append((canon, ca, cm))

    if not common_features:
        rep["reason"] = "no overlapping spatial features"
        return rep

    rep["compared"] = True
    rep["auto_rows"]   = int(len(a))
    rep["manual_rows"] = int(len(m))

    keep_a = ["_tid", "_fr"] + [ca for _, ca, _ in common_features]
    keep_m = ["_tid", "_fr"] + [cm for _, _, cm in common_features]
    a2 = a[keep_a].copy()
    m2 = m[keep_m].copy()
    merged = pd.merge(a2, m2, on=["_tid", "_fr"], suffixes=("_auto", "_manual"))
    rep["joined_rows"] = int(len(merged))

    per_feature = {}
    for canon, ca, cm in common_features:
        a_col = ca if ca != cm else f"{ca}_auto"
        m_col = cm if ca != cm else f"{cm}_manual"
        sa = pd.to_numeric(merged[a_col], errors="coerce")
        sm = pd.to_numeric(merged[m_col], errors="coerce")
        diff = (sa - sm).abs()
        valid = sa.notna() & sm.notna()
        n_valid = int(valid.sum())
        entry = {
            "n_valid_pairs": n_valid,
            "auto_mean":   float(sa.mean()) if n_valid else None,
            "manual_mean": float(sm.mean()) if n_valid else None,
            "auto_std":    float(sa.std())  if n_valid else None,
            "manual_std":  float(sm.std())  if n_valid else None,
            "mean_abs_diff":   float(diff.mean()) if n_valid else None,
            "median_abs_diff": float(diff.median()) if n_valid else None,
            "max_abs_diff":    float(diff.max()) if n_valid else None,
            "pearson_r": None,
        }
        if n_valid > 2 and sa[valid].std() > 0 and sm[valid].std() > 0:
            entry["pearson_r"] = float(sa[valid].corr(sm[valid]))
        per_feature[canon] = entry
    rep["per_feature"] = per_feature

    # Flag suspicious auto features (use entire auto CSV, not just joined).
    flags = {}
    for canon, ca, _ in common_features:
        col = pd.to_numeric(df_a[ca], errors="coerce")
        n = int(col.size)
        n_nan = int(col.isna().sum())
        std = float(col.std()) if n - n_nan > 1 else 0.0
        rng = (float(col.max()) - float(col.min())) if n - n_nan else 0.0
        mean_val = float(col.mean()) if n - n_nan else None
        flag = []
        if n - n_nan == 0:
            flag.append("all-NaN")
        if n_nan > 0.5 * n:
            flag.append("majority-NaN")
        if std < 1e-6 and n - n_nan > 0:
            flag.append("constant")
        if rng < 1e-3 and n - n_nan > 0:
            flag.append("near-constant")
        # extreme outliers: max > 100 * mean (only meaningful for positive distances)
        if (mean_val is not None and mean_val > 1e-9
                and float(col.max()) > 100 * mean_val):
            flag.append("extreme-outliers")
        flags[canon] = flag
    rep["auto_flags"] = flags

    return rep


# --------------------------------------------------------------------------- #
# 4. Spatial feature map inspection
# --------------------------------------------------------------------------- #
def inspect_spatial_maps(map_files: List[Path]) -> List[Dict[str, Any]]:
    out = []
    for p in map_files:
        entry: Dict[str, Any] = {
            "name": p.name, "path": str(p.relative_to(MP_ROOT)),
            "size_bytes": p.stat().st_size,
        }
        suf = p.suffix.lower()
        try:
            if suf == ".npy":
                arr = np.load(p, allow_pickle=False)
                entry.update(describe_array(arr))
            elif suf == ".csv":
                df = pd.read_csv(p)
                entry["shape"] = [int(len(df)), int(len(df.columns))]
                entry["columns"] = list(df.columns)
                num = df.select_dtypes(include="number")
                if num.shape[1] > 0:
                    arr = num.to_numpy(dtype=np.float64).flatten()
                    entry.update(describe_array(arr))
            elif suf in {".png", ".jpg", ".jpeg"}:
                arr = read_image_array(p)
                if arr is None:
                    entry["error"] = "cannot read image (no cv2 / PIL)"
                else:
                    entry.update(describe_array(arr))
                    entry.update(image_visual_summary(arr))
            else:
                entry["note"] = "unknown extension; skipped"
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
        out.append(entry)
    return out


def read_image_array(p: Path) -> Optional[np.ndarray]:
    if HAVE_CV2:
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if arr is not None:
            return arr
    if HAVE_PIL:
        return np.array(Image.open(p))
    return None


def describe_array(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr)
    entry = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    flat = arr.astype(np.float64).flatten()
    if flat.size == 0:
        return entry
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        entry["all_non_finite"] = True
        return entry
    entry["min"]  = float(finite.min())
    entry["max"]  = float(finite.max())
    entry["mean"] = float(finite.mean())
    entry["std"]  = float(finite.std())
    uniq = np.unique(finite[:200_000])  # cap for speed
    entry["n_unique_sample"] = int(uniq.size)
    # Classify values
    is_binary    = uniq.size <= 2
    is_all_zero  = float(finite.max()) == 0.0
    is_all_one   = float(finite.min()) == float(finite.max()) == 1.0
    is_degenerate = float(finite.std()) < 1e-9
    entry["value_class"] = (
        "binary"        if is_binary else
        "all-zero"      if is_all_zero else
        "all-one"       if is_all_one else
        "degenerate"    if is_degenerate else
        "continuous"
    )
    return entry


def image_visual_summary(arr: np.ndarray) -> Dict[str, Any]:
    out = {}
    a = np.asarray(arr)
    if a.ndim == 2:
        flat = a.flatten()
    elif a.ndim == 3:
        # H, W, C  → fold colour-tuples into a 1-D code so "unique colours" makes sense.
        flat_rgb = a.reshape(-1, a.shape[2])
        codes = flat_rgb.astype(np.int64) @ \
                (256 ** np.arange(a.shape[2]))[::-1]
        flat = codes
    else:
        return out
    n = flat.size
    sample = flat[:: max(1, n // 200_000)]  # subsample for unique count
    out["n_unique_colours_sample"] = int(np.unique(sample).size)
    # Mostly-flat heuristic: single most-common value > 90% of pixels
    vals, counts = np.unique(sample, return_counts=True)
    if counts.size:
        top_frac = float(counts.max() / counts.sum())
        out["top_value_fraction"] = top_frac
        out["mostly_flat"] = top_frac > 0.90
    # Mostly black / white
    if a.dtype.kind in {"u", "i"}:
        a_f = a.astype(np.float64)
    else:
        a_f = a.astype(np.float64)
    out["mean_intensity"] = float(a_f.mean())
    out["mostly_black"] = out["mean_intensity"] < 20.0
    out["mostly_white"] = out["mean_intensity"] > 235.0
    return out


# --------------------------------------------------------------------------- #
# 5. Debug image inspection
# --------------------------------------------------------------------------- #
def inspect_debug_images(paths: List[Path]) -> List[Dict[str, Any]]:
    out = []
    for p in paths:
        if not p.exists():
            out.append({"name": p.name, "exists": False})
            continue
        entry: Dict[str, Any] = {"name": p.name, "exists": True,
                                 "size_bytes": p.stat().st_size}
        arr = read_image_array(p)
        if arr is None:
            entry["error"] = "cannot read (no cv2 / PIL)"
            out.append(entry); continue
        entry.update(describe_array(arr))
        entry.update(image_visual_summary(arr))
        entry["empty_or_corrupt"] = bool(entry.get("mostly_flat", False)
                                         and (entry.get("mostly_black")
                                              or entry.get("mostly_white")))
        out.append(entry)
    return out


# --------------------------------------------------------------------------- #
# 6. Code analysis
# --------------------------------------------------------------------------- #
def analyse_code(code_files: List[str]) -> Dict[str, Any]:
    rep: Dict[str, Any] = {"files": []}

    method_keys = {
        "image_thresholding": [r"cv2\.threshold", r"adaptiveThreshold",
                               r"otsu", r"np\.where\("],
        "morphology":         [r"morphologyEx", r"erode\(", r"dilate\(",
                               r"opening", r"closing"],
        "distance_transform": [r"distanceTransform", r"distance_transform"],
        "kdtree":             [r"cKDTree", r"KDTree"],
        "manual_points":      [r"manual_points", r"hand_picked",
                               r"ANCHOR", r"anchor_points"],
        "homography":         [r"findHomography", r"perspectiveTransform",
                               r"\bhomography\b", r"\bH\s*=\s*np\."],
        "segmentation":       [r"segmentation", r"mask", r"semantic"],
        "trajectory_use":     [r"world_x", r"trajectory", r"track"],
    }

    for relpath in code_files:
        p = MP_ROOT / relpath
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            rep["files"].append({"path": relpath,
                                 "error": f"{type(e).__name__}: {e}"})
            continue

        methods_found = []
        for tag, patterns in method_keys.items():
            for pat in patterns:
                if re.search(pat, txt):
                    methods_found.append(tag); break

        # Hardcoded absolute paths
        hard_paths = re.findall(r"[A-Za-z]:[\\/][^\"'\s)]+", txt)
        # Suspicious constants — fixed thresholds, hardcoded image sizes etc.
        susp_consts = re.findall(
            r"\b(?:threshold|THRESH|EPS|epsilon|WALL|FLOOR|ALPHA|SCALE)\s*=\s*[\-\d.]+",
            txt)
        # Coordinate system hints
        coord_hints = []
        if "image_x" in txt and "world_x" in txt:
            coord_hints.append("uses both image_ and world_ coordinates")
        if "H @" in txt or "perspectiveTransform" in txt or "warpPerspective" in txt:
            coord_hints.append("applies a homography")
        if "flipud" in txt or "rot90" in txt or "transpose" in txt:
            coord_hints.append("flips/rotates an image axis")

        rep["files"].append({
            "path": relpath,
            "n_lines": txt.count("\n"),
            "methods_found": sorted(set(methods_found)),
            "hardcoded_paths": hard_paths[:10],
            "suspicious_constants": susp_consts[:15],
            "coord_hints": coord_hints,
        })

    # Best-guess for the auto encoder.
    likely_auto = None
    for f in rep["files"]:
        name = Path(f["path"]).name.lower()
        if "auto" in name and "encode" in name:
            likely_auto = f["path"]; break
    if likely_auto is None:
        for f in rep["files"]:
            if "auto" in Path(f["path"]).name.lower():
                likely_auto = f["path"]; break
    rep["likely_auto_encoder"] = likely_auto
    return rep


# --------------------------------------------------------------------------- #
# 7. Failure-mode detection
# --------------------------------------------------------------------------- #
def detect_failure_modes(auto_rep: Dict[str, Any],
                         manual_rep: Dict[str, Any],
                         compare_rep: Dict[str, Any],
                         map_reps: List[Dict[str, Any]],
                         debug_reps: List[Dict[str, Any]],
                         code_rep: Dict[str, Any]) -> List[str]:
    flags: List[str] = []

    # Distance-map degeneracy
    for m in map_reps:
        if m.get("value_class") in {"all-zero", "all-one", "degenerate"}:
            flags.append(f"distance maps degenerate: {m['name']} → "
                         f"{m['value_class']}")
        if m.get("mostly_flat"):
            flags.append(f"distance map mostly flat (top-value-fraction "
                         f"{m.get('top_value_fraction', 0):.2f}): {m['name']}")

    # Debug overlay empty
    for d in debug_reps:
        if d.get("empty_or_corrupt"):
            flags.append(f"debug image empty/flat: {d['name']}")

    # Per-CSV checks on auto
    auto_feats = (auto_rep or {}).get("key_features", {})
    for feat, info in auto_feats.items():
        if not info.get("present"):
            if feat in {"dist_to_obstacle", "dist_to_boundary"}:
                flags.append(f"auto CSV is missing the {feat} column")
            continue
        std = info.get("std")
        mn  = info.get("min"); mx = info.get("max")
        mean = info.get("mean")
        if info.get("n_unique") == 1:
            flags.append(f"auto {feat} is constant")
        elif std is not None and std < 1e-6:
            flags.append(f"auto {feat} has ~zero variance")
        if mean is not None and mn is not None and mx is not None:
            if mean > 1e-9 and mx > 100 * mean:
                flags.append(f"auto {feat} has extreme outliers "
                             f"(max={mx:.2f}, mean={mean:.2f})")

    # Comparison flags
    if compare_rep.get("compared"):
        for feat, info in compare_rep.get("per_feature", {}).items():
            r = info.get("pearson_r")
            mad = info.get("mean_abs_diff")
            if r is not None and r < 0.3:
                flags.append(f"auto vs manual {feat}: Pearson r={r:.2f} "
                             "(weak correlation)")
            if r is not None and r < 0.0:
                flags.append(f"auto vs manual {feat}: NEGATIVE correlation "
                             f"(r={r:.2f}) — sign flip likely")
            if mad is not None and info.get("manual_std"):
                if mad > 2 * info["manual_std"]:
                    flags.append(f"auto vs manual {feat}: mean-abs-diff "
                                 f"({mad:.2f}) > 2× manual std "
                                 f"({info['manual_std']:.2f})")
        for feat, fl in compare_rep.get("auto_flags", {}).items():
            for token in fl:
                flags.append(f"auto {feat}: {token}")

    # Inter-feature correlation among auto spatial features
    if auto_rep.get("exists"):
        try:
            df = pd.read_csv(auto_rep["path"])
            spatial_cols = [find_alias(df.columns, c) for c in
                            ("dist_to_obstacle", "dist_to_boundary",
                             "dist_to_entrance")]
            spatial_cols = [c for c in spatial_cols if c is not None]
            if len(spatial_cols) >= 2:
                corr = df[spatial_cols].apply(pd.to_numeric, errors="coerce")\
                                       .corr()
                for i, a in enumerate(spatial_cols):
                    for b in spatial_cols[i + 1:]:
                        r = corr.loc[a, b]
                        if pd.notna(r) and abs(r) > 0.95:
                            flags.append(f"spatial features highly correlated: "
                                         f"{a} ~ {b} (r={r:.2f})")
                        if pd.notna(r) and abs(r) > 0.999:
                            flags.append(f"auto features not meaningfully "
                                         f"different: {a} ≡ {b}")
        except Exception:
            pass

    # Coordinate-system mismatch hint
    if code_rep.get("likely_auto_encoder"):
        for f in code_rep["files"]:
            if f["path"] == code_rep["likely_auto_encoder"]:
                if ("uses both image_ and world_ coordinates"
                        in f.get("coord_hints", [])
                        and "applies a homography" not in f.get("coord_hints", [])):
                    flags.append("possible trajectory/image coordinate mismatch: "
                                 "auto encoder uses both world_ and image_ "
                                 "coordinates but no homography call detected")

    # Auto-vs-manual: missing entrance column on auto
    if (auto_rep.get("exists")
            and not auto_rep.get("key_features", {})
                            .get("dist_to_entrance", {}).get("present")
            and manual_rep.get("exists")
            and manual_rep.get("key_features", {})
                            .get("dist_to_entrance", {}).get("present")):
        flags.append("entry/exit detection unreliable: auto CSV has no "
                     "dist_to_entrance column (manual CSV does)")

    return flags


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
def fmt_num(x):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    try:
        f = float(x)
        if not np.isfinite(f):
            return "—"
        if abs(f) >= 1000 or (abs(f) < 0.01 and f != 0):
            return f"{f:.3e}"
        return f"{f:.3f}"
    except Exception:
        return str(x)


def render_csv_table(csv_label: str, rep: Dict[str, Any]) -> List[str]:
    out = [f"### CSV — `{csv_label}`", ""]
    if not rep.get("exists"):
        out += ["_File not present._", ""]
        return out
    if rep.get("error"):
        out += [f"**Could not read:** `{rep['error']}`", ""]
        return out

    out += [
        f"- Path: `{rep['path']}`",
        f"- Rows: **{rep['row_count']:,}**   ·   "
        f"Columns: **{len(rep['columns'])}**   ·   "
        f"Duplicates: **{rep['dup_rows']:,}**",
        f"- Track column: `{rep['track_col']}`   ·   "
        f"Frame column: `{rep['frame_col']}`   ·   "
        f"Unique tracks: "
        f"**{rep['n_unique_tracks'] if rep['n_unique_tracks'] is not None else '—'}**",
        "",
        "Columns: " + ", ".join(f"`{c}`" for c in rep["columns"]),
        "",
    ]
    miss = {c: n for c, n in rep["missing_per_col"].items() if n > 0}
    if miss:
        out += ["**Columns with missing values:** "
                + ", ".join(f"`{c}` ({n})" for c, n in miss.items()), ""]
    else:
        out += ["No missing values in any column.", ""]

    out += ["**Key feature summary:**", "",
            "| feature | present | min | max | mean | median | std | n_unique |",
            "|---|---|---|---|---|---|---|---|"]
    for feat in KEY_NUMERIC_COLS:
        info = rep["key_features"].get(feat, {"present": False})
        if not info.get("present"):
            out.append(f"| `{feat}` | no | — | — | — | — | — | — |")
        else:
            out.append(
                f"| `{feat}` (`{info['actual_col']}`) | yes | "
                f"{fmt_num(info['min'])} | {fmt_num(info['max'])} | "
                f"{fmt_num(info['mean'])} | {fmt_num(info['median'])} | "
                f"{fmt_num(info['std'])} | {info['n_unique']} |")
    out += [""]
    return out


def render_compare(rep: Dict[str, Any]) -> List[str]:
    out = ["## 3. Manual vs automatic encoding", ""]
    if not rep.get("compared"):
        out += [f"Comparison not run: {rep.get('reason', rep.get('error', '—'))}",
                ""]
        return out

    out += [
        f"- Auto rows: {rep['auto_rows']:,}   ·   "
        f"Manual rows: {rep['manual_rows']:,}   ·   "
        f"Joined rows (on track + frame): **{rep['joined_rows']:,}**",
        "",
        "**Per-feature comparison on joined rows:**", "",
        "| feature | pairs | auto mean | manual mean | mean |Δ| | "
        "median |Δ| | max |Δ| | Pearson r |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for feat, info in rep["per_feature"].items():
        out.append(
            f"| `{feat}` | {info['n_valid_pairs']:,} | "
            f"{fmt_num(info['auto_mean'])} | {fmt_num(info['manual_mean'])} | "
            f"{fmt_num(info['mean_abs_diff'])} | "
            f"{fmt_num(info['median_abs_diff'])} | "
            f"{fmt_num(info['max_abs_diff'])} | "
            f"{fmt_num(info['pearson_r'])} |")
    out += [""]

    flags = rep.get("auto_flags", {})
    if any(v for v in flags.values()):
        out += ["**Suspicious-auto flags:**", ""]
        for feat, lst in flags.items():
            if lst:
                out.append(f"- `{feat}`: {', '.join(lst)}")
        out += [""]
    return out


def render_spatial_maps(reps: List[Dict[str, Any]]) -> List[str]:
    out = ["## 4. Spatial feature maps", ""]
    if not reps:
        out += ["_No spatial feature map files found._", ""]
        return out
    out += ["| file | shape | dtype | min | max | mean | std | unique (sample) | class | mostly flat |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in reps:
        out.append(
            f"| `{r['name']}` | "
            f"{r.get('shape', '—')} | {r.get('dtype', '—')} | "
            f"{fmt_num(r.get('min'))} | {fmt_num(r.get('max'))} | "
            f"{fmt_num(r.get('mean'))} | {fmt_num(r.get('std'))} | "
            f"{r.get('n_unique_sample', '—')} | "
            f"{r.get('value_class', '—')} | "
            f"{fmt_num(r.get('mostly_flat'))} |")
    out += [""]
    errs = [r for r in reps if r.get("error")]
    if errs:
        out += ["_Errors:_", ""]
        for r in errs:
            out += [f"- `{r['name']}`: `{r['error']}`"]
        out += [""]
    return out


def render_debug_images(reps: List[Dict[str, Any]]) -> List[str]:
    out = ["## 5. Debug images", ""]
    if not reps:
        out += ["_No debug images found._", ""]
        return out
    out += ["| file | shape | unique colours (sample) | mean intensity | mostly black | mostly white | empty/corrupt |",
            "|---|---|---|---|---|---|---|"]
    for r in reps:
        if not r.get("exists"):
            out.append(f"| `{r['name']}` | (missing) | — | — | — | — | — |")
            continue
        out.append(
            f"| `{r['name']}` | {r.get('shape', '—')} | "
            f"{r.get('n_unique_colours_sample', '—')} | "
            f"{fmt_num(r.get('mean_intensity'))} | "
            f"{fmt_num(r.get('mostly_black'))} | "
            f"{fmt_num(r.get('mostly_white'))} | "
            f"{fmt_num(r.get('empty_or_corrupt'))} |")
    out += [""]
    return out


def render_code(rep: Dict[str, Any]) -> List[str]:
    out = ["## 6. Code analysis", ""]
    if not rep["files"]:
        out += ["_No code files matched the keywords._", ""]
        return out

    likely = rep.get("likely_auto_encoder")
    out += [f"**Likely auto-encoder script:** "
            f"`{likely}`" if likely else
            "**Likely auto-encoder script:** _could not infer_",
            ""]

    out += ["| script | lines | methods detected | hardcoded paths | coord hints |",
            "|---|---|---|---|---|"]
    for f in rep["files"]:
        if f.get("error"):
            out.append(f"| `{f['path']}` | — | error: {f['error']} | — | — |")
            continue
        out.append(
            f"| `{f['path']}` | {f['n_lines']} | "
            f"{', '.join(f['methods_found']) or '—'} | "
            f"{len(f['hardcoded_paths'])} | "
            f"{'; '.join(f['coord_hints']) or '—'} |")
    out += [""]

    # Detailed dump for the likely encoder, if any.
    if likely:
        for f in rep["files"]:
            if f["path"] == likely:
                out += [f"### Detail — `{likely}`", ""]
                if f.get("hardcoded_paths"):
                    out += ["**Hardcoded paths found:**", ""]
                    out += [f"- `{p}`" for p in f["hardcoded_paths"]]
                    out += [""]
                if f.get("suspicious_constants"):
                    out += ["**Suspicious constants:**", ""]
                    out += [f"- `{c.strip()}`" for c in f["suspicious_constants"]]
                    out += [""]
                if f.get("coord_hints"):
                    out += ["**Coordinate hints:** "
                            + "; ".join(f["coord_hints"]), ""]
                break

    return out


def render_failure_modes(flags: List[str]) -> List[str]:
    out = ["## 7. Likely Failure Modes", ""]
    if not flags:
        out += ["_No obvious failure modes detected by automated checks._", ""]
        return out
    # Deduplicate while preserving order
    seen = set(); uniq = []
    for f in flags:
        if f not in seen:
            seen.add(f); uniq.append(f)
    out += [f"- {f}" for f in uniq] + [""]
    return out


def render_intro(loc: Dict[str, Any]) -> List[str]:
    out = ["# Automatic Spatial Encoding — Diagnostic Report", "",
           "_Read-only audit of the current automatic spatial encoding system._",
           "_No retraining, no edits to encoders, no overwrites of CSVs._",
           "",
           "## 1. File inventory", "",
           "**CSVs:**", ""]
    for k, v in loc["csvs"].items():
        mark = "✓" if v["exists"] else "✗"
        size = f"{v['size_bytes']:,} bytes" if v["exists"] else "missing"
        out.append(f"- {mark} `{k}` — `{v['path']}` ({size})")
    out += ["", "**Other artefacts:**", ""]
    for k, v in loc["misc"].items():
        mark = "✓" if v["exists"] else "✗"
        out.append(f"- {mark} `{k}` — `{v['path']}`")
    out += ["", "**Spatial feature map files:**", ""]
    if loc["spatial_map_files"]:
        out += [f"- `{p}`" for p in loc["spatial_map_files"]]
    else:
        out += ["_None._"]
    out += ["", "**Code files (keyword-matched in trajectory-prediction/):**", ""]
    out += [f"- `{p}`" for p in loc["code_hits"]] or ["_None._"]
    out += [""]
    return out


# --------------------------------------------------------------------------- #
# Top-5 problem & recommendation derivation
# --------------------------------------------------------------------------- #
def derive_top_problems(flags: List[str]) -> List[str]:
    # Rank by category severity.
    severity = [
        ("auto features not meaningfully different", 0),
        ("NEGATIVE correlation",                     1),
        ("spatial features highly correlated",       2),
        ("constant",                                 3),
        ("degenerate",                               3),
        ("all-NaN",                                  3),
        ("majority-NaN",                             4),
        ("debug image empty/flat",                   5),
        ("distance maps degenerate",                 5),
        ("extreme outliers",                         6),
        ("mostly flat",                              7),
        ("possible trajectory/image coordinate mismatch", 4),
        ("missing the dist_to_",                     5),
        ("entry/exit detection unreliable",          6),
        ("weak correlation",                         8),
    ]
    scored = []
    for f in flags:
        s = 10
        for kw, sev in severity:
            if kw in f:
                s = sev; break
        scored.append((s, f))
    scored.sort()
    # de-dup
    seen = set(); top = []
    for _, f in scored:
        if f not in seen:
            seen.add(f); top.append(f)
        if len(top) >= 5: break
    return top


def derive_recommendations(flags: List[str],
                           code_rep: Dict[str, Any],
                           compare_rep: Dict[str, Any],
                           auto_rep: Dict[str, Any]) -> List[str]:
    recs: List[str] = []

    if any("highly correlated" in f or "not meaningfully different" in f
           for f in flags):
        recs.append("Re-examine the obstacle / boundary mask construction — "
                    "they may be derived from the same source image with only "
                    "small filter differences.")

    if any("NEGATIVE correlation" in f for f in flags):
        recs.append("Check for a sign flip or coordinate-axis inversion "
                    "between auto and manual encoders (e.g. inside-vs-outside "
                    "convention, or y-axis flip).")

    if any("possible trajectory/image coordinate mismatch" in f for f in flags):
        recs.append("Verify the homography is applied in the auto encoder "
                    "before querying distance maps — currently world_ and "
                    "image_ coordinates appear mixed without explicit warp.")

    if any("constant" in f or "degenerate" in f or "all-NaN" in f for f in flags):
        recs.append("Inspect the segmentation / thresholding step output "
                    "directly (save intermediate masks to a temp folder) — "
                    "the downstream distance transform is collapsing.")

    if any("extreme outliers" in f for f in flags):
        recs.append("Cap or clip distance-feature outputs in the auto encoder; "
                    "investigate which pixels produce the giant distance values.")

    if any("dist_to_entrance" in f for f in flags) or (
            auto_rep.get("exists")
            and not auto_rep["key_features"]["dist_to_entrance"]["present"]):
        recs.append("Add entrance/exit detection to the auto encoder, or "
                    "explicitly remove `dist_to_entrance` from downstream "
                    "feature lists so manual vs auto remain comparable.")

    if compare_rep.get("compared"):
        weak = [f for f, info in compare_rep["per_feature"].items()
                if (info.get("pearson_r") is not None
                    and info["pearson_r"] < 0.6)]
        if weak:
            recs.append("For weak-correlation features "
                        f"({', '.join(weak)}), overlay the auto distance "
                        "map on the same frame as the manual version and "
                        "inspect visually to find the divergence cause.")

    if code_rep.get("likely_auto_encoder"):
        recs.append("Walk through the likely auto encoder "
                    f"(`{code_rep['likely_auto_encoder']}`) end-to-end and "
                    "save each intermediate (raw mask → cleaned mask → "
                    "distance transform → sampled trajectory values).")

    if not recs:
        recs.append("No major issues flagged automatically — manually inspect "
                    "the debug overlay and the feature_correlation plot, then "
                    "decide whether the auto encoding is good enough to "
                    "freeze.")

    # Keep top 5.
    return recs[:5]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    # Windows consoles default to cp1252 and choke on non-ASCII characters
    # like the arrow used in some report strings — force UTF-8 if possible.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(f"[INFO]  cv2 available: {HAVE_CV2}   PIL available: {HAVE_PIL}")
    loc = locate_files()
    print(f"[INFO]  CSVs found: "
          f"{[k for k,v in loc['csvs'].items() if v['exists']]}")
    print(f"[INFO]  Spatial map files: {len(loc['spatial_map_files'])}")
    print(f"[INFO]  Code files matched: {len(loc['code_hits'])}")

    auto_path   = ENCODED / "trajectories_encoded_auto.csv"
    manual_path = ENCODED / "trajectories_encoded.csv"
    motion_paths = [ENCODED / "motion_dataset.csv",
                    ENCODED / "motion_dataset_v2.csv"]

    auto_rep   = inspect_csv(auto_path)
    manual_rep = inspect_csv(manual_path)
    motion_reps = {p.name: inspect_csv(p) for p in motion_paths}

    compare_rep = compare_manual_vs_auto(auto_path, manual_path)

    map_files = [MP_ROOT / p for p in loc["spatial_map_files"]]
    map_reps  = inspect_spatial_maps(map_files)

    debug_paths = [ENCODED / "debug_overlay.png",
                   ENCODED / "feature_correlation.png",
                   ENCODED / "feature_histograms.png"]
    debug_reps = inspect_debug_images(debug_paths)

    code_rep   = analyse_code(loc["code_hits"])

    flags = detect_failure_modes(auto_rep, manual_rep, compare_rep,
                                 map_reps, debug_reps, code_rep)

    # ---- Markdown ----
    md: List[str] = []
    md += render_intro(loc)
    md += ["## 2. Encoded CSV inspection", ""]
    md += render_csv_table("trajectories_encoded_auto.csv", auto_rep)
    md += render_csv_table("trajectories_encoded.csv (manual)", manual_rep)
    for name, rep in motion_reps.items():
        md += render_csv_table(name, rep)
    md += render_compare(compare_rep)
    md += render_spatial_maps(map_reps)
    md += render_debug_images(debug_reps)
    md += render_code(code_rep)
    md += render_failure_modes(flags)

    md += ["## 8. Plain-English summary", ""]
    if auto_rep.get("exists"):
        n_rows = auto_rep["row_count"]
        n_tracks = auto_rep["n_unique_tracks"]
        md += [f"The automatic encoder produced **{n_rows:,} rows** across "
               f"**{n_tracks if n_tracks is not None else '—'} unique tracks**. "
               "Its column set is "
               + ", ".join(f"`{c}`" for c in auto_rep["columns"]) + "."]
        # Compare with manual
        if manual_rep.get("exists"):
            md += [f"The manual encoder produced **{manual_rep['row_count']:,} "
                   f"rows** with columns "
                   + ", ".join(f"`{c}`" for c in manual_rep["columns"]) + "."]
        md += [""]

    if compare_rep.get("compared"):
        md += ["On the rows where the two encoders overlap, the spatial "
               "features can be compared directly. The Pearson correlations "
               "and mean-absolute-differences in Section 3 indicate how close "
               "each automatic feature is to its manual counterpart. Low "
               "correlations or large differences are highlighted as "
               "failure modes.", ""]
    else:
        md += ["Manual vs auto could not be compared "
               f"({compare_rep.get('reason', compare_rep.get('error', '—'))}).",
               ""]

    md += ["A complete machine-readable mirror of this report is at "
           f"`{REPORT_JSON.relative_to(MP_ROOT)}`.", ""]
    md += ["_End of report._"]

    REPORT_MD.write_text("\n".join(md), encoding="utf-8")

    # ---- JSON ----
    summary = {
        "file_inventory": loc,
        "auto_csv":   auto_rep,
        "manual_csv": manual_rep,
        "motion_csvs": motion_reps,
        "compare_manual_vs_auto": compare_rep,
        "spatial_maps":  map_reps,
        "debug_images":  debug_reps,
        "code_analysis": code_rep,
        "likely_failure_modes": flags,
    }
    REPORT_JSON.write_text(json.dumps(jsonable(summary), indent=2),
                           encoding="utf-8")

    # ---- Console output ----
    top_problems = derive_top_problems(flags)
    top_recs     = derive_recommendations(flags, code_rep, compare_rep,
                                          auto_rep)

    print()
    print(f"[OK]    Markdown report : {REPORT_MD}")
    print(f"[OK]    JSON summary    : {REPORT_JSON}")
    print()
    print("Top 5 suspected problems:")
    if top_problems:
        for i, p in enumerate(top_problems, 1):
            print(f"  {i}. {p}")
    else:
        print("  (none flagged)")
    print()
    print("Top 5 recommended next checks:")
    for i, r in enumerate(top_recs, 1):
        print(f"  {i}. {r}")


if __name__ == "__main__":
    main()
