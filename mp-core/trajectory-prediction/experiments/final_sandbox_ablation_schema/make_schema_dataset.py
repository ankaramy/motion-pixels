"""
make_schema_dataset.py
----------------------
Build the FULL Motion Pixels schema dataset from the rerun v2.1C encoded CSV.
Includes all features used by every ablation model (A/B/C/D) plus the
prediction targets. Duplicated ×10. Saves CSV + dataset_summary.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _paths import (
    ENCODED_CSV, DATASET_CSV, DATASET_MD, EXP,
    DUPLICATE_N, MIN_TRACK_LEN,
    STOP_THRESH_M, SHIFT_THRESH_RAD,
    FEATURES_D, TARGET_COLS,
)

ALIAS = {
    "track_id": ["track_id", "person_id", "pid"],
    "frame":    ["frame", "frame_number", "frame_idx"],
    "time":     ["time_s", "time"],
}


def alias(cols, key):
    for c in ALIAS[key]:
        if c in cols:
            return c
    raise SystemExit(f"[FATAL] no column for {key} in {cols}")


def wrap(a):
    """Wrap to (-π, π]."""
    return np.arctan2(np.sin(a), np.cos(a))


def per_track_features(g: pd.DataFrame, frame_col: str) -> pd.DataFrame:
    g = g.sort_values(frame_col).reset_index(drop=True)
    x = g["world_x"].to_numpy(dtype=np.float64)
    y = g["world_y"].to_numpy(dtype=np.float64)

    dx = np.zeros_like(x); dy = np.zeros_like(y)
    dx[1:] = x[1:] - x[:-1]
    dy[1:] = y[1:] - y[:-1]

    speed   = np.hypot(dx, dy)
    heading = np.arctan2(dy, dx)

    # Per-step heading change (turn rate).
    turn_rate = np.zeros_like(heading)
    if len(heading) > 1:
        turn_rate[1:] = wrap(np.diff(heading))

    is_stop  = (speed < STOP_THRESH_M).astype(np.float32)
    is_shift = (np.abs(turn_rate) > SHIFT_THRESH_RAD).astype(np.float32)

    heading_sin = np.sin(heading)
    heading_cos = np.cos(heading)

    g["delta_x"]       = dx
    g["delta_y"]       = dy
    g["speed"]         = speed
    g["heading_angle"] = heading
    g["heading_sin"]   = heading_sin
    g["heading_cos"]   = heading_cos
    g["turn_rate"]     = turn_rate
    g["is_stop"]       = is_stop
    g["is_shift"]      = is_shift

    # Prediction targets = NEXT step's (delta_x, delta_y). Last row's
    # target is unknown; mark with NaN and drop later.
    target_du = np.full_like(dx, np.nan)
    target_dv = np.full_like(dy, np.nan)
    target_du[:-1] = dx[1:]
    target_dv[:-1] = dy[1:]
    g["target_du"] = target_du
    g["target_dv"] = target_dv

    return g


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    EXP.mkdir(parents=True, exist_ok=True)
    if not ENCODED_CSV.exists():
        raise SystemExit(f"[FATAL] missing input: {ENCODED_CSV}")

    df = pd.read_csv(ENCODED_CSV)
    tid = alias(df.columns, "track_id")
    fr  = alias(df.columns, "frame")
    tm  = alias(df.columns, "time")
    print(f"[INFO]  Loaded {len(df):,} rows · {df[tid].nunique():,} tracks")

    lengths = df.groupby(tid).size()
    keep = lengths[lengths >= MIN_TRACK_LEN].index.tolist()
    df = df[df[tid].isin(keep)].copy()
    print(f"[INFO]  ≥{MIN_TRACK_LEN}-frame filter → "
          f"{len(df):,} rows · {df[tid].nunique():,} tracks")

    df = (df.groupby(tid, group_keys=False)
            .apply(lambda g: per_track_features(g, fr)))
    df = df.reset_index(drop=True)

    # Normalised u/v over the (filtered) dataset extent.
    x_min, x_max = float(df["world_x"].min()), float(df["world_x"].max())
    y_min, y_max = float(df["world_y"].min()), float(df["world_y"].max())
    df["u"] = (df["world_x"] - x_min) / max(1e-9, x_max - x_min)
    df["v"] = (df["world_y"] - y_min) / max(1e-9, y_max - y_min)

    # Drop rows whose target_du/target_dv is NaN (last frame per track).
    before = len(df)
    df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)
    print(f"[INFO]  Dropped {before - len(df):,} last-frame rows w/o target")

    # Duplicate ×10 with unique IDs (offset).
    orig_max = int(df[tid].max())
    dups = []
    for i in range(DUPLICATE_N):
        c = df.copy()
        c[tid] = c[tid] + i * (orig_max + 1)
        c["dup_idx"] = i
        dups.append(c)
    out = pd.concat(dups, ignore_index=True)
    out = out.rename(columns={tid: "track_id", fr: "frame", tm: "time"})
    print(f"[INFO]  Duplicated ×{DUPLICATE_N} → {len(out):,} rows · "
          f"{out['track_id'].nunique():,} unique IDs")

    needed = set(FEATURES_D + TARGET_COLS + ["track_id", "frame", "time"])
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise SystemExit(f"[FATAL] missing schema columns: {missing}")

    # Save dataset_extents for later (rollout u/v normalisation uses these).
    extents = {"x_min": x_min, "x_max": x_max,
               "y_min": y_min, "y_max": y_max}
    pd.Series(extents).to_csv(EXP / "world_extents.csv")
    print(f"[OK]    Saved world_extents.csv")

    out.to_csv(DATASET_CSV, index=False)
    print(f"[OK]    Wrote {DATASET_CSV.name} "
          f"({out.shape[0]:,} × {out.shape[1]})")

    # Summary md.
    md = []
    md.append("# Schema Dataset Summary")
    md.append("")
    md.append(f"- Source: `{ENCODED_CSV}`")
    md.append(f"- Output: `{DATASET_CSV}`")
    md.append(f"- World extent: x ∈ [{x_min:.2f}, {x_max:.2f}] m, "
              f"y ∈ [{y_min:.2f}, {y_max:.2f}] m")
    md.append(f"- Min-track-length filter: {MIN_TRACK_LEN} frames")
    md.append(f"- Duplication factor: ×{DUPLICATE_N}")
    md.append(f"- Final rows: **{len(out):,}**, unique tracks: "
              f"**{out['track_id'].nunique():,}**")
    md.append("")
    md.append("## Behaviour thresholds")
    md.append("")
    md.append(f"- `is_stop`  = 1 if per-step |Δ| < {STOP_THRESH_M:.3f} m")
    md.append(f"- `is_shift` = 1 if |turn_rate| > "
              f"{np.degrees(SHIFT_THRESH_RAD):.1f}°")
    md.append("")
    md.append("## Stats for derived behaviour features")
    md.append("")
    md.append("| feature | mean | std | min | max | non-zero frac |")
    md.append("|---|---|---|---|---|---|")
    for c in ["speed", "heading_angle", "turn_rate",
              "is_stop", "is_shift",
              "heading_sin", "heading_cos",
              "dist_to_obstacle", "dist_to_boundary", "dist_to_entrance",
              "target_du", "target_dv"]:
        if c not in out.columns:
            md.append(f"| `{c}` | (missing) | — | — | — | — |")
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        nz = float((s.abs() > 1e-6).mean())
        md.append(f"| `{c}` | {s.mean():.4f} | {s.std():.4f} | "
                  f"{s.min():.4f} | {s.max():.4f} | {nz:.2%} |")
    md.append("")
    md.append("## Schema (full set)")
    md.append("")
    md.append(", ".join(f"`{c}`" for c in out.columns))
    DATASET_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK]    Wrote {DATASET_MD.name}")


if __name__ == "__main__":
    main()
