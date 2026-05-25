"""
make_final_sandbox_dataset.py
-----------------------------
Build the duplicated overfit-style dataset from the rerun's spatial v2.1C
encoded CSV. Adds movement features (delta/speed/heading/turn-rate) and
normalised u/v coords. Writes dataset CSV + a summary md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import (
    ENCODED_CSV, DATASET_CSV, DATASET_MD,
    SANDBOX, DUPLICATE_N, MIN_TRACK_LEN, FEATURE_COLS,
)

ALIAS = {
    "track_id": ["track_id", "person_id", "pid"],
    "frame":    ["frame", "frame_number", "frame_idx"],
}


def alias(cols, key):
    for c in ALIAS[key]:
        if c in cols:
            return c
    raise SystemExit(f"[FATAL] column for {key} not found in {cols}")


def add_movement_features(g: pd.DataFrame, frame_col: str) -> pd.DataFrame:
    """Minimal deltas only — matches the production sandbox feature set.
    No speed / heading_sin / heading_cos / turn_rate (those biased the
    LSTM toward mean motion in the regressed run)."""
    g = g.sort_values(frame_col).reset_index(drop=True)
    x = g["world_x"].to_numpy(dtype=np.float64)
    y = g["world_y"].to_numpy(dtype=np.float64)
    dx = np.zeros_like(x); dy = np.zeros_like(y)
    dx[1:] = x[1:] - x[:-1]
    dy[1:] = y[1:] - y[:-1]
    g["delta_x"] = dx
    g["delta_y"] = dy
    return g


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    SANDBOX.mkdir(parents=True, exist_ok=True)
    if not ENCODED_CSV.exists():
        raise SystemExit(f"[FATAL] missing input: {ENCODED_CSV}")

    df = pd.read_csv(ENCODED_CSV)
    tid = alias(df.columns, "track_id")
    fr  = alias(df.columns, "frame")
    print(f"[INFO]  Loaded {len(df):,} rows · "
          f"{df[tid].nunique():,} tracks · "
          f"track col=`{tid}` frame col=`{fr}`")

    # Filter to tracks long enough to be useful.
    lengths = df.groupby(tid).size()
    keep_ids = lengths[lengths >= MIN_TRACK_LEN].index.tolist()
    df = df[df[tid].isin(keep_ids)].copy()
    print(f"[INFO]  After ≥{MIN_TRACK_LEN}-frame filter: "
          f"{len(df):,} rows · {df[tid].nunique():,} tracks")

    # Per-track movement features.
    df = (df.groupby(tid, group_keys=False)
            .apply(lambda g: add_movement_features(g, fr)))
    df = df.reset_index(drop=True)

    x_min, x_max = float(df["world_x"].min()), float(df["world_x"].max())
    y_min, y_max = float(df["world_y"].min()), float(df["world_y"].max())

    # Duplicate 10× with unique IDs (offset). Shape preserved per copy.
    orig_max = int(df[tid].max())
    dups = []
    for i in range(DUPLICATE_N):
        c = df.copy()
        c[tid] = c[tid] + i * (orig_max + 1)
        c["dup_idx"] = i
        dups.append(c)
    out = pd.concat(dups, ignore_index=True)
    out = out.rename(columns={tid: "track_id", fr: "frame"})
    print(f"[INFO]  Duplicated ×{DUPLICATE_N}: {len(out):,} rows · "
          f"{out['track_id'].nunique():,} unique IDs")

    # Final sanity — ensure all FEATURE_COLS are present.
    missing = [c for c in FEATURE_COLS if c not in out.columns]
    if missing:
        raise SystemExit(f"[FATAL] missing FEATURE_COLS: {missing}")

    out.to_csv(DATASET_CSV, index=False)
    print(f"[OK]    Wrote {DATASET_CSV.relative_to(DATASET_CSV.parents[3])} "
          f"({out.shape[0]:,} × {out.shape[1]})")

    # Summary md.
    feat_stats = []
    for c in FEATURE_COLS:
        s = pd.to_numeric(out[c], errors="coerce")
        feat_stats.append((c, float(s.mean()), float(s.std()),
                           float(s.min()), float(s.max())))
    md = []
    md.append("# Final Sandbox — Dataset Summary")
    md.append("")
    md.append(f"- Source: `{ENCODED_CSV}`")
    md.append(f"- Output: `{DATASET_CSV}`")
    md.append(f"- Min-track-length filter: {MIN_TRACK_LEN} frames")
    md.append(f"- Duplication factor: ×{DUPLICATE_N}")
    md.append(f"- Final rows: **{len(out):,}**, unique track_id: "
              f"**{out['track_id'].nunique():,}**")
    md.append(f"- World extent: x ∈ [{x_min:.2f}, {x_max:.2f}] m, "
              f"y ∈ [{y_min:.2f}, {y_max:.2f}] m")
    md.append("")
    md.append("## Feature stats")
    md.append("")
    md.append("| feature | mean | std | min | max |")
    md.append("|---|---|---|---|---|")
    for name, mu, sd, mn, mx in feat_stats:
        md.append(f"| `{name}` | {mu:.4f} | {sd:.4f} | {mn:.4f} | {mx:.4f} |")
    md.append("")
    md.append("## Columns")
    md.append("")
    md.append(", ".join(f"`{c}`" for c in out.columns))
    DATASET_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK]    Wrote {DATASET_MD.name}")


if __name__ == "__main__":
    main()
