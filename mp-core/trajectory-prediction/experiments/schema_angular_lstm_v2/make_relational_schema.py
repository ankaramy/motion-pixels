"""
make_relational_schema.py (v2)
------------------------------
Build the RELATIONAL feature schema for the angular LSTM experiment.

v2 difference vs v1:
    Motion magnitudes (du, dv, target_du, target_dv) are multiplied by
    MOTION_SCALE so the network sees motion at a learnable amplitude.
    u, v and every relational/percentile feature are kept untouched.

Output columns (in this order):

    track_id, frame_idx,
    u, v,
    du, dv,
    speed_rel,
    heading_sin, heading_cos,
    turn_rate_rel,
    is_stop, is_shift,
    obstacle_clearance_pct,
    boundary_clearance_pct,
    entrance_affinity_pct,
    local_space_openness,
    target_du, target_dv, target_turn_rate
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _paths import (
    ENCODED_CSV, SCHEMA_CSV, SCHEMA_MD, WORLD_EXTENTS_CSV,
    MIN_TRACK_LEN, STOP_THRESH_REL, SHIFT_THRESH_DEG,
    MOTION_SCALE, SCHEMA_COLS,
)


ALIAS = {
    "track_id": ["track_id", "person_id", "pid"],
    "frame":    ["frame", "frame_number", "frame_idx"],
}


def alias(cols, key):
    for c in ALIAS[key]:
        if c in cols:
            return c
    raise SystemExit(f"[FATAL] no column for {key!r} in {list(cols)}")


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def rank_pct(values: np.ndarray, ascending: bool = True) -> np.ndarray:
    s = pd.Series(values).rank(method="average", pct=True, na_option="keep")
    if not ascending:
        s = 1.0 - s
    return s.to_numpy(dtype=np.float64)


def per_track_motion(g: pd.DataFrame, frame_col: str,
                     u_min: float, u_max: float,
                     v_min: float, v_max: float) -> pd.DataFrame:
    g = g.sort_values(frame_col).reset_index(drop=True)

    u = g["u"].to_numpy(dtype=np.float64)
    v = g["v"].to_numpy(dtype=np.float64)

    du = np.zeros_like(u)
    dv = np.zeros_like(v)
    du[1:] = u[1:] - u[:-1]
    dv[1:] = v[1:] - v[:-1]

    speed_raw = np.hypot(du, dv)
    heading   = np.arctan2(dv, du)

    turn_rate = np.zeros_like(heading)
    if len(heading) > 1:
        turn_rate[1:] = wrap(np.diff(heading))

    g["du"]            = du
    g["dv"]            = dv
    g["_speed_raw"]    = speed_raw       # NOTE: unscaled, for speed_rel ranking
    g["heading_sin"]   = np.sin(heading)
    g["heading_cos"]   = np.cos(heading)
    g["_turn_raw"]     = turn_rate
    return g


def add_targets(g: pd.DataFrame, frame_col: str) -> pd.DataFrame:
    g = g.sort_values(frame_col).reset_index(drop=True)

    du = g["du"].to_numpy()
    dv = g["dv"].to_numpy()
    tr = g["turn_rate_rel"].to_numpy()

    t_du = np.full_like(du, np.nan)
    t_dv = np.full_like(dv, np.nan)
    t_tr = np.full_like(tr, np.nan)
    t_du[:-1] = du[1:]
    t_dv[:-1] = dv[1:]
    t_tr[:-1] = tr[1:]

    g["target_du"]        = t_du
    g["target_dv"]        = t_dv
    g["target_turn_rate"] = t_tr
    return g


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if not ENCODED_CSV.exists():
        raise SystemExit(f"[FATAL] missing input: {ENCODED_CSV}")

    print(f"[INFO]  Loading {ENCODED_CSV}")
    df = pd.read_csv(ENCODED_CSV)

    tid = alias(df.columns, "track_id")
    fr  = alias(df.columns, "frame")
    for needed in ("world_x", "world_y",
                   "dist_to_obstacle", "dist_to_boundary",
                   "dist_to_entrance"):
        if needed not in df.columns:
            raise SystemExit(f"[FATAL] missing required column {needed!r}")

    print(f"[INFO]  {len(df):,} rows · {df[tid].nunique():,} tracks")

    lengths = df.groupby(tid).size()
    keep = lengths[lengths >= MIN_TRACK_LEN].index.tolist()
    df = df[df[tid].isin(keep)].copy()
    print(f"[INFO]  >= {MIN_TRACK_LEN}-frame filter -> "
          f"{len(df):,} rows · {df[tid].nunique():,} tracks")

    x_min, x_max = float(df["world_x"].min()), float(df["world_x"].max())
    y_min, y_max = float(df["world_y"].min()), float(df["world_y"].max())
    df["u"] = (df["world_x"] - x_min) / max(1e-9, x_max - x_min)
    df["v"] = (df["world_y"] - y_min) / max(1e-9, y_max - y_min)

    pd.Series({"x_min": x_min, "x_max": x_max,
               "y_min": y_min, "y_max": y_max,
               "u_span": x_max - x_min,
               "v_span": y_max - y_min,
               "motion_scale": MOTION_SCALE}).to_csv(WORLD_EXTENTS_CSV)
    print(f"[OK]    Wrote {WORLD_EXTENTS_CSV.name}")

    df = (df.groupby(tid, group_keys=False)
            .apply(lambda g: per_track_motion(g, fr, 0, 1, 0, 1)))
    df = df.reset_index(drop=True)

    df["speed_rel"]     = rank_pct(df["_speed_raw"].to_numpy(), ascending=True)
    df["turn_rate_rel"] = np.clip(df["_turn_raw"].to_numpy() / np.pi, -1.0, 1.0)

    df["is_stop"]  = (df["speed_rel"] < STOP_THRESH_REL).astype(np.float32)
    df["is_shift"] = (np.abs(df["_turn_raw"]) * 180.0 / np.pi
                                                 > SHIFT_THRESH_DEG).astype(np.float32)

    df["obstacle_clearance_pct"] = rank_pct(
        df["dist_to_obstacle"].to_numpy(), ascending=True)
    df["boundary_clearance_pct"] = rank_pct(
        df["dist_to_boundary"].to_numpy(), ascending=True)
    df["entrance_affinity_pct"] = rank_pct(
        df["dist_to_entrance"].to_numpy(), ascending=False)

    df["local_space_openness"] = np.minimum(
        df["obstacle_clearance_pct"].to_numpy(),
        df["boundary_clearance_pct"].to_numpy(),
    )

    df = (df.groupby(tid, group_keys=False)
            .apply(lambda g: add_targets(g, fr)))
    df = df.reset_index(drop=True)

    before = len(df)
    df = df.dropna(subset=["target_du", "target_dv", "target_turn_rate"])
    df = df.reset_index(drop=True)
    print(f"[INFO]  Dropped {before - len(df):,} last-frame rows w/o target")

    # v2: apply MOTION_SCALE only to motion magnitudes. Heading + percentiles
    # are NOT touched. speed_rel was already computed from the unscaled magnitude.
    df["du"]        = df["du"]        * MOTION_SCALE
    df["dv"]        = df["dv"]        * MOTION_SCALE
    df["target_du"] = df["target_du"] * MOTION_SCALE
    df["target_dv"] = df["target_dv"] * MOTION_SCALE
    print(f"[INFO]  Applied MOTION_SCALE={MOTION_SCALE} to du/dv/target_du/target_dv")

    df = df.rename(columns={tid: "track_id", fr: "frame_idx"})
    missing = [c for c in SCHEMA_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"[FATAL] missing schema columns after build: "
                         f"{missing}")
    out = df[SCHEMA_COLS].copy()
    out = out.sort_values(["track_id", "frame_idx"]).reset_index(drop=True)

    out.to_csv(SCHEMA_CSV, index=False)
    print(f"[OK]    Wrote {SCHEMA_CSV}  ({out.shape[0]:,} x {out.shape[1]})")

    md = []
    md.append("# Relational Schema v2 — schema_angular_lstm_v2")
    md.append("")
    md.append(f"- Source: `{ENCODED_CSV}`")
    md.append(f"- Output: `{SCHEMA_CSV.name}`")
    md.append(f"- World extent: x in [{x_min:.2f}, {x_max:.2f}] m, "
              f"y in [{y_min:.2f}, {y_max:.2f}] m")
    md.append(f"- Tracks kept (>= {MIN_TRACK_LEN} frames): "
              f"**{out['track_id'].nunique():,}**")
    md.append(f"- Rows: **{len(out):,}**")
    md.append(f"- MOTION_SCALE: **{MOTION_SCALE}** (applied to "
              "`du`, `dv`, `target_du`, `target_dv`)")
    md.append("")
    md.append("## Behaviour thresholds")
    md.append(f"- `is_stop`  = 1 if `speed_rel`     < {STOP_THRESH_REL}")
    md.append(f"- `is_shift` = 1 if |turn rate|     > {SHIFT_THRESH_DEG}°")
    md.append("")
    md.append("## Per-feature stats")
    md.append("")
    md.append("| feature | mean | std | min | max |")
    md.append("|---|---|---|---|---|")
    for c in SCHEMA_COLS:
        if c in ("track_id", "frame_idx"):
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        md.append(f"| `{c}` | {s.mean():.4f} | {s.std():.4f} | "
                  f"{s.min():.4f} | {s.max():.4f} |")
    md.append("")
    md.append("## Schema columns")
    md.append("")
    md.append(", ".join(f"`{c}`" for c in SCHEMA_COLS))
    SCHEMA_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK]    Wrote {SCHEMA_MD.name}")


if __name__ == "__main__":
    main()
