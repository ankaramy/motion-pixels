"""
make_bridge_dataset.py
----------------------
Builds the schema_ablation_bridge dataset by porting the OLD
motion_dataset_v2 recipe onto the newest calibrated Skate 1 rerun.

Motion is kept in REAL METRIC space (du, dv in metres) — exactly like
the experiment that produced the angular-faithful trajectories
(308, 332, 72, 310). Position (u, v) is normalised to [0, 1] using
dataset-wide world extents. Spatial features are percentile-rank-based
"relational" 0–1 scalars per the new advisor schema.

Inputs
------
  mp-data/processed/rerun_macba_2026-05-19/spatial_v21C/trajectories_encoded.csv

Outputs (alongside this script)
-------------------------------
  schema_ablation_bridge_dataset.csv
  schema_summary.json                 (world_bounds_used + column docs)
  dataset_summary.md

Columns written (in order):
  trajectory_id, timestep,
  u, v,
  du, dv, speed, heading_sin, heading_cos, turn_rate,
  dist_to_obstacle_norm, dist_to_boundary_norm, entrance_affinity_norm,
  target_du, target_dv

Notes
-----
  * du, dv are NOT scaled and NOT normalised; they are world metres / step.
  * target_du, target_dv are next-step du, dv (last row per track dropped).
  * world_x, world_y are USED to compute u/v but are NOT emitted as features.
  * openness_lr_asymmetry is NOT in the v21C field set; the bridge skips
    it gracefully (no E feature set, see run_bridge_ablation.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MP_ROOT = HERE.parent.parent.parent.parent

# Input
ENCODED_CSV = (MP_ROOT / "mp-data" / "processed" / "rerun_macba_2026-05-19"
               / "spatial_v21C" / "trajectories_encoded.csv")

# Outputs
DATASET_CSV  = HERE / "schema_ablation_bridge_dataset.csv"
SCHEMA_JSON  = HERE / "schema_summary.json"
DATASET_MD   = HERE / "dataset_summary.md"

# Drop tracks shorter than this so we always have at least one full
# WINDOW + 1 target row. Real selection for rollouts happens later.
MIN_TRACK_LEN = 11

# Required input columns
INPUT_REQUIRED = {
    "track_id", "frame",
    "world_x", "world_y",
    "dist_to_obstacle", "dist_to_boundary", "dist_to_entrance",
}

OUTPUT_COLS = [
    "trajectory_id", "timestep",
    "u", "v",
    "du", "dv", "speed", "heading_sin", "heading_cos", "turn_rate",
    "dist_to_obstacle_norm", "dist_to_boundary_norm", "entrance_affinity_norm",
    "target_du", "target_dv",
]


def wrap(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def rank_pct(values: np.ndarray, ascending: bool = True) -> np.ndarray:
    """Convert raw values to a [0, 1] percentile rank.

    ascending=True   smallest → 0.0, largest → 1.0  (clearance)
    ascending=False  largest → 0.0, smallest → 1.0  (affinity)
    """
    s = pd.Series(values).rank(method="average", pct=True, na_option="keep")
    if not ascending:
        s = 1.0 - s
    return s.to_numpy(dtype=np.float64)


def per_track_features(g: pd.DataFrame) -> pd.DataFrame:
    """Compute motion features for one track (already sorted by frame)."""
    g = g.reset_index(drop=True).copy()

    wx = g["world_x"].to_numpy(dtype=np.float64)
    wy = g["world_y"].to_numpy(dtype=np.float64)

    du = np.zeros_like(wx)
    dv = np.zeros_like(wy)
    du[1:] = wx[1:] - wx[:-1]
    dv[1:] = wy[1:] - wy[:-1]

    speed   = np.hypot(du, dv)
    heading = np.arctan2(dv, du)

    turn_rate = np.zeros_like(heading)
    if len(heading) > 1:
        turn_rate[1:] = wrap(np.diff(heading))

    g["du"]          = du
    g["dv"]          = dv
    g["speed"]       = speed
    g["heading_sin"] = np.sin(heading)
    g["heading_cos"] = np.cos(heading)
    g["turn_rate"]   = turn_rate

    # next-step targets
    t_du = np.full_like(du, np.nan)
    t_dv = np.full_like(dv, np.nan)
    t_du[:-1] = du[1:]
    t_dv[:-1] = dv[1:]
    g["target_du"] = t_du
    g["target_dv"] = t_dv

    # sequential within-track timestep
    g["timestep"] = np.arange(len(g), dtype=np.int64)
    return g


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if not ENCODED_CSV.exists():
        raise SystemExit(f"[FATAL] missing input: {ENCODED_CSV}")

    print(f"[load]  {ENCODED_CSV}")
    df = pd.read_csv(ENCODED_CSV)
    missing = INPUT_REQUIRED - set(df.columns)
    if missing:
        raise SystemExit(f"[FATAL] missing input columns: {sorted(missing)}")
    print(f"        {len(df):,} rows · {df['track_id'].nunique():,} tracks")

    # ── track-length filter ───────────────────────────────────────────────
    lengths = df.groupby("track_id").size()
    keep_ids = lengths[lengths >= MIN_TRACK_LEN].index.tolist()
    df = df[df["track_id"].isin(keep_ids)].copy()
    print(f"[filter] >= {MIN_TRACK_LEN} frames → "
          f"{len(df):,} rows · {df['track_id'].nunique():,} tracks")

    # ── world bounds (used to map world_x/world_y ↔ u/v) ──────────────────
    xmin = float(df["world_x"].min()); xmax = float(df["world_x"].max())
    ymin = float(df["world_y"].min()); ymax = float(df["world_y"].max())
    xrng = max(xmax - xmin, 1e-9); yrng = max(ymax - ymin, 1e-9)
    print(f"[bounds] x ∈ [{xmin:.3f}, {xmax:.3f}]  y ∈ [{ymin:.3f}, {ymax:.3f}]")

    df["u"] = (df["world_x"] - xmin) / xrng
    df["v"] = (df["world_y"] - ymin) / yrng

    # ── per-track motion + next-step targets ──────────────────────────────
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    parts = []
    for tid, g in df.groupby("track_id", sort=False):
        parts.append(per_track_features(g.assign(track_id=tid)))
    df = pd.concat(parts, ignore_index=True)

    # ── relational spatial features (percentile rank, dataset-wide) ───────
    df["dist_to_obstacle_norm"] = rank_pct(
        df["dist_to_obstacle"].to_numpy(), ascending=True)
    df["dist_to_boundary_norm"] = rank_pct(
        df["dist_to_boundary"].to_numpy(), ascending=True)
    df["entrance_affinity_norm"] = rank_pct(
        df["dist_to_entrance"].to_numpy(), ascending=False)

    # ── drop last frame of each track (no target available) ───────────────
    before = len(df)
    df = df.dropna(subset=["target_du", "target_dv"]).reset_index(drop=True)
    print(f"[drop ] dropped {before - len(df):,} last-frame rows w/o target")

    # ── project to schema and emit ────────────────────────────────────────
    df = df.rename(columns={"track_id": "trajectory_id"})
    out = df[OUTPUT_COLS].copy()
    out = out.sort_values(["trajectory_id", "timestep"]).reset_index(drop=True)
    out.to_csv(DATASET_CSV, index=False)
    print(f"[ok]    wrote {DATASET_CSV.name}  "
          f"({out.shape[0]:,} × {out.shape[1]})")

    # ── schema_summary.json ───────────────────────────────────────────────
    summary = {
        "version": 1,
        "description": (
            "schema_ablation_bridge dataset — old motion_dataset_v2 recipe "
            "ported to the rerun_macba_2026-05-19 v21C encoding. Motion in "
            "real metric metres; positions normalised; spatial features "
            "percentile-rank relational."
        ),
        "normalization": {
            "u_v":              "global MinMax over dataset world_x/world_y",
            "du_dv_speed":      "raw metres/step (NOT scaled here; fit StandardScaler at train time)",
            "turn_rate":        "radians in (−π, π]",
            "dist_*_norm":      "percentile rank ascending (0 = closest, 1 = farthest)",
            "entrance_affinity_norm": "percentile rank descending (1 = closest to entrance)",
        },
        "world_bounds_used": {
            "xmin": xmin, "xmax": xmax,
            "ymin": ymin, "ymax": ymax,
            "xrng": xrng, "yrng": yrng,
        },
        "min_track_length_filter": MIN_TRACK_LEN,
        "n_rows":    int(len(out)),
        "n_tracks":  int(out["trajectory_id"].nunique()),
        "columns":   OUTPUT_COLS,
        "has_openness_lr_asymmetry": False,
        "source_encoded_csv": str(ENCODED_CSV),
    }
    SCHEMA_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ok]    wrote {SCHEMA_JSON.name}")

    # ── short markdown summary ────────────────────────────────────────────
    md_lines = []
    A = md_lines.append
    A("# schema_ablation_bridge — dataset summary")
    A("")
    A(f"- Source: `{ENCODED_CSV}`")
    A(f"- Rows: **{len(out):,}**")
    A(f"- Tracks: **{out['trajectory_id'].nunique()}**  "
      f"(filter: ≥ {MIN_TRACK_LEN} frames)")
    A(f"- World x: [{xmin:.3f}, {xmax:.3f}] m  (range {xrng:.3f} m)")
    A(f"- World y: [{ymin:.3f}, {ymax:.3f}] m  (range {yrng:.3f} m)")
    A("")
    A("## Column statistics")
    A("")
    A("| column | mean | std | min | max |")
    A("|---|---|---|---|---|")
    for c in OUTPUT_COLS:
        if c in ("trajectory_id", "timestep"):
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        A(f"| `{c}` | {s.mean():.4f} | {s.std():.4f} | "
          f"{s.min():.4f} | {s.max():.4f} |")
    A("")
    A("## Notes")
    A("")
    A("* `du`, `dv`, `target_du`, `target_dv` are in metres / step — no "
      "MOTION_SCALE applied. This matches the OLD ablation recipe.")
    A("* `openness_lr_asymmetry` is **NOT computed** here (the v21C "
      "encoding only emits `dist_to_obstacle`, `dist_to_boundary`, "
      "`dist_to_entrance`). The bridge experiment therefore stops at "
      "model `D_full_relational` (no model `E`).")
    DATASET_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[ok]    wrote {DATASET_MD.name}")


if __name__ == "__main__":
    main()
