"""
build_barcelona_v1_master.py
----------------------------
Thin assembly wrapper (NOT a new recipe). It reuses the FROZEN bridge recipe
functions from
    experiments/schema_ablation_bridge/make_bridge_dataset.py
(`per_track_features`, `rank_pct`, `wrap`, `MIN_TRACK_LEN`) verbatim, applied
PER RECORDING, then concatenates the 5 approved Barcelona_v1_encoded recordings
into a recording-level-split master dataset for Frozen Model C.

Per-recording normalization (matches the frozen recipe, scoped per site):
    u, v                    : MinMax over that recording's world_x / world_y
    du, dv, speed           : raw metres / step (NOT scaled)
    turn_rate               : radians in (-pi, pi]
    dist_to_obstacle_norm   : percentile rank ascending  (0=closest, 1=farthest)
    dist_to_boundary_norm   : percentile rank ascending
    entrance_affinity_norm  : percentile rank descending (diagnostic only)

Model C feature set (C_motion_position_spatial, 10 features) — confirmed from
run_bridge_ablation.py FEATURE_SETS:
    du, dv, speed, heading_sin, heading_cos, turn_rate, u, v,
    dist_to_obstacle_norm, dist_to_boundary_norm
Targets: target_du, target_dv.  (entrance_affinity_norm is Model D, EXCLUDED.)

Outputs -> new_datasets/Barcelona_v1_master_dataset/
    master_dataset.csv       (full + diagnostic columns, incl entrance_affinity_norm)
    model_C_dataset.csv       (exact Model C schema, NO entrance_affinity_norm)
    manifest.json
    dataset_validation.json
    dataset_schema_report.md   (written by a separate reporting step)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BRIDGE_DIR = HERE / "experiments" / "schema_ablation_bridge"
sys.path.insert(0, str(BRIDGE_DIR))
import make_bridge_dataset as bridge  # frozen recipe: per_track_features, rank_pct, MIN_TRACK_LEN

ENCODED_ROOT = Path(r"C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded")
OUT_DIR = Path(r"C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_master_dataset")

# Recording-level split (brief's suggested split; no production split exists for these).
SPLIT = {
    "esplanade_espanya_01": "train",
    "placa_catalunya_01":   "train",
    "placa_espanya_01":     "train",
    "stairs_montjuic_01":   "val",
    "red_bridge_combined_01": "test",
}
RECORDINGS = list(SPLIT.keys())

# Model C schema (exactly): identifiers + 10 features + 2 targets.
MODEL_C_COLS = [
    "recording_id", "trajectory_id", "timestep", "split",
    "u", "v",
    "du", "dv", "speed", "heading_sin", "heading_cos", "turn_rate",
    "dist_to_obstacle_norm", "dist_to_boundary_norm",
    "target_du", "target_dv",
]
# Master (diagnostic) schema: Model C cols + entrance_affinity_norm + raw provenance.
MASTER_EXTRA = [
    "entrance_affinity_norm",
    "src_track_id", "frame", "world_x", "world_y",
    "dist_to_obstacle", "dist_to_boundary", "dist_to_entrance",
]


def build_one(rec: str) -> tuple[pd.DataFrame, dict]:
    src = ENCODED_ROOT / rec / "spatial_v21C" / "trajectories_encoded.csv"
    if not src.exists():
        raise SystemExit(f"[FATAL] missing encoded CSV: {src}")
    df = pd.read_csv(src)
    missing = bridge.INPUT_REQUIRED - set(df.columns)
    if missing:
        raise SystemExit(f"[FATAL] {rec} missing input columns: {sorted(missing)}")

    n_rows_in, n_tracks_in = len(df), df["track_id"].nunique()

    # --- track-length filter (frozen: >= MIN_TRACK_LEN) ---
    lengths = df.groupby("track_id").size()
    keep = lengths[lengths >= bridge.MIN_TRACK_LEN].index
    df = df[df["track_id"].isin(keep)].copy()

    # --- per-recording u/v MinMax ---
    xmin, xmax = float(df["world_x"].min()), float(df["world_x"].max())
    ymin, ymax = float(df["world_y"].min()), float(df["world_y"].max())
    xrng, yrng = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
    df["u"] = (df["world_x"] - xmin) / xrng
    df["v"] = (df["world_y"] - ymin) / yrng

    # --- per-track motion + next-step targets (frozen function, verbatim) ---
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    parts = [bridge.per_track_features(g.assign(track_id=tid))
             for tid, g in df.groupby("track_id", sort=False)]
    df = pd.concat(parts, ignore_index=True)

    # --- per-recording relational spatial features (frozen rank_pct) ---
    df["dist_to_obstacle_norm"] = bridge.rank_pct(df["dist_to_obstacle"].to_numpy(), ascending=True)
    df["dist_to_boundary_norm"] = bridge.rank_pct(df["dist_to_boundary"].to_numpy(), ascending=True)
    df["entrance_affinity_norm"] = bridge.rank_pct(df["dist_to_entrance"].to_numpy(), ascending=False)

    # --- drop last frame per track (no target) ---
    df = df.dropna(subset=["target_du", "target_dv"]).reset_index(drop=True)

    # --- identifiers / split ---
    df["recording_id"] = rec
    df["src_track_id"] = df["track_id"].astype(int)
    df["trajectory_id"] = rec + "__" + df["track_id"].astype(int).astype(str)
    df["split"] = SPLIT[rec]

    meta = {
        "recording_id": rec,
        "source_encoded_csv": str(src),
        "split": SPLIT[rec],
        "rows_in": int(n_rows_in), "tracks_in": int(n_tracks_in),
        "rows_used": int(len(df)), "tracks_used": int(df["trajectory_id"].nunique()),
        "world_bounds": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
                         "xrng": xrng, "yrng": yrng},
        "normalization_ranges": {
            "u": [float(df["u"].min()), float(df["u"].max())],
            "v": [float(df["v"].min()), float(df["v"].max())],
            "dist_to_obstacle_norm": [float(df["dist_to_obstacle_norm"].min()),
                                       float(df["dist_to_obstacle_norm"].max())],
            "dist_to_boundary_norm": [float(df["dist_to_boundary_norm"].min()),
                                       float(df["dist_to_boundary_norm"].max())],
        },
        "notes": ("frozen bridge recipe (MIN_TRACK_LEN=%d) applied per-recording; "
                  "percentile-rank spatial features; entrance_affinity_norm kept "
                  "diagnostic-only (Model D feature, excluded from Model C)."
                  % bridge.MIN_TRACK_LEN),
    }
    print(f"  [{rec:24s}] split={SPLIT[rec]:5s} "
          f"{n_rows_in:>7,}->{len(df):>7,} rows  "
          f"{n_tracks_in:>5}->{df['trajectory_id'].nunique():>5} tracks")
    return df, meta


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Assembling Barcelona_v1 master dataset (frozen bridge recipe) ...")
    frames, metas = [], []
    for rec in RECORDINGS:
        df, meta = build_one(rec)
        frames.append(df); metas.append(meta)

    master = pd.concat(frames, ignore_index=True)

    # Order master columns: identifiers + Model C features + diagnostics.
    master_cols = (MODEL_C_COLS
                   + [c for c in MASTER_EXTRA if c in master.columns])
    master = master[master_cols]
    master_path = OUT_DIR / "master_dataset.csv"
    master.to_csv(master_path, index=False)

    model_c = master[MODEL_C_COLS].copy()
    model_c_path = OUT_DIR / "model_C_dataset.csv"
    model_c.to_csv(model_c_path, index=False)

    # manifest.json
    manifest = {
        "dataset_version": "Barcelona_v1_master",
        "encoded_source_root": str(ENCODED_ROOT),
        "recipe": "frozen make_bridge_dataset.py (per-recording), "
                  "C_motion_position_spatial Model C feature set",
        "min_track_length_filter": bridge.MIN_TRACK_LEN,
        "split_rule": "recording-level (each recording in exactly one split)",
        "recordings": metas,
        "totals": {
            "rows_master": int(len(master)),
            "tracks_master": int(master["trajectory_id"].nunique()),
        },
        "model_C_columns": MODEL_C_COLS,
        "excluded_from_model_C": ["entrance_affinity_norm"],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")

    print(f"\n[ok] master_dataset.csv  : {len(master):,} rows x {master.shape[1]} cols")
    print(f"[ok] model_C_dataset.csv : {len(model_c):,} rows x {model_c.shape[1]} cols")
    print(f"[ok] manifest.json")
    print(f"Output -> {OUT_DIR}")


if __name__ == "__main__":
    main()
