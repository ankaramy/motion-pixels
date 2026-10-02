"""
build_barcelona_v3_master.py
----------------------------
Builds the Barcelona **V3** Model C master dataset. It is a faithful mirror of
`build_barcelona_v1_master.py`: same FROZEN bridge recipe functions
(`per_track_features`, `rank_pct`, `wrap`, `MIN_TRACK_LEN`), same per-recording
normalization, same recording-level split, same Model C schema.

The ONLY change vs v1 is the source of the two spatial columns:
    v1:  dist_to_obstacle, dist_to_boundary   (old trajectory-coverage masks)
    v3:  dist_to_obstacle_v3_m, dist_to_walkable_boundary_v3_m
         (real manual architectural masks, Encoder V3)

This makes Barcelona_v1 vs Barcelona_v3 a clean A/B: identical trajectories,
identical motion/position/targets, identical recipe — only the architectural
spatial features differ.

Out-of-bounds (OOB) handling: V3 distances are NaN where a trajectory point maps
outside the plan-image crop (mainly red_bridge, ~27%). Old v1 masks had no NaN
(they were derived from the trajectories themselves). To keep the row/track set
identical to v1 we IMPUTE OOB distances to that recording's maximum finite
distance (i.e. "farthest / open") BEFORE percentile ranking. This is documented
in the manifest and schema report and is the only V3-specific data decision.

Outputs -> new_datasets/Barcelona_v3_manual_master_dataset/
    master_dataset.csv
    model_C_dataset.csv
    manifest.json
    dataset_validation.json
    dataset_schema_report.md
    feature_ranges.json
    split_report.md
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

import os
# Large data lives outside Git under MP_DATA_ROOT (docs/DATA_AVAILABILITY.md).
DATA_ROOT = Path(os.environ.get("MP_DATA_ROOT", HERE.parents[1] / "mp-data" / "external"))
ENCODED_ROOT = DATA_ROOT / "Barcelona_v3_manual_encoded"
OUT_DIR = DATA_ROOT / "Barcelona_v3_manual_master_dataset"

# recording-level split — identical to Barcelona_v1
SPLIT = {
    "esplanade_espanya_01": "train",
    "placa_catalunya_01":   "train",
    "placa_espanya_01":     "train",
    "stairs_montjuic_01":   "val",
    "red_bridge_combined_01": "test",
}
RECORDINGS = list(SPLIT.keys())

# V3 spatial source columns -> recipe spatial names
V3_OBST = "dist_to_obstacle_v3_m"
V3_BND = "dist_to_walkable_boundary_v3_m"

MODEL_C_COLS = [
    "recording_id", "trajectory_id", "timestep", "split",
    "u", "v",
    "du", "dv", "speed", "heading_sin", "heading_cos", "turn_rate",
    "dist_to_obstacle_norm", "dist_to_boundary_norm",
    "target_du", "target_dv",
]
MASTER_EXTRA = [
    "src_track_id", "frame", "world_x", "world_y",
    "dist_to_obstacle_v3_m", "dist_to_walkable_boundary_v3_m",
    "inbounds_v3",
]


def build_one(rec: str) -> tuple[pd.DataFrame, dict]:
    src = ENCODED_ROOT / rec / "spatial_v3_manual" / "trajectories_encoded_v3.csv"
    if not src.exists():
        raise SystemExit(f"[FATAL] missing V3 encoded CSV: {src}")
    df = pd.read_csv(src)
    need = {"track_id", "frame", "world_x", "world_y", V3_OBST, V3_BND}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"[FATAL] {rec} missing input columns: {sorted(missing)}")

    n_rows_in, n_tracks_in = len(df), df["track_id"].nunique()

    # --- track-length filter (frozen: >= MIN_TRACK_LEN) ---
    lengths = df.groupby("track_id").size()
    keep = lengths[lengths >= bridge.MIN_TRACK_LEN].index
    df = df[df["track_id"].isin(keep)].copy()

    # --- per-recording u/v MinMax (identical world coords to v1) ---
    xmin, xmax = float(df["world_x"].min()), float(df["world_x"].max())
    ymin, ymax = float(df["world_y"].min()), float(df["world_y"].max())
    xrng, yrng = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
    df["u"] = (df["world_x"] - xmin) / xrng
    df["v"] = (df["world_y"] - ymin) / yrng

    # --- OOB imputation: NaN V3 distances -> recording max finite (farthest/open) ---
    oob_obst = int(df[V3_OBST].isna().sum())
    oob_bnd = int(df[V3_BND].isna().sum())
    obst_max = float(np.nanmax(df[V3_OBST].to_numpy()))
    bnd_max = float(np.nanmax(df[V3_BND].to_numpy()))
    df[V3_OBST] = df[V3_OBST].fillna(obst_max)
    df[V3_BND] = df[V3_BND].fillna(bnd_max)
    if "inbounds_v3" not in df.columns:
        df["inbounds_v3"] = 1

    # --- per-track motion + next-step targets (frozen function, verbatim) ---
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    parts = [bridge.per_track_features(g.assign(track_id=tid))
             for tid, g in df.groupby("track_id", sort=False)]
    df = pd.concat(parts, ignore_index=True)

    # --- per-recording relational spatial features (frozen rank_pct, ascending) ---
    df["dist_to_obstacle_norm"] = bridge.rank_pct(df[V3_OBST].to_numpy(), ascending=True)
    df["dist_to_boundary_norm"] = bridge.rank_pct(df[V3_BND].to_numpy(), ascending=True)

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
        "oob_imputed_obstacle_rows": oob_obst,
        "oob_imputed_boundary_rows": oob_bnd,
        "oob_obstacle_pct": round(100 * oob_obst / max(n_rows_in, 1), 3),
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
                  "spatial features = percentile-rank of V3 manual-mask distances "
                  "(ascending); OOB distances imputed to recording max before "
                  "ranking." % bridge.MIN_TRACK_LEN),
    }
    print(f"  [{rec:24s}] split={SPLIT[rec]:5s} "
          f"{n_rows_in:>7,}->{len(df):>7,} rows  "
          f"{n_tracks_in:>5}->{df['trajectory_id'].nunique():>5} tracks  "
          f"OOB_obst={oob_obst} ({meta['oob_obstacle_pct']}%)")
    return df, meta


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Assembling Barcelona_v3 manual master dataset (frozen bridge recipe) ...")
    frames, metas = [], []
    for rec in RECORDINGS:
        df, meta = build_one(rec)
        frames.append(df); metas.append(meta)

    master = pd.concat(frames, ignore_index=True)
    master_cols = MODEL_C_COLS + [c for c in MASTER_EXTRA if c in master.columns]
    master = master[master_cols]
    master.to_csv(OUT_DIR / "master_dataset.csv", index=False)

    model_c = master[MODEL_C_COLS].copy()
    model_c.to_csv(OUT_DIR / "model_C_dataset.csv", index=False)

    # manifest.json
    manifest = {
        "dataset_version": "Barcelona_v3_manual_master",
        "encoded_source_root": str(ENCODED_ROOT),
        "recipe": "frozen make_bridge_dataset.py (per-recording), "
                  "C_motion_position_spatial Model C feature set; "
                  "spatial features from Encoder V3 manual masks",
        "min_track_length_filter": bridge.MIN_TRACK_LEN,
        "split_rule": "recording-level (identical to Barcelona_v1)",
        "spatial_source": {"obstacle": V3_OBST, "boundary": V3_BND,
                           "oob_imputation": "NaN -> recording max finite distance "
                                             "(farthest/open) before percentile rank"},
        "recordings": metas,
        "totals": {"rows_master": int(len(master)),
                   "tracks_master": int(master["trajectory_id"].nunique())},
        "model_C_columns": MODEL_C_COLS,
        "difference_vs_v1": "ONLY dist_to_obstacle_norm and dist_to_boundary_norm "
                            "differ (V3 real architecture vs v1 trajectory-coverage); "
                            "motion/position/targets identical.",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # feature_ranges.json
    franges = {}
    for c in MODEL_C_COLS:
        if c in ("recording_id", "trajectory_id", "timestep", "split"):
            continue
        s = pd.to_numeric(master[c], errors="coerce")
        franges[c] = {"min": float(s.min()), "max": float(s.max()),
                      "mean": float(s.mean()), "std": float(s.std()),
                      "n_nan": int(s.isna().sum())}
    (OUT_DIR / "feature_ranges.json").write_text(json.dumps(franges, indent=2), encoding="utf-8")

    # dataset_validation.json
    val = {
        "rows_master": int(len(master)),
        "tracks_master": int(master["trajectory_id"].nunique()),
        "n_nan_in_model_C": int(model_c[[c for c in MODEL_C_COLS
                                         if c not in ('recording_id','trajectory_id','split')]]
                                .isna().sum().sum()),
        "split_track_counts": {s: int(master[master.split == s]["trajectory_id"].nunique())
                               for s in ("train", "val", "test")},
        "split_row_counts": {s: int((master.split == s).sum())
                             for s in ("train", "val", "test")},
        "feature_order_model_C": [c for c in MODEL_C_COLS
                                  if c not in ('recording_id','trajectory_id','timestep','split',
                                               'target_du','target_dv')],
    }
    (OUT_DIR / "dataset_validation.json").write_text(json.dumps(val, indent=2), encoding="utf-8")

    # split_report.md
    sl = ["# Barcelona_v3 — Train/Val/Test Split Report", "",
          "Recording-level split (identical to Barcelona_v1 for fair comparison).", "",
          "| Recording | Split | Tracks | Rows | OOB obstacle % |",
          "|---|---|---|---|---|"]
    for meta in metas:
        sl.append(f"| {meta['recording_id']} | {meta['split']} | {meta['tracks_used']} "
                  f"| {meta['rows_used']} | {meta['oob_obstacle_pct']} |")
    sl += ["", "Totals by split:", "",
           "| Split | Tracks | Rows |", "|---|---|---|"]
    for s in ("train", "val", "test"):
        sl.append(f"| {s} | {val['split_track_counts'][s]} | {val['split_row_counts'][s]} |")
    (OUT_DIR / "split_report.md").write_text("\n".join(sl), encoding="utf-8")

    # dataset_schema_report.md
    dl = ["# Barcelona_v3 Model C — Dataset Schema Report", "",
          "Faithful V3 mirror of the Barcelona_v1 Model C dataset. Identical recipe,",
          "split, and schema; the **only** difference is that the two spatial features",
          "come from Encoder V3 real architectural masks instead of the old",
          "trajectory-coverage masks.", "",
          f"- Rows: **{len(master):,}**  Tracks: **{master['trajectory_id'].nunique():,}**",
          f"- Min track length filter: {bridge.MIN_TRACK_LEN}",
          f"- Split: recording-level (train=esplanade/catalunya/espanya, val=stairs, test=red_bridge)",
          "", "## Model C feature order (10) + targets", "",
          "```", ", ".join([c for c in MODEL_C_COLS
                             if c not in ('recording_id','trajectory_id','timestep','split',
                                          'target_du','target_dv')]),
          "targets: target_du, target_dv", "```", "",
          "## Normalization (per recording)", "",
          "- u, v: MinMax over recording world_x / world_y",
          "- du, dv, speed: raw metres/step (StandardScaler fit at train time)",
          "- turn_rate: radians in (-pi, pi]",
          "- dist_to_obstacle_norm, dist_to_boundary_norm: percentile rank ascending "
          "(0 = closest, 1 = farthest) of the V3 manual-mask distances",
          "- OOB V3 distances imputed to recording max finite distance before ranking",
          "", "## Column statistics", "",
          "| column | mean | std | min | max |", "|---|---|---|---|---|"]
    for c in MODEL_C_COLS:
        if c in ("recording_id", "trajectory_id", "timestep", "split"):
            continue
        s = pd.to_numeric(master[c], errors="coerce")
        dl.append(f"| `{c}` | {s.mean():.4f} | {s.std():.4f} | {s.min():.4f} | {s.max():.4f} |")
    (OUT_DIR / "dataset_schema_report.md").write_text("\n".join(dl), encoding="utf-8")

    print(f"\n[ok] master_dataset.csv  : {len(master):,} rows x {master.shape[1]} cols")
    print(f"[ok] model_C_dataset.csv : {len(model_c):,} rows x {model_c.shape[1]} cols")
    print(f"[ok] manifest / feature_ranges / dataset_validation / split_report / schema_report")
    print(f"Output -> {OUT_DIR}")


if __name__ == "__main__":
    main()
