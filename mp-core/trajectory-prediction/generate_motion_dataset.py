"""
generate_motion_dataset.py
--------------------------
Produce one row per trajectory timestep encoding motion state plus
directional spatial affordance.

Schema
------
Identifiers : trajectory_id, timestep
Inputs      : u, v, du, dv, speed, heading_sin, heading_cos, turn_rate,
              dist_to_obstacle_norm, dist_to_boundary_norm,
              delta_dist_to_obstacle, delta_dist_to_boundary,
              openness_ahead, openness_left, openness_right
Targets     : target_du, target_dv  (next-step displacement, no leakage)

Usage
-----
  python generate_motion_dataset.py
  python generate_motion_dataset.py --traj_csv PATH --calib_json PATH \\
      --spatial_dir PATH --output_dir PATH [--lookahead_m 1.5]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ── defaults ──────────────────────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent
MP_ROOT      = HERE.parent.parent
DEFAULT_TRAJ = MP_ROOT / "mp-data" / "processed" / "trajectories" / "trajectories_world.csv"
DEFAULT_CALIB   = MP_ROOT / "mp-data" / "raw" / "calibration" / "calib_skate1.json"
DEFAULT_SPATIAL = MP_ROOT / "mp-data" / "processed" / "spatial"
DEFAULT_OUT     = MP_ROOT / "mp-data" / "processed" / "encoded"

# ── tunable constants ─────────────────────────────────────────────────────────
MIN_TRAJ_LEN  = 3       # skip trajectories shorter than this (no usable target rows)
MIN_SPEED_MS  = 1e-6    # below this magnitude heading is treated as undefined


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def load_calib(path: Path) -> tuple[np.ndarray, float]:
    """
    Derive world→plan homography via cv2.findHomography(world_points, plan_points_px).
    Returns (H_3x3, meters_per_plan_pixel).
    """
    with open(path) as f:
        c = json.load(f)

    for key in ("world_points", "plan_points_px", "meters_per_plan_pixel"):
        if key not in c:
            raise KeyError(f"Calibration JSON missing required key: '{key}'")

    world_pts = np.array(c["world_points"],   dtype=np.float32)   # (N, 2)
    plan_pts  = np.array(c["plan_points_px"], dtype=np.float32)   # (N, 2)

    if len(world_pts) < 4:
        raise ValueError(f"Need ≥4 correspondence points, got {len(world_pts)}")

    H, mask = cv2.findHomography(world_pts, plan_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("cv2.findHomography returned None — check calibration points")

    n_inliers = int(mask.sum()) if mask is not None else len(world_pts)
    print(f"[calib]  findHomography: {n_inliers}/{len(world_pts)} inliers")

    return H.astype(np.float64), float(c["meters_per_plan_pixel"])


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def world_to_plan_px(H: np.ndarray, world_xy: np.ndarray) -> np.ndarray:
    """
    Map (N, 2) world-metre coords to (N, 2) plan-pixel coords [col, row]
    via a 3x3 perspective homography.
    """
    pts = world_xy.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)   # col = out[:,0], row = out[:,1]


def sample_dist_map(dist_map: np.ndarray,
                    plan_px: np.ndarray,
                    oob_val: float) -> np.ndarray:
    """
    Sample a (H, W) float32 distance map at (N, 2) float pixel coords [col, row].
    Out-of-bounds pixels and non-finite coords receive oob_val.
    """
    h, w    = dist_map.shape
    col_f   = plan_px[:, 0]
    row_f   = plan_px[:, 1]
    finite  = np.isfinite(col_f) & np.isfinite(row_f)
    col     = np.round(np.where(finite, col_f, 0)).astype(int)
    row     = np.round(np.where(finite, row_f, 0)).astype(int)
    in_bnd  = finite & (col >= 0) & (col < w) & (row >= 0) & (row < h)
    out     = np.full(len(plan_px), oob_val, dtype=np.float32)
    out[in_bnd] = dist_map[row[in_bnd], col[in_bnd]]
    return out


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (−π, π]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ─────────────────────────────────────────────────────────────────────────────
# Per-trajectory feature computation
# ─────────────────────────────────────────────────────────────────────────────

def process_trajectory(
    df:           pd.DataFrame,
    tid:          int | str,
    H:            np.ndarray,
    dist_obs:     np.ndarray,
    dist_bnd:     np.ndarray,
    max_obs:      float,
    max_bnd:      float,
    world_bounds: dict,
    lookahead_m:  float,
) -> pd.DataFrame | None:
    """
    Compute all features for one trajectory group.
    Returns a DataFrame with n−1 rows (last row dropped: no target available).
    Returns None if the trajectory is too short.
    """
    df = df.sort_values("frame_number").reset_index(drop=True)
    n  = len(df)
    if n < MIN_TRAJ_LEN:
        return None

    wx = df["world_x"].to_numpy(np.float64)
    wy = df["world_y"].to_numpy(np.float64)

    # ── normalized world position (global bounds, computed by caller)
    u = (wx - world_bounds["xmin"]) / world_bounds["xrng"]
    v = (wy - world_bounds["ymin"]) / world_bounds["yrng"]

    # ── per-trajectory displacement (no global contamination)
    du = np.zeros(n)
    dv = np.zeros(n)
    du[1:] = wx[1:] - wx[:-1]
    dv[1:] = wy[1:] - wy[:-1]
    speed = np.hypot(du, dv)

    # ── heading: atan2(dv, du), carry forward when stationary
    raw_h   = np.arctan2(dv, du)
    heading = np.zeros(n)
    for i in range(n):
        if speed[i] > MIN_SPEED_MS:
            heading[i] = raw_h[i]
        elif i > 0:
            heading[i] = heading[i - 1]
        # else: heading[0] = 0.0 (default for stationary first step)

    # ── turn rate: heading change wrapped to (−π, π]
    turn_rate    = np.zeros(n)
    turn_rate[1:] = wrap_angle(heading[1:] - heading[:-1])

    # ── map to plan pixels for distance sampling
    wxy     = np.stack([wx, wy], axis=1)
    plan_px = world_to_plan_px(H, wxy)

    # ── distance at current position (fallback = max = "far from obstacles")
    raw_obs  = sample_dist_map(dist_obs, plan_px, max_obs)
    raw_bnd  = sample_dist_map(dist_bnd, plan_px, max_bnd)
    obs_norm = raw_obs / max_obs
    bnd_norm = raw_bnd / max_bnd

    # ── delta distances normalized by respective maxima
    d_obs    = np.zeros(n)
    d_bnd    = np.zeros(n)
    d_obs[1:] = (raw_obs[1:] - raw_obs[:-1]) / max_obs
    d_bnd[1:] = (raw_bnd[1:] - raw_bnd[:-1]) / max_bnd

    # ── directional openness: sample dist_to_obstacle LOOKAHEAD_M metres
    #    in each direction (computed in world space, then projected to plan)
    def openness(angle_offset: float) -> np.ndarray:
        ang = heading + angle_offset
        lx  = wx + np.cos(ang) * lookahead_m
        ly  = wy + np.sin(ang) * lookahead_m
        lpx = world_to_plan_px(H, np.stack([lx, ly], axis=1))
        # fallback=0 → OOB point treated as no clearance (conservative)
        return sample_dist_map(dist_obs, lpx, 0.0) / max_obs

    oa  = openness(0.0)           # ahead
    ol  = openness(np.pi / 2)    # left  (+90°)
    or_ = openness(-np.pi / 2)   # right (−90°)

    # ── targets: next-step displacement (anti-leakage)
    #    last row is dropped below, so t_du[-1] is never used
    t_du    = np.empty(n)
    t_dv    = np.empty(n)
    t_du[:-1] = du[1:]
    t_dv[:-1] = dv[1:]
    t_du[-1]  = np.nan
    t_dv[-1]  = np.nan

    out = pd.DataFrame({
        "trajectory_id":           np.full(n, tid),
        "timestep":                np.arange(n),
        "u":                       u,
        "v":                       v,
        "du":                      du,
        "dv":                      dv,
        "speed":                   speed,
        "heading_sin":             np.sin(heading),
        "heading_cos":             np.cos(heading),
        "turn_rate":               turn_rate,
        "dist_to_obstacle_norm":   obs_norm,
        "dist_to_boundary_norm":   bnd_norm,
        "delta_dist_to_obstacle":  d_obs,
        "delta_dist_to_boundary":  d_bnd,
        "openness_ahead":          oa,
        "openness_left":           ol,
        "openness_right":          or_,
        "target_du":               t_du,
        "target_dv":               t_dv,
    })

    return out.iloc[:-1].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Schema summary
# ─────────────────────────────────────────────────────────────────────────────

def build_schema_summary(max_obs: float, max_bnd: float,
                         lookahead_m: float, world_bounds: dict) -> dict:
    return {
        "version": 1,
        "description": (
            "One row per trajectory timestep. "
            "Encodes motion state + directional spatial affordance. "
            "Targets are next-step displacements (no leakage). "
            "Last timestep of each trajectory is excluded."
        ),
        "normalization": {
            "u_v": "global MinMax over dataset world_x / world_y extents",
            "dist_features": (
                f"divided by map-wide max: "
                f"obs={max_obs:.4f} m, bnd={max_bnd:.4f} m"
            ),
            "du_dv_speed": (
                "raw metres/step — scale at training time with a "
                "split-fitted scaler to avoid leakage"
            ),
            "turn_rate": "radians in (−π, π], not rescaled",
        },
        "world_bounds_used": world_bounds,
        "lookahead_m": lookahead_m,
        "columns": {
            "trajectory_id": {
                "units": "integer",
                "interpretation": "Unique pedestrian track identifier",
                "reason": "Groups timesteps belonging to the same person; required for sequential modelling and evaluation split",
            },
            "timestep": {
                "units": "integer (0-indexed)",
                "interpretation": "Position within the trajectory; 0 = first step",
                "reason": "Allows absolute positioning within a sequence without encoding recording artifacts of raw frame numbers",
            },
            "u": {
                "units": "normalised [0, 1]",
                "interpretation": "World x position normalised by dataset-wide extents",
                "reason": "Gives the model a stable positional reference; global normalisation keeps positions comparable across trajectories",
            },
            "v": {
                "units": "normalised [0, 1]",
                "interpretation": "World y position normalised by dataset-wide extents",
                "reason": "Same as u for the vertical axis",
            },
            "du": {
                "units": "metres",
                "interpretation": "World x displacement from the previous timestep; 0 at first step",
                "reason": "Encodes instantaneous horizontal velocity without positional bias; primary motion signal for the model",
            },
            "dv": {
                "units": "metres",
                "interpretation": "World y displacement from the previous timestep; 0 at first step",
                "reason": "Same as du for the vertical axis",
            },
            "speed": {
                "units": "metres / step",
                "interpretation": "Euclidean magnitude of (du, dv); redundant with du+dv but explicit",
                "reason": (
                    "Explicit speed lets the model attend to magnitude separately from direction. "
                    "Kept despite derivability because LSTMs benefit from pre-computed magnitude "
                    "to avoid learning sqrt implicitly."
                ),
            },
            "heading_sin": {
                "units": "dimensionless [−1, 1]",
                "interpretation": "sin(atan2(dv, du)); part of unit-vector heading encoding",
                "reason": (
                    "Sine+cosine encoding avoids the discontinuity at ±π that a raw angle "
                    "would create, giving the model a smooth heading representation"
                ),
            },
            "heading_cos": {
                "units": "dimensionless [−1, 1]",
                "interpretation": "cos(atan2(dv, du)); paired with heading_sin",
                "reason": "See heading_sin",
            },
            "turn_rate": {
                "units": "radians (−π, π]",
                "interpretation": "Signed change in heading from previous timestep; 0 at first step",
                "reason": (
                    "Captures angular dynamics (curved vs straight paths). "
                    "Important for distinguishing intentional turns from noise."
                ),
            },
            "dist_to_obstacle_norm": {
                "units": "normalised [0, 1]",
                "interpretation": (
                    f"Distance from the agent's plan-pixel position to the nearest obstacle, "
                    f"divided by map maximum ({max_obs:.3f} m)"
                ),
                "reason": "Spatial affordance signal; low values indicate the agent is near an obstacle",
            },
            "dist_to_boundary_norm": {
                "units": "normalised [0, 1]",
                "interpretation": (
                    f"Distance to the walkable-area boundary, "
                    f"divided by map maximum ({max_bnd:.3f} m)"
                ),
                "reason": "Complements dist_to_obstacle: captures proximity to the edge of the navigable region",
            },
            "delta_dist_to_obstacle": {
                "units": "normalised (change / map_max)",
                "interpretation": "Change in dist_to_obstacle_norm from previous step; 0 at first step",
                "reason": (
                    "Encodes approach/retreat dynamics relative to obstacles. "
                    "Helps the model anticipate avoidance manoeuvres before the agent is already close."
                ),
            },
            "delta_dist_to_boundary": {
                "units": "normalised (change / map_max)",
                "interpretation": "Change in dist_to_boundary_norm from previous step; 0 at first step",
                "reason": "Same as delta_dist_to_obstacle but for the walkable boundary",
            },
            "openness_ahead": {
                "units": "normalised [0, 1]",
                "interpretation": (
                    f"dist_to_obstacle sampled at the plan pixel corresponding to "
                    f"{lookahead_m} m ahead in the current heading direction, "
                    f"divided by map maximum. "
                    "Out-of-bounds → 0 (treated as no clearance)."
                ),
                "reason": (
                    "Directional affordance ahead: high value means open space, "
                    "low value means an obstacle is nearby in that direction. "
                    "Breaks the symmetry of the scalar dist_to_obstacle feature."
                ),
            },
            "openness_left": {
                "units": "normalised [0, 1]",
                "interpretation": (
                    f"dist_to_obstacle sampled {lookahead_m} m to the left "
                    "(heading rotated +90°)"
                ),
                "reason": "Directional affordance; allows the model to prefer turns into open space",
            },
            "openness_right": {
                "units": "normalised [0, 1]",
                "interpretation": (
                    f"dist_to_obstacle sampled {lookahead_m} m to the right "
                    "(heading rotated −90°)"
                ),
                "reason": "Directional affordance; symmetric counterpart to openness_left",
            },
            "target_du": {
                "units": "metres",
                "interpretation": "World x displacement at the NEXT timestep (t+1); prediction target",
                "reason": (
                    "Predicting displacement rather than absolute position avoids "
                    "compounding positional error during autoregressive rollout. "
                    "Strictly next-step to prevent leakage."
                ),
            },
            "target_dv": {
                "units": "metres",
                "interpretation": "World y displacement at the NEXT timestep (t+1); prediction target",
                "reason": "Same as target_du for the vertical axis",
            },
        },
        "schema_alterations": [
            {
                "change": "frame_number removed",
                "reason": (
                    "Redundant with timestep for sequential modelling. "
                    "Raw frame indices encode recording-specific artifacts "
                    "(e.g. frame-rate gaps) rather than motion patterns."
                ),
            },
            {
                "change": "dist_to_entrance removed",
                "reason": (
                    "Not available in the distance-field pipeline (requires manual annotation). "
                    "Entrance proximity can be approximated through position features."
                ),
            },
            {
                "change": "du/dv/speed kept in raw metres, not normalised here",
                "reason": (
                    "Normalisation should be deferred to training time using a scaler "
                    "fitted on the training split only. Pre-normalising would leak "
                    "dataset statistics into held-out folds."
                ),
            },
            {
                "change": "openness_* sampled from dist_to_obstacle only",
                "reason": (
                    "Obstacle clearance determines forward navigability. "
                    "Boundary openness is less informative for turn decisions "
                    "and would double the number of directional features without "
                    "proportional benefit."
                ),
            },
            {
                "change": "openness_* uses point sampling at lookahead, not ray casting",
                "reason": (
                    "The dist_to_obstacle field is precomputed; point sampling is O(1) "
                    "and sufficient because the field already encodes obstacle proximity "
                    "at each location. Ray casting would require iterative traversal."
                ),
            },
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate motion-state + spatial-affordance dataset for LSTM training."
    )
    parser.add_argument("--traj_csv",    type=Path, default=DEFAULT_TRAJ)
    parser.add_argument("--calib_json",  type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--spatial_dir", type=Path, default=DEFAULT_SPATIAL)
    parser.add_argument("--output_dir",  type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lookahead_m", type=float, default=1.5,
                        help="Look-ahead distance in metres for directional openness (default: 1.5)")
    parser.add_argument("--out_csv",     type=str, default="motion_dataset.csv",
                        help="Output CSV filename (placed in output_dir)")
    args = parser.parse_args()

    # ── validate inputs
    for p, label in [
        (args.traj_csv,    "trajectory CSV"),
        (args.calib_json,  "calibration JSON"),
        (args.spatial_dir, "spatial directory"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    obs_path = args.spatial_dir / "dist_to_obstacle.npy"
    bnd_path = args.spatial_dir / "dist_to_boundary.npy"
    for p, label in [(obs_path, "dist_to_obstacle.npy"), (bnd_path, "dist_to_boundary.npy")]:
        if not p.exists():
            raise FileNotFoundError(f"Distance map not found: {p}  (run segment_plan.py first)")

    # ── load assets
    print(f"[load]  {args.traj_csv.name}")
    df_raw = pd.read_csv(args.traj_csv)

    # normalise column names across pipeline variants
    col_map = {
        "track_id":    "trajectory_id",
        "person_id":   "trajectory_id",
        "frame":       "frame_number",
        "frame_idx":   "frame_number",
    }
    df_raw = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})

    for col in ("trajectory_id", "frame_number", "world_x", "world_y"):
        if col not in df_raw.columns:
            raise ValueError(f"Required column '{col}' not found. Columns: {list(df_raw.columns)}")

    print(f"[load]  {args.calib_json.name}")
    H, mpp = load_calib(args.calib_json)

    print(f"[load]  distance maps")
    dist_obs = np.load(obs_path).astype(np.float32)
    dist_bnd = np.load(bnd_path).astype(np.float32)
    max_obs  = float(dist_obs.max())
    max_bnd  = float(dist_bnd.max())
    print(f"         dist_to_obstacle  shape={dist_obs.shape}  max={max_obs:.3f} m")
    print(f"         dist_to_boundary  shape={dist_bnd.shape}  max={max_bnd:.3f} m")
    print(f"         lookahead         {args.lookahead_m} m")

    # ── global position bounds for u/v normalisation
    xmin = float(df_raw["world_x"].min())
    xmax = float(df_raw["world_x"].max())
    ymin = float(df_raw["world_y"].min())
    ymax = float(df_raw["world_y"].max())
    xrng = xmax - xmin or 1.0   # guard against degenerate data
    yrng = ymax - ymin or 1.0
    world_bounds = {"xmin": xmin, "xmax": xmax, "xrng": xrng,
                    "ymin": ymin, "ymax": ymax, "yrng": yrng}
    print(f"[bounds] world_x=[{xmin:.2f}, {xmax:.2f}]  world_y=[{ymin:.2f}, {ymax:.2f}]")

    # ── process per trajectory
    groups   = df_raw.groupby("trajectory_id", sort=False)
    n_total  = len(groups)
    parts    = []
    skipped  = 0

    print(f"[proc]  {n_total} trajectories ...")
    for i, (tid, gdf) in enumerate(groups):
        result = process_trajectory(
            gdf, tid, H, dist_obs, dist_bnd,
            max_obs, max_bnd, world_bounds, args.lookahead_m,
        )
        if result is None:
            skipped += 1
            continue
        parts.append(result)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n_total} ...")

    if not parts:
        raise RuntimeError("No usable trajectories found — check MIN_TRAJ_LEN or data quality")

    dataset = pd.concat(parts, ignore_index=True)

    # ── sanity checks
    assert dataset[["target_du", "target_dv"]].isna().sum().sum() == 0, \
        "Unexpected NaN in targets — last-row drop logic failed"
    assert (dataset["trajectory_id"].value_counts() > 0).all()

    # ── write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_out    = args.output_dir / args.out_csv
    schema_out = args.output_dir / "schema_summary.json"

    dataset.to_csv(csv_out, index=False)

    schema = build_schema_summary(max_obs, max_bnd, args.lookahead_m, world_bounds)
    with open(schema_out, "w") as f:
        json.dump(schema, f, indent=2)

    # ── summary
    n_traj  = dataset["trajectory_id"].nunique()
    n_rows  = len(dataset)
    n_cols  = len(dataset.columns)
    print(f"\n[done]  trajectories used  : {n_traj:,}  (skipped short: {skipped})")
    print(f"        rows               : {n_rows:,}")
    print(f"        columns            : {n_cols}")
    print(f"        output CSV         : {csv_out}")
    print(f"        schema JSON        : {schema_out}")

    print("\n-- feature statistics --------------------------------------------------")
    feat_cols = [c for c in dataset.columns if c not in ("trajectory_id", "timestep")]
    stats = dataset[feat_cols].agg(["min", "mean", "max", "std"]).T
    print(stats.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
