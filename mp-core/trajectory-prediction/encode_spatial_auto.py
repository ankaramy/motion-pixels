"""
encode_spatial_auto.py
----------------------
Encode spatial features automatically from segmentation outputs.

Reads a trajectory CSV with world coordinates, maps them to image pixels
using a homography from the calibration JSON, samples the precomputed
distance fields produced by segment_plan.py, and writes an enriched CSV
plus a debug overlay.

Usage
-----
  python encode_spatial_auto.py --traj_csv TRAJ.csv
                                --spatial_dir SPATIAL_DIR
                                --calib_json CALIB.json
                                [--scale METERS_PER_PIXEL]
                                [--output_csv OUT.csv]
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE        = Path(__file__).resolve().parent
MP_ROOT     = HERE.parent.parent
DEFAULT_OUT = MP_ROOT / "mp-data" / "processed" / "encoded" / "trajectories_encoded_auto.csv"


def load_homography(calib_path: Path) -> np.ndarray:
    """Load 3x3 homography matrix from calibration JSON."""
    with open(calib_path) as f:
        calib = json.load(f)
    if "homography_matrix" not in calib:
        raise KeyError(f"'homography_matrix' missing from {calib_path}")
    H = np.asarray(calib["homography_matrix"], dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3, got {H.shape}")
    return H


def world_to_pixel_homography(world_x, world_y, H_world_to_image):
    """
    Map world (m) → image pixel coords via a 3x3 homography.

    cv2.perspectiveTransform requires shape (N, 1, 2) float32.
    Returns (pixel_x, pixel_y) as float arrays plus a finite-mask.
    """
    pts = np.column_stack([
        np.asarray(world_x, dtype=np.float64),
        np.asarray(world_y, dtype=np.float64),
    ])
    finite_in = np.isfinite(pts).all(axis=1)

    pts_safe = np.where(finite_in[:, None], pts, 0.0).astype(np.float32)
    pts_safe = pts_safe.reshape(-1, 1, 2)

    out = cv2.perspectiveTransform(pts_safe, H_world_to_image.astype(np.float64))
    out = out.reshape(-1, 2)

    pixel_x = out[:, 0]
    pixel_y = out[:, 1]

    finite_out = np.isfinite(pixel_x) & np.isfinite(pixel_y)
    valid      = finite_in & finite_out
    return pixel_x, pixel_y, valid


def sample_field(field, px_int, py_int, valid):
    """Sample 2D field at integer pixel coords. Returns NaN where invalid."""
    h, w = field.shape[:2]
    px_c = np.clip(px_int, 0, w - 1)
    py_c = np.clip(py_int, 0, h - 1)
    out = field[py_c, px_c].astype(np.float64)
    out[~valid] = np.nan
    return out


def save_debug_overlay(walkable_mask, px, py, dist_obs_vals, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(walkable_mask, cmap="gray")
    finite = np.isfinite(px) & np.isfinite(py) & np.isfinite(dist_obs_vals)
    sc = ax.scatter(px[finite], py[finite], c=dist_obs_vals[finite],
                    cmap="viridis", s=4, alpha=0.8)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("dist_to_obstacle")
    ax.set_title("Trajectory points coloured by distance to nearest obstacle")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Encode spatial features automatically from segmentation outputs.")
    parser.add_argument("--traj_csv", type=Path, required=True,
                        help="input trajectories_world.csv")
    parser.add_argument("--spatial_dir", type=Path, required=True,
                        help="directory with segment_plan.py outputs")
    parser.add_argument("--calib_json", type=Path, required=True,
                        help="calibration JSON containing 3x3 homography_matrix")
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUT,
                        help="output CSV path")
    parser.add_argument("--scale", type=float, default=None,
                        help="meters per pixel (optional, only used to interpret "
                             "distance values; not used for coordinate mapping)")
    args = parser.parse_args()

    dist_obs_path = args.spatial_dir / "dist_to_obstacle.npy"
    dist_bnd_path = args.spatial_dir / "dist_to_boundary.npy"
    walk_path     = args.spatial_dir / "walkable_mask.png"

    if not args.traj_csv.exists():
        raise FileNotFoundError(f"Trajectory CSV not found: {args.traj_csv}")
    if not args.calib_json.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {args.calib_json}")
    if not dist_obs_path.exists():
        raise FileNotFoundError(f"Missing: {dist_obs_path}")
    if not dist_bnd_path.exists():
        raise FileNotFoundError(f"Missing: {dist_bnd_path}")

    print(f"[INFO]  Traj CSV    : {args.traj_csv}")
    print(f"[INFO]  Spatial dir : {args.spatial_dir}")
    print(f"[INFO]  Calib JSON  : {args.calib_json}")
    if args.scale is not None:
        print(f"[INFO]  Scale       : {args.scale} m/px (distance interpretation only)")

    H = load_homography(args.calib_json)

    # The stored homography maps image-pixels → world (per the tracking pipeline).
    # We want world → image, so invert. If the input is already world → image,
    # invertibility still holds and inversion would just flip the direction —
    # we trust the convention used elsewhere in this codebase.
    try:
        H_world_to_image = np.linalg.inv(H)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Homography is singular and cannot be inverted: {e}")

    df = pd.read_csv(args.traj_csv)
    if "world_x" not in df.columns or "world_y" not in df.columns:
        raise ValueError("CSV must contain world_x and world_y columns.")

    dist_obs = np.load(dist_obs_path)
    dist_bnd = np.load(dist_bnd_path)

    pixel_x, pixel_y, valid = world_to_pixel_homography(
        df["world_x"].to_numpy(),
        df["world_y"].to_numpy(),
        H_world_to_image,
    )

    h, w = dist_obs.shape[:2]
    px_int = np.clip(np.round(np.where(valid, pixel_x, 0)).astype(int), 0, w - 1)
    py_int = np.clip(np.round(np.where(valid, pixel_y, 0)).astype(int), 0, h - 1)

    in_bounds = (
        (pixel_x >= 0) & (pixel_x < w) &
        (pixel_y >= 0) & (pixel_y < h)
    )
    valid_sample = valid & in_bounds

    df["pixel_x"] = np.where(valid, pixel_x, np.nan)
    df["pixel_y"] = np.where(valid, pixel_y, np.nan)
    df["dist_to_obstacle"] = sample_field(dist_obs, px_int, py_int, valid_sample)
    df["dist_to_boundary"] = sample_field(dist_bnd, px_int, py_int, valid_sample)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"[OK]    Encoded CSV : {args.output_csv}  ({len(df):,} rows)")
    n_dropped = int((~valid).sum())
    n_oob     = int((valid & ~in_bounds).sum())
    if n_dropped:
        print(f"[WARN]  Non-finite world coords      : {n_dropped:,}")
    if n_oob:
        print(f"[WARN]  Transformed out-of-bounds    : {n_oob:,}")

    walkable_mask = (cv2.imread(str(walk_path), cv2.IMREAD_GRAYSCALE)
                     if walk_path.exists() else np.zeros_like(dist_obs))
    debug_path = args.output_csv.parent / "debug_overlay.png"
    save_debug_overlay(
        walkable_mask,
        df["pixel_x"].to_numpy(),
        df["pixel_y"].to_numpy(),
        df["dist_to_obstacle"].to_numpy(),
        debug_path,
    )
    print(f"[OK]    Debug image : {debug_path}")

    obs = df["dist_to_obstacle"].dropna()
    bnd = df["dist_to_boundary"].dropna()
    if not obs.empty:
        print(f"\n[OK]   dist_to_obstacle  min/mean/max : "
              f"{obs.min():.3f} / {obs.mean():.3f} / {obs.max():.3f}")
    if not bnd.empty:
        print(f"[OK]   dist_to_boundary  min/mean/max : "
              f"{bnd.min():.3f} / {bnd.mean():.3f} / {bnd.max():.3f}")


if __name__ == "__main__":
    main()
