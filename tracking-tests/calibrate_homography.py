"""
calibrate_homography.py
-----------------------
Transforms tracked image-space trajectories into ground-plane world coordinates
using a homography computed from 4+ point correspondences.

Usage:
    python calibrate_homography.py \
        --traj_csv outputs/trajectories_image_space.csv \
        --calib_json calib.json \
        --out_csv outputs/trajectories_world.csv \
        --out_plot outputs/trajectories_world.png
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Homography helpers
# ---------------------------------------------------------------------------

def load_calibration(calib_json_path: str):
    """Load and validate image/world point pairs from a JSON file."""
    path = Path(calib_json_path)
    if not path.exists():
        sys.exit(f"[ERROR] calib_json not found: {path}")

    with open(path) as f:
        data = json.load(f)

    for key in ("image_points", "world_points"):
        if key not in data:
            sys.exit(f"[ERROR] calib_json is missing key: '{key}'")

    img_pts = np.array(data["image_points"], dtype=np.float32)
    world_pts = np.array(data["world_points"], dtype=np.float32)

    if img_pts.ndim != 2 or img_pts.shape[1] != 2:
        sys.exit("[ERROR] image_points must be a list of [x, y] pairs")
    if world_pts.ndim != 2 or world_pts.shape[1] != 2:
        sys.exit("[ERROR] world_points must be a list of [x, y] pairs")
    if len(img_pts) != len(world_pts):
        sys.exit("[ERROR] image_points and world_points must have the same number of entries")
    if len(img_pts) < 4:
        sys.exit("[ERROR] At least 4 point correspondences are required to compute a homography")

    return img_pts, world_pts


def compute_homography(img_pts: np.ndarray, world_pts: np.ndarray) -> np.ndarray:
    """Compute the 3x3 homography matrix from image -> world."""
    H, mask = cv2.findHomography(img_pts, world_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if H is None:
        sys.exit("[ERROR] cv2.findHomography failed — check that your point correspondences are correct")
    inliers = int(mask.sum()) if mask is not None else len(img_pts)
    print(f"[INFO] Homography computed. Inliers: {inliers}/{len(img_pts)}")
    return H


def apply_homography(H: np.ndarray, x: np.ndarray, y: np.ndarray):
    """
    Apply a 3x3 homography to arrays of (x, y) image coordinates.
    Returns (world_x, world_y) as float arrays of the same length.
    """
    # Stack as homogeneous column vectors: shape (3, N)
    pts = np.vstack([x, y, np.ones(len(x))])          # (3, N)
    projected = H @ pts                                 # (3, N)
    w = projected[2]
    world_x = projected[0] / w
    world_y = projected[1] / w
    return world_x, world_y


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_quality_plot(calib: dict, out_path: str) -> None:
    """
    Quality diagnostic plot in top-view pixel space:
      - Plan image as background (if available and path resolves)
      - Green circles: selected top-view correspondence pixels
      - Orange triangles: camera image points projected through H → plan pixels
      - Red lines: residual vectors
      - Per-point reprojection error annotation
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.image as mpimg
    except ImportError:
        print("[WARN] matplotlib not installed — skipping quality plot.")
        return

    plan_pts  = np.array(calib.get("plan_points_px", []),  dtype=np.float64)
    img_pts   = np.array(calib.get("image_points",   []),  dtype=np.float64)
    H_list    = calib.get("homography_matrix")
    mpp       = float(calib.get("meters_per_plan_pixel", 1.0))
    ox, oy    = calib.get("plan_origin_pixel", [0, 0])
    invert_y  = bool(calib.get("invert_plan_y", False))

    if len(plan_pts) == 0 or H_list is None:
        print("[WARN] calib.json missing points or homography — skipping quality plot.")
        return

    H = np.array(H_list, dtype=np.float64)

    # Project image points through H → world, then → plan pixels
    proj_world = cv2.perspectiveTransform(
        img_pts.reshape(-1, 1, 2).astype(np.float32), H
    ).reshape(-1, 2)

    if invert_y:
        proj_px_x = proj_world[:, 0] / mpp + ox
        proj_px_y = -proj_world[:, 1] / mpp + oy
    else:
        proj_px_x = proj_world[:, 0] / mpp + ox
        proj_px_y = proj_world[:, 1] / mpp + oy
    proj_plan = np.column_stack([proj_px_x, proj_px_y])

    res_px = np.hypot(proj_plan[:, 0] - plan_pts[:, 0],
                      proj_plan[:, 1] - plan_pts[:, 1])
    res_m  = res_px * mpp

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("white")

    # Background: plan image
    plan_img_path = calib.get("plan_image", "")
    bg_h = bg_w = None
    if plan_img_path and Path(plan_img_path).exists():
        try:
            bg = mpimg.imread(plan_img_path)
            ax.imshow(bg, zorder=0, origin="upper", alpha=0.6)
            bg_h, bg_w = bg.shape[:2]
        except Exception:
            pass

    # Draw residuals + points
    for i in range(len(plan_pts)):
        ax_pt, ay_pt = plan_pts[i]
        px_pt, py_pt = proj_plan[i]
        ax.plot([ax_pt, px_pt], [ay_pt, py_pt],
                color="red", linewidth=1.2, alpha=0.8, zorder=3)
        ax.scatter(ax_pt, ay_pt, s=90, color="lime",
                   edgecolors="white", linewidths=0.8, zorder=5)
        ax.annotate(str(i + 1), (ax_pt, ay_pt),
                    xytext=(6, -8), textcoords="offset points",
                    fontsize=9, color="lime", fontweight="bold")
        ax.scatter(px_pt, py_pt, s=60, color="orange",
                   edgecolors="white", linewidths=0.8, zorder=4, marker="^")
        ax.annotate(f"{res_m[i]:.3f} m", (px_pt, py_pt),
                    xytext=(6, 5), textcoords="offset points",
                    fontsize=7, color="orange")

    # Axis orientation: y-down (image convention)
    if bg_h:
        ax.set_xlim(0, bg_w)
        ax.set_ylim(bg_h, 0)
    else:
        all_y = np.concatenate([plan_pts[:, 1], proj_plan[:, 1]])
        ax.set_ylim(all_y.max() * 1.05, all_y.min() - all_y.ptp() * 0.05)

    ax.set_aspect("equal", adjustable="box")

    patches = [
        mpatches.Patch(color="lime",   label="Selected top-view point"),
        mpatches.Patch(color="orange", label="Projected camera point"),
        mpatches.Patch(color="red",    label="Residual"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.85)
    ax.set_title(
        f"Calibration quality  |  {len(plan_pts)} pairs  |  "
        f"mean {res_m.mean():.3f} m  |  median {np.median(res_m):.3f} m  |  "
        f"max {res_m.max():.3f} m",
        fontsize=9, pad=10
    )
    ax.set_xlabel("Top-view X (px)", fontsize=8)
    ax.set_ylabel("Top-view Y (px)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(False)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[INFO] Quality plot saved: {out.resolve()}")


def save_topdown_plot(df: pd.DataFrame, out_path: str):
    """
    Draw one polyline per track_id in world coordinates and save as PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    track_ids = df["track_id"].unique()
    cmap = plt.get_cmap("tab20", max(len(track_ids), 1))

    for i, tid in enumerate(sorted(track_ids)):
        track = df[df["track_id"] == tid].sort_values("frame")
        ax.plot(track["world_x"], track["world_y"],
                linewidth=0.8, alpha=0.7, color=cmap(i % 20))
        # mark start point
        ax.scatter(track["world_x"].iloc[0], track["world_y"].iloc[0],
                   s=12, color=cmap(i % 20), zorder=3)

    # Explicit bounds with 5 % padding so equal-aspect doesn't collapse narrow extents
    wx = df["world_x"].to_numpy()
    wy = df["world_y"].to_numpy()
    eps = 1e-6
    x_pad = 0.05 * (wx.max() - wx.min() + eps)
    y_pad = 0.05 * (wy.max() - wy.min() + eps)
    ax.set_xlim(wx.min() - x_pad, wx.max() + x_pad)
    ax.set_ylim(wy.min() - y_pad, wy.max() + y_pad)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_title(f"Top-down trajectories — {len(track_ids)} tracks")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Plot saved: {out.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Apply homography to transform tracked trajectories into world coordinates")
    ap.add_argument("--traj_csv",   required=True,                           help="Input CSV from track_people.py")
    ap.add_argument("--calib_json", required=True,                           help="JSON file with image_points and world_points")
    ap.add_argument("--out_csv",    default="outputs/trajectories_world.csv", help="Output CSV with world coordinates")
    ap.add_argument("--out_plot",         default="outputs/trajectories_world.png", help="Output top-down plot PNG")
    ap.add_argument("--out_quality_plot", default="outputs/calib_quality.png",
                    help="Calibration quality diagnostic plot (default: outputs/calib_quality.png)")
    args = ap.parse_args()

    # --- Load trajectory CSV ---
    traj_path = Path(args.traj_csv)
    if not traj_path.exists():
        sys.exit(f"[ERROR] traj_csv not found: {traj_path}")

    df = pd.read_csv(traj_path)
    if df.empty:
        sys.exit("[ERROR] traj_csv is empty — run track_people.py first")

    required_cols = {"frame", "track_id"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] traj_csv is missing columns: {missing}")

    # --- Choose image-space source columns ---
    if "foot_x" in df.columns and "foot_y" in df.columns:
        image_x = df["foot_x"].to_numpy(dtype=np.float64)
        image_y = df["foot_y"].to_numpy(dtype=np.float64)
        print("[INFO] Using foot_x / foot_y as image-space source")
    elif "cx" in df.columns and "cy" in df.columns:
        image_x = df["cx"].to_numpy(dtype=np.float64)
        image_y = df["cy"].to_numpy(dtype=np.float64)
        print("[INFO] foot_x/foot_y not found — falling back to cx / cy")
    else:
        sys.exit("[ERROR] traj_csv must contain either (foot_x, foot_y) or (cx, cy)")

    # --- Calibration ---
    img_pts, world_pts = load_calibration(args.calib_json)

    n_pairs = len(img_pts)
    if n_pairs == 4:
        print("[WARN] Only 4 correspondence pairs used; calibration may be fragile "
              "away from selected points. Consider re-running with 6-10 pairs.")
    print(f"[INFO] Using {n_pairs} correspondence pairs")

    H = compute_homography(img_pts, world_pts)

    # --- Per-pair reprojection errors ---
    src = img_pts.astype(np.float32)
    dst = world_pts.astype(np.float32)
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    errs_m = np.hypot(proj[:, 0] - dst[:, 0], proj[:, 1] - dst[:, 1])
    print(f"[INFO] Per-pair reprojection errors (m):")
    for i, e in enumerate(errs_m):
        print(f"       Pair {i+1:2d}: {e:.4f} m")
    print(f"[INFO] mean={errs_m.mean():.4f}  median={np.median(errs_m):.4f}  "
          f"max={errs_m.max():.4f}  [m]")

    # --- Transform ---
    world_x, world_y = apply_homography(H, image_x, image_y)

    df["image_x"] = image_x
    df["image_y"] = image_y
    df["world_x"] = np.round(world_x, 4)
    df["world_y"] = np.round(world_y, 4)

    # --- Save CSV ---
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] World-coordinate CSV saved: {out_csv.resolve()}")
    print(f"[INFO] Rows: {len(df)}  |  Tracks: {df['track_id'].nunique()}")
    print(f"[INFO] World extent: "
          f"X=[{world_x.min():.2f}, {world_x.max():.2f}] m, "
          f"Y=[{world_y.min():.2f}, {world_y.max():.2f}] m")

    # --- Save trajectory plot ---
    save_topdown_plot(df, args.out_plot)

    # --- Save calibration quality plot ---
    import json as _json
    with open(Path(args.calib_json), encoding="utf-8") as _f:
        _calib = _json.load(_f)
    save_quality_plot(_calib, args.out_quality_plot)


if __name__ == "__main__":
    main()
