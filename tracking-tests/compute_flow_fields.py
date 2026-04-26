"""
compute_flow_fields.py
----------------------
Generate a spatial flow-field analysis from tracked pedestrian trajectories.

Outputs:
  flow_field_vectors.csv  — one row per inter-frame step
  flow_field_cells.csv    — one row per grid cell (aggregated)
  flow_field_quiver.png   — clean quiver diagram for thesis boards

Usage:
    python compute_flow_fields.py \
        --traj_csv outputs/trajectories_world.csv \
        --out_dir  outputs/flow \
        --cell_size 1.0 \
        --min_vectors_per_cell 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Coordinate resolution
# ---------------------------------------------------------------------------

def resolve_coord_cols(df: pd.DataFrame):
    """
    Return (xcol, ycol, unit) based on column priority:
      1) world_x / world_y  → unit 'm'
      2) foot_x  / foot_y   → unit 'px'
      3) cx      / cy        → unit 'px'
    """
    if "world_x" in df.columns and "world_y" in df.columns:
        print("[INFO] Using world_x / world_y (metres)")
        return "world_x", "world_y", "m"
    if "foot_x" in df.columns and "foot_y" in df.columns:
        print("[INFO] Using foot_x / foot_y (pixels)")
        return "foot_x", "foot_y", "px"
    if "cx" in df.columns and "cy" in df.columns:
        print("[INFO] Using cx / cy (pixels)")
        return "cx", "cy", "px"
    sys.exit(
        "[ERROR] CSV must contain (world_x, world_y), (foot_x, foot_y), or (cx, cy). "
        "Check that the correct trajectory CSV is being passed."
    )


# ---------------------------------------------------------------------------
# Step vector computation
# ---------------------------------------------------------------------------

def compute_step_vectors(df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    """
    For each consecutive pair of observations within a track compute:
      dx, dy          — displacement in coordinate units
      dt              — elapsed seconds
      speed           — displacement / dt
      ux, uy          — unit direction vector (NaN if stationary)
      mid_x, mid_y    — midpoint of the step (used for cell assignment)

    Steps with dt <= 0 are dropped.
    """
    df = df.sort_values(["track_id", "frame"]).copy()
    g = df.groupby("track_id", sort=False)

    # Shift within each track to get previous observation
    df["_x0"]  = g[xcol].shift(1)
    df["_y0"]  = g[ycol].shift(1)
    df["_t0"]  = g["time_s"].shift(1)

    # Drop rows without a previous observation (first row of each track)
    steps = df.dropna(subset=["_x0", "_y0", "_t0"]).copy()

    steps["dx"] = steps[xcol]   - steps["_x0"]
    steps["dy"] = steps[ycol]   - steps["_y0"]
    steps["dt"] = steps["time_s"] - steps["_t0"]

    # Drop invalid steps (dt == 0 can happen with vid_stride or duplicate frames)
    steps = steps[steps["dt"] > 0].copy()

    dist = np.sqrt(steps["dx"] ** 2 + steps["dy"] ** 2)
    steps["speed"] = dist / steps["dt"]

    # Unit vector — NaN when the person did not move (dist == 0)
    safe_dist = dist.where(dist > 0)
    steps["ux"] = steps["dx"] / safe_dist
    steps["uy"] = steps["dy"] / safe_dist

    # Midpoint of the step → used to assign step to a grid cell
    steps["mid_x"] = (steps["_x0"] + steps[xcol]) / 2.0
    steps["mid_y"] = (steps["_y0"] + steps[ycol])  / 2.0

    keep = ["track_id", "frame", "time_s",
            "dx", "dy", "dt", "speed", "ux", "uy",
            "mid_x", "mid_y"]
    return steps[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Grid cell assignment and aggregation
# ---------------------------------------------------------------------------

def assign_cells(steps: pd.DataFrame, cell_size: float) -> pd.DataFrame:
    """Tag each step with the grid cell that its midpoint falls into."""
    steps = steps.copy()
    steps["cell_col"] = np.floor(steps["mid_x"] / cell_size).astype(int)
    steps["cell_row"] = np.floor(steps["mid_y"] / cell_size).astype(int)
    # Cell centre coordinates (for plotting)
    steps["cell_x"] = (steps["cell_col"] + 0.5) * cell_size
    steps["cell_y"] = (steps["cell_row"] + 0.5) * cell_size
    return steps


def aggregate_cells(steps: pd.DataFrame, min_vectors: int) -> pd.DataFrame:
    """
    Aggregate per cell:
      n_vectors           — number of steps that fall in this cell
      mean_dx, mean_dy    — average displacement vector
      mean_speed          — average scalar speed
      direction_consistency — magnitude of the mean unit vector
                             (1.0 = all steps point the same way,
                              0.0 = steps point in random directions)
    """
    # Main aggregation
    agg = (
        steps.groupby(["cell_col", "cell_row", "cell_x", "cell_y"], sort=False)
        .agg(
            n_vectors  = ("speed", "count"),
            mean_dx    = ("dx",    "mean"),
            mean_dy    = ("dy",    "mean"),
            mean_speed = ("speed", "mean"),
            mean_ux    = ("ux",    "mean"),   # mean of unit-x components
            mean_uy    = ("uy",    "mean"),   # mean of unit-y components
        )
        .reset_index()
    )

    # direction_consistency = |mean unit vector|
    agg["direction_consistency"] = np.sqrt(agg["mean_ux"] ** 2 + agg["mean_uy"] ** 2)
    agg.drop(columns=["mean_ux", "mean_uy"], inplace=True)

    # Apply minimum-vector threshold
    agg = agg[agg["n_vectors"] >= min_vectors].copy()

    return agg.sort_values(["cell_row", "cell_col"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quiver plot
# ---------------------------------------------------------------------------

def _render_top_view_bg(ax, top_view_image: str, alpha: float,
                         calib_json: str = ""):
    """
    Warp the top-view/plan image into world space using the calibration
    correspondences (plan_points_px -> world_points) and render as background.

    Unlike the origin+scale approach this correctly handles rotation, skew,
    and any perspective distortion present in the plan image.

    Returns (xmin, xmax, ymin, ymax) in world coords, or None on failure.
    """
    if not top_view_image or not calib_json:
        return None
    img_path   = Path(top_view_image)
    calib_path = Path(calib_json)
    if not img_path.exists() or not calib_path.exists():
        return None
    try:
        import cv2 as _cv2
        import json as _json

        with open(calib_path, encoding="utf-8") as f:
            calib = _json.load(f)

        plan_pts  = np.array(calib.get("plan_points_px", []), dtype=np.float32)
        world_pts = np.array(calib.get("world_points",   []), dtype=np.float32)
        if len(plan_pts) < 4:
            print("[WARN] calib_json needs >= 4 plan_points_px for background warp.")
            return None

        H_pw, _ = _cv2.findHomography(plan_pts, world_pts, _cv2.RANSAC, 3.0)
        if H_pw is None:
            print("[WARN] Could not compute plan->world homography for background.")
            return None

        img_bgr = _cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] Cannot read top_view_image: {img_path}")
            return None
        h_img, w_img = img_bgr.shape[:2]

        # World-space bounding box from all four image corners
        corners = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]],
                           dtype=np.float32)
        cw   = _cv2.perspectiveTransform(corners.reshape(-1, 1, 2),
                                          H_pw).reshape(-1, 2)
        xmin = float(cw[:, 0].min());  xmax = float(cw[:, 0].max())
        ymin = float(cw[:, 1].min());  ymax = float(cw[:, 1].max())

        # Output canvas resolution: match original image pixel density
        world_diag = float(np.hypot(xmax - xmin, ymax - ymin))
        img_diag   = float(np.hypot(w_img, h_img))
        res = min(img_diag / max(world_diag, 1e-6), 400.0)  # px per world unit

        out_w = max(int((xmax - xmin) * res), 2)
        out_h = max(int((ymax - ymin) * res), 2)

        # T: world coords -> canvas pixels
        T = np.array([[res, 0,   -xmin * res],
                      [0,   res, -ymin * res],
                      [0,   0,    1          ]], dtype=np.float64)

        # Combined: plan pixels -> world -> canvas pixels
        H_total    = T @ H_pw.astype(np.float64)
        warped_bgr = _cv2.warpPerspective(img_bgr, H_total, (out_w, out_h),
                                          flags=_cv2.INTER_LINEAR,
                                          borderMode=_cv2.BORDER_CONSTANT,
                                          borderValue=(255, 255, 255))
        warped_rgb = _cv2.cvtColor(warped_bgr, _cv2.COLOR_BGR2RGB)

        # imshow extent=[left, right, bottom, top] with origin='upper':
        # canvas row 0 -> world_y=ymin; last row -> world_y=ymax.
        # Setting top=ymin, bottom=ymax places row 0 at the lower y position.
        ax.imshow(warped_rgb, extent=[xmin, xmax, ymax, ymin],
                  aspect="auto", alpha=float(alpha), zorder=0)

        return xmin, xmax, ymin, ymax

    except Exception as e:
        print(f"[WARN] Could not warp top-view background: {e}")
        return None


def save_quiver_plot(cells: pd.DataFrame, unit: str, out_path: Path,
                     top_view_image: str = "", top_view_alpha: float = 0.25,
                     calib_json: str = "",
                     zoom_mode: str = "auto", zoom_padding: float = 0.08,
                     quiver_scale: float = 1.5, quiver_width: float = 0.006):
    """
    Clean quiver diagram:
      - Arrow direction  = mean displacement direction (mean_dx, mean_dy)
      - Arrow length     = proportional to mean_speed (normalised to cell_size)
      - Arrow colour     = direction_consistency (viridis: purple=random, yellow=uniform)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if cells.empty:
        ax.text(0.5, 0.5,
                "No cells met the minimum vector threshold.\nTry lowering --min_vectors_per_cell.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    x    = cells["cell_x"].to_numpy()
    y    = cells["cell_y"].to_numpy()
    dx   = cells["mean_dx"].to_numpy()
    dy   = cells["mean_dy"].to_numpy()
    cons = cells["direction_consistency"].to_numpy()

    # Normalise arrow length: scale each arrow so that max-speed = 0.8 * cell_size,
    # then apply quiver_scale to boost overall visibility.
    speed     = cells["mean_speed"].to_numpy()
    cell_size = float(cells["cell_x"].diff().abs().replace(0, np.nan).dropna().min()
                      if len(cells) > 1 else 1.0)
    speed_max = speed.max() if speed.max() > 0 else 1.0
    norm_factor = speed / speed_max          # [0, 1]
    arrow_len   = norm_factor * cell_size * 0.8 * quiver_scale

    # Direction from (mean_dx, mean_dy), magnitude from arrow_len
    mag = np.sqrt(dx ** 2 + dy ** 2)
    safe_mag = np.where(mag > 0, mag, np.nan)
    ux = dx / safe_mag
    uy = dy / safe_mag
    adx = ux * arrow_len
    ady = uy * arrow_len

    # Colour by direction_consistency
    norm  = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cmap  = plt.get_cmap("viridis")
    colors = cmap(norm(cons))

    ax.quiver(
        x, y, adx, ady,
        color=colors,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=quiver_width,
        headwidth=4,
        headlength=5,
        headaxislength=4,
        alpha=0.9,
    )

    # Scatter dots at cell centres (shows coverage even for near-zero-speed cells)
    ax.scatter(x, y, s=12, color="#555555", zorder=2, linewidths=0)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax,
                 label="Direction consistency  (0 = random,  1 = uniform)",
                 fraction=0.03, pad=0.02)

    # Background image — returns world extent of full plan image, or None
    img_extent = _render_top_view_bg(ax, top_view_image, top_view_alpha, calib_json)

    # Compute axis bounds — union of image world extent and data bounds
    eps = 1e-6
    pad = zoom_padding
    dxmin, dxmax = float(x.min()), float(x.max())
    dymin, dymax = float(y.min()), float(y.max())
    if img_extent is not None:
        ixmin, ixmax, iymin, iymax = img_extent
        x_lo = min(ixmin, dxmin)
        x_hi = max(ixmax, dxmax)
        y_lo = min(iymin, dymin)
        y_hi = max(iymax, dymax)
    else:
        x_lo, x_hi, y_lo, y_hi = dxmin, dxmax, dymin, dymax
    x_rng = x_hi - x_lo + eps
    y_rng = y_hi - y_lo + eps
    ax.set_xlim(x_lo - pad * x_rng, x_hi + pad * x_rng)
    ax.set_ylim(y_lo - pad * y_rng, y_hi + pad * y_rng)

    unit_label = "m" if unit == "m" else "px"
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"X ({unit_label})", fontsize=9)
    ax.set_ylabel(f"Y ({unit_label})", fontsize=9)
    ax.set_title(
        f"Pedestrian flow field  |  {len(cells)} cells  |  arrow length ∝ speed",
        fontsize=10, pad=10
    )
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.3, color="#eeeeee")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#cccccc")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[INFO] Quiver plot saved: {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compute spatial flow field from pedestrian trajectories."
    )
    ap.add_argument("--traj_csv",             required=True,
                    help="Trajectory CSV (from track_people.py or calibrate_homography.py)")
    ap.add_argument("--out_dir",              default="outputs/flow",
                    help="Output directory for CSV and PNG files")
    ap.add_argument("--cell_size",            type=float, default=1.0,
                    help="Grid cell size in coordinate units (metres if world coords, pixels otherwise)")
    ap.add_argument("--min_vectors_per_cell", type=int,   default=3,
                    help="Minimum number of step vectors a cell must contain to appear in output")
    ap.add_argument("--top_view_image",      default="",
                    help="Optional top-view/plan image to render as plot background")
    ap.add_argument("--top_view_alpha",      type=float, default=0.25,
                    help="Opacity of the top-view background image (default: 0.25)")
    ap.add_argument("--calib_json",          default="",
                    help="calib.json — required for correct image alignment")
    ap.add_argument("--zoom_mode",           default="auto", choices=["auto", "full"],
                    help="'auto' = zoom to data extent + padding; 'full' = full calibrated image extent")
    ap.add_argument("--zoom_padding",        type=float, default=0.08,
                    help="Fractional padding around data in auto zoom mode (default: 0.08)")
    ap.add_argument("--quiver_scale",        type=float, default=1.5,
                    help="Arrow length multiplier — larger = longer arrows (default: 1.5)")
    ap.add_argument("--quiver_width",        type=float, default=0.006,
                    help="Arrow shaft width as fraction of axes width (default: 0.006)")
    args = ap.parse_args()

    # --- Load & validate ---
    csv_path = Path(args.traj_csv)
    if not csv_path.exists():
        sys.exit(f"[ERROR] traj_csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        sys.exit("[ERROR] CSV is empty.")

    for col in ("frame", "time_s", "track_id"):
        if col not in df.columns:
            sys.exit(f"[ERROR] Required column missing: '{col}'")

    xcol, ycol, unit = resolve_coord_cols(df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] {df['track_id'].nunique()} tracks, {len(df)} observations")

    # --- Compute ---
    steps = compute_step_vectors(df, xcol, ycol)
    print(f"[INFO] {len(steps)} valid steps (dt > 0)")

    if steps.empty:
        sys.exit("[ERROR] No valid steps found. Check that the CSV has more than one "
                 "observation per track and that time_s values are increasing.")

    steps = assign_cells(steps, args.cell_size)
    cells = aggregate_cells(steps, args.min_vectors_per_cell)
    print(f"[INFO] {len(cells)} cells (min_vectors_per_cell={args.min_vectors_per_cell})")
    if not cells.empty:
        print(f"[INFO] World extent: "
              f"X=[{cells['cell_x'].min():.2f}, {cells['cell_x'].max():.2f}] {unit}, "
              f"Y=[{cells['cell_y'].min():.2f}, {cells['cell_y'].max():.2f}] {unit}")

    if cells.empty:
        print("[WARN] No cells passed the threshold — try lowering --min_vectors_per_cell.")

    # --- Save ---
    vectors_path = out_dir / "flow_field_vectors.csv"
    cells_path   = out_dir / "flow_field_cells.csv"
    quiver_path  = out_dir / "flow_field_quiver.png"

    steps.to_csv(vectors_path, index=False)
    cells.to_csv(cells_path,   index=False)
    save_quiver_plot(cells, unit, quiver_path,
                     top_view_image=args.top_view_image,
                     top_view_alpha=args.top_view_alpha,
                     calib_json=args.calib_json,
                     zoom_mode=args.zoom_mode,
                     zoom_padding=args.zoom_padding,
                     quiver_scale=args.quiver_scale,
                     quiver_width=args.quiver_width)

    print("[INFO] Saved:")
    print(" -", vectors_path)
    print(" -", cells_path)
    print(" -", quiver_path)


if __name__ == "__main__":
    main()
