"""
compute_bottlenecks.py
----------------------
Detect congestion and bottleneck zones from pedestrian trajectory data.

For each spatial grid cell the script computes:
  - how many observations fell there (density proxy)
  - how many unique tracks passed through
  - mean and median speed
  - fraction of observations classified as stops
  - a bottleneck score (see formula below)

Bottleneck score formula
------------------------
  density_score  = log1p(observation_count) / log1p(global_max_obs_count)   ∈ [0, 1]
  slowness_score = 1 - (median_speed / global_max_median_speed)              ∈ [0, 1]
  bottleneck_score = density_score × slowness_score × (1 + stop_fraction)

Rationale:
  • density_score   — cells with more activity score higher (log-scale dampens outliers)
  • slowness_score  — slower cells score higher; cells at global max speed score 0
  • (1 + stop_fraction) — multiplier 1.0→2.0; cells where people fully stop are doubled

Usage:
    python compute_bottlenecks.py \
        --traj_csv outputs/trajectories_world.csv \
        --out_dir  outputs/bottlenecks \
        --cell_size 1.0 \
        --stop_speed_threshold 0.3 \
        --min_obs_per_cell 5 \
        --top_k 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Coordinate resolution  (shared pattern across pipeline)
# ---------------------------------------------------------------------------

def resolve_coord_cols(df: pd.DataFrame):
    """
    Return (xcol, ycol, unit) using column priority:
      1) world_x / world_y  → 'm'
      2) foot_x  / foot_y   → 'px'
      3) cx      / cy        → 'px'
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
        "Check that the correct trajectory CSV is passed."
    )


# ---------------------------------------------------------------------------
# Per-observation speed (computed internally from consecutive detections)
# ---------------------------------------------------------------------------

def attach_speed(df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    """
    Compute per-step speed from consecutive observations within each track.
    Speed is assigned to the later of the two observations.
    First observation in each track receives NaN speed.
    """
    df = df.sort_values(["track_id", "frame"]).copy()
    grp = df.groupby("track_id", sort=False)

    prev_x  = grp[xcol].shift(1)
    prev_y  = grp[ycol].shift(1)
    prev_t  = grp["time_s"].shift(1)

    dx   = df[xcol]    - prev_x
    dy   = df[ycol]    - prev_y
    dt   = df["time_s"] - prev_t
    dist = np.sqrt(dx ** 2 + dy ** 2)

    # speed is NaN where dt <= 0 or no previous observation
    df["_speed"] = np.where(dt > 0, dist / dt, np.nan)
    return df


# ---------------------------------------------------------------------------
# Cell assignment
# ---------------------------------------------------------------------------

def assign_cells(df: pd.DataFrame, cell_size: float) -> pd.DataFrame:
    """Assign each observation to a grid cell using floor division."""
    df = df.copy()
    df["cell_col"] = np.floor(df["_x"] / cell_size).astype(int)
    df["cell_row"] = np.floor(df["_y"] / cell_size).astype(int)
    # Cell centre in coordinate units (for plotting)
    df["cell_x"]   = (df["cell_col"] + 0.5) * cell_size
    df["cell_y"]   = (df["cell_row"] + 0.5) * cell_size
    return df


# ---------------------------------------------------------------------------
# Per-cell aggregation
# ---------------------------------------------------------------------------

def aggregate_cells(df: pd.DataFrame, stop_speed: float, min_obs: int) -> pd.DataFrame:
    """
    Compute per-cell metrics.
    stop_fraction = share of observations where speed <= stop_speed_threshold.
    Observations with NaN speed (first in track) are excluded from speed stats
    but still counted in observation_count.
    """
    # is_stop: True if speed known and below threshold
    df["_is_stop"] = (df["_speed"] <= stop_speed) & df["_speed"].notna()

    def cell_stats(g):
        speeds = g["_speed"].dropna()
        return pd.Series({
            "observation_count": len(g),
            "unique_track_count": g["track_id"].nunique(),
            "mean_speed":   float(speeds.mean())   if len(speeds) else np.nan,
            "median_speed": float(speeds.median())  if len(speeds) else np.nan,
            "stop_fraction": float(g["_is_stop"].mean()),
        })

    cells = (
        df.groupby(["cell_col", "cell_row", "cell_x", "cell_y"], sort=False)
        .apply(cell_stats)
        .reset_index()
    )

    # Apply minimum observation threshold
    cells = cells[cells["observation_count"] >= min_obs].copy()
    return cells.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bottleneck score
# ---------------------------------------------------------------------------

def compute_bottleneck_scores(cells: pd.DataFrame) -> pd.DataFrame:
    """
    bottleneck_score = density_score × slowness_score × (1 + stop_fraction)

    density_score  = log1p(obs_count) / log1p(max_obs_count)    ∈ [0, 1]
    slowness_score = 1 - (median_speed / max_median_speed)       ∈ [0, 1]

    Cells with NaN median_speed (no valid speed observations) receive
    slowness_score = 0 so they do not falsely rank as high-severity.
    """
    cells = cells.copy()

    max_obs   = cells["observation_count"].max()
    cells["density_score"] = np.log1p(cells["observation_count"]) / np.log1p(max_obs)

    max_speed = cells["median_speed"].max()
    if pd.isna(max_speed) or max_speed == 0:
        cells["slowness_score"] = 0.0
    else:
        cells["slowness_score"] = 1.0 - (cells["median_speed"].fillna(max_speed) / max_speed)

    cells["bottleneck_score"] = (
        cells["density_score"]
        * cells["slowness_score"]
        * (1.0 + cells["stop_fraction"])
    )

    return cells.sort_values("bottleneck_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Heatmap plot
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


def save_heatmap(cells: pd.DataFrame, unit: str, out_path: Path, top_k: int,
                 top_view_image: str = "", top_view_alpha: float = 0.25,
                 calib_json: str = "",
                 zoom_mode: str = "auto", zoom_padding: float = 0.08):
    """
    Static PNG heatmap: cell colour encodes bottleneck_score.
    Top-k cells are annotated with their rank.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if cells.empty:
        ax.text(0.5, 0.5,
                "No cells met the minimum observation threshold.\n"
                "Try lowering --min_obs_per_cell.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    score = cells["bottleneck_score"].to_numpy()
    norm  = mcolors.Normalize(vmin=0.0, vmax=score.max() if score.max() > 0 else 1.0)
    cmap  = plt.get_cmap("YlOrRd")

    # Infer cell size from minimum nonzero spacing between cell centres
    all_x = np.sort(cells["cell_x"].unique())
    if len(all_x) > 1:
        cell_size_est = float(np.diff(all_x).min())
    else:
        cell_size_est = float(cells["cell_x"].iloc[0]) if len(cells) else 1.0
    half = cell_size_est / 2.0

    # Draw Rectangle patches positioned in data coordinates — geometrically
    # correct regardless of figure size, DPI, or aspect ratio adjustments.
    for _, row in cells.iterrows():
        color = cmap(norm(row["bottleneck_score"]))
        rect = plt.Rectangle(
            (row["cell_x"] - half, row["cell_y"] - half),
            cell_size_est, cell_size_est,
            facecolor=color, edgecolor="none", alpha=0.9, zorder=3,
        )
        ax.add_patch(rect)

    # Colorbar via ScalarMappable (no scatter object to attach to)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Bottleneck score", fraction=0.03, pad=0.02)

    # Annotate top-k cells with their rank
    top = cells.head(min(top_k, len(cells)))
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        ax.text(row["cell_x"], row["cell_y"], str(rank),
                ha="center", va="center", fontsize=6,
                color="white", fontweight="bold", zorder=5)

    # Background image — returns world extent of full plan image, or None
    img_extent = _render_top_view_bg(ax, top_view_image, top_view_alpha, calib_json)

    # Compute axis bounds — union of image world extent and data bounds
    eps = 1e-6
    pad = zoom_padding
    dxmin = cells["cell_x"].min() - half
    dxmax = cells["cell_x"].max() + half
    dymin = cells["cell_y"].min() - half
    dymax = cells["cell_y"].max() + half
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
        f"Bottleneck heatmap  |  {len(cells)} cells  |  top {min(top_k, len(cells))} ranked",
        fontsize=10, pad=10
    )
    ax.tick_params(labelsize=7)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#cccccc")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[INFO] Heatmap saved: {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Detect congestion and bottleneck zones from pedestrian trajectories."
    )
    ap.add_argument("--traj_csv",             required=True,
                    help="Trajectory CSV (from track_people.py or calibrate_homography.py)")
    ap.add_argument("--out_dir",              default="outputs/bottlenecks",
                    help="Directory for output CSV and PNG files")
    ap.add_argument("--cell_size",            type=float, default=1.0,
                    help="Grid cell size in coordinate units (metres or pixels)")
    ap.add_argument("--stop_speed_threshold", type=float, default=0.3,
                    help="Speed at or below which an observation is classified as a stop "
                         "(metres/s for world coords, px/s for image coords)")
    ap.add_argument("--min_obs_per_cell",     type=int,   default=5,
                    help="Minimum observations a cell must have to appear in output")
    ap.add_argument("--top_k",               type=int,   default=10,
                    help="Number of top-scoring bottleneck cells to report separately")
    ap.add_argument("--top_view_image",      default="",
                    help="Optional top-view/plan image to render as plot background")
    ap.add_argument("--top_view_alpha",      type=float, default=0.25,
                    help="Opacity of the top-view background image (default: 0.25)")
    ap.add_argument("--calib_json",          default="",
                    help="calib.json from calibrate_homography_interactive.py — required for correct image alignment")
    ap.add_argument("--zoom_mode",           default="auto", choices=["auto", "full"],
                    help="'auto' = zoom to data extent + padding; 'full' = full calibrated image extent")
    ap.add_argument("--zoom_padding",        type=float, default=0.08,
                    help="Fractional padding around data in auto zoom mode (default: 0.08)")
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

    n_tracks = df["track_id"].nunique()
    print(f"[INFO] {n_tracks} tracks, {len(df)} observations")

    # --- Attach speed, alias coordinate columns for internal use ---
    df = attach_speed(df, xcol, ycol)
    df["_x"] = df[xcol]
    df["_y"] = df[ycol]

    # --- Grid assignment ---
    df = assign_cells(df, args.cell_size)

    # --- Aggregate ---
    cells = aggregate_cells(df, args.stop_speed_threshold, args.min_obs_per_cell)
    if cells.empty:
        print("[WARN] No cells passed the minimum observation threshold. "
              "Try lowering --min_obs_per_cell or --cell_size.")

    # --- Score ---
    cells = compute_bottleneck_scores(cells)

    # --- Save ---
    cells_path     = out_dir / "bottleneck_cells.csv"
    top_cells_path = out_dir / "bottleneck_top_cells.csv"
    heatmap_path   = out_dir / "bottleneck_heatmap.png"

    cells.to_csv(cells_path, index=False)
    cells.head(args.top_k).to_csv(top_cells_path, index=False)
    save_heatmap(cells, unit, heatmap_path, top_k=args.top_k,
                 top_view_image=args.top_view_image,
                 top_view_alpha=args.top_view_alpha,
                 calib_json=args.calib_json,
                 zoom_mode=args.zoom_mode,
                 zoom_padding=args.zoom_padding)

    print(f"[INFO] {len(cells)} cells scored  |  top {min(args.top_k, len(cells))} saved separately")
    if not cells.empty:
        print(f"[INFO] World extent: "
              f"X=[{cells['cell_x'].min():.2f}, {cells['cell_x'].max():.2f}] {unit}, "
              f"Y=[{cells['cell_y'].min():.2f}, {cells['cell_y'].max():.2f}] {unit}")
    print("[INFO] Saved:")
    print(" -", cells_path)
    print(" -", top_cells_path)
    print(" -", heatmap_path)

    # Print top-k summary to terminal
    if not cells.empty:
        top = cells.head(args.top_k)[
            ["cell_x", "cell_y", "observation_count", "unique_track_count",
             "median_speed", "stop_fraction", "bottleneck_score"]
        ]
        print(f"\nTop {min(args.top_k, len(cells))} bottleneck cells:")
        print(top.to_string(index=True, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
