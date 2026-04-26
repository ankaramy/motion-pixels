"""
compute_linger_zones.py
-----------------------
Detect and summarise linger zones from tracked pedestrian trajectories.

Pipeline:
  1. Compute per-step speed from consecutive detections within each track.
  2. Label observations as stops (speed <= stop_speed_threshold).
  3. Merge consecutive stop observations per track into dwell events.
  4. Classify each dwell event by duration: brief / pause / linger / long_wait.
  5. Cluster dwell event positions with DBSCAN → linger zones.
  6. Aggregate per-zone statistics and save outputs.

Outputs:
  dwell_events_enriched.csv   — one row per dwell event with class and zone_id
  linger_zones.csv            — one row per DBSCAN cluster
  linger_zones_plot.png       — thesis-ready diagram

Usage:
    python compute_linger_zones.py \
        --traj_csv  outputs/trajectories_world.csv \
        --out_dir   outputs/linger \
        --stop_speed_threshold 0.3 \
        --pause_s 1.0 \
        --linger_s 5.0 \
        --long_wait_s 15.0 \
        --cluster_eps 1.5 \
        --cluster_min_samples 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import DBSCAN


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Class labels in ascending duration order
CLASS_ORDER  = ["brief", "pause", "linger", "long_wait"]
CLASS_COLORS = {
    "brief":     "#5B8FF9",   # muted blue  — very short pause, possible noise
    "pause":     "#f4a261",   # orange      — noticeable stop
    "linger":    "#e76f51",   # red-orange  — sustained stop
    "long_wait": "#9b2226",   # dark red    — long occupation of space
}


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
        "[ERROR] CSV must contain (world_x, world_y), (foot_x, foot_y), or (cx, cy)."
    )


# ---------------------------------------------------------------------------
# Speed computation
# ---------------------------------------------------------------------------

def attach_speed(df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    """
    Compute per-step speed and assign it to each observation.
    Speed is the displacement from the previous observation divided by dt.
    First observation in each track receives NaN.
    """
    df = df.sort_values(["track_id", "frame"]).copy()
    grp = df.groupby("track_id", sort=False)

    prev_x = grp[xcol].shift(1)
    prev_y = grp[ycol].shift(1)
    prev_t = grp["time_s"].shift(1)

    dx   = df[xcol]     - prev_x
    dy   = df[ycol]     - prev_y
    dt   = df["time_s"] - prev_t
    dist = np.sqrt(dx ** 2 + dy ** 2)

    df["_speed"] = np.where(dt > 0, dist / dt, np.nan)
    return df


# ---------------------------------------------------------------------------
# Dwell event extraction
# ---------------------------------------------------------------------------

def classify_duration(duration_s: float,
                       pause_s: float, linger_s: float, long_wait_s: float) -> str:
    """
    Four-class duration label:
      brief     — duration <  pause_s      (short pause, may be noise)
      pause     — pause_s  <= duration <  linger_s
      linger    — linger_s <= duration <  long_wait_s
      long_wait — duration >= long_wait_s
    """
    if duration_s >= long_wait_s:
        return "long_wait"
    if duration_s >= linger_s:
        return "linger"
    if duration_s >= pause_s:
        return "pause"
    return "brief"


def extract_dwell_events(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    stop_speed: float,
    pause_s: float,
    linger_s: float,
    long_wait_s: float,
) -> pd.DataFrame:
    """
    For each track merge consecutive stop-labelled observations into dwell events.
    A stop observation is one where _speed <= stop_speed and speed is not NaN.
    (First observation of each track, which has NaN speed, is never a stop.)

    Returns a DataFrame with one row per dwell event.
    """
    df = df.copy()
    df["_is_stop"] = (df["_speed"] <= stop_speed) & df["_speed"].notna()

    events = []

    for tid, g in df.groupby("track_id", sort=False):
        g = g.sort_values("frame").copy()
        if g.empty:
            continue

        is_stop = g["_is_stop"].to_numpy().astype(int)
        idx     = np.arange(len(g))

        # Split on every change between stop / not-stop
        boundaries = np.where(np.diff(is_stop) != 0)[0] + 1
        segments   = np.split(idx, boundaries)

        for seg in segments:
            if len(seg) == 0 or is_stop[seg[0]] == 0:
                continue

            seg_df  = g.iloc[seg]
            dur     = float(seg_df["time_s"].iloc[-1] - seg_df["time_s"].iloc[0])

            events.append({
                "track_id":    int(tid),
                "start_frame": int(seg_df["frame"].iloc[0]),
                "end_frame":   int(seg_df["frame"].iloc[-1]),
                "start_time_s": float(seg_df["time_s"].iloc[0]),
                "end_time_s":   float(seg_df["time_s"].iloc[-1]),
                "duration_s":  dur,
                "x":           float(seg_df[xcol].mean()),
                "y":           float(seg_df[ycol].mean()),
                "dwell_class": classify_duration(dur, pause_s, linger_s, long_wait_s),
            })

    return pd.DataFrame(events)


# ---------------------------------------------------------------------------
# DBSCAN clustering
# ---------------------------------------------------------------------------

def cluster_dwell_events(
    events: pd.DataFrame,
    eps: float,
    min_samples: int,
) -> pd.DataFrame:
    """
    Cluster dwell event (x, y) positions with DBSCAN.
    Adds a zone_id column:  -1 = noise (no zone), 0+ = zone index.
    """
    if events.empty:
        events = events.copy()
        events["zone_id"] = pd.array([], dtype=int)
        return events

    coords = events[["x", "y"]].to_numpy()
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)

    events = events.copy()
    events["zone_id"] = labels
    return events


def aggregate_zones(events: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-zone summary statistics.
    Noise events (zone_id == -1) are excluded.
    Zones are sorted by total_duration_s descending.
    """
    zoned = events[events["zone_id"] >= 0]
    if zoned.empty:
        return pd.DataFrame(columns=[
            "zone_id", "n_events", "unique_tracks",
            "mean_duration_s", "median_duration_s", "total_duration_s",
            "centroid_x", "centroid_y",
        ])

    zones = (
        zoned.groupby("zone_id")
        .agg(
            n_events        = ("track_id",   "count"),
            unique_tracks   = ("track_id",   "nunique"),
            mean_duration_s = ("duration_s", "mean"),
            median_duration_s = ("duration_s", "median"),
            total_duration_s  = ("duration_s", "sum"),
            centroid_x      = ("x", "mean"),
            centroid_y      = ("y", "mean"),
        )
        .reset_index()
        .sort_values("total_duration_s", ascending=False)
        .reset_index(drop=True)
    )
    return zones


# ---------------------------------------------------------------------------
# Plot
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


def save_plot(
    df: pd.DataFrame,
    events: pd.DataFrame,
    zones: pd.DataFrame,
    xcol: str,
    ycol: str,
    unit: str,
    out_path: Path,
    top_view_image: str = "",
    top_view_alpha: float = 0.25,
    calib_json: str = "",
    zoom_mode: str = "auto",
    zoom_padding: float = 0.08,
):
    """
    Thesis-ready diagram:
      - Faint grey lines: all pedestrian trajectories
      - Coloured dots: dwell events by class
      - Numbered circles: linger zone centroids
    """
    fig, ax = plt.subplots(figsize=(11, 11))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # --- Background trajectories ---
    for tid in df["track_id"].unique():
        t = df[df["track_id"] == tid].sort_values("frame")
        ax.plot(t[xcol], t[ycol], linewidth=0.5, alpha=0.25, color="#bbbbbb", zorder=1)

    # --- Dwell events coloured by class ---
    if not events.empty:
        for cls in CLASS_ORDER:
            subset = events[events["dwell_class"] == cls]
            if subset.empty:
                continue
            ax.scatter(
                subset["x"], subset["y"],
                s=30, color=CLASS_COLORS[cls], alpha=0.8,
                linewidths=0.3, edgecolors="white",
                label=cls, zorder=3,
            )

    # --- Linger zone centroids ---
    if not zones.empty:
        # Scale circle size by total_duration_s (visual weight = time spent)
        max_dur = zones["total_duration_s"].max()
        for _, row in zones.iterrows():
            radius_norm = 0.3 + 0.7 * (row["total_duration_s"] / max(max_dur, 1.0))
            # Draw a light filled circle at centroid
            circle = plt.Circle(
                (row["centroid_x"], row["centroid_y"]),
                radius=radius_norm,
                color="#2d3142", alpha=0.15, zorder=4,
            )
            ax.add_patch(circle)
            # Zone label
            ax.text(
                row["centroid_x"], row["centroid_y"],
                f"Z{int(row['zone_id'])}",
                ha="center", va="center",
                fontsize=7, fontweight="bold", color="#2d3142",
                zorder=5,
            )

    # --- Legend ---
    handles = [
        mpatches.Patch(color=CLASS_COLORS[c], label=c)
        for c in CLASS_ORDER
        if not events.empty and c in events["dwell_class"].values
    ]
    if handles:
        ax.legend(handles=handles, title="Dwell class",
                  fontsize=8, title_fontsize=8, frameon=False,
                  loc="upper right")

    # Background image — returns world extent of full plan image, or None
    img_extent = _render_top_view_bg(ax, top_view_image, top_view_alpha, calib_json)

    # Compute axis bounds
    all_x = np.concatenate([df[xcol].to_numpy(),
                             events["x"].to_numpy() if not events.empty else np.array([])])
    all_y = np.concatenate([df[ycol].to_numpy(),
                             events["y"].to_numpy() if not events.empty else np.array([])])

    # Compute axis bounds — union of image world extent and data bounds
    eps = 1e-6
    pad = zoom_padding
    if len(all_x) > 0:
        dxmin, dxmax = float(all_x.min()), float(all_x.max())
        dymin, dymax = float(all_y.min()), float(all_y.max())
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
    elif img_extent is not None:
        ixmin, ixmax, iymin, iymax = img_extent
        ax.set_xlim(ixmin, ixmax)
        ax.set_ylim(iymin, iymax)

    unit_label = "m" if unit == "m" else "px"
    n_zones  = len(zones) if not zones.empty else 0
    n_events = len(events) if not events.empty else 0

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"X ({unit_label})", fontsize=9)
    ax.set_ylabel(f"Y ({unit_label})", fontsize=9)
    ax.set_title(
        f"Linger zones  |  {n_events} dwell events  |  {n_zones} zones  "
        f"(Z = DBSCAN cluster)",
        fontsize=10, pad=10,
    )
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#cccccc")
    ax.grid(True, linewidth=0.3, color="#eeeeee")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[INFO] Plot saved: {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Detect linger zones from pedestrian trajectory data."
    )
    ap.add_argument("--traj_csv",             required=True,
                    help="Trajectory CSV (from track_people.py or calibrate_homography.py)")
    ap.add_argument("--out_dir",              default="outputs/linger",
                    help="Output directory")
    ap.add_argument("--stop_speed_threshold", type=float, default=0.3,
                    help="Speed at or below which an observation is a stop "
                         "(m/s for world coords, px/s for image coords)")
    ap.add_argument("--pause_s",              type=float, default=1.0,
                    help="Minimum duration (s) to classify a dwell event as 'pause'")
    ap.add_argument("--linger_s",             type=float, default=5.0,
                    help="Minimum duration (s) to classify as 'linger'")
    ap.add_argument("--long_wait_s",          type=float, default=15.0,
                    help="Minimum duration (s) to classify as 'long_wait'")
    ap.add_argument("--cluster_eps",          type=float, default=1.5,
                    help="DBSCAN eps: max distance between events in the same zone "
                         "(same units as coordinates)")
    ap.add_argument("--cluster_min_samples",  type=int,   default=3,
                    help="DBSCAN min_samples: minimum events to form a zone")
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
    print(f"[INFO] World extent: "
          f"X=[{df[xcol].min():.2f}, {df[xcol].max():.2f}] {unit}, "
          f"Y=[{df[ycol].min():.2f}, {df[ycol].max():.2f}] {unit}")

    # --- Speed ---
    df = attach_speed(df, xcol, ycol)

    # --- Dwell events ---
    events = extract_dwell_events(
        df, xcol, ycol,
        stop_speed=args.stop_speed_threshold,
        pause_s=args.pause_s,
        linger_s=args.linger_s,
        long_wait_s=args.long_wait_s,
    )
    print(f"[INFO] {len(events)} dwell events extracted")

    if events.empty:
        print("[WARN] No stop observations found. Try increasing --stop_speed_threshold.")

    # --- Cluster ---
    events = cluster_dwell_events(events, eps=args.cluster_eps,
                                  min_samples=args.cluster_min_samples)

    n_zones = int((events["zone_id"] >= 0).sum()) if not events.empty else 0
    print(f"[INFO] {events['zone_id'].nunique() - (1 if -1 in events['zone_id'].values else 0)} "
          f"linger zones detected  ({(events['zone_id'] == -1).sum()} noise events)")

    # --- Zone aggregation ---
    zones = aggregate_zones(events)

    # --- Save ---
    events_path = out_dir / "dwell_events_enriched.csv"
    zones_path  = out_dir / "linger_zones.csv"
    plot_path   = out_dir / "linger_zones_plot.png"

    events.to_csv(events_path, index=False)
    zones.to_csv(zones_path,   index=False)
    save_plot(df, events, zones, xcol, ycol, unit, plot_path,
              top_view_image=args.top_view_image,
              top_view_alpha=args.top_view_alpha,
              calib_json=args.calib_json,
              zoom_mode=args.zoom_mode,
              zoom_padding=args.zoom_padding)

    print("[INFO] Saved:")
    print(" -", events_path)
    print(" -", zones_path)
    print(" -", plot_path)

    # Terminal summary
    if not zones.empty:
        print(f"\nLinger zones summary (top {min(10, len(zones))}):")
        cols = ["zone_id", "n_events", "unique_tracks",
                "median_duration_s", "total_duration_s", "centroid_x", "centroid_y"]
        print(zones[cols].head(10).to_string(index=False, float_format="{:.2f}".format))


if __name__ == "__main__":
    main()
