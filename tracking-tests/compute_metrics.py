import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_speed(df: pd.DataFrame, xcol: str, ycol: str, unit: str) -> pd.DataFrame:
    """
    Adds per-observation speed and step distance using consecutive detections per track.
    Output column names are unit-aware: speed_px_s / speed_m_s, step_dist_px / step_dist_m.
    """
    df = df.sort_values(["track_id", "frame"]).copy()

    df["_prev_x"] = df.groupby("track_id")[xcol].shift(1)
    df["_prev_y"] = df.groupby("track_id")[ycol].shift(1)
    df["prev_time_s"] = df.groupby("track_id")["time_s"].shift(1)

    dx = df[xcol] - df["_prev_x"]
    dy = df[ycol] - df["_prev_y"]
    dt = df["time_s"] - df["prev_time_s"]

    dist = np.sqrt(dx * dx + dy * dy)

    speed_col = f"speed_{unit}_s"
    dist_col  = f"step_dist_{unit}"

    df[speed_col] = np.where((dt > 0) & np.isfinite(dist), dist / dt, np.nan)
    df[dist_col]  = dist

    df.drop(columns=["_prev_x", "_prev_y", "prev_time_s"], inplace=True)
    return df


def rolling_median_speed(df: pd.DataFrame, speed_col: str, window: int) -> pd.Series:
    """Median filter per track to reduce jitter. window is in observations."""
    if window <= 1:
        return df[speed_col]
    return (
        df.groupby("track_id")[speed_col]
        .transform(lambda s: s.rolling(window=window, min_periods=max(2, window // 2), center=True).median())
    )


def extract_dwell_events(
    df: pd.DataFrame,
    speed_smooth_col: str,
    stop_speed: float,
    min_dwell_s: float,
    xcol: str,
    ycol: str,
) -> pd.DataFrame:
    """
    One row per continuous stop segment:
    track_id, start_time_s, end_time_s, duration_s, x_mean, y_mean, n_obs
    """
    stop = (df[speed_smooth_col] <= stop_speed) & df[speed_smooth_col].notna()
    df = df.copy()
    df["is_stop"] = stop

    events = []
    for tid, g in df.groupby("track_id", sort=False):
        g = g.sort_values("frame").copy()
        if g.empty:
            continue

        stop_int = g["is_stop"].astype(int).to_numpy()
        idx = np.arange(len(g))
        boundaries = np.where(np.diff(stop_int) != 0)[0] + 1
        splits = np.split(idx, boundaries)

        for seg in splits:
            if len(seg) == 0:
                continue
            if stop_int[seg[0]] != 1:
                continue

            seg_g = g.iloc[seg]
            start_t  = float(seg_g["time_s"].iloc[0])
            end_t    = float(seg_g["time_s"].iloc[-1])
            duration = end_t - start_t

            if duration >= min_dwell_s:
                events.append(
                    {
                        "track_id":     int(tid),
                        "start_time_s": start_t,
                        "end_time_s":   end_t,
                        "duration_s":   float(duration),
                        "cx_mean":      float(seg_g["cx"].mean()),
                        "cy_mean":      float(seg_g["cy"].mean()),
                        "n_obs":        int(len(seg_g)),
                    }
                )

    return pd.DataFrame(events)


def compute_shift_flags(df: pd.DataFrame,
                        min_dist_px: float = 5.0,
                        angle_deg: float = 65.0) -> pd.Series:
    """
    Per-observation flag (is_shift=1) when a direction change exceeds angle_deg.
    Uses cx/cy (image space). Compares consecutive step vectors; both must have
    displacement >= min_dist_px. First two rows of each track are always 0.
    """
    df = df.sort_values(["track_id", "frame"]).copy()
    result = pd.Series(0, index=df.index)
    for tid, g in df.groupby("track_id", sort=False):
        g = g.sort_values("frame")
        xs  = g["cx"].to_numpy(dtype=float)
        ys  = g["cy"].to_numpy(dtype=float)
        idx = g.index.to_numpy()
        for i in range(2, len(g)):
            v1x, v1y = xs[i-1] - xs[i-2], ys[i-1] - ys[i-2]
            v2x, v2y = xs[i]   - xs[i-1], ys[i]   - ys[i-1]
            m1 = (v1x**2 + v1y**2) ** 0.5
            m2 = (v2x**2 + v2y**2) ** 0.5
            if m1 < min_dist_px or m2 < min_dist_px:
                continue
            cos_a = np.clip((v1x*v2x + v1y*v2y) / (m1 * m2), -1.0, 1.0)
            if np.degrees(np.arccos(cos_a)) > angle_deg:
                result[idx[i]] = 1
    return result


def resolve_coord_mode(df: pd.DataFrame, mode: str):
    """
    Returns (xcol, ycol, unit) based on --coord_mode and available columns.
    Exits with a clear message if the requested mode is unavailable.
    """
    has_world = "world_x" in df.columns and "world_y" in df.columns
    has_image = "cx" in df.columns and "cy" in df.columns

    if mode == "world":
        if not has_world:
            raise ValueError("--coord_mode world requested but world_x/world_y not found in CSV. "
                             "Run calibrate_homography.py first.")
        return "world_x", "world_y", "m"

    if mode == "image":
        if not has_image:
            raise ValueError("--coord_mode image requested but cx/cy not found in CSV.")
        return "cx", "cy", "px"

    # auto: prefer world if available
    if has_world:
        print("[compute_metrics] Auto mode: using world_x/world_y (metres)")
        return "world_x", "world_y", "m"

    print("[compute_metrics] Auto mode: using cx/cy (pixels)")
    return "cx", "cy", "px"


def main():
    ap = argparse.ArgumentParser(description="Compute basic movement metrics from trajectories CSV.")
    ap.add_argument("--traj_csv",          default="outputs/trajectories_image_space.csv")
    ap.add_argument("--out_dir",           default="outputs/metrics")
    ap.add_argument("--min_track_len",     type=int,   default=15,   help="Ignore tracks shorter than this (# observations).")
    ap.add_argument("--smooth_window",     type=int,   default=5,    help="Rolling median window for speed smoothing.")
    ap.add_argument("--speed_smoothing_window", type=int, default=5,
                    help="Rolling mean window applied after median smoothing to reduce flicker (default: 5). "
                         "Set to 1 to disable.")
    ap.add_argument("--stop_speed_px_s",   type=float, default=25.0, help="Stop threshold in px/s (image mode).")
    ap.add_argument("--stop_speed_m_s",    type=float, default=0.3,  help="Stop threshold in m/s (world mode).")
    ap.add_argument("--min_dwell_s",       type=float, default=2.0,  help="Minimum duration (s) to count as a dwell event.")
    ap.add_argument("--shift_angle_deg",   type=float, default=65.0,
                    help="Minimum direction change (degrees) to flag as a trajectory shift (default: 65).")
    ap.add_argument("--shift_min_dist_px", type=float, default=5.0,
                    help="Minimum step displacement (pixels) for shift detection (default: 5).")
    ap.add_argument("--coord_mode",        default="auto", choices=["auto", "image", "world"],
                    help="Which coordinates to use for speed/path metrics. "
                         "auto = use world_x/world_y if present, else cx/cy.")
    args = ap.parse_args()

    traj_csv = Path(args.traj_csv)
    if not traj_csv.exists():
        raise FileNotFoundError(f"Trajectory CSV not found: {traj_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(traj_csv)

    required_base = {"frame", "time_s", "track_id", "cx", "cy"}
    missing = required_base - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in trajectories CSV: {sorted(missing)}")

    # --- Coordinate mode resolution ---
    xcol, ycol, unit = resolve_coord_mode(df, args.coord_mode)
    stop_speed = args.stop_speed_m_s if unit == "m" else args.stop_speed_px_s

    speed_col        = f"speed_{unit}_s"
    speed_smooth_col = f"speed_smooth_{unit}_s"
    dist_col         = f"step_dist_{unit}"
    path_len_col     = f"path_len_{unit}"

    # --- Filter short tracks ---
    lens = df.groupby("track_id")["frame"].count()
    keep_ids = set(lens[lens >= args.min_track_len].index.tolist())
    df = df[df["track_id"].isin(keep_ids)].copy()
    if df.empty:
        raise ValueError("No tracks left after filtering; adjust --min_track_len.")

    # --- Compute speed ---
    df = compute_speed(df, xcol, ycol, unit)
    df[speed_smooth_col] = rolling_median_speed(df, speed_col, window=args.smooth_window)

    # Secondary mean pass to damp residual flicker
    if args.speed_smoothing_window > 1:
        df[speed_smooth_col] = (
            df.groupby("track_id")[speed_smooth_col]
            .transform(lambda s: s.rolling(
                window=args.speed_smoothing_window,
                min_periods=1,
                center=True,
            ).mean())
        )

    # --- speed_per_observation.csv ---
    obs_cols = ["frame", "time_s", "track_id", "cx", "cy",
                speed_col, speed_smooth_col, dist_col]
    df[obs_cols].to_csv(out_dir / "speed_per_observation.csv", index=False)

    # --- speed_per_track.csv ---
    summary = (
        df.groupby("track_id")
        .agg(
            n_obs              = ("frame",          "count"),
            t_start_s          = ("time_s",         "min"),
            t_end_s            = ("time_s",         "max"),
            **{f"mean_speed_{unit}_s":   (speed_smooth_col, "mean")},
            **{f"median_speed_{unit}_s": (speed_smooth_col, "median")},
            **{f"p95_speed_{unit}_s":    (speed_smooth_col,
                                          lambda s: float(np.nanpercentile(s.to_numpy(), 95)))},
            **{path_len_col:             (dist_col, "sum")},
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "speed_per_track.csv", index=False)

    # --- dwell_events.csv ---
    dwells = extract_dwell_events(
        df,
        speed_smooth_col=speed_smooth_col,
        stop_speed=stop_speed,
        min_dwell_s=args.min_dwell_s,
        xcol="cx", ycol="cy",   # dwell position always in image space for overlay compatibility
    )
    dwells.to_csv(out_dir / "dwell_events.csv", index=False)

    # --- stop_flags_per_observation.csv ---
    stop_flag = (df[speed_smooth_col] <= stop_speed) & df[speed_smooth_col].notna()
    df_out = df[["frame", "time_s", "track_id", "cx", "cy", speed_smooth_col]].copy()
    df_out["is_stop"] = stop_flag.astype(int)
    df_out.to_csv(out_dir / "stop_flags_per_observation.csv", index=False)

    # --- shift_flags_per_observation.csv ---
    shift_flag = compute_shift_flags(df,
                                     min_dist_px=args.shift_min_dist_px,
                                     angle_deg=args.shift_angle_deg)
    df_shift = df[["frame", "time_s", "track_id", "cx", "cy"]].copy()
    df_shift["is_shift"] = shift_flag.astype(int)
    df_shift.to_csv(out_dir / "shift_flags_per_observation.csv", index=False)

    print(f"[compute_metrics] coord_mode={args.coord_mode} → using {xcol}/{ycol} (unit: {unit})")
    print("[compute_metrics] Saved:")
    print(" -", out_dir / "speed_per_observation.csv")
    print(" -", out_dir / "speed_per_track.csv")
    print(" -", out_dir / "dwell_events.csv")
    print(" -", out_dir / "stop_flags_per_observation.csv")
    print(" -", out_dir / "shift_flags_per_observation.csv")


if __name__ == "__main__":
    main()
