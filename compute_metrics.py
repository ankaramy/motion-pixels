import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_speed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds per-observation speed in px/s using consecutive detections per track.
    Expects columns: track_id, frame, time_s, cx, cy
    """
    df = df.sort_values(["track_id", "frame"]).copy()

    # previous observation per track
    df["prev_cx"] = df.groupby("track_id")["cx"].shift(1)
    df["prev_cy"] = df.groupby("track_id")["cy"].shift(1)
    df["prev_time_s"] = df.groupby("track_id")["time_s"].shift(1)

    dx = df["cx"] - df["prev_cx"]
    dy = df["cy"] - df["prev_cy"]
    dt = df["time_s"] - df["prev_time_s"]

    dist = np.sqrt(dx * dx + dy * dy)

    # Guard against dt==0 or missing
    df["speed_px_s"] = np.where((dt > 0) & np.isfinite(dist), dist / dt, np.nan)
    df["step_dist_px"] = dist
    return df


def rolling_median_speed(df: pd.DataFrame, window: int) -> pd.Series:
    """Median filter per track to reduce jitter. window is in observations."""
    if window <= 1:
        return df["speed_px_s"]
    return (
        df.groupby("track_id")["speed_px_s"]
        .transform(lambda s: s.rolling(window=window, min_periods=max(2, window // 2), center=True).median())
    )


def extract_dwell_events(
    df: pd.DataFrame,
    stop_speed_px_s: float,
    min_dwell_s: float,
) -> pd.DataFrame:
    """
    One row per continuous stop segment:
    track_id, start_time_s, end_time_s, duration_s, cx_mean, cy_mean, n_obs
    """
    stop = (df["speed_smooth_px_s"] <= stop_speed_px_s) & df["speed_smooth_px_s"].notna()
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
            start_t = float(seg_g["time_s"].iloc[0])
            end_t = float(seg_g["time_s"].iloc[-1])
            duration = end_t - start_t

            if duration >= min_dwell_s:
                events.append(
                    {
                        "track_id": int(tid),
                        "start_time_s": start_t,
                        "end_time_s": end_t,
                        "duration_s": float(duration),
                        "cx_mean": float(seg_g["cx"].mean()),
                        "cy_mean": float(seg_g["cy"].mean()),
                        "n_obs": int(len(seg_g)),
                    }
                )

    return pd.DataFrame(events)


def main():
    ap = argparse.ArgumentParser(description="Compute basic movement metrics from trajectories CSV.")
    ap.add_argument("--traj_csv", default="outputs/trajectories_image_space.csv")
    ap.add_argument("--out_dir", default="outputs/metrics")
    ap.add_argument("--min_track_len", type=int, default=15, help="Ignore tracks shorter than this (# observations).")
    ap.add_argument("--smooth_window", type=int, default=5, help="Rolling median window for speed smoothing.")
    ap.add_argument("--stop_speed_px_s", type=float, default=25.0, help="Speed threshold (px/s) below which we call it a stop.")
    ap.add_argument("--min_dwell_s", type=float, default=2.0, help="Minimum duration (s) to count as a dwell event.")
    args = ap.parse_args()

    traj_csv = Path(args.traj_csv)
    if not traj_csv.exists():
        raise FileNotFoundError(f"Trajectory CSV not found: {traj_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(traj_csv)
    required = {"frame", "time_s", "track_id", "cx", "cy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in trajectories CSV: {sorted(missing)}")

    lens = df.groupby("track_id")["frame"].count()
    keep_ids = set(lens[lens >= args.min_track_len].index.tolist())
    df = df[df["track_id"].isin(keep_ids)].copy()
    if df.empty:
        raise ValueError("No tracks left after filtering; adjust --min_track_len.")

    df = compute_speed(df)
    df["speed_smooth_px_s"] = rolling_median_speed(df, window=args.smooth_window)

    obs_cols = ["frame", "time_s", "track_id", "cx", "cy", "speed_px_s", "speed_smooth_px_s", "step_dist_px"]
    df[obs_cols].to_csv(out_dir / "speed_per_observation.csv", index=False)

    summary = (
        df.groupby("track_id")
        .agg(
            n_obs=("frame", "count"),
            t_start_s=("time_s", "min"),
            t_end_s=("time_s", "max"),
            mean_speed_px_s=("speed_smooth_px_s", "mean"),
            median_speed_px_s=("speed_smooth_px_s", "median"),
            p95_speed_px_s=("speed_smooth_px_s", lambda s: float(np.nanpercentile(s.to_numpy(), 95))),
            path_len_px=("step_dist_px", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "speed_per_track.csv", index=False)

    dwells = extract_dwell_events(df, stop_speed_px_s=args.stop_speed_px_s, min_dwell_s=args.min_dwell_s)
    dwells.to_csv(out_dir / "dwell_events.csv", index=False)

    stop_flag = (df["speed_smooth_px_s"] <= args.stop_speed_px_s) & df["speed_smooth_px_s"].notna()
    df_out = df[["frame", "time_s", "track_id", "cx", "cy", "speed_smooth_px_s"]].copy()
    df_out["is_stop"] = stop_flag.astype(int)
    df_out.to_csv(out_dir / "stop_flags_per_observation.csv", index=False)

    print("[compute_metrics] Saved:")
    print(" -", out_dir / "speed_per_observation.csv")
    print(" -", out_dir / "speed_per_track.csv")
    print(" -", out_dir / "dwell_events.csv")
    print(" -", out_dir / "stop_flags_per_observation.csv")


if __name__ == "__main__":
    main()