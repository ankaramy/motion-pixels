import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse


def speed_to_color_bgr(speed, stop_thr=25.0, mid_thr=80.0):
    """Map speed (px/s) to a BGR color."""
    if speed is None or (isinstance(speed, float) and np.isnan(speed)):
        return (180, 180, 180)  # gray
    if speed <= stop_thr:
        return (0, 0, 255)      # red
    if speed < mid_thr:
        return (0, 255, 255)    # yellow
    return (0, 255, 0)          # green


def build_dwell_lookup(dwell_csv: Path):
    """
    Returns dict: track_id -> list of (start_time_s, end_time_s)
    Handles missing or empty dwell CSVs gracefully.
    """
    if not dwell_csv or not Path(dwell_csv).exists():
        return {}

    try:
        if Path(dwell_csv).stat().st_size == 0:
            return {}
    except Exception:
        pass

    try:
        dw = pd.read_csv(dwell_csv)
    except pd.errors.EmptyDataError:
        return {}

    if dw.empty:
        return {}

    required = {"track_id", "start_time_s", "end_time_s"}
    if not required.issubset(set(dw.columns)):
        return {}

    lookup = defaultdict(list)
    for _, r in dw.iterrows():
        tid = int(r["track_id"])
        lookup[tid].append((float(r["start_time_s"]), float(r["end_time_s"])))

    for tid in lookup:
        lookup[tid].sort(key=lambda x: x[0])

    return dict(lookup)


def in_dwell_interval(dwell_lookup, tid: int, t_s: float) -> bool:
    intervals = dwell_lookup.get(tid)
    if not intervals:
        return False
    for (a, b) in intervals:
        if a <= t_s <= b:
            return True
    return False


def main(
    video_path,
    traj_csv="outputs/trajectories_image_space.csv",
    out_dir="outputs",
    max_tracks_draw=9999,
    min_track_len=15,
    draw_tail=60,
    thickness=2,
    # metrics
    speed_obs_csv="outputs/metrics/speed_per_observation.csv",
    stop_flags_csv="outputs/metrics/stop_flags_per_observation.csv",
    dwell_events_csv="outputs/metrics/dwell_events.csv",
    stop_speed_px_s=25.0,
    mid_speed_px_s=80.0,
    show_legend=True,
    # heatmap
    heatmap_video="outputs/metrics/density_heatmap.mp4",
    heatmap_alpha=0.20,
    use_heatmap=True,
):
    print(f"[draw_trajectories] video_path = {video_path}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_csv = Path(traj_csv)
    df = pd.read_csv(traj_csv)
    if df.empty:
        raise ValueError("Trajectory CSV is empty.")

    lens = df.groupby("track_id")["frame"].count()
    keep_ids = set(lens[lens >= min_track_len].index.tolist())
    df = df[df["track_id"].isin(keep_ids)].copy()

    speed_obs_csv = Path(speed_obs_csv)
    stop_flags_csv = Path(stop_flags_csv)
    dwell_events_csv = Path(dwell_events_csv)

    if speed_obs_csv.exists():
        sp = pd.read_csv(speed_obs_csv)
        if "speed_smooth_px_s" in sp.columns:
            df = df.merge(
                sp[["frame", "track_id", "speed_smooth_px_s"]],
                on=["frame", "track_id"],
                how="left",
            )
        else:
            print("[draw_trajectories] WARNING: speed_obs missing speed_smooth_px_s; speed coloring disabled.")
            df["speed_smooth_px_s"] = np.nan
    else:
        print(f"[draw_trajectories] WARNING: missing {speed_obs_csv}; speed coloring disabled.")
        df["speed_smooth_px_s"] = np.nan

    if stop_flags_csv.exists():
        sf = pd.read_csv(stop_flags_csv)
        if "is_stop" in sf.columns:
            df = df.merge(
                sf[["frame", "track_id", "is_stop"]],
                on=["frame", "track_id"],
                how="left",
            )
        else:
            print("[draw_trajectories] WARNING: stop_flags missing is_stop; stop ring disabled.")
            df["is_stop"] = 0
    else:
        print(f"[draw_trajectories] WARNING: missing {stop_flags_csv}; stop ring disabled.")
        df["is_stop"] = 0

    dwell_lookup = build_dwell_lookup(dwell_events_csv)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    heat_cap = None
    heatmap_video = Path(heatmap_video) if heatmap_video else None
    if use_heatmap and heatmap_video and heatmap_video.exists():
        heat_cap = cv2.VideoCapture(str(heatmap_video))
        if not heat_cap.isOpened():
            print(f"[draw_trajectories] WARNING: Could not open heatmap video: {heatmap_video}")
            heat_cap = None
    else:
        if use_heatmap:
            print(f"[draw_trajectories] WARNING: Heatmap not found: {heatmap_video}. Continuing without heatmap.")

    out_video = str(out_dir / "trajectories_metrics_overlay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Failed to initialize VideoWriter for trajectories_metrics_overlay.mp4")

    df.sort_values(["frame", "track_id"], inplace=True)
    by_frame = df.groupby("frame")

    history = defaultdict(list)
    last_speed = {}
    last_stop = {}

    frame_idx = -1
    first_frame_img = None
    a_heat = float(np.clip(heatmap_alpha, 0.0, 1.0))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        t_s = frame_idx / fps

        if first_frame_img is None:
            first_frame_img = frame.copy()

        # Blend heatmap first
        if heat_cap is not None:
            ok_h, hframe = heat_cap.read()
            if ok_h:
                if hframe.shape[1] != w or hframe.shape[0] != h:
                    hframe = cv2.resize(hframe, (w, h), interpolation=cv2.INTER_LINEAR)
                frame = cv2.addWeighted(frame, 1.0 - a_heat, hframe, a_heat, 0)
            else:
                heat_cap.release()
                heat_cap = None

        if frame_idx in by_frame.groups:
            chunk = by_frame.get_group(frame_idx)
            for _, row in chunk.iterrows():
                tid = int(row["track_id"])
                pt = (int(row["cx"]), int(row["cy"]))
                history[tid].append(pt)
                if len(history[tid]) > draw_tail:
                    history[tid] = history[tid][-draw_tail:]

                last_speed[tid] = row.get("speed_smooth_px_s", np.nan)
                is_stop = row.get("is_stop", 0)
                try:
                    last_stop[tid] = int(is_stop) if not pd.isna(is_stop) else 0
                except Exception:
                    last_stop[tid] = 0

        if show_legend:
            cv2.putText(
                frame,
                "Speed: green=fast  yellow=mid  red=slow/stop | Red ring=stop | Blue ring=dwell",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            if use_heatmap and (heat_cap is not None):
                cv2.putText(
                    frame,
                    f"Heatmap alpha={a_heat:.2f}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

        drawn = 0
        for tid, pts in history.items():
            if drawn >= max_tracks_draw:
                break
            if len(pts) < 2:
                continue

            speed = last_speed.get(tid, np.nan)
            color = speed_to_color_bgr(speed, stop_thr=stop_speed_px_s, mid_thr=mid_speed_px_s)

            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, color, thickness)

            head = pts[-1]
            cv2.putText(frame, f"ID {tid}", head, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if last_stop.get(tid, 0) == 1:
                cv2.circle(frame, head, 14, (0, 0, 255), 2)

            if in_dwell_interval(dwell_lookup, tid, t_s):
                cv2.circle(frame, head, 22, (255, 0, 0), 2)

            drawn += 1

        writer.write(frame)

    cap.release()
    if heat_cap is not None:
        heat_cap.release()
    writer.release()

    if first_frame_img is None:
        raise RuntimeError("No frames read; cannot write trajectories_static.png")

    static = first_frame_img.copy()
    mean_speed_by_tid = (
        df.groupby("track_id")["speed_smooth_px_s"].mean()
        if "speed_smooth_px_s" in df.columns
        else pd.Series(dtype=float)
    )

    for tid, g in df.groupby("track_id"):
        pts = g.sort_values("frame")[["cx", "cy"]].to_numpy().astype(int)
        if len(pts) < 2:
            continue
        mean_sp = float(mean_speed_by_tid.get(tid, np.nan)) if not mean_speed_by_tid.empty else np.nan
        col = speed_to_color_bgr(mean_sp, stop_thr=stop_speed_px_s, mid_thr=mid_speed_px_s)
        cv2.polylines(static, [pts.reshape(-1, 1, 2)], False, col, thickness)

    out_img = str(out_dir / "trajectories_metrics_static.png")
    cv2.imwrite(out_img, static)

    print("[draw_trajectories] Saved:")
    print(" -", out_video)
    print(" -", out_img)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--traj_csv", default="outputs/trajectories_image_space.csv")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_track_len", type=int, default=15)
    ap.add_argument("--draw_tail", type=int, default=60)
    ap.add_argument("--thickness", type=int, default=2)

    ap.add_argument("--speed_obs_csv", default="outputs/metrics/speed_per_observation.csv")
    ap.add_argument("--stop_flags_csv", default="outputs/metrics/stop_flags_per_observation.csv")
    ap.add_argument("--dwell_events_csv", default="outputs/metrics/dwell_events.csv")

    ap.add_argument("--stop_speed_px_s", type=float, default=25.0)
    ap.add_argument("--mid_speed_px_s", type=float, default=80.0)
    ap.add_argument("--no_legend", action="store_true")

    ap.add_argument("--heatmap_video", default="outputs/metrics/density_heatmap.mp4")
    ap.add_argument("--heatmap_alpha", type=float, default=0.20)
    ap.add_argument("--no_heatmap", action="store_true")

    args = ap.parse_args()

    main(
        video_path=args.video,
        traj_csv=args.traj_csv,
        out_dir=args.out_dir,
        min_track_len=args.min_track_len,
        draw_tail=args.draw_tail,
        thickness=args.thickness,
        speed_obs_csv=args.speed_obs_csv,
        stop_flags_csv=args.stop_flags_csv,
        dwell_events_csv=args.dwell_events_csv,
        stop_speed_px_s=args.stop_speed_px_s,
        mid_speed_px_s=args.mid_speed_px_s,
        show_legend=(not args.no_legend),
        heatmap_video=args.heatmap_video,
        heatmap_alpha=args.heatmap_alpha,
        use_heatmap=(not args.no_heatmap),
    )