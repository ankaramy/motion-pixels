import argparse
import subprocess
import shutil
import sys
from pathlib import Path
import imageio_ffmpeg

HERE         = Path(__file__).resolve().parent
MP_ROOT      = HERE.parent.parent
MP_DATA      = MP_ROOT / "mp-data"
TRACKING_OUT = MP_DATA / "outputs" / "tracking"
BEHAVIOR_OUT = MP_DATA / "outputs" / "behavior"
MODELS_DIR   = MP_DATA / "models"


def run_script(script_name, extra_args=None, check=True):
    cmd = [sys.executable, script_name]
    if extra_args:
        cmd += extra_args
    print("\nRunning:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        msg = f"[WARN] {script_name} exited with code {result.returncode}"
        if check:
            raise RuntimeError(msg)
        print(msg)
        return False
    return True


def compress_video(src: Path, max_mb: float, crf: int, scale: int) -> None:
    """
    Re-encode src in-place to H.264 MP4.
    If the result still exceeds max_mb, retries once with crf+6 and half the width.
    """
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = src.with_suffix(".tmp.mp4")

    def _encode(crf_val: int, scale_val: int) -> float:
        cmd = [
            ffmpeg, "-y", "-i", str(src),
            "-vf", f"scale={scale_val}:-2",
            "-c:v", "libx264", "-preset", "veryfast",
            "-crf", str(crf_val),
            "-movflags", "+faststart",
            "-an",
            str(tmp),
        ]
        subprocess.run(cmd, check=True)
        return tmp.stat().st_size / (1024 * 1024)

    size_mb = _encode(crf, scale)
    print(f"[compress] {src.name}: {size_mb:.1f} MB  (crf={crf}, scale={scale})")

    if size_mb > max_mb:
        print(f"[compress] {size_mb:.1f} MB > {max_mb} MB target — retrying with stronger settings")
        size_mb = _encode(min(crf + 6, 51), max(scale // 2, 320))
        print(f"[compress] Retry: {size_mb:.1f} MB")
        if size_mb > max_mb:
            print(f"[WARN] Could not reach {max_mb} MB — keeping best result ({size_mb:.1f} MB)")

    tmp.replace(src)


def clean_outputs(out_dir: Path):
    """Delete everything inside outputs/ while keeping the directory itself."""
    if not out_dir.exists():
        return
    for p in out_dir.iterdir():
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
        except Exception as e:
            print(f"[WARN] Could not remove {p}: {e}")


def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def ensure_calibration(
    video: Path,
    calib_json: str | None,
    top_view_image: str | None,
    recalibrate: bool,
):
    """
    If calib_json is requested but missing (or recalibrate=True),
    launch the interactive calibration UI to create it.
    """
    if not calib_json:
        return

    calib_path = Path(calib_json)
    need_interactive = recalibrate or (not calib_path.exists())

    if not need_interactive:
        return

    if not top_view_image:
        raise ValueError(
            "Calibration requires --top_view_image when calib.json is missing or when using --recalibrate."
        )

    top_view_path = Path(top_view_image)
    if not top_view_path.exists():
        raise FileNotFoundError(f"Top-view image not found: {top_view_path}")

    section("Interactive calibration — top-view point picking")
    if recalibrate and calib_path.exists():
        print(f"[INFO] --recalibrate enabled -> reopening calibration UI and overwriting: {calib_path}")
    else:
        print(f"[INFO] Calibration file not found -> launching interactive calibration UI: {calib_path}")

    # IMPORTANT:
    # This assumes your interactive script accepts --top_view_image.
    # If Claude built it with --plan_image instead, rename that argument there too.
    run_script("calibrate_homography_interactive.py", [
        "--video", str(video),
        "--top_view_image", str(top_view_path),
        "--out_json", str(calib_path),
    ], check=True)

    if not calib_path.exists():
        raise RuntimeError("Calibration was cancelled or failed: calib.json was not created.")


def main(
    video_path: str,
    calib_json: str = None,
    top_view_image: str = None,
    recalibrate: bool = False,
    run_flow_fields: bool = False,
    run_bottlenecks: bool = False,
    run_linger_zones: bool = False,
    show_behavior_on_video: bool = False,
    max_video_mb: float = 30.0,
    output_crf: int = 26,
    output_scale: int = 1280,
    blur_faces: bool = False,
    face_height_ratio: float = 0.32,
    face_width_ratio: float = 0.65,
    face_y_offset_ratio: float = 0.05,
    blur_kernel: int = 51,
    # agent labels / HUD
    highlight_top_k_speed: int = 0,
    highlight_top_k_dwell: int = 0,
    show_hud: bool = False,
    # analysis plot appearance
    top_view_alpha: float = 0.25,
    zoom_mode: str = "auto",
    zoom_padding: float = 0.08,
):
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    # If calibration is requested, create calib.json interactively if needed.
    ensure_calibration(
        video=video,
        calib_json=calib_json,
        top_view_image=top_view_image,
        recalibrate=recalibrate,
    )

    # After ensure_calibration, validate calib_json if it was requested.
    if calib_json and not Path(calib_json).exists():
        raise FileNotFoundError(f"calib_json not found: {calib_json}")

    out_dir = TRACKING_OUT
    behavior_dir = BEHAVIOR_OUT
    flow_dir = behavior_dir / "flow_fields"
    bn_dir = behavior_dir / "bottlenecks"
    linger_dir = behavior_dir / "linger_zones"

    out_dir.mkdir(parents=True, exist_ok=True)
    behavior_dir.mkdir(parents=True, exist_ok=True)
    clean_outputs(out_dir)

    image_traj_csv = str(out_dir / "trajectories_image_space.csv")
    world_traj_csv = str(out_dir / "trajectories_world.csv")
    metrics_dir    = str(out_dir / "metrics")

    # ── 1) Tracking ──────────────────────────────────────────────────────────
    section("Stage 1 — Tracking")
    run_script("track_people.py", [
        "--video", str(video),
        "--out_dir", str(out_dir),
        "--model_path", str(MODELS_DIR / "yolov8s.pt"),
        "--tracker_cfg", str(HERE / "bytetrack_mp.yaml"),
        "--imgsz", "1280",
        "--conf", "0.25",
        "--iou", "0.50",
        "--max_det", "200",
        "--vid_stride", "1",
        "--save_video",
        "--save_csv",
        "--draw",
    ])

    # ── 1b) Homography calibration (optional) ───────────────────────────────
    calibrated = False
    if calib_json:
        section("Stage 1b — Homography calibration (world coordinates)")
        calibrated = run_script("calibrate_homography.py", [
            "--traj_csv", image_traj_csv,
            "--calib_json", calib_json,
            "--out_csv", world_traj_csv,
            "--out_plot", str(out_dir / "trajectories_world_plot.png"),
        ], check=False)

        if calibrated:
            print("[INFO] World-coordinate CSV ready:", world_traj_csv)
        else:
            print("[WARN] Calibration failed — falling back to image-space analysis.")

    # Choose analysis coordinate space
    if calibrated:
        analysis_traj_csv = world_traj_csv
        analysis_stop_speed = "0.3"      # m/s
        analysis_cell_size = "1.0"       # metres
        analysis_cluster_eps = "1.5"     # metres
        analysis_unit_note = "world (metres)"
    else:
        analysis_traj_csv = image_traj_csv
        analysis_stop_speed = "25"       # px/s
        analysis_cell_size = "80"        # pixels
        analysis_cluster_eps = "80"      # pixels
        analysis_unit_note = "image (pixels)"

    print(f"\n[INFO] Analysis coordinate space: {analysis_unit_note}")
    print(f"[INFO] Metrics + behavior source CSV: {analysis_traj_csv}")

    # ── 2) Core metrics ──────────────────────────────────────────────────────
    section("Stage 2 — Core metrics")
    run_script("compute_metrics.py", [
        "--traj_csv", analysis_traj_csv,
        "--out_dir", metrics_dir,
        "--min_track_len", "15",
        "--smooth_window", "5",
        "--stop_speed_px_s", analysis_stop_speed,
        "--min_dwell_s", "2.0",
    ])

    # ── 3) Optional behavioral analysis ─────────────────────────────────────
    flow_ok = False
    bn_ok = False
    linger_ok = False

    # Shared top-view background args for analysis plots
    tv_args = []
    if top_view_image:
        tv_args += ["--top_view_image", str(top_view_image),
                    "--top_view_alpha", str(top_view_alpha)]
    if calib_json and Path(calib_json).exists():
        tv_args += ["--calib_json", str(calib_json)]
    tv_args += ["--zoom_mode", zoom_mode, "--zoom_padding", str(zoom_padding)]

    if run_flow_fields:
        section("Stage 4a — Flow fields")
        flow_ok = run_script("compute_flow_fields.py", [
            "--traj_csv", analysis_traj_csv,
            "--out_dir", str(flow_dir),
            "--cell_size", analysis_cell_size,
            "--min_vectors_per_cell", "3",
        ] + tv_args, check=False)

    if run_bottlenecks:
        section("Stage 4b — Bottleneck detection")
        bn_ok = run_script("compute_bottlenecks.py", [
            "--traj_csv", analysis_traj_csv,
            "--out_dir", str(bn_dir),
            "--cell_size", analysis_cell_size,
            "--stop_speed_threshold", analysis_stop_speed,
            "--min_obs_per_cell", "5",
            "--top_k", "10",
        ] + tv_args, check=False)

    if run_linger_zones:
        section("Stage 4c — Linger zones")
        linger_ok = run_script("compute_linger_zones.py", [
            "--traj_csv", analysis_traj_csv,
            "--out_dir", str(linger_dir),
            "--stop_speed_threshold", analysis_stop_speed,
            "--pause_s", "1.0",
            "--linger_s", "5.0",
            "--long_wait_s", "15.0",
            "--cluster_eps", analysis_cluster_eps,
            "--cluster_min_samples", "3",
        ] + tv_args, check=False)

    # ── 5) Trajectory overlay video (always image-space) ────────────────────
    section("Stage 5 — Trajectory + metrics overlay video")
    draw_args = [
        "--video", str(video),
        "--traj_csv", image_traj_csv,
        "--out_dir", str(out_dir),
        "--speed_obs_csv", f"{metrics_dir}/speed_per_observation.csv",
        "--stop_flags_csv", f"{metrics_dir}/stop_flags_per_observation.csv",
        "--dwell_events_csv", f"{metrics_dir}/dwell_events.csv",
        "--shift_flags_csv", f"{metrics_dir}/shift_flags_per_observation.csv",
        "--no_heatmap",
        # Speed-color thresholds must match the unit of the speed column.
        # Flag names still say px_s but values are in the correct unit for each case.
        "--stop_speed_px_s", "0.3"  if calibrated else "25.0",
        "--mid_speed_px_s",  "1.5"  if calibrated else "80.0",
    ]

    if blur_faces:
        draw_args += [
            "--blur_faces",
            "--face_height_ratio", str(face_height_ratio),
            "--face_width_ratio", str(face_width_ratio),
            "--face_y_offset_ratio", str(face_y_offset_ratio),
            "--face_blur_kernel", str(blur_kernel),
        ]

    if highlight_top_k_speed > 0:
        draw_args += ["--highlight_top_k_speed", str(highlight_top_k_speed)]
    if highlight_top_k_dwell > 0:
        draw_args += ["--highlight_top_k_dwell", str(highlight_top_k_dwell)]
    if show_hud:
        draw_args.append("--show_hud")

    if show_behavior_on_video:
        if calibrated:
            print(
                "[INFO] --show_behavior_on_video skipped for behavioral overlays:\n"
                "       Behavioral data is in world coordinates (metres) and does not\n"
                "       map directly to video pixels. Use the top-view outputs in\n"
                "       outputs/behavior/ and outputs/trajectories_world_plot.png instead."
            )
        else:
            draw_args.append("--show_metrics")
            if flow_ok:
                draw_args += ["--flow_csv", str(flow_dir / "flow_field_cells.csv")]
            else:
                print("[INFO] Flow overlay skipped (stage did not run or failed)")
            if bn_ok:
                draw_args += ["--bottleneck_csv", str(bn_dir / "bottleneck_cells.csv")]
            else:
                print("[INFO] Bottleneck overlay skipped (stage did not run or failed)")
            if linger_ok:
                draw_args += ["--linger_csv", str(linger_dir / "linger_zones.csv")]
            else:
                print("[INFO] Linger overlay skipped (stage did not run or failed)")

    run_script("draw_trajectories.py", draw_args)

    # ── 6) Compress final outputs in-place to H.264 ─────────────────────────
    section("Stage 6 — Compress outputs")
    for vid in [
        out_dir / "tracked.mp4",
        out_dir / "trajectories_metrics_overlay.mp4",
    ]:
        if vid.exists():
            compress_video(vid, max_mb=max_video_mb, crf=output_crf, scale=output_scale)
        else:
            print(f"[WARN] {vid.name} not found — skipping compression")

    section("Pipeline complete")
    print(f"  Output folder : {out_dir.resolve()}")
    if calibrated:
        print(f"  World CSV     : {Path(world_traj_csv).resolve()}")
        print(f"  Top-view plot : {(out_dir / 'trajectories_world_plot.png').resolve()}")
    if run_flow_fields or run_bottlenecks or run_linger_zones:
        print(f"  Behavior data : {behavior_dir.resolve()}")
        print(f"  Coord space   : {analysis_unit_note}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Motion Pixels pipeline")

    ap.add_argument("--video", required=True, help="Path to input video")

    ap.add_argument(
        "--calib_json",
        default=None,
        help="Path to calib.json. If provided, calibration will run after tracking. "
             "If the file does not exist, the interactive calibration UI will open."
    )
    ap.add_argument(
        "--top_view_image",
        default=None,
        help="Path to top-view image used for interactive calibration. "
             "Required when calib.json does not exist yet or when using --recalibrate."
    )
    ap.add_argument(
        "--recalibrate",
        action="store_true",
        help="Force the interactive calibration UI to open even if calib.json already exists."
    )

    # Behavioral analysis flags
    ap.add_argument("--run_flow_fields", action="store_true",
                    help="Run compute_flow_fields.py -> outputs/behavior/flow_fields/")
    ap.add_argument("--run_bottlenecks", action="store_true",
                    help="Run compute_bottlenecks.py -> outputs/behavior/bottlenecks/")
    ap.add_argument("--run_linger_zones", action="store_true",
                    help="Run compute_linger_zones.py -> outputs/behavior/linger_zones/")
    ap.add_argument("--show_behavior_on_video", action="store_true",
                    help="Overlay behavioral outputs on the trajectory video. "
                         "Only active in image-space mode. With calibration, "
                         "use the top-view outputs instead.")

    # Agent labels & HUD
    ap.add_argument("--highlight_top_k_speed", type=int, default=0,
                    help="Label the N fastest tracks on the output video (0 = off)")
    ap.add_argument("--highlight_top_k_dwell", type=int, default=0,
                    help="Label the N longest-dwelling tracks on the output video (0 = off)")
    ap.add_argument("--show_hud", action="store_true",
                    help="Show global-metrics HUD on the output video")

    # Analysis plot appearance
    ap.add_argument("--top_view_alpha", type=float, default=0.25,
                    help="Opacity of the top-view background in analysis plots (default: 0.25)")
    ap.add_argument("--zoom_mode", default="auto", choices=["auto", "full"],
                    help="Analysis plot zoom: 'auto' = data extent, 'full' = full image extent")
    ap.add_argument("--zoom_padding", type=float, default=0.08,
                    help="Fractional padding around data in auto zoom mode (default: 0.08)")

    # Face blur
    ap.add_argument("--blur_faces", action="store_true",
                    help="Blur the estimated face area of each detected person")
    ap.add_argument("--face_height_ratio", type=float, default=0.32,
                    help="Fraction of bbox height used as face blur region (default: 0.32)")
    ap.add_argument("--face_width_ratio", type=float, default=0.65,
                    help="Fraction of bbox width used as face blur region, centred (default: 0.65)")
    ap.add_argument("--face_y_offset_ratio", type=float, default=0.05,
                    help="Vertical offset from bbox top before blur starts (default: 0.05)")
    ap.add_argument("--face_blur_kernel", type=int, default=51,
                    help="Gaussian kernel size; applied twice for strong anonymisation (default: 51)")

    # Output compression
    ap.add_argument("--max_video_mb", type=float, default=30.0,
                    help="Target max file size in MB for each output video (default: 30)")
    ap.add_argument("--output_crf", type=int, default=26,
                    help="H.264 CRF quality (lower = better; default: 26)")
    ap.add_argument("--output_scale", type=int, default=1280,
                    help="Output video width in pixels; height is scaled proportionally (default: 1280)")

    args = ap.parse_args()
    main(
        video_path=args.video,
        calib_json=args.calib_json,
        top_view_image=args.top_view_image,
        recalibrate=args.recalibrate,
        run_flow_fields=args.run_flow_fields,
        run_bottlenecks=args.run_bottlenecks,
        run_linger_zones=args.run_linger_zones,
        show_behavior_on_video=args.show_behavior_on_video,
        max_video_mb=args.max_video_mb,
        output_crf=args.output_crf,
        output_scale=args.output_scale,
        blur_faces=args.blur_faces,
        face_height_ratio=args.face_height_ratio,
        face_width_ratio=args.face_width_ratio,
        face_y_offset_ratio=args.face_y_offset_ratio,
        blur_kernel=args.face_blur_kernel,
        highlight_top_k_speed=args.highlight_top_k_speed,
        highlight_top_k_dwell=args.highlight_top_k_dwell,
        show_hud=args.show_hud,
        top_view_alpha=args.top_view_alpha,
        zoom_mode=args.zoom_mode,
        zoom_padding=args.zoom_padding,
    )