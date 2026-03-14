import subprocess
import shutil
import sys
from pathlib import Path
import imageio_ffmpeg

# 🔴 CHANGE THIS TO YOUR REAL VIDEO
VIDEO_PATH = "C:/Users/OWNER/Desktop/IAAC/IAAC_Thesis/SecondYOLOtest/input_video.mp4"

SCALE_WIDTH = 1280
FPS = 24
CRF = 28
PRESET = "veryfast"

TRACKED_TRANSPOSE = 1
OVERLAY_TRANSPOSE = None


def run_script(script_name, extra_args=None):
    cmd = [sys.executable, script_name]
    if extra_args:
        cmd += extra_args
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def compress_mp4_for_miro(in_path, out_path, transpose=None):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    vf_parts = []
    if transpose in (1, 2):
        vf_parts.append(f"transpose={transpose}")
    vf_parts.append(f"scale={SCALE_WIDTH}:-2")
    vf_parts.append(f"fps={FPS}")
    vf = ",".join(vf_parts)

    cmd = [
        ffmpeg, "-y",
        "-noautorotate",
        "-i", str(in_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", PRESET,
        "-crf", str(CRF),
        "-movflags", "+faststart",
        "-map_metadata", "-1",
        "-metadata:s:v:0", "rotate=0",
        str(out_path)
    ]
    subprocess.run(cmd, check=True)
    print(f"Compressed: {out_path}")


def clean_outputs(out_dir: Path):
    """
    Delete everything inside outputs/, including folders like outputs/metrics/.
    Keeps the outputs/ directory itself.
    """
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


def main():
    video = Path(VIDEO_PATH)
    if not video.exists():
        raise FileNotFoundError(f"VIDEO_PATH does not exist: {VIDEO_PATH}")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # Clean outputs safely (files + folders)
    clean_outputs(out_dir)

    # 1) Tracking
    run_script("track_people.py", [
        "--video", str(video),
        "--out_dir", "outputs",
        "--model_path", "yolov8s.pt",
        "--tracker_cfg", "bytetrack_mp.yaml",
        "--imgsz", "1280",
        "--conf", "0.25",
        "--iou", "0.50",
        "--max_det", "200",
        "--vid_stride", "1",
        "--save_video",
        "--save_csv",
        "--draw",
    ])

    # 2) Compute metrics (creates outputs/metrics/*.csv)
    run_script("compute_metrics.py", [
        "--traj_csv", "outputs/trajectories_image_space.csv",
        "--out_dir", "outputs/metrics",
        "--min_track_len", "15",
        "--smooth_window", "5",
        "--stop_speed_px_s", "25",
        "--min_dwell_s", "2.0",
    ])

    # 3) ✅ Compute density heatmap video (heatmap-only frames)
    run_script("compute_heatmap.py", [
        "--video", str(video),
        "--traj_csv", "outputs/trajectories_image_space.csv",
        "--out_path", "outputs/metrics/density_heatmap.mp4",
        "--cell_size", "60",
        "--blur", "61",
    ])

    # 4) Draw trajectories WITH metrics + heatmap overlay
    run_script("draw_trajectories.py", [
        "--video", str(video),
        "--traj_csv", "outputs/trajectories_image_space.csv",
        "--out_dir", "outputs",
        "--speed_obs_csv", "outputs/metrics/speed_per_observation.csv",
        "--stop_flags_csv", "outputs/metrics/stop_flags_per_observation.csv",
        "--dwell_events_csv", "outputs/metrics/dwell_events.csv",
        "--heatmap_video", "outputs/metrics/density_heatmap.mp4",
        "--heatmap_alpha", "0.20",
    ])

    # 5) Compress
    tracked = out_dir / "tracked.mp4"
    overlay = out_dir / "trajectories_metrics_overlay.mp4"
    heatmap = out_dir / "metrics" / "density_heatmap.mp4"

    if tracked.exists():
        compress_mp4_for_miro(tracked, out_dir / "tracked_miro.mp4", transpose=TRACKED_TRANSPOSE)
    else:
        print("[WARN] tracked.mp4 not found — skipping tracked_miro.mp4")

    if overlay.exists():
        compress_mp4_for_miro(overlay, out_dir / "trajectories_metrics_overlay_miro.mp4", transpose=OVERLAY_TRANSPOSE)
    else:
        print("[WARN] trajectories_metrics_overlay.mp4 not found — skipping overlay_miro.mp4")

    # Optional: also compress heatmap-only video (handy for Miro)
    if heatmap.exists():
        compress_mp4_for_miro(heatmap, out_dir / "density_heatmap_miro.mp4", transpose=OVERLAY_TRANSPOSE)
    else:
        print("[WARN] density_heatmap.mp4 not found — skipping density_heatmap_miro.mp4")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()