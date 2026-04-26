from pathlib import Path
import subprocess
import imageio_ffmpeg

HERE         = Path(__file__).resolve().parent
MP_ROOT      = HERE.parent.parent
TRACKING_OUT = MP_ROOT / "mp-data" / "outputs" / "tracking"


def compress_mp4_for_miro(
    in_path: str,
    out_path: str,
    scale_width: int = 1280,
    fps: int = 24,
    crf: int = 28,
    preset: str = "veryfast",
):
    """
    Re-encode to H.264 MP4 (Miro-friendly) and reduce size.
    Uses imageio-ffmpeg's bundled ffmpeg binary (no system ffmpeg needed).
    """
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    in_path = str(Path(in_path))
    out_path = str(Path(out_path))

    cmd = [
        ffmpeg, "-y",
        "-i", in_path,
        "-vf", f"scale={scale_width}:-2,fps={fps}",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-movflags", "+faststart",
        out_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Saved: {out_path}")


def main():
    out_dir = TRACKING_OUT

    # A) tracked.mp4 -> tracked_miro.mp4
    compress_mp4_for_miro(
        in_path=out_dir / "tracked.mp4",
        out_path=out_dir / "tracked_miro.mp4",
        scale_width=1280,
        fps=24,
        crf=28,
        preset="veryfast",
    )

    # B) trajectories_overlay.mp4 -> trajectories_overlay_miro.mp4 (if it exists)
    overlay = out_dir / "trajectories_overlay.mp4"
    if overlay.exists():
        compress_mp4_for_miro(
            in_path=overlay,
            out_path=out_dir / "trajectories_overlay_miro.mp4",
            scale_width=1280,
            fps=24,
            crf=28,
            preset="veryfast",
        )
    else:
        print("Note: outputs/trajectories_overlay.mp4 not found, skipping.")


if __name__ == "__main__":
    main()