"""
run_reconstruction.py
---------------------
Dedicated entry point for depth-based reconstruction experiments.

Current backend: Depth Anything (placeholder — inference not yet wired).

Pipeline stages:
  1. Validate input video
  2. Create output folder structure
  3. Extract frames at the requested stride / max-frame limit
  4. [TODO] Run Depth Anything inference on extracted frames
  5. [TODO] Post-process depth maps for spatial / 3-D analysis

Outputs (under --out_dir / reconstruction / <video_stem> /):
  frames/   — extracted RGB frames (PNG)
  depth/    — depth maps (PNG + optional .npy) — populated once inference is wired
  logs/     — run metadata JSON

Usage:
    python run_reconstruction.py --video footage.mp4
    python run_reconstruction.py --video footage.mp4 --frame_stride 5 --max_frames 200 --save_depth
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# Depth Anything placeholder
# ---------------------------------------------------------------------------

def run_depth_anything(frames_dir: Path, depth_dir: Path, args) -> int:
    """
    Placeholder for Depth Anything monocular depth estimation.

    Expected inputs
    ---------------
    frames_dir : Path
        Directory containing extracted RGB frames named frame_XXXXXX.png.
        Frames have already been filtered by --frame_stride and --max_frames.
    depth_dir : Path
        Output directory.  For each input frame <name>.png the inference
        function should write:
          - <name>_depth.png   — 8-bit or 16-bit normalised depth visualisation
          - <name>_depth.npy   — raw float32 depth array (shape H×W, metres or
                                 relative units depending on the model variant)
    args : argparse.Namespace
        Full CLI args, so future options (model size, metric vs relative,
        encoder backbone, etc.) can be threaded through without changing this
        function signature.

    Expected outputs and downstream use
    ------------------------------------
    The depth maps produced here are intended to support several thesis
    experiment branches:

    1. Spatial reference overlay
       Project tracked foot-positions (world_x, world_y from
       calibrate_homography.py) onto the depth canvas to cross-validate
       homography quality: depth discontinuities should align with physical
       boundaries visible in the camera image.

    2. Architectural visualisation
       Per-frame depth maps can be rendered as point clouds
       (open3d.geometry.PointCloud) and composited with the top-view plan
       image for thesis board figures that show both the camera perspective
       and the inferred geometry.

    3. Pseudo-3D reconstruction
       Accumulate depth maps across keyframes (using tracked homographies or
       COLMAP camera poses) to build a dense pseudo-3D model of the monitored
       space.  This acts as a lightweight alternative to full NeRF / Gaussian
       Splatting for quick spatial validation.

    How to wire real inference
    --------------------------
    # TODO (Step A): Install Depth Anything v2
    #   git clone https://github.com/DepthAnything/Depth-Anything-V2
    #   pip install -r Depth-Anything-V2/requirements.txt
    #   Download a checkpoint, e.g. depth_anything_v2_vitb.pth

    # TODO (Step B): Import the model inside this function:
    #   import sys
    #   sys.path.insert(0, "Depth-Anything-V2")
    #   from depth_anything_v2.dpt import DepthAnythingV2
    #   model = DepthAnythingV2(encoder="vitb", features=128, out_channels=[96,192,384,768])
    #   model.load_state_dict(torch.load("depth_anything_v2_vitb.pth", map_location="cpu"))
    #   model.eval()

    # TODO (Step C): Loop over sorted(frames_dir.glob("*.png")), run
    #   depth = model.infer_image(cv2.imread(str(frame_path)))
    #   and save outputs to depth_dir.

    Returns
    -------
    int
        Number of depth maps written (0 while placeholder is active).
    """
    # --- Placeholder logic (safe to run before inference is wired) -----------
    n_frames = len(sorted(frames_dir.glob("*.png")))
    print(f"[reconstruction] Depth Anything backend placeholder ready. "
          f"Frame extraction completed.")
    print(f"[reconstruction] {n_frames} frame(s) available in: {frames_dir}")
    print(f"[reconstruction] Depth output will go to:          {depth_dir}")
    print(f"[reconstruction] Wire inference via the TODO steps in run_depth_anything().")
    return 0


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path: Path, frames_dir: Path,
                   frame_stride: int, max_frames: int) -> list[Path]:
    """
    Read video, save every <frame_stride>-th frame as PNG.
    Stops after <max_frames> frames are saved (0 = no limit).
    Returns list of saved frame paths.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps                = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[reconstruction] Video: {video_path.name}  "
          f"| {total_video_frames} frames  | {fps:.2f} fps")
    print(f"[reconstruction] Stride: every {frame_stride} frame(s)"
          + (f"  |  max {max_frames}" if max_frames > 0 else ""))

    saved: list[Path] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_stride == 0:
            out_path = frames_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(out_path), frame)
            saved.append(out_path)
            if max_frames > 0 and len(saved) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    print(f"[reconstruction] Extracted {len(saved)} frame(s) → {frames_dir}")
    return saved


# ---------------------------------------------------------------------------
# Output folder setup
# ---------------------------------------------------------------------------

def setup_output_dirs(out_dir: Path, video_stem: str) -> tuple[Path, Path, Path, Path]:
    """
    Create and return (run_dir, frames_dir, depth_dir, logs_dir).
    """
    run_dir    = out_dir / "reconstruction" / video_stem
    frames_dir = run_dir / "frames"
    depth_dir  = run_dir / "depth"
    logs_dir   = run_dir / "logs"

    for d in (frames_dir, depth_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    return run_dir, frames_dir, depth_dir, logs_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Depth-based reconstruction entry point (Depth Anything backend)."
    )
    ap.add_argument("--video",        required=True,
                    help="Input video file")
    ap.add_argument("--out_dir",      default="outputs",
                    help="Root output directory (default: outputs/)")
    ap.add_argument("--frame_stride", type=int, default=1,
                    help="Save every N-th frame (default: 1 = all frames)")
    ap.add_argument("--max_frames",   type=int, default=0,
                    help="Stop after saving this many frames; 0 = no limit (default: 0)")
    ap.add_argument("--save_frames",  action="store_true", default=True,
                    help="Extract and save frames (default: True)")
    ap.add_argument("--no_save_frames", dest="save_frames", action="store_false",
                    help="Skip frame extraction (useful when frames already exist)")
    ap.add_argument("--save_depth",   action="store_true",
                    help="Run depth estimation and save depth maps (requires inference to be wired)")
    args = ap.parse_args()

    # --- Validate input ---
    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"[ERROR] Video not found: {video_path}")

    out_dir = Path(args.out_dir)

    if args.frame_stride < 1:
        sys.exit("[ERROR] --frame_stride must be >= 1")

    # --- Create output structure ---
    run_dir, frames_dir, depth_dir, logs_dir = setup_output_dirs(out_dir, video_path.stem)
    print(f"[reconstruction] Output root: {run_dir.resolve()}")

    t_start = time.time()

    # --- Frame extraction ---
    saved_frames: list[Path] = []
    if args.save_frames:
        saved_frames = extract_frames(video_path, frames_dir,
                                      args.frame_stride, args.max_frames)
    else:
        # Re-use frames already on disk (e.g. from a previous run)
        saved_frames = sorted(frames_dir.glob("*.png"))
        print(f"[reconstruction] Skipping extraction — "
              f"{len(saved_frames)} existing frame(s) found in {frames_dir}")

    if not saved_frames:
        print("[WARN] No frames available. Depth estimation skipped.")
        return

    # --- Depth estimation ---
    # TODO: replace the placeholder below with real Depth Anything inference
    #       once the model and checkpoint are available.
    #       Set --save_depth to activate this branch.
    n_depth = 0
    if args.save_depth:
        n_depth = run_depth_anything(frames_dir, depth_dir, args)
    else:
        print(f"[reconstruction] Depth estimation skipped "
              f"(pass --save_depth to activate).")

    # --- Write run log ---
    elapsed = time.time() - t_start
    log = {
        "video":        str(video_path.resolve()),
        "frame_stride": args.frame_stride,
        "max_frames":   args.max_frames,
        "frames_saved": len(saved_frames),
        "depth_saved":  n_depth,
        "elapsed_s":    round(elapsed, 2),
    }
    log_path = logs_dir / "run_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    # --- Summary ---
    print(f"\n[reconstruction] Done in {elapsed:.1f}s")
    print(f"  frames extracted : {len(saved_frames)}")
    print(f"  depth maps saved : {n_depth}")
    print(f"  log              : {log_path.resolve()}")


if __name__ == "__main__":
    main()
