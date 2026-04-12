import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Compute density heatmap video from trajectories (heatmap-only frames).")
    ap.add_argument("--video", required=True, help="Input video path (used for size/fps).")
    ap.add_argument("--traj_csv", default="outputs/trajectories_image_space.csv")
    ap.add_argument("--out_path", default="outputs/metrics/density_heatmap.mp4")
    ap.add_argument("--cell_size", type=int, default=40, help="Grid cell size in pixels.")
    ap.add_argument("--blur", type=int, default=31, help="Gaussian blur kernel (odd).")
    ap.add_argument("--max_count", type=float, default=0.0, help="If >0, fixed normalization max (stabilizes colors).")
    ap.add_argument("--label", default="Density heatmap", help="Text label to draw on the heatmap.")
    ap.add_argument("--static_png", default=None,
                    help="If provided, save a cumulative density heatmap PNG at this path.")
    args = ap.parse_args()

    traj_csv = Path(args.traj_csv)
    if not traj_csv.exists():
        raise FileNotFoundError(f"Missing trajectories CSV: {traj_csv}")

    df = pd.read_csv(traj_csv)
    if df.empty:
        raise ValueError("Trajectory CSV is empty.")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter: {out_path}")

    cell = max(5, int(args.cell_size))
    gw = int(np.ceil(w / cell))
    gh = int(np.ceil(h / cell))

    # Group trajectory points by frame for fast lookup
    df.sort_values(["frame", "track_id"], inplace=True)
    by_frame = df.groupby("frame")

    # Compute normalization max
    if args.max_count and args.max_count > 0:
        norm_max = float(args.max_count)
    else:
        # Estimate a stable max using the 99th percentile of per-frame max cell counts
        per_frame_max = []
        for _, g in by_frame:
            xs = np.clip((g["cx"].to_numpy() // cell).astype(int), 0, gw - 1)
            ys = np.clip((g["cy"].to_numpy() // cell).astype(int), 0, gh - 1)
            grid = np.zeros((gh, gw), dtype=np.float32)
            for x, y in zip(xs, ys):
                grid[y, x] += 1.0
            per_frame_max.append(float(grid.max()) if grid.size else 0.0)

        norm_max = max(1.0, float(np.percentile(per_frame_max, 99))) if per_frame_max else 1.0

    # Rewind video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = -1

    # Ensure blur kernel is odd and >= 3
    k = int(args.blur)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1

    while True:
        ok, _frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # Build grid counts for this frame
        grid = np.zeros((gh, gw), dtype=np.float32)
        if frame_idx in by_frame.groups:
            g = by_frame.get_group(frame_idx)
            xs = np.clip((g["cx"].to_numpy() // cell).astype(int), 0, gw - 1)
            ys = np.clip((g["cy"].to_numpy() // cell).astype(int), 0, gh - 1)
            for x, y in zip(xs, ys):
                grid[y, x] += 1.0

        # Upsample to full resolution
        heat = cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)

        # Smooth
        heat = cv2.GaussianBlur(heat, (k, k), 0)

        # Normalize to [0,1]
        heat_norm = np.clip(heat / norm_max, 0.0, 1.0)
        heat_u8 = (heat_norm * 255).astype(np.uint8)

        # Colorize (JET)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

        # Label
        cv2.putText(
            heat_color,
            f"{args.label} (cell={cell}px)",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Write heatmap-only frame
        writer.write(heat_color)

    cap.release()
    writer.release()
    print("[compute_heatmap] Saved:", out_path)

    # --- Optional: cumulative static heatmap PNG ---
    if args.static_png:
        # Accumulate all trajectory points across every frame into one grid
        cumulative = np.zeros((gh, gw), dtype=np.float32)
        xs = np.clip((df["cx"].to_numpy() // cell).astype(int), 0, gw - 1)
        ys = np.clip((df["cy"].to_numpy() // cell).astype(int), 0, gh - 1)
        for x, y in zip(xs, ys):
            cumulative[y, x] += 1.0

        # Upsample, blur, normalise — same pipeline as per-frame video
        heat = cv2.resize(cumulative, (w, h), interpolation=cv2.INTER_LINEAR)
        heat = cv2.GaussianBlur(heat, (k, k), 0)
        cum_max = max(1.0, float(heat.max()))
        heat_norm = np.clip(heat / cum_max, 0.0, 1.0)
        heat_u8 = (heat_norm * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

        png_path = Path(args.static_png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(png_path), heat_color)
        print("[compute_heatmap] Static PNG saved:", png_path)


if __name__ == "__main__":
    main()