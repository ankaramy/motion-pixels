import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
# from config import VIDEO_PATH, OUT_DIR  # No longer needed, CLI args used

def blur_box(frame, xyxy, ksize=31):
    """Simple privacy blur for a bounding box region."""
    x1, y1, x2, y2 = map(int, xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    roi = frame[y1:y2, x1:x2]
    k = max(3, ksize | 1)  # ensure odd >=3
    roi_blur = cv2.GaussianBlur(roi, (k, k), 0)
    frame[y1:y2, x1:x2] = roi_blur
    return frame

def point_in_poly(pt, poly):
    """pt: (x,y), poly: np.array shape (N,2)"""
    return cv2.pointPolygonTest(poly.astype(np.float32), pt, False) >= 0

def main(
    video_path="input_video.mp4",
    out_dir="outputs",
    model_path="yolov8s.pt",
    tracker_cfg="bytetrack_mp.yaml",
    # --- core detection/tracking params ---
    imgsz=1280,
    conf=0.25,
    iou=0.50,
    classes=(0,),        # 0 = person in COCO
    device=0,            # 0 for GPU, "cpu" for CPU
    half=True,           # FP16 on GPU (set False on CPU)
    max_det=200,         # cap detections per frame
    vid_stride=1,        # process every Nth frame (1 = all frames)
    # --- output/visualization ---
    save_video=True,
    save_csv=True,
    draw=True,
    anonymize=False,     # set True to blur each person box
    blur_ksize=31,
    # --- optional ROI filter (only keep tracks in ROI) ---
    use_roi=False,
    roi_polygon=None,    # list of (x,y) points in image coordinates
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video writer
    writer = None
    if save_video:
        out_video = str(out_dir / "tracked.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))

    # ROI polygon
    roi_poly = None
    if use_roi:
        if not roi_polygon or len(roi_polygon) < 3:
            raise ValueError("use_roi=True requires roi_polygon with >= 3 points")
        roi_poly = np.array(roi_polygon, dtype=np.int32)

    model = YOLO(model_path)

    rows = []
    frame_idx = -1

    # We call model.track() per frame with persist=True so IDs stay stable.
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # optional skipping for speed
        if vid_stride > 1 and (frame_idx % vid_stride != 0):
            continue

        results = model.track(
            source=frame,
            persist=True,          # crucial: keep track IDs across frames
            tracker=tracker_cfg,   # ByteTrack config
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=list(classes),
            device=device,
            half=half,
            max_det=max_det,
            verbose=False,
        )

        r = results[0]
        boxes = r.boxes

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            for tid, bb, c in zip(ids, xyxy, confs):
                x1, y1, x2, y2 = bb
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)

                # ROI filter (optional)
                if use_roi and roi_poly is not None:
                    if not point_in_poly((cx, cy), roi_poly):
                        continue

                t_sec = frame_idx / fps
                rows.append({
                    "frame": frame_idx,
                    "time_s": t_sec,
                    "track_id": int(tid),
                    "conf": float(c),
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                    "cx": cx, "cy": cy
                })

                if anonymize:
                    frame = blur_box(frame, (x1, y1, x2, y2), ksize=blur_ksize)

                if draw:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {tid}", (int(x1), int(y1) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw ROI polygon (optional)
        if use_roi and roi_poly is not None and draw:
            cv2.polylines(frame, [roi_poly], isClosed=True, color=(255, 255, 0), thickness=2)

        if writer is not None:
            writer.write(frame)

        # optional preview (press q to quit)
        # cv2.imshow("track", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if save_csv:
        df = pd.DataFrame(rows)
        csv_path = out_dir / "trajectories_image_space.csv"
        df.to_csv(csv_path, index=False)

        # also save a “per-track polyline” summary (useful for diagramming)
        if not df.empty:
            summary = (df.sort_values(["track_id", "frame"])
                         .groupby("track_id")
                         .agg(
                             start_frame=("frame", "min"),
                             end_frame=("frame", "max"),
                             n_obs=("frame", "count"),
                             mean_conf=("conf", "mean")
                         ).reset_index())
            summary.to_csv(out_dir / "tracks_summary.csv", index=False)

    print(f"Done. Processed ~{nframes} frames (stride={vid_stride}). Output in: {out_dir.resolve()}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--model_path", default="yolov8n.pt")
    ap.add_argument("--tracker_cfg", default="bytetrack_mp.yaml")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--max_det", type=int, default=200)
    ap.add_argument("--vid_stride", type=int, default=1)
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--save_csv", action="store_true")
    ap.add_argument("--draw", action="store_true")
    ap.add_argument("--anonymize", action="store_true")
    ap.add_argument("--blur_ksize", type=int, default=31)
    ap.add_argument("--use_roi", action="store_true")
    # For ROI polygon, you may want to add custom parsing if needed
    args = ap.parse_args()

    main(
        video_path=args.video,
        out_dir=args.out_dir,
        model_path=args.model_path,
        tracker_cfg=args.tracker_cfg,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        vid_stride=args.vid_stride,
        save_video=args.save_video,
        save_csv=args.save_csv,
        draw=args.draw,
        anonymize=args.anonymize,
        blur_ksize=args.blur_ksize,
        use_roi=args.use_roi,
        # roi_polygon=args.roi_polygon, # Add if you want to support ROI from CLI
    )
