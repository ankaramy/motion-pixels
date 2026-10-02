import cv2
import time
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
# from config import VIDEO_PATH, OUT_DIR  # No longer needed, CLI args used

def blur_box(frame, xyxy, ksize=31):
    """Simple privacy  blur for a bounding box region."""
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


def rotation_from_meta(cap):
    """
    Decide how to rotate frames upright for YOLO based on the video's rotation
    metadata. Returns a cv2.ROTATE_* code, or None for no rotation.

    OpenCV here returns RAW (un-rotated) frames (CAP_PROP_ORIENTATION_AUTO=0),
    so phone videos with a rotation flag arrive sideways. We rotate only for
    inference; detections are mapped back to raw coords (see map_box_back), so
    exported coordinates stay in the original raw OpenCV frame space and remain
    consistent with calib.json. Validated for 90 (clockwise) on this dataset.
    """
    try:
        if cap.get(cv2.CAP_PROP_ORIENTATION_AUTO) >= 1:
            return None  # OpenCV already auto-rotates -> frames are upright already
        meta = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    except Exception:
        return None
    angle = int(round(meta)) % 360 if meta else 0
    return {
        90:  cv2.ROTATE_90_CLOCKWISE,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
    }.get(angle)


def map_point_back(x, y, rot, w_raw, h_raw):
    """Map a point from the rotated (upright) frame back to raw-frame coords."""
    if rot == cv2.ROTATE_90_CLOCKWISE:
        return y, (h_raw - 1) - x
    if rot == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return (w_raw - 1) - y, x
    if rot == cv2.ROTATE_180:
        return (w_raw - 1) - x, (h_raw - 1) - y
    return x, y


def map_box_back(x1, y1, x2, y2, rot, w_raw, h_raw):
    """
    Map a bbox from rotated (upright) coords back to raw-frame coords.
    Returns axis-aligned (x1,y1,x2,y2) plus the footpoint (bottom-center in the
    UPRIGHT frame = true ground contact) mapped into raw coords.
    """
    foot_x, foot_y = map_point_back((x1 + x2) / 2.0, y2, rot, w_raw, h_raw)
    pts = [map_point_back(px, py, rot, w_raw, h_raw)
           for px, py in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys), foot_x, foot_y


def sanitize_fps(fps, default=30.0, lo=1.0, hi=120.0):
    """
    Clamp absurd/invalid capture FPS to a value the mp4v codec can encode.
    Some phone clips report e.g. 240 fps (slow-mo) whose timebase the MPEG-4
    container rejects, causing a silent 0-byte VideoWriter. Used ONLY for the
    output video playback rate — never for time_s / speed math (those keep the
    real capture fps so temporal metrics stay correct).
    """
    try:
        f = float(fps)
    except (TypeError, ValueError):
        return default
    if not (f == f) or f <= lo or f > hi:   # NaN or out of sane range
        return default
    return f


def open_video_writer(path, fps, size):
    """
    Open a cv2.VideoWriter robustly: sanitize fps, try mp4v then avc1, and
    fail loudly rather than return a silent non-functional (0-byte) writer.
    """
    wfps = sanitize_fps(fps)
    if abs(wfps - float(fps if fps else 0)) > 1e-3:
        print(f"[writer] capture fps={fps} is out of range -> using {wfps} fps for output video")
    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), wfps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(
        f"VideoWriter failed to initialize for {path} "
        f"(fps={fps} -> {wfps}, size={size}). Refusing to emit a 0-byte file.")


def main(
    video_path="input_video.mp4",
    out_dir="outputs",
    model_path="yolov8s.pt",
    tracker_cfg="bytetrack_mp.yaml",
    auto_orient=True,    # rotate frames upright for inference, export raw coords
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

    # Orientation-aware inference: rotate frames upright for YOLO, but keep all
    # exported coordinates in the raw (w x h) OpenCV space so calib.json stays valid.
    rot = rotation_from_meta(cap) if auto_orient else None
    if rot is not None:
        print(f"[orient] rotation metadata detected -> frames rotated upright for "
              f"inference; detections mapped back to raw {w}x{h} coordinates")

    # Video writer (fps sanitized; fails loudly instead of writing a 0-byte file)
    writer = None
    if save_video:
        out_video = str(out_dir / "tracked.mp4")
        writer = open_video_writer(out_video, fps, (w, h))

    # ROI polygon
    roi_poly = None
    if use_roi:
        if not roi_polygon or len(roi_polygon) < 3:
            raise ValueError("use_roi=True requires roi_polygon with >= 3 points")
        roi_poly = np.array(roi_polygon, dtype=np.int32)

    model = YOLO(model_path)

    rows = []
    frame_idx = -1

    # Progress logging state (lightweight, throttled — see periodic block below)
    t_start = time.time()
    last_log = t_start
    seen_ids = set()
    print(f"[track] starting: ~{nframes} frames, fps={fps:.3f}, imgsz={imgsz}, conf={conf}", flush=True)

    # We call model.track() per frame with persist=True so IDs stay stable.
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # optional skipping for speed
        if vid_stride > 1 and (frame_idx % vid_stride != 0):
            continue

        # Rotate upright for inference only; raw `frame` is kept for output/draw.
        infer_frame = cv2.rotate(frame, rot) if rot is not None else frame

        results = model.track(
            source=infer_frame,
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
                # Map detection from upright-inference space back to raw coords.
                if rot is not None:
                    x1, y1, x2, y2, foot_x, foot_y = map_box_back(
                        float(x1), float(y1), float(x2), float(y2), rot, w, h)
                else:
                    foot_x = float((x1 + x2) / 2.0)
                    foot_y = float(y2)
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)

                # ROI filter (optional) — uses footpoint (ground contact) not bbox center
                if use_roi and roi_poly is not None:
                    if not point_in_poly((foot_x, foot_y), roi_poly):
                        continue

                t_sec = frame_idx / fps
                rows.append({
                    "frame": frame_idx,
                    "time_s": t_sec,
                    "track_id": int(tid),
                    "conf": float(c),
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                    "cx": cx, "cy": cy,
                    "foot_x": foot_x, "foot_y": foot_y,
                })
                seen_ids.add(int(tid))

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

        # Throttled progress logging (every ~15s): frame, rate, ETA, track count
        now = time.time()
        if now - last_log >= 15.0:
            elapsed = now - t_start
            done = frame_idx + 1
            rate = done / elapsed if elapsed > 0 else 0.0
            eta_min = ((nframes - done) / rate / 60.0) if (rate > 0 and nframes > 0) else float("nan")
            pct = (100.0 * done / nframes) if nframes > 0 else 0.0
            print(f"[track] frame {done}/{nframes} ({pct:.1f}%)  "
                  f"elapsed {elapsed/60:.1f}min  rate {rate:.1f} fps  "
                  f"eta {eta_min:.1f}min  tracks {len(seen_ids)}", flush=True)
            last_log = now

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
    ap.add_argument("--auto_orient", dest="auto_orient", action="store_true", default=True,
                    help="Rotate frames upright for inference using rotation metadata; "
                         "detections are mapped back to raw coords (default: on)")
    ap.add_argument("--no_auto_orient", dest="auto_orient", action="store_false",
                    help="Disable orientation-aware inference (feed raw frames to YOLO)")
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
        auto_orient=args.auto_orient,
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
