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


def blur_face_region(frame, x1, y1, x2, y2,
                     height_ratio: float, width_ratio: float,
                     y_offset_ratio: float, ksize: int):
    """
    Blur an estimated face ROI within a person bounding box.
    ROI is centred horizontally and placed near the top of the box.
    Applied twice for stronger de-identification.
    All coordinates are clamped to image bounds.
    """
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return
    # Horizontal: centred within bbox
    cx   = (x1 + x2) / 2.0
    rfx1 = max(0,              int(cx - bw * width_ratio / 2))
    rfx2 = min(frame.shape[1], int(cx + bw * width_ratio / 2))
    # Vertical: offset from top, then height_ratio tall
    rfy1 = max(0,              int(y1 + bh * y_offset_ratio))
    rfy2 = min(frame.shape[0], int(y1 + bh * (y_offset_ratio + height_ratio)))
    if rfx2 <= rfx1 or rfy2 <= rfy1:
        return
    k   = max(3, ksize | 1)   # ensure odd >= 3
    roi = frame[rfy1:rfy2, rfx1:rfx2]
    roi = cv2.GaussianBlur(roi, (k, k), 0)
    roi = cv2.GaussianBlur(roi, (k, k), 0)   # second pass for stronger anonymisation
    frame[rfy1:rfy2, rfx1:rfx2] = roi


def in_dwell_interval(dwell_lookup, tid: int, t_s: float) -> bool:
    intervals = dwell_lookup.get(tid)
    if not intervals:
        return False
    for (a, b) in intervals:
        if a <= t_s <= b:
            return True
    return False


# ---------------------------------------------------------------------------
# Metric overlay helpers (new)
# ---------------------------------------------------------------------------

def load_csv_safe(path) -> pd.DataFrame:
    """Load a CSV file. Returns an empty DataFrame if missing, empty, or unreadable."""
    if path is None:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def infer_cell_size_px(df: pd.DataFrame, col: str = "cell_x") -> int:
    """
    Estimate cell size in pixels from the spacing between sorted cell centres.
    Falls back to 60 if the column is missing or there is only one unique value.
    """
    if col not in df.columns:
        return 60
    vals = np.sort(df[col].unique())
    if len(vals) < 2:
        return 60
    return max(1, int(np.diff(vals).min()))


def draw_bottleneck_overlay(frame: np.ndarray,
                             cells_df: pd.DataFrame,
                             top_n: int = 5,
                             alpha: float = 0.25) -> np.ndarray:
    """
    Draw the top-N bottleneck cells as semi-transparent orange rectangles
    with a 'B=score' label.
    Expects columns: cell_x, cell_y, bottleneck_score (pixel-space coordinates).
    """
    needed = {"cell_x", "cell_y", "bottleneck_score"}
    if cells_df.empty or not needed.issubset(cells_df.columns):
        return frame

    top = cells_df.nlargest(top_n, "bottleneck_score")
    cell_size = infer_cell_size_px(cells_df)
    half = cell_size // 2

    overlay = frame.copy()
    color_fill    = (0, 100, 255)   # orange (BGR)
    color_outline = (0, 80, 200)

    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        cx_px = int(row["cell_x"])
        cy_px = int(row["cell_y"])

        x0, y0 = cx_px - half, cy_px - half
        x1, y1 = cx_px + half, cy_px + half

        cv2.rectangle(overlay, (x0, y0), (x1, y1), color_fill, -1)      # filled
        cv2.rectangle(frame,   (x0, y0), (x1, y1), color_outline, 1)    # outline

        cv2.putText(frame, f"#{rank}",
                    (x0 + 3, y0 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Blend fill semi-transparently
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    return frame


def draw_linger_overlay(frame: np.ndarray, zones_df: pd.DataFrame) -> np.ndarray:
    """
    Draw linger zone centroids as labelled magenta circles.
    Expects columns: zone_id, centroid_x, centroid_y.
    mean_duration_s is shown if present.
    Coordinates must be in pixel space.
    """
    needed = {"zone_id", "centroid_x", "centroid_y"}
    if zones_df.empty or not needed.issubset(zones_df.columns):
        return frame

    color = (200, 0, 200)   # magenta (BGR)

    for _, row in zones_df.iterrows():
        cx_px = int(row["centroid_x"])
        cy_px = int(row["centroid_y"])
        zid   = int(row["zone_id"])

        # Inner semi-transparent fill for visual weight
        ov = frame.copy()
        cv2.circle(ov, (cx_px, cy_px), 18, color, -1)
        cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)
        # Double-ring outline — stronger than a dwell ring
        cv2.circle(frame, (cx_px, cy_px), 20, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx_px, cy_px), 14, color, 1, cv2.LINE_AA)

        label = f"L{zid}"
        if "mean_duration_s" in row and not pd.isna(row["mean_duration_s"]):
            label += f" {row['mean_duration_s']:.1f}s"

        cv2.putText(frame, label,
                    (cx_px + 15, cy_px + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(frame, label,
                    (cx_px + 14, cy_px + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    return frame


def draw_flow_overlay(frame: np.ndarray,
                       flow_df: pd.DataFrame,
                       sample_n: int = 40) -> np.ndarray:
    """
    Draw a sampled subset of flow vectors as cyan arrows.
    Expects columns: cell_x, cell_y, mean_dx, mean_dy, mean_speed.
    Coordinates and displacements must be in pixel space.
    Arrow length is normalised so the fastest cell maps to ~40px.
    """
    needed = {"cell_x", "cell_y", "mean_dx", "mean_dy"}
    if flow_df.empty or not needed.issubset(flow_df.columns):
        return frame

    # Sample evenly across the available rows
    df = flow_df.sample(n=min(sample_n, len(flow_df)), random_state=0)

    color = (255, 220, 0)   # cyan-yellow (BGR) — visible on dark and light backgrounds

    # Normalise arrow length: longest arrow = 40 px
    speed_col = "mean_speed" if "mean_speed" in df.columns else None
    if speed_col:
        speed_max = df[speed_col].max()
        speed_max = speed_max if speed_max > 0 else 1.0
    else:
        mag = np.sqrt(df["mean_dx"] ** 2 + df["mean_dy"] ** 2)
        speed_max = mag.max() if mag.max() > 0 else 1.0

    target_len = 40.0   # pixels

    for _, row in df.iterrows():
        x0 = int(row["cell_x"])
        y0 = int(row["cell_y"])
        dx = float(row["mean_dx"])
        dy = float(row["mean_dy"])

        mag = np.sqrt(dx * dx + dy * dy)
        if mag == 0:
            continue

        spd = float(row[speed_col]) if speed_col else mag
        arrow_len = (spd / speed_max) * target_len

        ux = dx / mag
        uy = dy / mag
        x1 = int(x0 + ux * arrow_len)
        y1 = int(y0 + uy * arrow_len)

        cv2.arrowedLine(frame, (x0, y0), (x1, y1),
                        color, 1, cv2.LINE_AA, tipLength=0.35)

    return frame


def draw_metrics_legend(frame: np.ndarray,
                         n_bottleneck: int,
                         n_linger: int,
                         n_flow: int) -> np.ndarray:
    """
    Compact legend in the bottom-right corner summarising overlay counts.
    """
    lines = [
        "Overlays",
        f"  Bottleneck: {n_bottleneck}",
        f"  Linger:     {n_linger}",
        f"  Flow:       {n_flow}",
    ]
    h_frame, w_frame = frame.shape[:2]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness  = 1
    line_h     = 18
    pad        = 8

    # Measure widest line
    text_w = max(cv2.getTextSize(l, font, font_scale, thickness)[0][0] for l in lines)
    box_h  = line_h * len(lines) + pad * 2
    box_w  = text_w + pad * 2

    x0 = w_frame - box_w - 16
    y0 = h_frame - box_h - 16

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    for i, line in enumerate(lines):
        ty = y0 + pad + (i + 1) * line_h - 4
        cv2.putText(frame, line, (x0 + pad, ty),
                    font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)

    return frame


def _draw_speed_legend(frame: np.ndarray, panel_opacity: float, font_scale: float) -> None:
    """Compact semi-transparent speed legend, bottom-left corner."""
    items = [
        ("head",   None,            "Speed"),
        ("fill",   (0,   255,   0), "Fast"),
        ("fill",   (0,   255, 255), "Medium"),
        ("fill",   (0,     0, 255), "Slow"),
        ("cross",  (230, 230, 230), "Paused"),
        ("ring",   (0,  165, 255),  "Dwell"),
        ("arrow",  (0,  200, 255),  "Direction change"),
        ("linger", (200,   0, 200), "Linger zone"),
    ]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    pad    = 12
    dot_r  = 6
    line_h = 24
    box_h  = pad * 2 + line_h * len(items)
    box_w  = max(
        cv2.getTextSize(label, font, font_scale, 1)[0][0]
        for _, _, label in items
    ) + pad * 2 + dot_r * 2 + 8
    x0, y0 = 16, frame.shape[0] - box_h - 16

    ov = frame.copy()
    cv2.rectangle(ov, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(ov, panel_opacity, frame, 1.0 - panel_opacity, 0, frame)

    for i, (style, col, label) in enumerate(items):
        cy = y0 + pad + i * line_h + line_h // 2
        cx = x0 + pad + dot_r
        if style == "head":
            cv2.putText(frame, label, (x0 + pad, cy + 4),
                        font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            continue
        elif style == "fill":
            cv2.circle(frame, (cx, cy), dot_r, col, -1, cv2.LINE_AA)
        elif style == "cross":
            cv2.line(frame, (cx - dot_r, cy), (cx + dot_r, cy), col, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy - dot_r), (cx, cy + dot_r), col, 1, cv2.LINE_AA)
        elif style == "ring":
            cv2.circle(frame, (cx, cy), dot_r, col, 1, cv2.LINE_AA)
        elif style == "arrow":
            # Mini bent-arrow matching the on-frame marker
            cv2.arrowedLine(frame, (cx - dot_r, cy), (cx, cy),     col, 1, cv2.LINE_AA, tipLength=0.0)
            cv2.arrowedLine(frame, (cx, cy),          (cx, cy - dot_r), col, 1, cv2.LINE_AA, tipLength=0.5)
        elif style == "linger":
            cv2.circle(frame, (cx, cy), dot_r,     col, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), dot_r + 3, col, 1, cv2.LINE_AA)
        cv2.putText(frame, label, (cx + dot_r + 5, cy + 4),
                    font, font_scale, (210, 210, 210), 1, cv2.LINE_AA)


def _draw_info_panel(frame: np.ndarray, n_active: int, n_paused: int,
                     n_dwelling: int, n_shifts: int, n_linger_zones: int,
                     panel_opacity: float, font_scale: float) -> None:
    """Live counter panel, top-right corner."""
    rows = [
        ("Active",       n_active),
        ("Paused",       n_paused),
        ("Dwelling",     n_dwelling),
        ("Dir. shifts",  n_shifts),
        ("Linger zones", n_linger_zones),
    ]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    pad    = 12
    line_h = 24
    box_h  = pad * 2 + line_h * len(rows)

    # Size box to widest rendered string
    box_w = max(
        cv2.getTextSize(f"{lbl}:  {val}", font, font_scale, 1)[0][0]
        for lbl, val in rows
    ) + pad * 2

    x0 = frame.shape[1] - box_w - 16
    y0 = 16

    ov = frame.copy()
    cv2.rectangle(ov, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(ov, panel_opacity, frame, 1.0 - panel_opacity, 0, frame)

    for i, (lbl, val) in enumerate(rows):
        ty = y0 + pad + i * line_h + line_h // 2 + 4
        cv2.putText(frame, f"{lbl}:  {val}", (x0 + pad, ty),
                    font, font_scale, (210, 210, 210), 1, cv2.LINE_AA)


def _draw_hud(frame: np.ndarray, n_active: int, avg_speed, max_speed,
              n_stops: int, n_dwellers: int, unit: str,
              panel_opacity: float, font_scale: float) -> None:
    """Compact global-metrics HUD panel, top-left corner."""
    def _fmt(v):
        return f"{v:.2f} {unit}" if (v is not None and not np.isnan(v)) else "n/a"

    rows = [
        ("Active",   str(n_active)),
        ("Avg spd",  _fmt(avg_speed)),
        ("Max spd",  _fmt(max_speed)),
        ("Stops",    str(n_stops)),
        ("Dwellers", str(n_dwellers)),
    ]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    pad    = 8
    line_h = 20
    box_h  = pad * 2 + line_h * len(rows)
    box_w  = max(
        cv2.getTextSize(f"{lbl}: {val}", font, font_scale, 1)[0][0]
        for lbl, val in rows
    ) + pad * 2

    x0, y0 = 16, 16
    ov = frame.copy()
    cv2.rectangle(ov, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(ov, panel_opacity, frame, 1.0 - panel_opacity, 0, frame)

    for i, (lbl, val) in enumerate(rows):
        ty = y0 + pad + i * line_h + line_h // 2 + 4
        cv2.putText(frame, f"{lbl}: {val}", (x0 + pad, ty),
                    font, font_scale, (210, 210, 210), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    video_path,
    traj_csv="outputs/trajectories_image_space.csv",
    out_dir="outputs",
    max_tracks_draw=9999,
    min_track_len=15,
    draw_tail=90,
    tail_fade=5,
    thickness=2,
    current_marker_size=5,
    current_marker_outline=(20, 20, 20),
    marker_size=4,
    panel_opacity=0.70,
    font_scale=0.42,
    # metrics
    speed_obs_csv="outputs/metrics/speed_per_observation.csv",
    stop_flags_csv="outputs/metrics/stop_flags_per_observation.csv",
    dwell_events_csv="outputs/metrics/dwell_events.csv",
    shift_flags_csv="outputs/metrics/shift_flags_per_observation.csv",
    stop_speed_px_s=25.0,
    mid_speed_px_s=80.0,
    show_legend=True,
    # heatmap
    heatmap_video=None,
    heatmap_alpha=0.20,
    use_heatmap=True,
    # spatial metric overlays (new)
    flow_csv=None,
    bottleneck_csv=None,
    linger_csv=None,
    show_metrics=False,
    # face blur
    blur_faces=False,
    face_height_ratio=0.32,
    face_width_ratio=0.65,
    face_y_offset_ratio=0.05,
    blur_kernel=51,
    # selective agent labels
    highlight_top_k_speed=0,
    highlight_top_k_dwell=0,
    # live HUD
    show_hud=False,
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

    speed_col = None
    unit = "px/s"
    if speed_obs_csv.exists():
        sp = pd.read_csv(speed_obs_csv)
        if "speed_smooth_px_s" in sp.columns:
            speed_col = "speed_smooth_px_s"
            unit = "px/s"
        elif "speed_smooth_m_s" in sp.columns:
            speed_col = "speed_smooth_m_s"
            unit = "m/s"
        elif "speed_m_s" in sp.columns:
            speed_col = "speed_m_s"
            unit = "m/s"
        if speed_col:
            df = df.merge(
                sp[["frame", "track_id", speed_col]],
                on=["frame", "track_id"],
                how="left",
            )
            print(f"[draw_trajectories] Using speed column: {speed_col} ({unit})")

            # Adaptive thresholds: use actual speed distribution so the
            # red/yellow/green split is meaningful regardless of coordinate scale.
            if unit == "m/s":
                speeds_all = df[speed_col].dropna().to_numpy()
                if len(speeds_all) >= 10:
                    p20 = float(np.percentile(speeds_all, 20))
                    p65 = float(np.percentile(speeds_all, 65))
                    p95 = float(np.percentile(speeds_all, 95))
                    stop_speed_px_s = p20
                    mid_speed_px_s  = p65
                    print(f"[draw_trajectories] Adaptive thresholds (m/s): "
                          f"stop={stop_speed_px_s:.3f}  mid={mid_speed_px_s:.3f}  p95={p95:.3f}")
        else:
            print(f"[draw_trajectories] WARN: no speed column found in {speed_obs_csv} -> drawing in grey")
    else:
        print(f"[draw_trajectories] WARN: missing {speed_obs_csv} -> drawing in grey")

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

    shift_flags_csv = Path(shift_flags_csv)
    if shift_flags_csv.exists():
        shf = pd.read_csv(shift_flags_csv)
        if "is_shift" in shf.columns:
            df = df.merge(
                shf[["frame", "track_id", "is_shift"]],
                on=["frame", "track_id"],
                how="left",
            )
        else:
            df["is_shift"] = 0
    else:
        df["is_shift"] = 0

    dwell_lookup = build_dwell_lookup(dwell_events_csv)

    # --- Compute per-track highlight sets (Parts 1 & 2) ---
    highlight_speed_ids: set = set()
    highlight_dwell_ids: set = set()

    if highlight_top_k_speed > 0 and speed_col and speed_col in df.columns:
        avg_sp = df.groupby("track_id")[speed_col].mean().dropna()
        highlight_speed_ids = set(int(t) for t in avg_sp.nlargest(highlight_top_k_speed).index)

    if highlight_top_k_dwell > 0 and dwell_lookup:
        dwell_totals = {
            tid: sum(b - a for a, b in intervals)
            for tid, intervals in dwell_lookup.items()
        }
        sorted_dwell = sorted(dwell_totals.items(), key=lambda x: x[1], reverse=True)
        highlight_dwell_ids = set(int(t) for t, _ in sorted_dwell[:highlight_top_k_dwell])

    # --- Load spatial metric CSVs (new) ---
    flow_df       = load_csv_safe(flow_csv)       if show_metrics else pd.DataFrame()
    bottleneck_df = load_csv_safe(bottleneck_csv) if show_metrics else pd.DataFrame()
    linger_df     = load_csv_safe(linger_csv)     if show_metrics else pd.DataFrame()

    if show_metrics:
        def _warn_missing(name, path):
            if path and not Path(path).exists():
                print(f"[draw_trajectories] WARNING: {name} not found at {path} — overlay skipped.")
        _warn_missing("flow_csv",       flow_csv)
        _warn_missing("bottleneck_csv", bottleneck_csv)
        _warn_missing("linger_csv",     linger_csv)

    n_bottleneck_shown = min(5,         len(bottleneck_df)) if not bottleneck_df.empty else 0
    n_linger_shown     = len(linger_df) if not linger_df.empty else 0
    n_flow_shown       = min(40,        len(flow_df))       if not flow_df.empty else 0

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
        if use_heatmap and heatmap_video:
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
    last_shift = {}
    last_foot = {}
    last_bbox = {}   # (x1, y1, x2, y2) in camera pixel coords, for highlighted pedestrians

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
                if "foot_x" in row.index and "foot_y" in row.index:
                    _fx, _fy = row.get("foot_x"), row.get("foot_y")
                    pt = (int(_fx), int(_fy)) if not (pd.isna(_fx) or pd.isna(_fy)) \
                         else (int(row["cx"]), int(row["cy"]))
                else:
                    pt = (int(row["cx"]), int(row["cy"]))
                history[tid].append(pt)
                if len(history[tid]) > draw_tail:
                    history[tid] = history[tid][-draw_tail:]

                last_speed[tid] = row.get(speed_col, np.nan) if speed_col else np.nan
                is_stop = row.get("is_stop", 0)
                try:
                    last_stop[tid] = int(is_stop) if not pd.isna(is_stop) else 0
                except Exception:
                    last_stop[tid] = 0
                is_shift = row.get("is_shift", 0)
                try:
                    last_shift[tid] = int(is_shift) if not pd.isna(is_shift) else 0
                except Exception:
                    last_shift[tid] = 0

                if "foot_x" in row.index and "foot_y" in row.index:
                    fx, fy = row.get("foot_x"), row.get("foot_y")
                    if not (pd.isna(fx) or pd.isna(fy)):
                        last_foot[tid] = (int(fx), int(fy))

                # Keep the most recent bbox for each track (used by highlight drawing below)
                if all(c in row.index for c in ("x1", "y1", "x2", "y2")):
                    bx1, by1, bx2, by2 = row["x1"], row["y1"], row["x2"], row["y2"]
                    if not any(pd.isna(v) for v in (bx1, by1, bx2, by2)):
                        last_bbox[tid] = (int(bx1), int(by1), int(bx2), int(by2))

                # Blur upper bbox region as face approximation
                if blur_faces and all(c in row.index for c in ("x1", "y1", "x2", "y2")):
                    blur_face_region(frame,
                                     row["x1"], row["y1"], row["x2"], row["y2"],
                                     face_height_ratio, face_width_ratio,
                                     face_y_offset_ratio, blur_kernel)

        if show_legend:
            _draw_speed_legend(frame, panel_opacity, font_scale)

        n_shifts = 0
        drawn = 0
        for tid, pts in history.items():
            if drawn >= max_tracks_draw:
                break
            if len(pts) < 2:
                continue

            speed = last_speed.get(tid, np.nan)
            color = speed_to_color_bgr(speed, stop_thr=stop_speed_px_s, mid_thr=mid_speed_px_s)

            # Temporal fading: tail_fade steps, oldest→dim, newest→full brightness
            n = len(pts)
            n_steps = min(tail_fade, n - 1)
            for s in range(n_steps):
                i0 = int(s       * (n - 1) / n_steps)
                i1 = int((s + 1) * (n - 1) / n_steps)
                seg = pts[i0:i1 + 1]
                if len(seg) < 2:
                    continue
                a = 1.0 if n_steps == 1 else 0.18 + 0.82 * s / (n_steps - 1)
                seg_color = tuple(int(c * a) for c in color)
                seg_thick = max(1, round(thickness * (0.5 + 0.5 * a)))
                cv2.polylines(frame, [np.array(seg, dtype=np.int32)],
                              False, seg_color, seg_thick, cv2.LINE_AA)

            head = pts[-1]

            # Paused: white crosshair with dark shadow (readable on light and dark backgrounds)
            if last_stop.get(tid, 0) == 1:
                arm = marker_size + 5
                cv2.line(frame, (head[0] - arm, head[1]), (head[0] + arm, head[1]),
                         (30, 30, 30), 3, cv2.LINE_AA)
                cv2.line(frame, (head[0], head[1] - arm), (head[0], head[1] + arm),
                         (30, 30, 30), 3, cv2.LINE_AA)
                cv2.line(frame, (head[0] - arm, head[1]), (head[0] + arm, head[1]),
                         (230, 230, 230), 1, cv2.LINE_AA)
                cv2.line(frame, (head[0], head[1] - arm), (head[0], head[1] + arm),
                         (230, 230, 230), 1, cv2.LINE_AA)

            # Dwelling: small amber ring — subtle "staying in place" indicator
            if in_dwell_interval(dwell_lookup, tid, t_s):
                cv2.circle(frame, head, marker_size + 8, (20, 20, 20),  2, cv2.LINE_AA)
                cv2.circle(frame, head, marker_size + 8, (0, 165, 255), 1, cv2.LINE_AA)

            # Direction change: bent-arrow showing incoming and outgoing heading
            if last_shift.get(tid, 0) == 1 and len(pts) >= 2:
                v2x = pts[-1][0] - pts[-2][0]
                v2y = pts[-1][1] - pts[-2][1]
                m2 = (v2x**2 + v2y**2) ** 0.5
                if m2 > 0:
                    n_shifts += 1
                    ux, uy = v2x / m2, v2y / m2
                    arm = marker_size + 9
                    p_back = (int(head[0] - ux * arm), int(head[1] - uy * arm))
                    px_, py_ = -uy, ux   # 90° left as visual proxy for "turned"
                    p_out  = (int(head[0] + px_ * arm), int(head[1] + py_ * arm))
                    cv2.arrowedLine(frame, p_back, head,  (20, 20, 20),  3, cv2.LINE_AA, tipLength=0.0)
                    cv2.arrowedLine(frame, head,   p_out, (20, 20, 20),  3, cv2.LINE_AA, tipLength=0.4)
                    cv2.arrowedLine(frame, p_back, head,  (0, 200, 255), 1, cv2.LINE_AA, tipLength=0.0)
                    cv2.arrowedLine(frame, head,   p_out, (0, 200, 255), 1, cv2.LINE_AA, tipLength=0.4)

            drawn += 1

        # --- Second pass: foot markers on top of all trails ---
        foot_drawn = 0
        for tid, pts in history.items():
            if foot_drawn >= max_tracks_draw:
                break
            if not pts:
                continue
            foot = last_foot.get(tid, pts[-1])
            marker_color = speed_to_color_bgr(
                last_speed.get(tid, np.nan),
                stop_thr=stop_speed_px_s, mid_thr=mid_speed_px_s,
            )
            cv2.circle(frame, foot, current_marker_size, marker_color,           -1, cv2.LINE_AA)
            cv2.circle(frame, foot, current_marker_size, current_marker_outline,   1, cv2.LINE_AA)

            # Selective agent label (Part 1)
            if tid in highlight_speed_ids or tid in highlight_dwell_ids:
                spd = last_speed.get(tid, np.nan)
                tag_parts = [f"#{tid}"]
                if tid in highlight_speed_ids:
                    tag_parts.append(f"spd:{'?' if np.isnan(spd) else f'{spd:.1f}'}")
                if tid in highlight_dwell_ids:
                    dw_total = sum(b - a for a, b in dwell_lookup.get(tid, []))
                    tag_parts.append(f"dw:{dw_total:.0f}s")
                label = " ".join(tag_parts)
                lx = foot[0] + current_marker_size + 4
                ly = foot[1] - current_marker_size - 4
                cv2.putText(frame, label, (lx + 1, ly + 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 2, cv2.LINE_AA)
                cv2.putText(frame, label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (240, 240, 240), 1, cv2.LINE_AA)

                # Green tracking box for highlighted pedestrians only
                if tid in last_bbox:
                    bx1, by1, bx2, by2 = last_bbox[tid]
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2, cv2.LINE_AA)

            foot_drawn += 1

        # --- Spatial metric overlays (new) ---
        if show_metrics:
            if not bottleneck_df.empty:
                frame = draw_bottleneck_overlay(frame, bottleneck_df, top_n=5)
            if not linger_df.empty:
                frame = draw_linger_overlay(frame, linger_df)
            if not flow_df.empty:
                frame = draw_flow_overlay(frame, flow_df, sample_n=40)
            draw_metrics_legend(frame, n_bottleneck_shown, n_linger_shown, n_flow_shown)

        # Info panel drawn after trajectory loop so n_shifts is fully counted
        n_active_now   = sum(1 for p in history.values() if p)
        n_paused_now   = sum(1 for tid in history if last_stop.get(tid, 0) == 1 and history[tid])
        n_dwelling_now = sum(1 for tid in history
                             if history[tid] and in_dwell_interval(dwell_lookup, tid, t_s))
        n_linger_z = len(linger_df) if not linger_df.empty else 0

        if show_legend:
            _draw_info_panel(frame, n_active_now, n_paused_now, n_dwelling_now, n_shifts,
                             n_linger_z, panel_opacity, font_scale)

        # Live HUD (Part 2)
        if show_hud:
            active_speeds = [last_speed.get(tid, np.nan) for tid in history if history[tid]]
            valid_speeds  = [s for s in active_speeds if not np.isnan(s)]
            avg_spd = float(np.mean(valid_speeds)) if valid_speeds else float("nan")
            max_spd = float(np.max(valid_speeds))  if valid_speeds else float("nan")
            _draw_hud(frame, n_active_now, avg_spd, max_spd,
                      n_paused_now, n_dwelling_now, unit,
                      panel_opacity, font_scale)

        writer.write(frame)

    cap.release()
    if heat_cap is not None:
        heat_cap.release()
    writer.release()

    if first_frame_img is None:
        raise RuntimeError("No frames read; cannot write trajectories_static.png")

    static = first_frame_img.copy()
    mean_speed_by_tid = (
        df.groupby("track_id")[speed_col].mean()
        if speed_col and speed_col in df.columns
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
    ap.add_argument("--tail_length", type=int, default=None,
                    help="Number of recent positions kept per track (default: 90)")
    ap.add_argument("--tail_fade",   type=int, default=5,
                    help="Number of brightness steps along the tail (default: 5)")
    ap.add_argument("--draw_tail",   type=int, default=90,
                    help=argparse.SUPPRESS)  # backward-compat alias for --tail_length
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--current_marker_size",  type=int,   default=5,
                    help="Radius of the foot-position dot in pixels (default: 5)")
    ap.add_argument("--current_marker_outline", type=str, default="20,20,20",
                    help="BGR outline color of the foot-position dot as 'B,G,R' (default: 20,20,20)")
    ap.add_argument("--marker_size",   type=int,   default=4,
                    help="Radius of head-position dot in pixels (default: 4)")
    ap.add_argument("--panel_opacity", type=float, default=0.70,
                    help="Opacity of semi-transparent legend panel (default: 0.6)")
    ap.add_argument("--font_scale",    type=float, default=0.42,
                    help="Font scale for legend text (default: 0.42)")

    ap.add_argument("--speed_obs_csv", default="outputs/metrics/speed_per_observation.csv")
    ap.add_argument("--stop_flags_csv", default="outputs/metrics/stop_flags_per_observation.csv")
    ap.add_argument("--dwell_events_csv", default="outputs/metrics/dwell_events.csv")
    ap.add_argument("--shift_flags_csv", default="outputs/metrics/shift_flags_per_observation.csv",
                    help="Per-observation trajectory shift flags from compute_metrics.py")

    ap.add_argument("--stop_speed_px_s", type=float, default=25.0)
    ap.add_argument("--mid_speed_px_s", type=float, default=80.0)
    ap.add_argument("--no_legend", action="store_true")

    ap.add_argument("--heatmap_video", default="outputs/metrics/density_heatmap.mp4")
    ap.add_argument("--heatmap_alpha", type=float, default=0.20)
    ap.add_argument("--no_heatmap", action="store_true")

    # Spatial metric overlays (new)
    ap.add_argument("--flow_csv",       default=None,
                    help="flow_field_cells.csv from compute_flow_fields.py (image-space)")
    ap.add_argument("--bottleneck_csv", default=None,
                    help="bottleneck_cells.csv from compute_bottlenecks.py (image-space)")
    ap.add_argument("--linger_csv",     default=None,
                    help="linger_zones.csv from compute_linger_zones.py (image-space)")
    ap.add_argument("--show_metrics",   action="store_true",
                    help="Enable spatial metric overlays (requires image-space CSVs)")

    ap.add_argument("--highlight_top_k_speed", type=int, default=0,
                    help="Label the N fastest tracks with their speed on the video (0 = off)")
    ap.add_argument("--highlight_top_k_dwell", type=int, default=0,
                    help="Label the N longest-dwelling tracks with their dwell time (0 = off)")
    ap.add_argument("--show_hud", action="store_true",
                    help="Show a compact global-metrics HUD (active, avg/max speed, stops, dwellers)")

    ap.add_argument("--blur_faces",        action="store_true",
                    help="Blur the upper portion of each person bbox to anonymise faces")
    ap.add_argument("--face_height_ratio",   type=float, default=0.32,
                    help="Fraction of bbox height used as face blur region (default: 0.32)")
    ap.add_argument("--face_width_ratio",    type=float, default=0.65,
                    help="Fraction of bbox width used as face blur region, centred (default: 0.65)")
    ap.add_argument("--face_y_offset_ratio", type=float, default=0.05,
                    help="Vertical offset from bbox top before blur starts (default: 0.05)")
    ap.add_argument("--face_blur_kernel",    type=int,   default=51,
                    help="Gaussian kernel applied twice for strong anonymisation (default: 51)")

    args = ap.parse_args()

    _cmo = tuple(int(v) for v in args.current_marker_outline.split(","))

    main(
        video_path=args.video,
        traj_csv=args.traj_csv,
        out_dir=args.out_dir,
        min_track_len=args.min_track_len,
        draw_tail=args.tail_length if args.tail_length is not None else args.draw_tail,
        tail_fade=args.tail_fade,
        thickness=args.thickness,
        current_marker_size=args.current_marker_size,
        current_marker_outline=_cmo,
        marker_size=args.marker_size,
        panel_opacity=args.panel_opacity,
        font_scale=args.font_scale,
        speed_obs_csv=args.speed_obs_csv,
        stop_flags_csv=args.stop_flags_csv,
        dwell_events_csv=args.dwell_events_csv,
        shift_flags_csv=args.shift_flags_csv,
        stop_speed_px_s=args.stop_speed_px_s,
        mid_speed_px_s=args.mid_speed_px_s,
        show_legend=(not args.no_legend),
        heatmap_video=args.heatmap_video,
        heatmap_alpha=args.heatmap_alpha,
        use_heatmap=(not args.no_heatmap),
        flow_csv=args.flow_csv,
        bottleneck_csv=args.bottleneck_csv,
        linger_csv=args.linger_csv,
        show_metrics=args.show_metrics,
        blur_faces=args.blur_faces,
        face_height_ratio=args.face_height_ratio,
        face_width_ratio=args.face_width_ratio,
        face_y_offset_ratio=args.face_y_offset_ratio,
        blur_kernel=args.face_blur_kernel,
        highlight_top_k_speed=args.highlight_top_k_speed,
        highlight_top_k_dwell=args.highlight_top_k_dwell,
        show_hud=args.show_hud,
    )
