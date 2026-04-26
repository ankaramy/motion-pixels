"""
Generate a structured PDF documentation for the Motion Pixels pipeline.
Designed to be uploaded as a ChatGPT project source.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, Preformatted
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import os

OUTPUT = os.path.join(os.path.dirname(__file__), "MotionPixels_Pipeline_Documentation.pdf")

# ── Styles ────────────────────────────────────────────────────────────────────

def build_styles():
    base = getSampleStyleSheet()
    s = {}

    s["title"] = ParagraphStyle("title",
        fontName="Helvetica-Bold", fontSize=22, leading=28,
        textColor=colors.HexColor("#1a1a2e"), spaceAfter=6, alignment=TA_CENTER)

    s["subtitle"] = ParagraphStyle("subtitle",
        fontName="Helvetica", fontSize=12, leading=16,
        textColor=colors.HexColor("#4a4a6a"), spaceAfter=4, alignment=TA_CENTER)

    s["date"] = ParagraphStyle("date",
        fontName="Helvetica-Oblique", fontSize=10,
        textColor=colors.HexColor("#888888"), spaceAfter=20, alignment=TA_CENTER)

    s["h1"] = ParagraphStyle("h1",
        fontName="Helvetica-Bold", fontSize=16, leading=20,
        textColor=colors.HexColor("#1a1a2e"), spaceBefore=18, spaceAfter=6,
        borderPad=4)

    s["h2"] = ParagraphStyle("h2",
        fontName="Helvetica-Bold", fontSize=13, leading=17,
        textColor=colors.HexColor("#2d3561"), spaceBefore=12, spaceAfter=4)

    s["h3"] = ParagraphStyle("h3",
        fontName="Helvetica-Bold", fontSize=11, leading=15,
        textColor=colors.HexColor("#444466"), spaceBefore=8, spaceAfter=3)

    s["body"] = ParagraphStyle("body",
        fontName="Helvetica", fontSize=10, leading=15,
        textColor=colors.HexColor("#222222"), spaceAfter=6, alignment=TA_JUSTIFY)

    s["bullet"] = ParagraphStyle("bullet",
        fontName="Helvetica", fontSize=10, leading=14,
        textColor=colors.HexColor("#222222"), spaceAfter=3,
        leftIndent=16, bulletIndent=4)

    s["code_label"] = ParagraphStyle("code_label",
        fontName="Helvetica-Bold", fontSize=9,
        textColor=colors.HexColor("#555555"), spaceAfter=2, spaceBefore=6)

    s["code"] = ParagraphStyle("code",
        fontName="Courier", fontSize=8.5, leading=13,
        textColor=colors.HexColor("#1a1a1a"),
        backColor=colors.HexColor("#f4f4f8"),
        borderColor=colors.HexColor("#ccccdd"),
        borderWidth=0.5, borderPad=6,
        leftIndent=4, rightIndent=4,
        spaceAfter=8)

    s["io_label"] = ParagraphStyle("io_label",
        fontName="Helvetica-Bold", fontSize=9,
        textColor=colors.HexColor("#2d3561"), spaceAfter=2)

    s["note"] = ParagraphStyle("note",
        fontName="Helvetica-Oblique", fontSize=9, leading=13,
        textColor=colors.HexColor("#555577"),
        leftIndent=12, spaceAfter=6)

    return s


# ── Helper builders ───────────────────────────────────────────────────────────

def hr(story):
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#ccccdd"), spaceAfter=6))


def section_header(story, text, s):
    story.append(Spacer(1, 6))
    hr(story)
    story.append(Paragraph(text, s["h1"]))


def module_header(story, filename, tagline, s):
    story.append(Spacer(1, 10))
    story.append(Paragraph(f'<font color="#2d3561">■</font> <b>{filename}</b>', s["h2"]))
    story.append(Paragraph(tagline, s["note"]))


def io_table(story, inputs, outputs, s):
    """Render a compact inputs / outputs two-column table."""
    def fmt(items):
        return "\n".join(f"• {i}" for i in items)

    data = [
        [Paragraph("INPUTS", s["io_label"]), Paragraph("OUTPUTS", s["io_label"])],
        [Paragraph(fmt(inputs), s["code"]),  Paragraph(fmt(outputs), s["code"])],
    ]
    tbl = Table(data, colWidths=[8.5*cm, 8.5*cm])
    tbl.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eeeef8")),
        ("LINEBELOW", (0,0), (-1,0), 0.5, colors.HexColor("#aaaacc")),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#ccccdd")),
        ("INNERGRID", (0,0), (-1,-1), 0.3, colors.HexColor("#ddddee")),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 6))


def code_block(story, label, code_text, s):
    story.append(Paragraph(label, s["code_label"]))
    story.append(Preformatted(code_text, s["code"]))


def key_params_table(story, rows, s):
    """rows = list of (param, default, description)"""
    header = [Paragraph(h, s["io_label"]) for h in ["Parameter", "Default", "Description"]]
    data = [header] + [[Paragraph(str(c), s["code"]) for c in row] for row in rows]
    col_w = [4.5*cm, 3*cm, 9.5*cm]
    tbl = Table(data, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eeeef8")),
        ("LINEBELOW", (0,0), (-1,0), 0.5, colors.HexColor("#aaaacc")),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#ccccdd")),
        ("INNERGRID", (0,0), (-1,-1), 0.3, colors.HexColor("#ddddee")),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("FONTSIZE", (0,1), (-1,-1), 9),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 6))


# ── Content ───────────────────────────────────────────────────────────────────

def build_story(s):
    story = []

    # ── Cover ──────────────────────────────────────────────────────────────
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph("Motion Pixels Pipeline", s["title"]))
    story.append(Paragraph("Codebase Reference & Architecture Documentation", s["subtitle"]))
    story.append(Paragraph("IAAC Thesis · March 2026", s["date"]))
    story.append(Spacer(1, 0.5*cm))
    hr(story)
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph(
        "This document describes the full structure, data flow, and operational logic of "
        "the Motion Pixels pedestrian-tracking and behavioural-analysis pipeline. "
        "It is intended as a machine-readable project reference for AI assistants "
        "(e.g. ChatGPT) to provide contextually accurate advice about the codebase.",
        s["body"]))

    story.append(Spacer(1, 0.5*cm))

    # ── 1. System Overview ─────────────────────────────────────────────────
    section_header(story, "1. System Overview", s)

    story.append(Paragraph(
        "The pipeline converts a raw surveillance or field video into multi-layer "
        "behavioural intelligence. It is structured as a series of independent, "
        "file-based modules that are orchestrated by a single entry-point script. "
        "All intermediate results are written to disk as CSV, PNG, or MP4 so that "
        "any module can be re-run or inspected in isolation.",
        s["body"]))

    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Pipeline stages (in execution order):", s["h3"]))
    stages = [
        ("1", "track_people.py", "YOLOv8 + ByteTrack detection & tracking → image-space trajectories"),
        ("2", "calibrate_homography.py", "Project image coords to world/plan coords via homography"),
        ("3", "compute_metrics.py", "Derive speed, dwell events, stops from raw trajectories"),
        ("4", "compute_flow_fields.py", "Aggregate step vectors into spatial flow grid"),
        ("5", "compute_bottlenecks.py", "Score cells by density × slowness → congestion map"),
        ("6", "compute_linger_zones.py", "DBSCAN clustering of stop events → linger zone polygons"),
        ("7", "compute_heatmap.py", "Gaussian density heatmap video from frame presence"),
        ("8", "draw_trajectories.py", "Annotated output video with all overlays + face blur"),
    ]
    data = [[Paragraph(c, s["body"]) for c in row] for row in
            [["#", "Script", "Purpose"]] + stages]
    col_w = [1*cm, 5.5*cm, 10.5*cm]
    tbl = Table(data, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2d3561")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.HexColor("#f4f4f8"), colors.white]),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#aaaacc")),
        ("INNERGRID", (0,0), (-1,-1), 0.3, colors.HexColor("#ddddee")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.4*cm))

    # Data flow diagram (ASCII)
    code_block(story, "Data flow between modules:", """\
[Video file]
    │
    ▼
track_people.py ──────────────────► trajectories_image_space.csv
                                              │
               ┌──────────────────────────────┤
               │                              │
               ▼                              ▼
  calibrate_homography.py         compute_metrics.py
               │                              │
               ▼                              ├──► compute_flow_fields.py
  trajectories_world_space.csv    metrics.csv │
               │                              ├──► compute_bottlenecks.py
               └──────────┬───────────────────┘
                          │                   └──► compute_linger_zones.py
                          ▼
              compute_heatmap.py  ──► heatmap.mp4
              draw_trajectories.py ──► tracked_annotated.mp4
""", s)

    # ── 2. Directory & File Inventory ─────────────────────────────────────
    section_header(story, "2. Directory & File Inventory", s)

    story.append(Paragraph(
        "All scripts live flat in the project root. Outputs are written to "
        "<b>outputs/&lt;video_stem&gt;/</b>. No package structure or imports "
        "between modules — each script is self-contained.",
        s["body"]))

    files = [
        ("run_pipeline.py",                    "Orchestrator — calls all modules in order"),
        ("track_people.py",                    "Detection & tracking (YOLOv8 + ByteTrack)"),
        ("compute_metrics.py",                 "Speed, dwell, stop metrics from trajectories"),
        ("compute_heatmap.py",                 "Gaussian density heatmap video"),
        ("draw_trajectories.py",               "Final annotated video + overlays + face blur"),
        ("calibrate_homography.py",            "Batch homography projection (CLI)"),
        ("calibrate_homography_interactive.py","GUI tool for picking calibration points"),
        ("compute_flow_fields.py",             "Spatial flow field & quiver plot"),
        ("compute_bottlenecks.py",             "Congestion scoring & heatmap"),
        ("compute_linger_zones.py",            "DBSCAN linger-zone detection"),
        ("plot_topdown_trajectories.py",       "Clean plan-view trajectory diagram"),
        ("bytetrack_mp.yaml",                  "ByteTrack tracker config (thresholds, buffer)"),
        ("calib.json",                         "Calibration point correspondences (auto-written)"),
        ("outputs/",                           "All generated CSVs, PNGs, and MP4s"),
    ]
    data = [[Paragraph(f, s["code"]), Paragraph(d, s["body"])] for f, d in files]
    tbl = Table([[Paragraph("File / Directory", s["io_label"]),
                  Paragraph("Role", s["io_label"])]] + data,
                colWidths=[6*cm, 11*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eeeef8")),
        ("LINEBELOW", (0,0), (-1,0), 0.5, colors.HexColor("#aaaacc")),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#ccccdd")),
        ("INNERGRID", (0,0), (-1,-1), 0.3, colors.HexColor("#ddddee")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(tbl)

    # ── 3. Module Reference ────────────────────────────────────────────────
    section_header(story, "3. Module Reference", s)

    # ── 3.1 run_pipeline.py ───────────────────────────────────────────────
    module_header(story, "run_pipeline.py",
        "Master orchestrator. Runs the full pipeline end-to-end from a single CLI call.", s)

    story.append(Paragraph("How it works:", s["h3"]))
    story.append(Paragraph(
        "Calls each module as a subprocess (via importlib or direct function calls). "
        "Resolves output paths, checks for already-completed steps (skip logic), "
        "and handles optional flags such as face blurring. "
        "All parameters are set at the top of the file as Python constants — "
        "there is no external config file for the pipeline itself.",
        s["body"]))

    code_block(story, "Invocation:", "python run_pipeline.py --video input_macba.MOV [--blur-faces] [--skip-tracking]", s)

    key_params_table(story, [
        ("VIDEO_PATH",       "CLI arg",    "Path to input video"),
        ("CONFIDENCE",       "0.3",        "YOLO detection confidence threshold"),
        ("IOU_THRESHOLD",    "0.45",       "NMS IoU threshold for YOLO"),
        ("GRID_ROWS/COLS",   "20 / 20",   "Grid resolution for heatmap, flow, bottleneck analysis"),
        ("DWELL_THRESHOLD",  "0.5 m/s",   "Speed below which a frame counts as a dwell"),
        ("STOP_THRESHOLD",   "0.2 m/s",   "Speed below which a frame counts as a full stop"),
        ("BLUR_FACES",       "False",      "Enable YOLOv8-face face blurring in output video"),
    ], s)

    io_table(story,
        ["Input video file (any OpenCV-readable format)"],
        ["outputs/<stem>/tracked.mp4",
         "outputs/<stem>/trajectories_image_space.csv",
         "outputs/<stem>/trajectories_world_space.csv (if calib.json present)",
         "outputs/<stem>/metrics.csv",
         "outputs/<stem>/flow_field.csv + flow_quiver.png",
         "outputs/<stem>/bottleneck_cells.csv + bottleneck_heatmap.png",
         "outputs/<stem>/linger_zones.csv + linger_zones.png",
         "outputs/<stem>/heatmap.mp4",
         "outputs/<stem>/tracked_annotated.mp4"], s)

    # ── 3.2 track_people.py ───────────────────────────────────────────────
    module_header(story, "track_people.py",
        "Person detection and multi-object tracking. The first and most compute-intensive step.", s)

    story.append(Paragraph("How it works:", s["h3"]))
    story.append(Paragraph(
        "Loads a YOLOv8 model (yolov8n.pt or yolov8s.pt) and runs it on every frame "
        "with the ByteTrack tracker specified in bytetrack_mp.yaml. "
        "For each detected person the bounding-box centroid (cx, cy) plus "
        "frame index, timestamp, and track_id are appended to a running list. "
        "At the end the list is written to CSV and an annotated preview video "
        "(tracked.mp4) is saved.",
        s["body"]))

    code_block(story, "Core tracking call:", """\
results = model.track(
    source=video_path,
    conf=confidence,
    iou=iou_threshold,
    classes=[0],           # person class only
    tracker="bytetrack_mp.yaml",
    stream=True,
    persist=True,
)""", s)

    story.append(Paragraph("Output CSV schema:", s["h3"]))
    code_block(story, "trajectories_image_space.csv columns:",
        "track_id, frame_idx, timestamp_s, cx, cy, bbox_x1, bbox_y1, bbox_x2, bbox_y2", s)

    io_table(story,
        ["video file", "yolov8n.pt or yolov8s.pt", "bytetrack_mp.yaml"],
        ["tracked.mp4 (preview)", "trajectories_image_space.csv"], s)

    # ── 3.3 compute_metrics.py ────────────────────────────────────────────
    module_header(story, "compute_metrics.py",
        "Derives per-frame kinematic metrics from raw centroid trajectories.", s)

    story.append(Paragraph("How it works:", s["h3"]))
    story.append(Paragraph(
        "Operates on one track_id at a time. Computes frame-to-frame Euclidean "
        "displacement in pixels (or metres if world-space coords are provided), "
        "converts to speed using the known fps, then applies a rolling-median "
        "smoother to suppress jitter. Dwell and stop flags are set by threshold "
        "comparison. A trajectory-shift detector flags frames where a track_id "
        "reappears after a gap (potential ID reassignment).",
        s["body"]))

    code_block(story, "Key metric derivations:", """\
# Speed (pixels/s or m/s)
dist = sqrt((cx[t] - cx[t-1])**2 + (cy[t] - cy[t-1])**2)
speed_raw = dist * fps
speed = rolling_median(speed_raw, window=5)

# Dwell / stop classification
is_dwell = speed < DWELL_THRESHOLD
is_stop  = speed < STOP_THRESHOLD

# Cumulative path length
path_length = cumsum(dist)""", s)

    story.append(Paragraph("Output CSV schema:", s["h3"]))
    code_block(story, "metrics.csv columns:",
        "track_id, frame_idx, timestamp_s, cx, cy, speed_px_s, speed_m_s,\n"
        "is_dwell, is_stop, path_length_px, path_length_m, shift_flag", s)

    io_table(story,
        ["trajectories_image_space.csv (or world_space variant)",
         "fps value from original video"],
        ["metrics.csv"], s)

    # ── 3.4 compute_heatmap.py ────────────────────────────────────────────
    module_header(story, "compute_heatmap.py",
        "Produces a presence-density heatmap video overlaid on the original footage.", s)

    story.append(Paragraph("How it works:", s["h3"]))
    story.append(Paragraph(
        "For each frame, accumulates centroid positions into a 2-D grid. "
        "Applies Gaussian blur (σ tunable) to the grid to create a smooth density "
        "surface. Normalises to [0, 255], maps to the JET colormap, and alpha-blends "
        "the heatmap over the original frame. The rolling accumulation window can be "
        "set to 'full' (all frames so far) or a fixed number of frames.",
        s["body"]))

    key_params_table(story, [
        ("GRID_W / GRID_H",  "match video", "Accumulation grid resolution"),
        ("SIGMA",            "15",           "Gaussian blur radius (pixels)"),
        ("ALPHA",            "0.5",          "Heatmap blend opacity"),
        ("WINDOW_FRAMES",    "None (full)",  "Rolling window size; None = cumulative"),
    ], s)

    io_table(story,
        ["trajectories_image_space.csv", "original video (for background frames)"],
        ["heatmap.mp4"], s)

    # ── 3.5 draw_trajectories.py ──────────────────────────────────────────
    module_header(story, "draw_trajectories.py",
        "Main annotated video renderer. Composites all analysis layers onto the video.", s)

    story.append(Paragraph("How it works:", s["h3"]))
    story.append(Paragraph(
        "For each frame, looks up all active tracks from metrics.csv and draws: "
        "(1) trajectory tail coloured by speed (green→yellow→red); "
        "(2) dwell rings at positions where is_dwell is True; "
        "(3) direction-change indicators; "
        "(4) optional heatmap underlay; "
        "(5) flow-field quiver overlay; "
        "(6) bottleneck cell shading; "
        "(7) linger-zone polygon outlines. "
        "Face blurring uses a second YOLOv8-face model to detect and blur faces "
        "independently of the tracking step.",
        s["body"]))

    story.append(Paragraph("Key visual encodings:", s["h3"]))
    enc = [
        ("Trajectory colour", "Speed: green (fast) → yellow → red (slow/stopped)"),
        ("Ring diameter",     "Proportional to dwell duration at that location"),
        ("Cell shading",      "Bottleneck score: transparent→orange→dark red"),
        ("Quiver arrows",     "Mean step vector per grid cell (flow field)"),
        ("Polygon outline",   "Linger zone boundary (colour = zone category)"),
        ("Blurred region",    "Face bounding box, blurred with high-σ Gaussian"),
    ]
    data = [[Paragraph(a, s["body"]), Paragraph(b, s["body"])] for a, b in enc]
    tbl = Table([[Paragraph("Element", s["io_label"]),
                  Paragraph("Encoding", s["io_label"])]] + data,
                colWidths=[5*cm, 12*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eeeef8")),
        ("LINEBELOW", (0,0), (-1,0), 0.5, colors.HexColor("#aaaacc")),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#ccccdd")),
        ("INNERGRID", (0,0), (-1,-1), 0.3, colors.HexColor("#ddddee")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 6))

    io_table(story,
        ["original video", "metrics.csv", "flow_field.csv (optional)",
         "bottleneck_cells.csv (optional)", "linger_zones.csv (optional)"],
        ["tracked_annotated.mp4"], s)

    # ── 3.6 calibrate_homography.py ───────────────────────────────────────
    module_header(story, "calibrate_homography.py",
        "Reprojects image-space trajectories to metric world coordinates.", s)

    story.append(Paragraph("How it works:", s["h3"]))
    story.append(Paragraph(
        "Reads calibration point correspondences from calib.json "
        "(4+ point pairs: image pixel ↔ plan/world metre). "
        "Uses cv2.findHomography (RANSAC) to compute a 3×3 perspective transform. "
        "Applies the transform to every (cx, cy) centroid in the trajectory CSV "
        "and writes a new CSV with (wx, wy) columns in metres. "
        "Also produces a scatter plot of the projected points overlaid on the plan image.",
        s["body"]))

    code_block(story, "Homography application:", """\
H, _ = cv2.findHomography(img_pts, world_pts, cv2.RANSAC)
pts_h = np.hstack([cx, cy, ones])  # homogeneous
world_h = (H @ pts_h.T).T
wx = world_h[:,0] / world_h[:,2]
wy = world_h[:,1] / world_h[:,2]""", s)

    io_table(story,
        ["trajectories_image_space.csv", "calib.json", "plan image (optional, for plot)"],
        ["trajectories_world_space.csv", "topdown_plot.png"], s)

    # ── 3.7 calibrate_homography_interactive.py ───────────────────────────
    module_header(story, "calibrate_homography_interactive.py",
        "GUI tool for picking calibration point correspondences.", s)

    story.append(Paragraph("How it works:", s["h3"]))
    story.append(Paragraph(
        "Opens two OpenCV windows side by side: the camera frame and the plan/top-view image. "
        "The user clicks matching points alternately in each window. "
        "Additional features: zoom/pan with scroll wheel, "
        "snap-to-corner detection within a configurable radius, "
        "scale-reference mode (click two points and enter real-world distance to set scale), "
        "and live preview of the projected point cloud. "
        "Saves correspondences to calib.json on exit.",
        s["body"]))

    io_table(story,
        ["A video frame (frame.png) or video file", "Plan/top-view image (top-view.png)"],
        ["calib.json (image_pts, world_pts arrays)"], s)

    # ── 3.8 compute_flow_fields.py ────────────────────────────────────────
    module_header(story, "compute_flow_fields.py",
        "Aggregates per-person step vectors into a spatial flow field.", s)

    story.append(Paragraph("How it works:", s["h3"]))
    story.append(Paragraph(
        "Divides the image (or world) space into a regular grid. "
        "For each consecutive centroid pair within a track, computes the displacement "
        "vector (dx, dy) and bins it into the cell containing its midpoint. "
        "Per cell, computes mean direction vector and a direction-consistency score "
        "(mean of cosine similarities between individual vectors and the cell mean). "
        "Outputs a CSV of cell centres + mean vectors + consistency scores, "
        "plus a quiver plot PNG.",
        s["body"]))

    key_params_table(story, [
        ("GRID_ROWS",         "20",   "Number of grid rows"),
        ("GRID_COLS",         "20",   "Number of grid columns"),
        ("MIN_VECTORS",       "3",    "Minimum step-vectors to include a cell"),
        ("CONSISTENCY_MIN",   "0.0",  "Filter cells below this consistency score"),
    ], s)

    io_table(story,
        ["trajectories_image_space.csv (or world_space)"],
        ["flow_field.csv", "flow_quiver.png"], s)

    # ── 3.9 compute_bottlenecks.py ────────────────────────────────────────
    module_header(story, "compute_bottlenecks.py",
        "Scores each spatial cell for congestion using density and speed.", s)

    story.append(Paragraph("How it works:", s["h3"]))
    story.append(Paragraph(
        "For each grid cell, counts total observations (density) and records "
        "the mean speed and fraction of frames where is_stop is True. "
        "Computes a composite bottleneck score: "
        "log(density+1) × (1/(mean_speed+ε)) × (1 + stop_fraction). "
        "Normalises scores to [0, 1] and renders a heatmap PNG with the cell grid overlaid.",
        s["body"]))

    code_block(story, "Bottleneck score formula:", """\
density_score = log(n_obs + 1)          # log scale to reduce dominance of hubs
speed_score   = 1.0 / (mean_speed + ε)  # ε avoids division by zero
stop_score    = 1.0 + stop_fraction      # boost cells with frequent stops
bottleneck    = density_score * speed_score * stop_score
# then normalise across all cells to [0, 1]""", s)

    io_table(story,
        ["metrics.csv", "video frame (for background of heatmap PNG)"],
        ["bottleneck_cells.csv", "bottleneck_heatmap.png"], s)

    # ── 3.10 compute_linger_zones.py ──────────────────────────────────────
    module_header(story, "compute_linger_zones.py",
        "Identifies areas of prolonged dwelling using stop-event clustering.", s)

    story.append(Paragraph("How it works:", s["h3"]))
    story.append(Paragraph(
        "Extracts all frames where is_stop is True from metrics.csv. "
        "Groups consecutive stop-frames per track into 'stop events' with a centroid "
        "and duration. Filters events shorter than MIN_STOP_DURATION. "
        "Runs DBSCAN on event centroids to find spatial clusters. "
        "Each cluster is classified by its median duration: "
        "brief (< 5 s), pause (5–30 s), linger (30–120 s), long_wait (> 120 s). "
        "The convex hull of cluster points becomes the zone polygon. "
        "Outputs a CSV of zones and a PNG overlay.",
        s["body"]))

    key_params_table(story, [
        ("MIN_STOP_DURATION", "2.0 s",   "Minimum stop event duration to include"),
        ("DBSCAN_EPS",        "30 px",   "DBSCAN neighbourhood radius"),
        ("DBSCAN_MIN_SAMPLES","3",       "Minimum points to form a cluster"),
        ("CATEGORY_THRESHOLDS","5/30/120 s","Brief / pause / linger / long_wait boundaries"),
    ], s)

    io_table(story,
        ["metrics.csv"],
        ["linger_zones.csv", "linger_zones.png"], s)

    # ── 3.11 plot_topdown_trajectories.py ─────────────────────────────────
    module_header(story, "plot_topdown_trajectories.py",
        "Generates a clean plan-view trajectory diagram for publication or reporting.", s)

    story.append(Paragraph("How it works:", s["h3"]))
    story.append(Paragraph(
        "Reads world-space or image-space trajectory data and plots each track as "
        "a polyline using matplotlib. Three colouring modes are supported: "
        "by track_id (categorical palette), by speed (sequential colourmap), "
        "or by dwell status (binary). "
        "Optional Y-axis inversion corrects image coordinate convention for plan images. "
        "Saves a high-DPI PNG.",
        s["body"]))

    io_table(story,
        ["trajectories_world_space.csv or trajectories_image_space.csv",
         "metrics.csv (if colour = speed or dwell)"],
        ["topdown_trajectories.png"], s)

    # ── 4. Key Data Contracts ─────────────────────────────────────────────
    section_header(story, "4. Key Data Contracts (CSV Schemas)", s)

    story.append(Paragraph(
        "All modules communicate through CSV files. The schemas below are the "
        "canonical contracts — any module that reads or writes these files must "
        "respect the column names exactly.",
        s["body"]))

    schemas = [
        ("trajectories_image_space.csv",
         "track_id (int), frame_idx (int), timestamp_s (float),\n"
         "cx (float), cy (float),\n"
         "bbox_x1, bbox_y1, bbox_x2, bbox_y2 (float)"),

        ("trajectories_world_space.csv",
         "track_id (int), frame_idx (int), timestamp_s (float),\n"
         "wx (float), wy (float)   ← in metres after homography"),

        ("metrics.csv",
         "track_id, frame_idx, timestamp_s, cx, cy,\n"
         "speed_px_s, speed_m_s (float),\n"
         "is_dwell (bool), is_stop (bool),\n"
         "path_length_px, path_length_m (float),\n"
         "shift_flag (bool)"),

        ("flow_field.csv",
         "cell_row, cell_col (int),\n"
         "cx_px, cy_px (float) — cell centre in image space,\n"
         "mean_dx, mean_dy (float) — mean step vector,\n"
         "n_vectors (int), consistency (float 0–1)"),

        ("bottleneck_cells.csv",
         "cell_row, cell_col (int),\n"
         "cx_px, cy_px (float),\n"
         "n_obs, mean_speed_px_s, stop_fraction (float),\n"
         "bottleneck_score (float, normalised 0–1)"),

        ("linger_zones.csv",
         "zone_id (int), category (str: brief/pause/linger/long_wait),\n"
         "centroid_x, centroid_y (float),\n"
         "n_events (int), median_duration_s (float),\n"
         "hull_points (JSON list of [x,y] pairs)"),
    ]

    for name, cols in schemas:
        story.append(Paragraph(name, s["h3"]))
        story.append(Preformatted(cols, s["code"]))

    # ── 5. Configuration & Tuning ──────────────────────────────────────────
    section_header(story, "5. Configuration & Tuning Guide", s)

    story.append(Paragraph(
        "All tuning parameters are plain Python constants at the top of each "
        "script (or in run_pipeline.py for global ones). There is no central "
        "config file — the design intention is that each run is fully reproducible "
        "by inspecting the script that produced it.",
        s["body"]))

    tuning = [
        ("Fewer false detections",
         "Raise CONFIDENCE in run_pipeline.py (try 0.4–0.5). "
         "Also increase IOU_THRESHOLD slightly."),
        ("Smoother speed curves",
         "Increase the rolling-median window in compute_metrics.py (default 5). "
         "Larger windows lag but suppress jitter."),
        ("More/fewer linger zones",
         "Adjust DBSCAN_EPS (larger = merge more stops into one zone) "
         "and MIN_STOP_DURATION (higher = ignore brief pauses)."),
        ("Finer spatial analysis",
         "Increase GRID_ROWS / GRID_COLS in run_pipeline.py. "
         "More cells = finer resolution but noisier with low pedestrian counts."),
        ("Better homography accuracy",
         "Use more than 4 calibration points. Points near image edges are most "
         "important. Avoid collinear point arrangements."),
        ("Face blur quality",
         "Switch YOLO face model to a larger variant. "
         "Increase blur kernel sigma for stronger anonymisation."),
        ("Slow heatmap video",
         "Reduce WINDOW_FRAMES to a rolling window (e.g. 30 frames) "
         "to see temporal evolution rather than cumulative density."),
    ]

    for issue, fix in tuning:
        story.append(Paragraph(f"▸ <b>{issue}</b>", s["bullet"]))
        story.append(Paragraph(fix, ParagraphStyle("indent_body",
            parent=s["body"], leftIndent=24, spaceAfter=6)))

    # ── 6. Dependencies ────────────────────────────────────────────────────
    section_header(story, "6. Dependencies", s)

    deps = [
        ("ultralytics",  "≥ 8.0",   "YOLOv8 model loading, inference, ByteTrack integration"),
        ("opencv-python","≥ 4.8",   "Video I/O, drawing, face blur, homography (cv2)"),
        ("numpy",        "≥ 1.24",  "Array operations throughout"),
        ("pandas",       "≥ 2.0",   "CSV read/write, per-track groupby operations"),
        ("scipy",        "≥ 1.11",  "Gaussian filter (heatmap), DBSCAN (linger zones)"),
        ("scikit-learn", "≥ 1.3",   "DBSCAN clustering in compute_linger_zones.py"),
        ("matplotlib",   "≥ 3.7",   "Quiver plots, topdown diagrams, PNG output"),
        ("torch",        "≥ 2.0",   "Backend for ultralytics (CUDA optional)"),
    ]
    data = [[Paragraph(a, s["code"]), Paragraph(b, s["body"]), Paragraph(c, s["body"])]
            for a, b, c in deps]
    tbl = Table([[Paragraph("Package", s["io_label"]),
                  Paragraph("Version", s["io_label"]),
                  Paragraph("Used for", s["io_label"])]] + data,
                colWidths=[4.5*cm, 2.5*cm, 10*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eeeef8")),
        ("LINEBELOW", (0,0), (-1,0), 0.5, colors.HexColor("#aaaacc")),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#ccccdd")),
        ("INNERGRID", (0,0), (-1,-1), 0.3, colors.HexColor("#ddddee")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 6))
    code_block(story, "Install all dependencies:",
        "pip install ultralytics opencv-python numpy pandas scipy scikit-learn matplotlib", s)

    # ── 7. Known Limitations & Open Issues ────────────────────────────────
    section_header(story, "7. Known Limitations & Open Issues", s)

    limitations = [
        ("No cross-module parameter synchronisation",
         "Each script has its own GRID_ROWS/COLS constants. If you change the grid "
         "resolution in one script you must also change it in all others manually."),
        ("Homography assumes flat ground plane",
         "The projective transform is only valid when the tracked persons move on a "
         "flat surface. Sloped terrain or multi-level areas will introduce metric errors."),
        ("ByteTrack ID fragmentation on occlusion",
         "Long occlusions cause track IDs to change. The shift_flag in metrics.csv "
         "partially flags this but does not re-link broken tracks automatically."),
        ("Speed computed in image space",
         "speed_px_s is in pixels/second. speed_m_s is only valid if world-space "
         "trajectories are available (i.e. calibration has been run)."),
        ("No GPU memory management",
         "The YOLO model loads to the first available CUDA device. For long videos "
         "on small GPUs, VRAM may be exhausted; fall back with device='cpu'."),
        ("Face blur is a separate YOLO pass",
         "draw_trajectories.py runs a second model inference pass per frame for face "
         "detection, roughly doubling rendering time when blur is enabled."),
    ]
    for title, desc in limitations:
        story.append(Paragraph(f"▸ <b>{title}</b>", s["bullet"]))
        story.append(Paragraph(desc, ParagraphStyle("indent_body",
            parent=s["body"], leftIndent=24, spaceAfter=6)))

    # ── 8. Suggested Next Steps ────────────────────────────────────────────
    section_header(story, "8. Suggested Next Steps / Research Directions", s)

    nextsteps = [
        "Implement track re-identification (Re-ID) to stitch fragmented ByteTrack IDs across occlusions.",
        "Add a dedicated calibration-validation step that measures reprojection error and flags poor calibrations.",
        "Unify grid parameters into a single config dataclass shared by all analysis modules.",
        "Export linger zones and flow fields as GeoJSON for overlay in GIS / spatial design tools (e.g. Rhino/Grasshopper).",
        "Build a Streamlit or Gradio dashboard for interactive parameter tuning without editing source files.",
        "Add statistical significance testing for bottleneck and linger zone comparisons across different videos.",
        "Explore SORT or Deep-OC-SORT as alternatives to ByteTrack for improved handling of crowded scenes.",
        "Integrate with Open3D or PointCloud tools if LiDAR data becomes available for the same sites.",
    ]
    for step in nextsteps:
        story.append(Paragraph(f"• {step}", s["bullet"]))

    story.append(Spacer(1, 1*cm))
    hr(story)
    story.append(Paragraph(
        "End of document — generated 2026-03-19 · Motion Pixels Pipeline · IAAC Thesis",
        s["date"]))

    return story


# ── Build PDF ─────────────────────────────────────────────────────────────────

def main():
    s = build_styles()
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="Motion Pixels Pipeline — Codebase Documentation",
        author="IAAC Thesis",
        subject="Pedestrian tracking and behavioural analysis pipeline reference",
    )
    story = build_story(s)
    doc.build(story)
    print(f"PDF written to: {OUTPUT}")


if __name__ == "__main__":
    main()
