#!/usr/bin/env python3
"""
calibrate_homography_interactive.py  -  Motion Pixels pipeline
==============================================================
Method B interactive homography calibration.

Enforced workflow (in order):
  STEP 1  Scale -- click exactly 2 points on the TOP-VIEW image, then enter
                   the real-world distance between them (metres).
                   Computes meters_per_pixel; first click becomes
                   the world-coordinate origin.

  STEP 2  Correspondence -- alternate between clicking a camera point and
                   the matching top-view point.  Requires 4+ complete pairs.

  STEP 3  Solve -- press  S  to compute the homography and save calib.json.

World coordinate convention
  world_x = (top_view_x - origin_x) * meters_per_pixel
  world_y = (top_view_y - origin_y) * meters_per_pixel
  (negate world_y if --invert_plan_y is set)

Reprojection-error quality thresholds (metres)
  GOOD  < 0.30 m  -- sub-foot accuracy, suitable for most spatial analysis
  OK    < 1.00 m  -- acceptable for coarse analysis
  POOR  >= 1.00 m -- recalibrate; check point correspondences
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# -- Optional Tkinter for small dialogs ---------------------------------------
try:
    import tkinter as tk
    from tkinter import messagebox, simpledialog

    _tk_root = None

    def _tk():
        global _tk_root
        if _tk_root is None:
            _tk_root = tk.Tk()
            _tk_root.withdraw()
            _tk_root.lift()
        return _tk_root

    def ask_float(title, prompt):
        _tk()
        return simpledialog.askfloat(title, prompt)

    def ask_ok(title, msg):
        _tk()
        return messagebox.askokcancel(title, msg)

    def show_info(title, msg):
        _tk()
        messagebox.showinfo(title, msg)

except ImportError:
    def ask_float(title, prompt):
        try:
            return float(input(f"\n[{title}] {prompt}: "))
        except ValueError:
            return None

    def ask_ok(title, msg):
        return input(f"\n[{title}] {msg} [y/N]: ").strip().lower().startswith("y")

    def show_info(title, msg):
        print(f"\n[{title}] {msg}")


# -- Constants ----------------------------------------------------------------
WIN_CAM      = "Camera Frame  -  Motion Pixels Calibration"
WIN_TOP_VIEW = "Top-View Image  -  Motion Pixels Calibration"
DW, DH   = 900, 660   # display canvas size (pixels)
HEADER_H = 32          # header bar height (pixels)

# BGR colours
C_CAM    = (0,   200, 255)   # orange-yellow: camera points
C_TV     = (60,  230,  90)   # green: top-view points
C_SCALE  = (200,  60, 240)   # magenta: scale reference points
C_TEXT   = (230, 230, 230)
C_SHADOW = (10,   10,  10)
C_GOOD   = (60,  210,  60)
C_OK     = (60,  200, 255)
C_POOR   = (50,   50, 240)
C_ACTIVE = (0,   140, 220)   # blue: this window expects input
C_WAIT   = (60,   60,  60)   # dark: this window is idle

FONT = cv2.FONT_HERSHEY_SIMPLEX

ERR_GOOD = 0.30   # metres -- see quality thresholds in module docstring
ERR_OK   = 1.00


# -- State --------------------------------------------------------------------
class State:
    """All mutable session state.  Passed to callbacks via closure."""

    def __init__(self, invert_y: bool):
        # Phase:
        #   "scale"          -- collecting scale reference points on top-view image
        #   "correspondence" -- collecting camera <-> top-view pairs
        self.mode = "scale"

        # Scale calibration (top-view image coords)
        self.scale_pts: list       = []    # 0, 1, or 2 [x, y] in top-view image pixels
        self.top_view_origin: list = None  # [x, y]  first scale click = world origin
        self.scale_distance_m: float = None
        self.meters_per_pixel: float = None

        # Correspondence pairs (parallel lists)
        self.img_pts      : list = []    # [[px, py], ...]  camera image pixels
        self.top_view_pts : list = []    # [[px, py], ...]  top-view image pixels
        self.world_pts    : list = []    # [[X,  Y ], ...]  world metres

        # True  -> next click expected in camera window
        # False -> next click expected in top-view window
        self.pending_cam: bool = True

        # Zoom / pan per window
        # pan is in screen pixels: screen_xy = image_xy * zoom + pan
        self.zoom_cam     = 1.0;  self.pan_cam     = [0.0, 0.0]
        self.zoom_top_view = 1.0;  self.pan_top_view = [0.0, 0.0]

        # Optional snap-to-corner (camera window)
        self.snap    = False
        self.corners = None   # Nx2 float32 detected corners in camera image

        # Y-axis convention
        self.invert_y: bool = invert_y

        # Solved homography (stored after S-key solve, used for live residuals plot)
        self.H         = None
        self.errs      = None
        self.worst_idx = None   # index of the pair with the highest reprojection error

        # Scale factors: display_pixels / original_pixels  (set in run())
        # Used by save_calibration_json to convert back to original image coords.
        self.cam_display_scale: float = 1.0
        self.top_display_scale: float = 1.0

        # Misc
        self.show_help    = True
        self.dirty        = False   # unsaved changes
        self.source_label = ""      # stored in JSON

        # Current cursor image-coords (for snap ring display)
        self.cur_cam      = (0, 0)
        self.cur_top_view = (0, 0)


# -- Coordinate math ----------------------------------------------------------
def s2i(sx, sy, zoom, pan):
    """Screen -> image coordinates (float)."""
    return (sx - pan[0]) / zoom, (sy - pan[1]) / zoom


def i2s(ix, iy, zoom, pan):
    """Image -> screen coordinates (int tuple)."""
    return int(ix * zoom + pan[0]), int(iy * zoom + pan[1])


def zoom_at_cursor(zoom, pan, mx, my, factor, z_min=0.10, z_max=12.0):
    """Return (new_zoom, new_pan) after zooming by factor centred on (mx, my)."""
    new_zoom = max(z_min, min(z_max, zoom * factor))
    ratio    = new_zoom / zoom
    return new_zoom, [mx - (mx - pan[0]) * ratio,
                      my - (my - pan[1]) * ratio]


def nearest_pt(pts, ix, iy, thr=22):
    """Index of nearest stored point within image-coord threshold, or -1."""
    if not pts:
        return -1
    a = np.array(pts, dtype=float)
    d = np.hypot(a[:, 0] - ix, a[:, 1] - iy)
    i = int(np.argmin(d))
    return i if d[i] < thr else -1


def snap_to_corner(cur, corners, thr=16):
    """Snap cursor to nearest detected corner within thr pixels, or return cur."""
    if corners is None or len(corners) == 0:
        return cur
    d = np.hypot(corners[:, 0] - cur[0], corners[:, 1] - cur[1])
    i = int(np.argmin(d))
    return (int(corners[i, 0]), int(corners[i, 1])) if d[i] < thr else cur


# -- World coordinate conversion ----------------------------------------------
def top_view_px_to_world(px, py, origin, mpp, invert_y):
    """
    Convert a top-view image pixel to world metres.

    Convention (documented in module docstring):
        world_x = (px - origin[0]) * mpp
        world_y = (py - origin[1]) * mpp   [negated if invert_y]
    """
    wx = (px - origin[0]) * mpp
    wy = (py - origin[1]) * mpp
    if invert_y:
        wy = -wy
    return [round(wx, 6), round(wy, 6)]


# -- Homography & diagnostics -------------------------------------------------
def solve_homography(img_pts, world_pts):
    """
    Compute homography mapping camera image pixels -> world metres.
    Returns (H, n_inliers, error_dict) or (None, 0, None).
    """
    n = min(len(img_pts), len(world_pts))
    if n < 4:
        return None, 0, None
    src = np.array(img_pts[:n],   dtype=np.float32)
    dst = np.array(world_pts[:n], dtype=np.float32)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None, 0, None
    n_inliers = int(mask.sum()) if mask is not None else n
    errs = _reprojection_errors(src, dst, H)
    return H, n_inliers, errs


def _reprojection_errors(src, dst, H):
    proj  = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    dists = np.hypot(proj[:, 0] - dst[:, 0], proj[:, 1] - dst[:, 1])
    return {
        "mean":   float(np.mean(dists)),
        "median": float(np.median(dists)),
        "max":    float(np.max(dists)),
        "per_pt": dists.tolist(),
    }


def quality_label(mean_err):
    """Return (label_str, BGR_colour) for a mean reprojection error in metres."""
    if mean_err is None:
        return "need 4+ pairs", C_TEXT
    if mean_err < ERR_GOOD:
        return f"GOOD  ({mean_err:.3f} m)", C_GOOD
    if mean_err < ERR_OK:
        return f"OK    ({mean_err:.3f} m)", C_OK
    return     f"POOR  ({mean_err:.3f} m)", C_POOR


# -- Drawing helpers ----------------------------------------------------------
def stext(img, text, pos, scale=0.42, col=C_TEXT, thick=1):
    """Render text with a 1-pixel dark shadow for readability."""
    x, y = int(pos[0]), int(pos[1])
    cv2.putText(img, text, (x+1, y+1), FONT, scale, C_SHADOW, thick+1, cv2.LINE_AA)
    cv2.putText(img, text, (x,   y  ), FONT, scale, col,      thick,   cv2.LINE_AA)


def draw_numbered_point(img, sx, sy, idx, col):
    """Draw a filled circle with a 1-based index label at screen coords."""
    cv2.circle(img, (sx, sy), 10, C_SHADOW, -1, cv2.LINE_AA)
    cv2.circle(img, (sx, sy),  9, col,      -1, cv2.LINE_AA)
    stext(img, str(idx + 1), (sx + 12, sy + 5), 0.42, C_TEXT)


def draw_header(canvas, text, active):
    """Render a coloured header bar at the top of the canvas."""
    col = C_ACTIVE if active else C_WAIT
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], HEADER_H), col, -1)
    stext(canvas, text, (10, 22), 0.48, (255, 255, 255), 1)


# -- Canvas rendering ---------------------------------------------------------
def render_canvas(base, state, window, live_err):
    """
    Produce a display-ready canvas for 'camera' or 'top_view' window.
    Applies zoom/pan, draws all overlays, header, and status bar.
    """
    zoom = state.zoom_cam      if window == "camera" else state.zoom_top_view
    pan  = state.pan_cam       if window == "camera" else state.pan_top_view
    h, w = base.shape[:2]

    # Zoom/pan via affine warp (gray fill outside the image)
    M      = np.float32([[zoom, 0, pan[0]], [0, zoom, pan[1]]])
    canvas = cv2.warpAffine(base, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(40, 40, 40))

    # Header bar
    n_pairs = min(len(state.img_pts), len(state.top_view_pts))

    if state.mode == "scale":
        n_sc = len(state.scale_pts)
        if window == "top_view":
            step_txt = (
                "STEP 1 / SCALE  -  Click scale point 1 on this top-view image" if n_sc == 0 else
                "STEP 1 / SCALE  -  Click scale point 2 on this top-view image" if n_sc == 1 else
                "STEP 1 / SCALE  -  Enter the real distance in the dialog..."
            )
            draw_header(canvas, step_txt, active=True)
        else:
            draw_header(canvas,
                        "STEP 1 / SCALE  -  Set scale in the top-view window first",
                        active=False)

    else:  # correspondence mode
        if window == "camera" and state.pending_cam:
            draw_header(canvas,
                        f"STEP 2 / CORRESPONDENCE  -  Click camera point #{n_pairs + 1}",
                        active=True)
        elif window == "top_view" and not state.pending_cam:
            draw_header(canvas,
                        f"STEP 2 / CORRESPONDENCE  -  Click matching top-view point #{n_pairs + 1}",
                        active=True)
        else:
            waiting = "top-view" if state.pending_cam else "camera"
            draw_header(canvas,
                        f"STEP 2 / CORRESPONDENCE  -  Waiting for {waiting} click",
                        active=False)

    # Scale reference points (top-view window only)
    if window == "top_view":
        for i, pt in enumerate(state.scale_pts):
            sx, sy = i2s(pt[0], pt[1], zoom, pan)
            cv2.circle(canvas, (sx, sy), 8, C_SHADOW, -1, cv2.LINE_AA)
            cv2.circle(canvas, (sx, sy), 7, C_SCALE,  -1, cv2.LINE_AA)
            stext(canvas, f"S{i+1}", (sx + 10, sy + 4), 0.38, C_SCALE)

        if len(state.scale_pts) == 2:
            sa = i2s(*state.scale_pts[0], zoom, pan)
            sb = i2s(*state.scale_pts[1], zoom, pan)
            cv2.line(canvas, sa, sb, C_SCALE, 1, cv2.LINE_AA)
            if state.scale_distance_m is not None:
                mid = ((sa[0] + sb[0]) // 2, (sa[1] + sb[1]) // 2)
                stext(canvas, f"{state.scale_distance_m:.2f} m", (mid[0] + 4, mid[1] - 6),
                      0.40, C_SCALE)

    # Correspondence points
    pts = state.img_pts       if window == "camera"   else state.top_view_pts
    col = C_CAM               if window == "camera"   else C_TV
    for i, pt in enumerate(pts):
        sx, sy = i2s(pt[0], pt[1], zoom, pan)
        draw_numbered_point(canvas, sx, sy, i, col)

    # Ghost ring: camera point clicked but top-view not yet matched
    if window == "camera" and not state.pending_cam and state.img_pts:
        pt = state.img_pts[-1]
        sx, sy = i2s(pt[0], pt[1], zoom, pan)
        cv2.circle(canvas, (sx, sy), 11, (200, 200, 0), 2, cv2.LINE_AA)

    # Snap ring around cursor (camera window only)
    if window == "camera" and state.snap and state.corners is not None:
        sx, sy = i2s(*state.cur_cam, zoom, pan)
        cv2.circle(canvas, (sx, sy), 18, (200, 200, 0), 1, cv2.LINE_AA)

    # Status bar (bottom)
    qlbl, qcol = quality_label(live_err)
    lines = []

    if state.mode == "scale":
        if state.meters_per_pixel is not None:
            lines.append((f"Scale: {state.meters_per_pixel:.6f} m/px  |  "
                           f"Origin: {state.top_view_origin}", C_SCALE))
        else:
            lines.append(("Scale: not set -- click 2 points on top-view image", C_SCALE))
    else:
        lines.append((f"Pairs: {n_pairs} / 4 min (6-10 recommended)  |  Quality: {qlbl}", qcol))
        lines.append((f"Scale: {state.meters_per_pixel:.5f} m/px  |  "
                       f"Origin: {state.top_view_origin}", C_TEXT))

    if state.show_help and state.mode == "correspondence":
        lines.append(("U: undo  |  D: delete pair  |  R: reset  |  S: solve & save  |  "
                       "C: snap  |  Scroll: zoom  |  Mid-drag: pan  |  H: hide hints",
                       (150, 150, 150)))
    elif state.show_help:
        lines.append(("Click 2 top-view points, then enter the real distance.  H: hide hints",
                       (150, 150, 150)))

    for i, (txt, c) in enumerate(reversed(lines)):
        y = canvas.shape[0] - 8 - i * 19
        stext(canvas, txt, (8, y), 0.37, c)

    return canvas


# -- Image loading ------------------------------------------------------------
def load_source_frame(args):
    """Load camera frame from --frame_image or extract from --video."""
    if args.frame_image:
        img = cv2.imread(str(args.frame_image))
        if img is None:
            sys.exit(f"[ERROR] Cannot read frame image: {args.frame_image}")
        print(f"[INFO] Loaded camera frame: {args.frame_image}")
        return img, str(args.frame_image)
    # Video path
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"[ERROR] Could not read frame {args.frame_index} from video.")
    label = f"{args.video}@frame{args.frame_index}"
    print(f"[INFO] Loaded camera frame: {label}")
    return frame, label


def load_top_view_image(args):
    """Load top-view (or plan) image. Exits clearly on failure."""
    img = cv2.imread(str(args.top_view_image))
    if img is None:
        sys.exit(f"[ERROR] Cannot read top-view image: {args.top_view_image}")
    print(f"[INFO] Loaded top-view image: {args.top_view_image}")
    return img


def fit_to_display(img, max_w=DW, max_h=DH - HEADER_H - 60):
    """Scale image down to fit in canvas (never upscale)."""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
    return img


def pad_to_width(img, target_w):
    """Pad image with gray on the right to reach target_w pixels."""
    h, w = img.shape[:2]
    if w >= target_w:
        return img
    pad = np.full((h, target_w - w, 3), 40, dtype=np.uint8)
    return np.hstack([img, pad])


# -- Output: calibration JSON -------------------------------------------------
def save_calibration_json(state, H, n_inliers, errs, args):
    """
    Write calib.json with ALL pixel coordinates in ORIGINAL image space.

    The interactive tool works on fit_to_display-scaled canvases, so every
    clicked coordinate is in "display pixels."  Downstream scripts
    (calibrate_homography.py, analysis scripts) load the full-resolution plan
    image and apply the homography to full-resolution tracking coordinates, so
    everything stored here must be in original pixel units.

    Conversion:
        original_coord = display_coord / display_scale
        mpp_original   = mpp_display   * display_scale
        world_points   are in metres — invariant under this change.
        H_original     maps original camera pixels → world metres.
    """
    n   = len(state.img_pts)
    ts  = state.top_display_scale   # top-view: display_px / original_px  (≤1)
    cs  = state.cam_display_scale   # camera  : display_px / original_px  (≤1)

    # --- Convert plan (top-view) coordinates to original pixel space ----------
    def _orig_plan(pt):
        return [round(pt[0] / ts, 3), round(pt[1] / ts, 3)]

    orig_origin   = [round(state.top_view_origin[0] / ts, 3),
                     round(state.top_view_origin[1] / ts, 3)]
    mpp_orig      = state.meters_per_pixel * ts      # metres / original_plan_px
    orig_plan_pts = [_orig_plan(p) for p in state.top_view_pts]

    # --- Recompute world_points from original-scale plan pixels ---------------
    # (numerically identical to state.world_pts, just cleaner source)
    orig_world_pts = [
        top_view_px_to_world(pt[0], pt[1], orig_origin, mpp_orig, state.invert_y)
        for pt in orig_plan_pts
    ]

    # --- Convert camera coordinates to original pixel space -------------------
    orig_img_pts = [[round(p[0] / cs, 3), round(p[1] / cs, 3)]
                    for p in state.img_pts]

    # --- Recompute H in original-camera-pixel → world space -------------------
    # This H can be directly applied to full-resolution tracking data from
    # track_people.py (foot_x/foot_y or cx/cy in original camera pixels).
    H_orig        = H          # fallback if recomputation somehow fails
    n_inliers_out = n_inliers
    if len(orig_img_pts) >= 4:
        _src = np.array(orig_img_pts,    dtype=np.float32)
        _dst = np.array(orig_world_pts,  dtype=np.float32)
        _H, _mask = cv2.findHomography(_src, _dst, cv2.RANSAC, 5.0)
        if _H is not None:
            H_orig        = _H
            n_inliers_out = int(_mask.sum()) if _mask is not None else n

    # --- Sanity check: roundtrip for pair 1 -----------------------------------
    if H_orig is not None and len(orig_img_pts) >= 1:
        _p  = cv2.perspectiveTransform(
                  np.array([[orig_img_pts[0]]], dtype=np.float32),
                  H_orig).reshape(-1, 2)[0]
        _rx = _p[0] / mpp_orig + orig_origin[0]
        _ry = _p[1] / mpp_orig + orig_origin[1]
        print(f"[SANITY] Pair 1 (original pixel space):")
        print(f"   plan_px (orig)      : {[round(v,1) for v in orig_plan_pts[0]]}")
        print(f"   world_pt (m)        : {[round(v,4) for v in orig_world_pts[0]]}")
        print(f"   H(cam_px) → world   : [{_p[0]:.4f}, {_p[1]:.4f}]")
        print(f"   world → plan_px     : [{_rx:.1f}, {_ry:.1f}]  "
              f"(target: {[round(v,1) for v in orig_plan_pts[0]]})")

    # --- Build JSON -----------------------------------------------------------
    data = {
        "version":  1,
        "method":   "plan_scale_plus_correspondences",
        "plan_image":            str(args.top_view_image),
        "source_frame":          state.source_label,
        "invert_plan_y":         state.invert_y,
        # All pixel values below are in ORIGINAL (full-resolution) image space
        "meters_per_plan_pixel": mpp_orig,
        "plan_origin_pixel":     orig_origin,
        "scale_reference": {
            "plan_point_1": _orig_plan(state.scale_pts[0]),
            "plan_point_2": _orig_plan(state.scale_pts[1]),
            "distance_m":   state.scale_distance_m,
        },
        "image_points":   orig_img_pts,
        "plan_points_px": orig_plan_pts,
        "world_points":   [list(p) for p in orig_world_pts],
        # H maps ORIGINAL camera pixels → world metres
        "homography_matrix": H_orig.tolist() if H_orig is not None else None,
        "diagnostics": {
            "n_pairs":                     n,
            "n_inliers":                   n_inliers_out,
            "cam_display_scale":           round(cs, 6),
            "top_display_scale":           round(ts, 6),
            "mean_reprojection_error_m":   round(errs["mean"],   4) if errs else None,
            "median_reprojection_error_m": round(errs["median"], 4) if errs else None,
            "max_reprojection_error_m":    round(errs["max"],    4) if errs else None,
        },
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    print(f"[INFO] Calibration saved: {out.resolve()}")
    print(f"[INFO] Coordinates stored in original pixel space "
          f"(cam_scale={cs:.4f}, plan_scale={ts:.4f})")
    return data


# -- Output: preview PNG ------------------------------------------------------
def render_preview(cam_img, top_view_img, state, out_path):
    """Save a side-by-side PNG with numbered correspondence and scale points."""
    h = max(cam_img.shape[0], top_view_img.shape[0])

    def pad_h(img, target_h):
        dh = target_h - img.shape[0]
        if dh <= 0:
            return img
        return np.pad(img, ((0, dh), (0, 0), (0, 0)),
                      mode="constant", constant_values=40)

    cam_c      = pad_h(cam_img.copy(),      h)
    top_view_c = pad_h(top_view_img.copy(), h)

    for i, pt in enumerate(state.img_pts):
        draw_numbered_point(cam_c, int(pt[0]), int(pt[1]), i, C_CAM)
    for i, pt in enumerate(state.top_view_pts):
        draw_numbered_point(top_view_c, int(pt[0]), int(pt[1]), i, C_TV)
    for i, pt in enumerate(state.scale_pts):
        sx, sy = int(pt[0]), int(pt[1])
        cv2.circle(top_view_c, (sx, sy), 7, C_SCALE, -1, cv2.LINE_AA)
        stext(top_view_c, f"S{i+1}", (sx + 9, sy + 4), 0.42, C_SCALE)

    divider = np.full((h, 6, 3), 200, dtype=np.uint8)
    preview  = np.hstack([cam_c, divider, top_view_c])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), preview)
    print(f"[INFO] Preview image saved: {out_path}")


# -- Output: top-down scatter plot (matplotlib, optional) ---------------------
def render_topdown_plot(state, out_path):
    """
    Save a matplotlib scatter of world-coordinate points.
    When H is available (after solve), also plots projected camera points
    and residual vectors so calibration quality is visible.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[WARN] matplotlib not installed -- skipping top-down plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    wpts = np.array(state.world_pts, dtype=float)

    # Selected world points (green)
    ax.scatter(wpts[:, 0], wpts[:, 1], c="tab:green", s=90, zorder=4,
               edgecolors="white", linewidths=0.6, label="Selected top-view point")
    for i, (wx, wy) in enumerate(wpts):
        ax.annotate(str(i + 1), (wx, wy), textcoords="offset points",
                    xytext=(6, 4), fontsize=9, color="tab:green")

    # Projected image points + residuals — only when H is available
    if state.H is not None and len(state.img_pts) >= 4:
        src = np.array(state.img_pts, dtype=np.float32)
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2),
                                         state.H).reshape(-1, 2)

        errs_per_pt = state.errs.get("per_pt", []) if state.errs else []
        worst_idx   = getattr(state, "worst_idx", None)

        for i, ((wx, wy), (px, py)) in enumerate(zip(wpts, proj)):
            is_worst = (i == worst_idx)
            # Residual line: crimson + thick for worst pair, muted red otherwise
            ax.plot([wx, px], [wy, py],
                    color="crimson" if is_worst else "red",
                    linewidth=2.2   if is_worst else 1.0,
                    alpha=1.0       if is_worst else 0.65,
                    zorder=3)
            # Projected point marker
            ax.scatter(px, py,
                       c="crimson"    if is_worst else "tab:orange",
                       s=130          if is_worst else 60,
                       marker="X"     if is_worst else "^",
                       edgecolors="white", linewidths=0.8, zorder=5)
            # Error annotation
            if i < len(errs_per_pt):
                label_txt = f"{errs_per_pt[i]:.3f}m" + (" !" if is_worst else "")
                ax.annotate(label_txt, (px, py),
                            textcoords="offset points", xytext=(5, -11),
                            fontsize=7,
                            color="crimson"   if is_worst else "tab:orange",
                            fontweight="bold" if is_worst else "normal")

        # Outlier annotation box on plot canvas
        if state.errs and worst_idx is not None:
            worst_err = errs_per_pt[worst_idx] if worst_idx < len(errs_per_pt) else 0.0
            if worst_err > 5.0 or (state.errs["median"] > 0
                                   and worst_err > 3.0 * state.errs["median"]):
                ax.annotate(
                    f"[WARN] Pair {worst_idx + 1} is a strong outlier ({worst_err:.2f} m)",
                    xy=(0.02, 0.02), xycoords="axes fraction",
                    fontsize=8, color="crimson",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="crimson", alpha=0.85),
                )

        mean_e    = state.errs["mean"] if state.errs else float("nan")
        max_e     = state.errs["max"]  if state.errs else float("nan")
        worst_lbl = f"  |  worst: Pair {worst_idx + 1}" if worst_idx is not None else ""
        ax.set_title(
            f"Calibration quality  |  {len(wpts)} pairs  |  "
            f"mean {mean_e:.3f} m  |  max {max_e:.3f} m{worst_lbl}",
            fontsize=9
        )
        handles = [
            mpatches.Patch(color="tab:green",  label="Selected top-view point"),
            mpatches.Patch(color="tab:orange", label="Projected camera point"),
            mpatches.Patch(color="crimson",    label="Worst-error pair"),
            mpatches.Patch(color="red",        label="Residual"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="upper right")
    else:
        ax.set_title(f"Calibration — {len(wpts)} world points  (solve to see residuals)",
                     fontsize=9)

    ax.set_xlabel("World X (m)", fontsize=9)
    ax.set_ylabel("World Y (m)", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[INFO] Top-down plot saved: {out_path}")


# -- Main interactive session -------------------------------------------------
def run(args):
    # Load images
    cam_raw,      source_label = load_source_frame(args)
    top_view_raw               = load_top_view_image(args)

    cam_base      = fit_to_display(cam_raw)
    top_view_base = fit_to_display(top_view_raw)

    # Pad both to the same width so windows are consistent
    target_w      = max(cam_base.shape[1], top_view_base.shape[1], DW)
    cam_base      = pad_to_width(cam_base,      target_w)
    top_view_base = pad_to_width(top_view_base, target_w)

    state              = State(invert_y=args.invert_plan_y)
    state.source_label = source_label

    # Record how much fit_to_display scaled each image so that
    # save_calibration_json can convert clicked coords back to original pixels.
    # Height is used (unaffected by pad_to_width horizontal padding).
    state.cam_display_scale = cam_base.shape[0] / cam_raw.shape[0]
    state.top_display_scale = top_view_base.shape[0] / top_view_raw.shape[0]
    print(f"[INFO] Display scale — camera: {state.cam_display_scale:.4f}  "
          f"top-view: {state.top_display_scale:.4f}")

    print("[INFO] Click corresponding points on camera frame and top-view image.")

    # Detect corners in camera image for optional snap assist
    gray = cv2.cvtColor(cam_base, cv2.COLOR_BGR2GRAY)
    c    = cv2.goodFeaturesToTrack(gray, maxCorners=400, qualityLevel=0.01, minDistance=10)
    state.corners = c.reshape(-1, 2).astype(np.float32) if c is not None else None

    # OpenCV windows
    cv2.namedWindow(WIN_CAM,      cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_TOP_VIEW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CAM,      target_w, DH)
    cv2.resizeWindow(WIN_TOP_VIEW, target_w, DH)
    cv2.moveWindow(WIN_CAM,       20,            60)
    cv2.moveWindow(WIN_TOP_VIEW,  target_w + 50, 60)

    # Scale calibration: called when 2nd scale point is clicked
    def finalize_scale():
        p1, p2  = state.scale_pts
        px_dist = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if px_dist < 2:
            show_info("Error", "The two scale points are the same or too close.\n"
                               "Please click again.")
            state.scale_pts.clear()
            return

        dist_m = ask_float(
            "Scale Reference",
            f"Pixel distance between S1 and S2: {px_dist:.1f} px\n\n"
            "Enter the REAL distance between those two points (metres):"
        )
        if dist_m is None or dist_m <= 0:
            show_info("Cancelled", "Scale not set. Click the two scale points again.")
            state.scale_pts.clear()
            return

        state.meters_per_pixel = dist_m / px_dist
        state.top_view_origin  = list(p1)   # first click = world origin (0, 0)
        state.scale_distance_m = dist_m
        mpp                    = state.meters_per_pixel

        print(f"\n[SCALE] {px_dist:.1f} px = {dist_m:.3f} m  ->  "
              f"{mpp:.6f} m/px  |  origin = {p1}")

        show_info(
            "Scale Calibrated",
            f"Pixel distance : {px_dist:.1f} px\n"
            f"Real distance  : {dist_m:.3f} m\n"
            f"Scale          : {mpp:.6f} m/px\n\n"
            f"World origin (0, 0) = top-view pixel S1 = {p1}\n"
            f"{'Y axis is INVERTED (world_y = -...)' if state.invert_y else 'Y axis is normal'}\n\n"
            "STEP 2 -- Now collect correspondence pairs:\n"
            "  Click a point in the CAMERA window.\n"
            "  Click the matching point in the TOP-VIEW window.\n"
            "  Repeat. Minimum 4 pairs; 6-10 recommended for robustness.\n\n"
            "Keys:  U = undo   R = reset   S = solve & save   Q = quit"
        )

        state.mode        = "correspondence"
        state.pending_cam = True

    # Per-window pan drag state (middle-button)
    _drag = {"cam": False, "top_view": False}
    _last = {"cam": (0, 0), "top_view": (0, 0)}

    # Mouse callback: camera window
    def on_cam(ev, sx, sy, flags, _):
        ix, iy = s2i(sx, sy, state.zoom_cam, state.pan_cam)
        cur    = (int(ix), int(iy))
        if state.snap and state.corners is not None:
            cur = snap_to_corner(cur, state.corners)
        state.cur_cam = cur

        if ev == cv2.EVENT_MBUTTONDOWN:
            _drag["cam"] = True;  _last["cam"] = (sx, sy)
        elif ev == cv2.EVENT_MBUTTONUP:
            _drag["cam"] = False
        elif ev == cv2.EVENT_MOUSEMOVE and _drag["cam"]:
            dx, dy = sx - _last["cam"][0], sy - _last["cam"][1]
            state.pan_cam[0] += dx;  state.pan_cam[1] += dy
            _last["cam"] = (sx, sy)
        elif ev == cv2.EVENT_MOUSEWHEEL:
            factor = 1.15 if flags > 0 else 1.0 / 1.15
            state.zoom_cam, state.pan_cam = zoom_at_cursor(
                state.zoom_cam, state.pan_cam, sx, sy, factor)
        elif ev == cv2.EVENT_LBUTTONDOWN:
            # Only accept clicks during correspondence mode when camera is expected
            if state.mode != "correspondence" or not state.pending_cam:
                return
            state.img_pts.append(list(cur))
            state.pending_cam = False
            state.dirty       = True
            print(f"[cam  #{len(state.img_pts)}]  img_px = {cur}")

    # Mouse callback: top-view window
    def on_top_view(ev, sx, sy, flags, _):
        ix, iy = s2i(sx, sy, state.zoom_top_view, state.pan_top_view)
        cur    = (int(ix), int(iy))
        state.cur_top_view = cur

        if ev == cv2.EVENT_MBUTTONDOWN:
            _drag["top_view"] = True;  _last["top_view"] = (sx, sy)
        elif ev == cv2.EVENT_MBUTTONUP:
            _drag["top_view"] = False
        elif ev == cv2.EVENT_MOUSEMOVE and _drag["top_view"]:
            dx, dy = sx - _last["top_view"][0], sy - _last["top_view"][1]
            state.pan_top_view[0] += dx;  state.pan_top_view[1] += dy
            _last["top_view"] = (sx, sy)
        elif ev == cv2.EVENT_MOUSEWHEEL:
            factor = 1.15 if flags > 0 else 1.0 / 1.15
            state.zoom_top_view, state.pan_top_view = zoom_at_cursor(
                state.zoom_top_view, state.pan_top_view, sx, sy, factor)
        elif ev == cv2.EVENT_LBUTTONDOWN:

            # SCALE MODE: collect the two scale reference points
            if state.mode == "scale":
                if len(state.scale_pts) >= 2:
                    return
                state.scale_pts.append(list(cur))
                print(f"[scale] S{len(state.scale_pts)}: {cur}")
                if len(state.scale_pts) == 2:
                    finalize_scale()
                return

            # CORRESPONDENCE MODE: collect the top-view half of a pair
            if state.mode == "correspondence":
                if state.pending_cam:
                    return  # waiting for camera click, ignore top-view click
                # Convert top-view pixel -> world metres
                world = top_view_px_to_world(cur[0], cur[1],
                                              state.top_view_origin,
                                              state.meters_per_pixel,
                                              state.invert_y)
                state.top_view_pts.append(list(cur))
                state.world_pts.append(world)
                state.pending_cam = True
                state.dirty       = True
                print(f"[top-view #{len(state.top_view_pts)}]  "
                      f"top_view_px = {cur}  "
                      f"world = ({world[0]:.3f}, {world[1]:.3f}) m")

    cv2.setMouseCallback(WIN_CAM,      on_cam)
    cv2.setMouseCallback(WIN_TOP_VIEW, on_top_view)

    # Startup instructions popup
    show_info(
        "Motion Pixels  -  Calibration (Method B)",
        "WORKFLOW\n\n"
        "STEP 1  SCALE  (top-view window on the right):\n"
        "  Click point S1 on the top-view image.\n"
        "  Click point S2 on the top-view image.\n"
        "  Enter the real distance between them in metres.\n\n"
        "STEP 2  CORRESPONDENCE  (both windows):\n"
        "  Click a point in the camera window.\n"
        "  Click the matching point in the top-view window.\n"
        "  Minimum 4 pairs required; 6-10 pairs recommended for\n"
        "  robust calibration across the full scene.\n\n"
        "STEP 3  SOLVE & SAVE:\n"
        "  Press  S  to compute and save calib.json.\n\n"
        "KEYS\n"
        "  U   undo last pair\n"
        "  R   reset all pairs\n"
        "  S   solve & save\n"
        "  C   toggle snap-to-corner\n"
        "  H   toggle key hints\n"
        "  Q   quit\n\n"
        "MOUSE\n"
        "  Left-click   add point\n"
        "  Scroll       zoom (centred on cursor)\n"
        "  Middle-drag  pan"
    )

    # Main event loop
    live_err = None

    while True:
        # Recompute live quality estimate whenever we have 4+ complete pairs
        n_pairs = min(len(state.img_pts), len(state.top_view_pts))
        if n_pairs >= 4 and len(state.world_pts) >= 4:
            _, _, errs = solve_homography(state.img_pts, state.world_pts)
            live_err   = errs["mean"] if errs else None
        else:
            live_err = None

        cam_canvas      = render_canvas(cam_base,      state, "camera",   live_err)
        top_view_canvas = render_canvas(top_view_base, state, "top_view", live_err)

        cv2.imshow(WIN_CAM,      cam_canvas)
        cv2.imshow(WIN_TOP_VIEW, top_view_canvas)

        try:
            key = cv2.waitKeyEx(30)
        except AttributeError:
            key = cv2.waitKey(30)

        if key == -1:
            continue

        char = chr(key & 0xFF).lower() if 0 < (key & 0xFF) < 128 else ""

        # Q: quit
        if char == "q":
            if state.dirty and not ask_ok("Quit Without Saving",
                                           "You have unsaved changes.\nQuit anyway?"):
                continue
            break

        # S: solve & save
        elif char == "s":
            if state.mode == "scale":
                show_info("Cannot Save", "Complete Step 1 (scale) first.")
                continue
            n = min(len(state.img_pts), len(state.top_view_pts))
            if n < 4:
                show_info("Cannot Save",
                          f"Need 4+ correspondence pairs.  Currently have {n}.")
                continue

            H, n_inliers, errs = solve_homography(state.img_pts, state.world_pts)
            if H is None:
                show_info("Error", "Homography computation failed.\n"
                                   "Check that your points are not collinear.")
                continue

            qlbl, _ = quality_label(errs["mean"] if errs else None)
            print(f"\n-- Homography solved ------------------------------------------")
            print(f"   Pairs    : {n}")
            print(f"   Inliers  : {n_inliers}")
            worst_idx = None
            if errs:
                per_pt    = errs["per_pt"]
                worst_idx = int(np.argmax(per_pt))
                print(f"   Mean err : {errs['mean']:.4f} m")
                print(f"   Median   : {errs['median']:.4f} m")
                print(f"   Max      : {errs['max']:.4f} m  (Pair {worst_idx + 1})")
                for i, e in enumerate(per_pt):
                    marker = "  <-- WORST" if i == worst_idx else ""
                    print(f"   Pair {i+1:2d}  : {e:.4f} m{marker}")
                # Outlier warnings
                worst_err = per_pt[worst_idx]
                if worst_err > 5.0:
                    print(f"\n[WARN] Pair {worst_idx + 1} has very large reprojection "
                          f"error: {worst_err:.3f} m")
                if errs["median"] > 0 and worst_err > 3.0 * errs["median"]:
                    print(f"[WARN] Calibration contains a strong outlier. "
                          f"Recheck point ordering or landmark matching.")
                    print(f"[WARN] Consider pressing D to delete Pair {worst_idx + 1}, "
                          f"then S to re-solve.")
            print(f"   Quality  : {qlbl}")
            print(f"---------------------------------------------------------------")

            state.H         = H
            state.errs      = errs
            state.worst_idx = worst_idx

            save_calibration_json(state, H, n_inliers, errs, args)

            if args.out_preview:
                render_preview(cam_base, top_view_base, state, args.out_preview)

            if args.out_plot:
                render_topdown_plot(state, args.out_plot)

            state.dirty = False

            # Build optional outlier warning for dialog
            _outlier_warn = ""
            if errs and worst_idx is not None:
                _worst_err = errs["per_pt"][worst_idx]
                if _worst_err > 5.0 or (errs["median"] > 0
                                        and _worst_err > 3.0 * errs["median"]):
                    _outlier_warn = (
                        f"\n\n[WARN] Pair {worst_idx + 1} is a strong outlier "
                        f"({_worst_err:.3f} m).\n"
                        "Press D to delete it, then S to re-solve."
                    )

            show_info(
                "Saved",
                f"calib.json written to:\n{Path(args.out_json).resolve()}\n\n"
                f"Pairs     : {n}{'  [WARN: only 4 — add more for robustness]' if n == 4 else ''}\n"
                f"Inliers   : {n_inliers}\n"
                f"Quality   : {qlbl}{_outlier_warn}\n\n"
                "For best results collect 6-10 well-spread pairs.\n"
                "Keep adding pairs and press S again to re-save,\n"
                "or press Q to quit."
            )

        # U: undo last pair
        elif char == "u":
            # If a camera point was clicked but not yet paired, remove only it
            if not state.pending_cam and state.img_pts:
                state.img_pts.pop()
                state.pending_cam = True
                print("[undo] Pending camera point removed.")
            elif state.img_pts:
                state.img_pts.pop()
                if state.top_view_pts: state.top_view_pts.pop()
                if state.world_pts:    state.world_pts.pop()
                state.pending_cam = True
                n_remain = min(len(state.img_pts), len(state.top_view_pts))
                print(f"[undo] Last pair removed.  Pairs remaining: {n_remain}")

        # D: delete a specific pair by index
        elif char == "d":
            if state.mode != "correspondence":
                continue
            n_complete = min(len(state.img_pts), len(state.top_view_pts))
            # Subtract any dangling unpaired camera point
            if not state.pending_cam:
                n_complete = max(0, n_complete - 1)
            if n_complete == 0:
                print("[delete] No complete pairs to delete.")
                continue
            # Print current errors as a reminder if available
            if state.errs and state.errs.get("per_pt"):
                print("\nCurrent per-pair reprojection errors (from last solve):")
                for i, e in enumerate(state.errs["per_pt"]):
                    marker = "  <-- WORST" if i == state.worst_idx else ""
                    print(f"  Pair {i+1}: {e:.4f} m{marker}")
            raw = ask_float("Delete Pair",
                            f"Enter pair number to delete (1–{n_complete}):")
            if raw is None:
                continue
            idx = int(raw) - 1
            if idx < 0 or idx >= n_complete:
                show_info("Invalid", f"Pair number must be between 1 and {n_complete}.")
                continue
            # Remove a pending (unpaired) camera point before deleting a complete pair
            if not state.pending_cam and state.img_pts:
                state.img_pts.pop()
                state.pending_cam = True
            state.img_pts.pop(idx)
            if idx < len(state.top_view_pts):
                state.top_view_pts.pop(idx)
            if idx < len(state.world_pts):
                state.world_pts.pop(idx)
            state.pending_cam = True
            state.dirty       = True
            state.H           = None
            state.errs        = None
            state.worst_idx   = None
            remaining = min(len(state.img_pts), len(state.top_view_pts))
            print(f"[delete] Pair {idx + 1} removed.  "
                  f"Complete pairs remaining: {remaining}.  "
                  f"Press S to re-solve.")

        # R: reset all pairs
        elif char == "r":
            if ask_ok("Reset Pairs", "Delete ALL correspondence pairs?"):
                state.img_pts.clear()
                state.top_view_pts.clear()
                state.world_pts.clear()
                state.pending_cam = True
                print("[reset] All pairs cleared.")

        # C: toggle snap-to-corner
        elif char == "c":
            state.snap = not state.snap
            print(f"[snap] {'ON' if state.snap else 'OFF'}")

        # H: toggle help hints
        elif char == "h":
            state.show_help = not state.show_help

    cv2.destroyAllWindows()
    print("[INFO] Session ended.")


# -- CLI ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Interactive homography calibration (Method B) - Motion Pixels pipeline.\n\n"
            "Workflow:\n"
            "  1. Click 2 scale points on the top-view image, enter real distance (metres).\n"
            "  2. Collect 4+ camera <-> top-view point pairs.\n"
            "  3. Press S to solve and save calib.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",        metavar="PATH",
                     help="Input video. Frame at --frame_index is used for calibration.")
    src.add_argument("--frame_image",  metavar="PATH",
                     help="Static camera frame image (PNG/JPG). Alternative to --video.")

    # Preferred argument name
    ap.add_argument("--top_view_image", metavar="PATH", default=None,
                    help="Top-view (or plan) image used for homography calibration. "
                         "Preferred form of this argument.")
    # Legacy backward-compatible alias
    ap.add_argument("--plan_image", metavar="PATH", default=None,
                    help="Legacy alias for --top_view_image. "
                         "Accepted for backward compatibility; prefer --top_view_image.")

    ap.add_argument("--out_json",    default="calib.json",
                    help="Output calibration JSON path (default: calib.json).")
    ap.add_argument("--out_preview", metavar="PATH", default=None,
                    help="Optional: side-by-side calibration preview PNG.")
    ap.add_argument("--out_plot",    metavar="PATH", default=None,
                    help="Optional: matplotlib top-down scatter of world points PNG.")
    ap.add_argument("--frame_index", type=int, default=0,
                    help="Frame index to extract from video (default: 0).")
    ap.add_argument("--invert_plan_y", action="store_true",
                    help="Negate Y: world_y = -(top_view_y - origin_y) * mpp. "
                         "Use when Y increases upward in your top-view. Default: off.")

    args = ap.parse_args()

    # Resolve top_view_image: prefer --top_view_image, fall back to --plan_image
    top_view_image = args.top_view_image or args.plan_image
    if not top_view_image:
        raise ValueError("You must provide --top_view_image (or legacy --plan_image)")
    args.top_view_image = top_view_image   # normalise to single attribute

    if args.plan_image and not args.top_view_image:
        print("[WARN] --plan_image is a legacy alias. "
              "Please switch to --top_view_image in new scripts.")

    # Validate file existence
    if args.video and not Path(args.video).exists():
        ap.error(f"Video file not found: {args.video}")
    if args.frame_image and not Path(args.frame_image).exists():
        ap.error(f"Frame image not found: {args.frame_image}")
    if not Path(args.top_view_image).exists():
        ap.error(f"Top-view image not found: {args.top_view_image}")

    run(args)


if __name__ == "__main__":
    main()
