"""
MOTION PIXELS - Manual Mask Annotation Tool (Brush + Polygon)
=============================================================

Manually paint architectural walkable / obstacle masks for Encoder V3
preparation. The old encoder derived obstacles from trajectory coverage, not
architecture; these manually approved masks replace that.

This tool does NOT run any encoder, rebuild any dataset, or train any model.
It only reads a source image and writes mask PNGs + metadata.

SEMANTICS
---------
GREEN / walkable : areas pedestrians can physically move through -- plazas,
    sidewalks, crosswalks, walkable bridges, walkable stairs, ramps, paths.
RED / obstacle   : areas pedestrians cannot pass / that physically constrain
    movement -- buildings, walls, fences, planters, vegetation islands,
    fountains, monuments, railings, kiosks, permanent furniture, hard barriers.
Do NOT mark as obstacles: shadows, road markings, temporary vehicles, image
    texture, lighting differences.

USAGE
-----
    python manual_annotation/annotate_masks_v3.py --recording placa_catalunya_01
    python manual_annotation/annotate_masks_v3.py --image path\\to\\img.png --recording custom_id

CONTROLS
--------
    b              brush mode
    p              polygon mode
    1              select WALKABLE (green)
    2              select OBSTACLE (red)
    3              select ERASE (brush mode only)
    left-drag      paint (brush mode)
    left-click     add polygon vertex (polygon mode)
    backspace      remove last polygon vertex
    enter          fill current polygon
    esc            cancel current polygon
    [ / ]          smaller / larger brush
    s              save
    h              toggle on-screen help / controls panel
    q              save + quit (asks to confirm if there are unsaved changes)
    space          (startup screen only) begin annotating
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_ROOT = os.path.join(REPO_ROOT, "mp-data", "annotations", "manual_masks_v3")
# Per-recording data root (plans/calibration are also committed under mp-data/recordings/).
DATASETS_ROOT = os.environ.get("MP_DATA_ROOT", os.path.join(REPO_ROOT, "mp-data", "external"))

# recording_id -> source image filename
RECORDING_IMAGE = {
    "stairs_montjuic_01":     "stairs-montjuic-1.png",
    "red_bridge_combined_01": "red-bridge-combined.png",
    "esplanade_espanya_01":   "esplanade-espanya.png",
    "placa_espanya_01":       "placa-espanya.png",
    "placa_catalunya_01":     "placa-catalunya.png",
}

# colours (BGR)
COL_WALK = (0, 200, 0)
COL_OBST = (0, 0, 220)


def resolve_source(recording_id, explicit_image=None):
    """Return a path to the source image, preferring the plan/ copy."""
    if explicit_image:
        if not os.path.exists(explicit_image):
            sys.exit(f"[ERROR] --image not found: {explicit_image}")
        return explicit_image

    fname = RECORDING_IMAGE.get(recording_id)
    candidates = []
    if fname:
        candidates.append(os.path.join(DATASETS_ROOT, recording_id, "plan", fname))
        candidates.append(os.path.join(DATASETS_ROOT, fname))
    # also try a generic plan folder glob fallback
    plan_dir = os.path.join(DATASETS_ROOT, recording_id, "plan")
    if os.path.isdir(plan_dir):
        for f in sorted(os.listdir(plan_dir)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                candidates.append(os.path.join(plan_dir, f))
    for c in candidates:
        if os.path.exists(c):
            return c
    sys.exit(f"[ERROR] Could not resolve a source image for '{recording_id}'.\n"
             f"        Tried:\n        " + "\n        ".join(candidates))


class Annotator:
    WIN = "MotionPixels - Manual Mask Annotation (h = help)"

    def __init__(self, recording_id, source_path, max_w=1500, max_h=850):
        self.recording_id = recording_id
        self.source_path = source_path
        self.img = cv2.imread(source_path)
        if self.img is None:
            sys.exit(f"[ERROR] Failed to read image: {source_path}")
        self.h, self.w = self.img.shape[:2]

        self.outdir = os.path.join(OUT_ROOT, recording_id)
        os.makedirs(self.outdir, exist_ok=True)

        self.wmask = np.zeros((self.h, self.w), np.uint8)
        self.omask = np.zeros((self.h, self.w), np.uint8)
        self._load_existing()

        # display scale (fit on screen) -- we map mouse coords back to full res
        self.scale = min(max_w / self.w, max_h / self.h, 1.0)

        self.mode = "brush"        # 'brush' | 'polygon'
        self.active = "obstacle"   # 'walkable' | 'obstacle' | 'erase'
        self.brush = max(6, int(0.01 * max(self.h, self.w)))
        self.poly = []             # current polygon vertices (full-res)
        self.mouse = (0, 0)        # current mouse in full-res
        self.drawing = False
        self.last_pt = None
        self.dirty = False

        # ---- UX / HUD state ----
        self.startup = True          # full-screen help until SPACE
        self.quit_dialog = False     # unsaved-changes dialog
        self.show_controls = True    # toggle controls panel with 'h'
        self.flash_until = 0.0       # [SAVED] flash deadline (time.time)
        self.flash_text = ""

    # -- persistence --------------------------------------------------------
    def _p(self, name):
        return os.path.join(self.outdir, name)

    def _load_existing(self):
        wp, op = self._p("walkable_mask_v3_manual.png"), self._p("obstacle_mask_v3_manual.png")
        loaded = False
        if os.path.exists(wp):
            m = cv2.imread(wp, cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape == (self.h, self.w):
                self.wmask = (m > 127).astype(np.uint8) * 255
                loaded = True
        if os.path.exists(op):
            m = cv2.imread(op, cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape == (self.h, self.w):
                self.omask = (m > 127).astype(np.uint8) * 255
                loaded = True
        if loaded:
            print(f"[RESUME] Loaded existing masks from {self.outdir}")

    # -- mask editing (mutually exclusive) ----------------------------------
    def _apply(self, scratch):
        """scratch>0 is the affected region; write according to active target."""
        sel = scratch > 0
        if self.active == "walkable":
            self.wmask[sel] = 255
            self.omask[sel] = 0
        elif self.active == "obstacle":
            self.omask[sel] = 255
            self.wmask[sel] = 0
        else:  # erase
            self.wmask[sel] = 0
            self.omask[sel] = 0
        self.dirty = True

    def _paint_segment(self, p0, p1):
        scratch = np.zeros((self.h, self.w), np.uint8)
        if p0 is None:
            cv2.circle(scratch, p1, self.brush, 255, -1)
        else:
            cv2.line(scratch, p0, p1, 255, thickness=self.brush * 2)
            cv2.circle(scratch, p1, self.brush, 255, -1)
        self._apply(scratch)

    def _fill_polygon(self):
        if len(self.poly) < 3:
            print("[POLY] Need >= 3 vertices to fill.")
            return
        scratch = np.zeros((self.h, self.w), np.uint8)
        cv2.fillPoly(scratch, [np.array(self.poly, np.int32)], 255)
        # erase target not allowed in polygon mode; force walk/obst
        if self.active == "erase":
            self.active = "obstacle"
        self._apply(scratch)
        print(f"[POLY] Filled {self.active} polygon ({len(self.poly)} verts).")
        self.poly = []

    # -- mouse --------------------------------------------------------------
    def on_mouse(self, event, x, y, flags, _):
        if self.startup or self.quit_dialog:
            return  # ignore painting while a modal overlay is up
        fx, fy = int(x / self.scale), int(y / self.scale)
        fx = max(0, min(self.w - 1, fx))
        fy = max(0, min(self.h - 1, fy))
        self.mouse = (fx, fy)

        if self.mode == "brush":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.last_pt = None
                self._paint_segment(None, (fx, fy))
                self.last_pt = (fx, fy)
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self._paint_segment(self.last_pt, (fx, fy))
                self.last_pt = (fx, fy)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.last_pt = None
        else:  # polygon
            if event == cv2.EVENT_LBUTTONDOWN:
                self.poly.append((fx, fy))

    # -- rendering ----------------------------------------------------------
    def _composite(self, with_cursor=True):
        overlay = self.img.copy()
        overlay[self.wmask > 0] = COL_WALK
        overlay[self.omask > 0] = COL_OBST
        comp = cv2.addWeighted(self.img, 0.5, overlay, 0.5, 0)

        # polygon in progress
        if self.poly:
            col = COL_WALK if self.active == "walkable" else COL_OBST
            for i, pt in enumerate(self.poly):
                cv2.circle(comp, pt, max(2, self.brush // 3), col, -1)
                if i > 0:
                    cv2.line(comp, self.poly[i - 1], pt, col, 2)
            cv2.line(comp, self.poly[-1], self.mouse, (255, 255, 255), 1)

        if with_cursor and self.mode == "brush":
            col = {"walkable": COL_WALK, "obstacle": COL_OBST,
                   "erase": (255, 255, 255)}[self.active]
            cv2.circle(comp, self.mouse, self.brush, col, 1)

        disp = cv2.resize(comp, (int(self.w * self.scale), int(self.h * self.scale)),
                          interpolation=cv2.INTER_AREA)
        self._draw_hud(disp)
        self._draw_legend(disp)
        self._draw_flash(disp)
        return disp

    # -- HUD / overlays -----------------------------------------------------
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def _panel(self, disp, x, y, lines, pad=8, scale=0.5, line_h=22,
               bg=(20, 20, 20), alpha=0.65):
        """Draw a translucent text panel. `lines` = list of (text, color)."""
        H, W = disp.shape[:2]
        widths = [cv2.getTextSize(t, self.FONT, scale, 1)[0][0] for t, _ in lines]
        pw = (max(widths) if widths else 0) + pad * 2
        ph = line_h * len(lines) + pad * 2
        x2, y2 = min(x + pw, W), min(y + ph, H)
        if x2 <= x or y2 <= y:
            return y
        roi = disp[y:y2, x:x2]
        rect = np.empty_like(roi)
        rect[:] = bg
        cv2.addWeighted(rect, alpha, roi, 1 - alpha, 0, roi)
        cy = y + pad + 14
        for t, c in lines:
            cv2.putText(disp, t, (x + pad, cy), self.FONT, scale, c, 1, cv2.LINE_AA)
            cy += line_h
        return y2

    def _active_style(self):
        return {"walkable": ("ACTIVE: WALKABLE", COL_WALK),
                "obstacle": ("ACTIVE: OBSTACLE", COL_OBST),
                "erase":    ("ACTIVE: ERASE", (255, 255, 255))}[self.active]

    def _draw_hud(self, disp):
        white = (255, 255, 255)
        grey = (180, 180, 180)
        head = (120, 200, 255)
        act_text, act_col = self._active_style()
        mode_lines = [
            ("==== MODE ====", head),
            (f"Mode: {self.mode.upper()}", white),
            (act_text, act_col),
            (f"Brush size: {self.brush} px", white),
            ("UNSAVED *" if self.dirty else "saved", (0, 165, 255) if self.dirty else grey),
        ]
        bottom = self._panel(disp, 10, 10, mode_lines)
        if self.show_controls:
            ctrl_lines = [
                ("== CONTROLS ==", head),
                ("1 = Walkable (Green)", COL_WALK),
                ("2 = Obstacle (Red)", COL_OBST),
                ("3 = Erase", white),
                ("b = Brush Mode", white),
                ("p = Polygon Mode", white),
                ("[ / ] = Brush -/+", white),
                ("Enter = Fill Polygon", white),
                ("Backspace = Undo Vertex", white),
                ("Esc = Cancel Polygon", white),
                ("s = Save", white),
                ("q = Save + Quit", white),
                ("h = Toggle Help", white),
            ]
            self._panel(disp, 10, bottom + 8, ctrl_lines, scale=0.45, line_h=19)

    def _draw_legend(self, disp):
        H, W = disp.shape[:2]
        lines = [("Green = Walkable", COL_WALK), ("Red = Obstacle", COL_OBST)]
        pw = max(cv2.getTextSize(t, self.FONT, 0.5, 1)[0][0] for t, _ in lines) + 16
        ph = 22 * len(lines) + 16
        self._panel(disp, W - pw - 10, H - ph - 10, lines)

    def _draw_flash(self, disp):
        if time.time() >= self.flash_until:
            return
        W = disp.shape[1]
        text = self.flash_text
        (tw, th), _ = cv2.getTextSize(text, self.FONT, 0.7, 2)
        x, y = (W - tw) // 2 - 16, 16
        x2, y2 = x + tw + 32, y + th + 24
        roi = disp[y:y2, max(0, x):x2]
        rect = np.empty_like(roi)
        rect[:] = (0, 120, 0)
        cv2.addWeighted(rect, 0.85, roi, 0.15, 0, roi)
        cv2.putText(disp, text, (max(8, x) + 16, y + th + 8), self.FONT, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

    def _modal(self, disp, lines, dim=0.78):
        """Centered full-screen modal. `lines` = list of (text, color, scale)."""
        H, W = disp.shape[:2]
        black = np.zeros_like(disp)
        out = cv2.addWeighted(black, dim, disp, 1 - dim, 0)
        heights = [cv2.getTextSize(t, self.FONT, s, 2)[0][1] + 14 for t, _, s in lines]
        total = sum(heights)
        cy = (H - total) // 2 + heights[0]
        for (t, c, s), hh in zip(lines, heights):
            (tw, _), _ = cv2.getTextSize(t, self.FONT, s, 2)
            cv2.putText(out, t, ((W - tw) // 2, cy), self.FONT, s, c, 2, cv2.LINE_AA)
            cy += hh
        return out

    def _draw_startup(self, disp):
        w, g, r, y = (255, 255, 255), COL_WALK, COL_OBST, (120, 200, 255)
        lines = [
            ("MOTION PIXELS", y, 1.0),
            ("Manual Architectural Annotation", w, 0.6),
            ("", w, 0.4),
            ("GREEN = Walkable  (sidewalks, plazas, stairs, bridges, paths)", g, 0.6),
            ("RED = Obstacle  (trees, planters, fountains, monuments, buildings, barriers)", r, 0.6),
            ("", w, 0.4),
            ("1 Walkable    2 Obstacle    3 Erase", w, 0.6),
            ("b Brush    p Polygon    [ ] Brush size", w, 0.6),
            ("s Save    q Save + Quit    h Toggle Help", w, 0.6),
            ("", w, 0.4),
            ("Press SPACE to begin", y, 0.8),
        ]
        return self._modal(disp, lines)

    def _draw_quit_dialog(self, disp):
        w, y = (255, 255, 255), (120, 200, 255)
        lines = [
            ("UNSAVED CHANGES", (0, 165, 255), 0.9),
            ("Save before exit?", w, 0.6),
            ("", w, 0.4),
            ("Y = Save + Exit", COL_WALK, 0.7),
            ("N = Exit Without Saving", COL_OBST, 0.7),
            ("Esc = Cancel", w, 0.7),
        ]
        return self._modal(disp, lines)

    def _clean_overlay(self):
        overlay = self.img.copy()
        overlay[self.wmask > 0] = COL_WALK
        overlay[self.omask > 0] = COL_OBST
        return cv2.addWeighted(self.img, 0.5, overlay, 0.5, 0)

    # -- save ---------------------------------------------------------------
    def save(self):
        overlap = int(np.logical_and(self.wmask > 0, self.omask > 0).sum())
        if overlap:  # safety: should never happen due to mutual exclusion
            self.omask[self.wmask > 0] = 0
            overlap = 0
        cv2.imwrite(self._p("annotation_rgb.png"), self.img)
        cv2.imwrite(self._p("walkable_mask_v3_manual.png"), self.wmask)
        cv2.imwrite(self._p("obstacle_mask_v3_manual.png"), self.omask)
        cv2.imwrite(self._p("overlay_manual.png"), self._clean_overlay())

        wpx = int((self.wmask > 0).sum())
        opx = int((self.omask > 0).sum())
        total = self.w * self.h
        meta = {
            "recording_id": self.recording_id,
            "source_image_path": self.source_path,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "image_width": self.w,
            "image_height": self.h,
            "brush_size_last": self.brush,
            "walkable_pixels": wpx,
            "obstacle_pixels": opx,
            "overlap_pixels": overlap,
            "walkable_coverage_percent": round(100 * wpx / total, 3),
            "obstacle_coverage_percent": round(100 * opx / total, 3),
            "notes": "",
        }
        with open(self._p("metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self.dirty = False
        ts = datetime.now().strftime("%H:%M:%S")
        self.flash_text = f"[SAVED] {self.recording_id}  {ts}"
        self.flash_until = time.time() + 2.0
        print(f"[SAVE] {self.outdir}\n       walkable {wpx}px "
              f"({meta['walkable_coverage_percent']}%)  "
              f"obstacle {opx}px ({meta['obstacle_coverage_percent']}%)  "
              f"overlap {overlap}px")

    # -- help ---------------------------------------------------------------
    @staticmethod
    def help():
        print(__doc__.split("CONTROLS")[1])

    # -- main loop ----------------------------------------------------------
    def run(self):
        print(f"[OPEN] {self.recording_id}  ({self.w}x{self.h})  "
              f"src={self.source_path}")
        self.help()
        cv2.namedWindow(self.WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WIN, self.on_mouse)
        while True:
            disp = self._composite()
            if self.startup:
                disp = self._draw_startup(disp)
            elif self.quit_dialog:
                disp = self._draw_quit_dialog(disp)
            cv2.imshow(self.WIN, disp)
            k = cv2.waitKey(20) & 0xFF

            # --- modal: startup help (must press SPACE to begin) ---
            if self.startup:
                if k == ord(" "):
                    self.startup = False
                    print("[BEGIN] annotation started.")
                elif k in (ord("q"), 27):
                    break
                continue

            # --- modal: unsaved-changes quit dialog ---
            if self.quit_dialog:
                if k in (ord("y"), ord("Y")):
                    self.save()
                    break
                elif k in (ord("n"), ord("N")):
                    print("[QUIT] exited without saving.")
                    break
                elif k == 27:
                    self.quit_dialog = False
                continue

            if k == 255:
                continue
            if k == ord("q"):
                if self.dirty:
                    self.quit_dialog = True   # ask before exit
                else:
                    break
            elif k == ord("s"):
                self.save()
            elif k == ord("b"):
                self.mode = "brush"; print("[MODE] brush")
            elif k == ord("p"):
                self.mode = "polygon"; self.poly = []; print("[MODE] polygon")
            elif k == ord("1"):
                self.active = "walkable"; print("[TARGET] walkable")
            elif k == ord("2"):
                self.active = "obstacle"; print("[TARGET] obstacle")
            elif k == ord("3"):
                if self.mode == "brush":
                    self.active = "erase"; print("[TARGET] erase")
                else:
                    print("[TARGET] erase is brush-mode only")
            elif k == ord("["):
                self.brush = max(1, self.brush - 2); print(f"[BRUSH] {self.brush}")
            elif k == ord("]"):
                self.brush += 2; print(f"[BRUSH] {self.brush}")
            elif k == ord("h"):
                self.show_controls = not self.show_controls
                self.help()
            elif k == 8:    # backspace
                if self.poly:
                    self.poly.pop()
            elif k == 13:   # enter
                if self.mode == "polygon":
                    self._fill_polygon()
            elif k == 27:   # esc
                if self.poly:
                    self.poly = []; print("[POLY] cancelled")
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Manual architectural mask annotation (V3).")
    ap.add_argument("--recording", required=True, help="recording id, e.g. placa_catalunya_01")
    ap.add_argument("--image", default=None, help="explicit source image path (optional)")
    ap.add_argument("--max-width", type=int, default=1500)
    ap.add_argument("--max-height", type=int, default=850)
    args = ap.parse_args()

    src = resolve_source(args.recording, args.image)
    Annotator(args.recording, src, args.max_width, args.max_height).run()


if __name__ == "__main__":
    main()
