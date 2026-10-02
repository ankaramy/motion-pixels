"""
generate_trace_dust_hero_bg_v2.py
---------------------------------
Trace + Dust HERO — BACKGROUND V2 (Plaça Catalunya).

The movement layer (traces + dust + glow + palette) is FROZEN: it is rebuilt by
importing the exact functions/params from `generate_trace_dust_hero.py`. Only the
architectural background linework is changed.

Problem with V1: Canny on the satellite plan produced "contour-noodle" texture
(trees, shadows) that did not read as an urban plan.

OSM availability: `osmnx` is NOT installed, and the project calibration carries NO
geographic reference (no lat/lon / CRS — the world frame is a *local* metric frame
from the plan→world homography). Real OSM vector data therefore cannot be aligned
to our coordinates. Per the brief's Option C, we instead build clean, CAD-style
linework from the existing plan:
  - strong bilateral pre-filter (flatten texture)
  - Canny with high thresholds
  - HoughLinesP -> long STRAIGHT structural lines (buildings/roads/plaza edges)
  - connected-component length filter for a faint "detail" layer
Trees/shadows are curved/short -> dropped by the long-line filter -> no noodles.

Outputs (movement identical across all three):
  placa_catalunya_trace_dust_hero_black_osm.png     (clean structural linework)
  placa_catalunya_trace_dust_hero_black_hybrid.png  (structural + faint detail)
  placa_catalunya_trace_dust_background_comparison.png  (V1 Canny | OSM-style | hybrid)

VISUALIZATION ONLY. No retracking, recalibration, model inference, or source-data
modification.

Usage:
    python generate_trace_dust_hero_bg_v2.py
"""

from pathlib import Path
import sys

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from PIL import Image

HERO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HERO_DIR))
import generate_trace_dust_hero as hero   # frozen movement layer + params


OUT_DIR = hero.OUT_DIR
OUT_OSM    = OUT_DIR / "placa_catalunya_trace_dust_hero_black_osm.png"
OUT_HYBRID = OUT_DIR / "placa_catalunya_trace_dust_hero_black_hybrid.png"
OUT_CMP    = OUT_DIR / "placa_catalunya_trace_dust_background_comparison.png"
OUT_REPORT = hero.OUT_REPORT

# Background V2 style (only the plan linework changes)
PLAN_OPACITY_CLEAN = 0.34          # cleaner lines can sit a touch brighter
PLAN_TINT = np.array([0.72, 0.80, 0.95])
HYBRID_DETAIL_W = 0.45

# Hough / cleaning params
BILATERAL = (9, 80, 80)
CANNY = (70, 175)
HOUGH_THRESH = 55
HOUGH_MIN_LEN = 42                 # px on the warped-plan grid -> only long lines
HOUGH_MAX_GAP = 6
CC_MIN_DIAG = 34                   # min bbox diagonal (px) for detail components
CC_MIN_AREA = 18


# ---------------------------------------------------------------------------
# Clean CAD-style linework from the plan
# ---------------------------------------------------------------------------

def clean_linework(warped_rgb):
    gray = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.bilateralFilter(gray, *BILATERAL)
    edges = cv2.Canny(g, *CANNY)

    # structural: long straight lines only (buildings/roads/plaza geometry)
    canvas = np.zeros(gray.shape, dtype=np.float32)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=HOUGH_THRESH,
                            minLineLength=HOUGH_MIN_LEN, maxLineGap=HOUGH_MAX_GAP)
    n_lines = 0
    if lines is not None:
        n_lines = len(lines)
        for x1, y1, x2, y2 in lines[:, 0, :]:
            cv2.line(canvas, (x1, y1), (x2, y2), 1.0, 1, cv2.LINE_AA)
    structural = gaussian_filter(canvas, 0.5)
    if structural.max() > 0:
        structural /= structural.max()

    # detail: keep only long connected edge components (drop tree/shadow specks)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    keep = np.zeros(edges.shape, dtype=np.float32)
    for i in range(1, num):
        w = stats[i, cv2.CC_STAT_WIDTH]; h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if np.hypot(w, h) >= CC_MIN_DIAG and area >= CC_MIN_AREA:
            keep[lab == i] = 1.0
    detail = gaussian_filter(keep, 0.6)
    if detail.max() > 0:
        detail /= detail.max()
    return structural, detail, n_lines


# ---------------------------------------------------------------------------
# Compose (frozen movement + glow; only plan layer differs)
# ---------------------------------------------------------------------------

def compose_bg(M, P_intensity, plan_opacity, radii, weights, glow_strength):
    """Identical to hero.compose math (CORE_GAIN, bloom, rolloff frozen); the only
    change is which plan-linework intensity image P is used and its opacity."""
    b = hero.bloom(M, radii, weights)
    P_rgb = np.dstack([P_intensity] * 3) * PLAN_TINT
    out = hero.CORE_GAIN * M + glow_strength * b + plan_opacity * P_rgb
    out = out / (1.0 + 0.16 * out)
    return np.clip(out, 0, 1)


def resize_to(M, intensity):
    Hh, Ww = M.shape[:2]
    return cv2.resize(intensity, (Ww, Hh), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    warped_rgb, extent = hero.warp_plan_to_world(hero.DEF_PLAN, hero.DEF_CALIB)

    # --- rebuild the FROZEN movement layer exactly ---
    polylines, dust_xy, n_total, n_tracks = hero.load(hero.DEF_TRAJ, extent)
    D, xs, ys, vmax = hero.density_field(polylines, extent)
    M = hero.render_layer(extent, lambda ax: hero.draw_movement(
        ax, polylines, dust_xy, D, xs, ys, vmax))
    Hh, Ww = M.shape[:2]

    # --- backgrounds ---
    v1_edges = hero.plan_linework(warped_rgb)                 # current (noisy Canny)
    structural, detail, n_lines = clean_linework(warped_rgb)  # clean CAD-style

    P_v1     = resize_to(M, v1_edges)
    P_osm    = resize_to(M, structural)
    P_hybrid = resize_to(M, np.clip(structural + HYBRID_DETAIL_W * detail, 0, 1))

    radii, weights, gs = hero.GLOW["base"]

    hero_v1     = compose_bg(M, P_v1,     hero.PLAN_OPACITY,  radii, weights, gs)
    hero_osm    = compose_bg(M, P_osm,    PLAN_OPACITY_CLEAN, radii, weights, gs)
    hero_hybrid = compose_bg(M, P_hybrid, PLAN_OPACITY_CLEAN, radii, weights, gs)

    Image.fromarray((hero_osm * 255).astype(np.uint8)).save(OUT_OSM)
    Image.fromarray((hero_hybrid * 255).astype(np.uint8)).save(OUT_HYBRID)
    print(f"[INFO] Saved {OUT_OSM}  ({Ww}x{Hh})")
    print(f"[INFO] Saved {OUT_HYBRID}  ({Ww}x{Hh})")

    # --- comparison: V1 Canny | OSM-style | hybrid (movement identical) ---
    panels = [hero_v1, hero_osm, hero_hybrid]
    pw = 1366
    imgs = []
    for p in panels:
        im = Image.fromarray((p * 255).astype(np.uint8))
        im = im.resize((pw, int(im.height * pw / im.width)), Image.LANCZOS)
        imgs.append(im)
    gap = 16
    ph = imgs[0].height
    canvas = Image.new("RGB", (pw * 3 + gap * 2, ph), "black")
    for i, im in enumerate(imgs):
        canvas.paste(im, (i * (pw + gap), 0))
    canvas.save(OUT_CMP)
    print(f"[INFO] Saved {OUT_CMP}")

    print(f"[INFO] Hough structural lines: {n_lines}")
    update_report(n_lines, n_tracks, n_total, (Ww, Hh))


def update_report(n_lines, n_tracks, n_total, size):
    Ww, Hh = size
    section = f"""

---

# Background V2 (architectural linework upgrade)

**Movement layer frozen** (traces / dust / glow / palette / black background /
clipping / output size unchanged — rebuilt from `generate_trace_dust_hero.py`).
Only the plan linework changed. **Visualization-only; no retracking, recalibration,
inference, or source-data modification.**

## Was OSM data available?
**No.** `osmnx` is not installed, the live Overpass endpoint was not usable from
this environment, and — more fundamentally — the project calibration has **no
geographic reference** (no lat/lon or CRS; the world frame is a *local* metric
frame produced by the plan→world homography). Real OSM vector data cannot be
aligned to these local coordinates, so Option A/B (true OSM) is not feasible here.

## Source of vector linework
Per the brief's **Option C fallback**, clean CAD-style linework is extracted from
the existing warped plan image:
- bilateral pre-filter `{BILATERAL}` to flatten tree/shadow texture,
- Canny `{CANNY}`,
- **HoughLinesP** (threshold {HOUGH_THRESH}, minLineLength {HOUGH_MIN_LEN}px,
  maxLineGap {HOUGH_MAX_GAP}) → **{n_lines}** long STRAIGHT structural lines
  (buildings, roads, plaza edges). Curved/short tree & shadow edges are not long
  straight lines, so they are dropped — eliminating the "contour-noodle" texture.
- A faint **detail** layer keeps only long connected edge components
  (bbox diagonal ≥ {CC_MIN_DIAG}px, area ≥ {CC_MIN_AREA}px) for the hybrid.

## CRS / projection / alignment
No CRS is used (none available). The linework is computed on the **warped-plan
pixel grid** (already in the plan→world frame via `warp_plan_to_world`) and
resized to the movement-layer pixel grid, so it is pixel-aligned to the movement
and the plan extent by construction. No recalibration.

## Line opacity / style
Thin cool-grey lines (tint {list(np.round(PLAN_TINT,2))}) over black at opacity
**{PLAN_OPACITY_CLEAN}** (vs the V1 Canny background at {hero.PLAN_OPACITY}); hybrid
adds the detail layer at weight {HYBRID_DETAIL_W}. No fills, no labels/text.

## Outputs
- `{OUT_OSM.name}` — clean structural (Hough) linework. **Recommended.**
- `{OUT_HYBRID.name}` — structural + faint long-edge detail.
- `{OUT_CMP.name}` — comparison: V1 Canny | OSM-style (clean) | hybrid
  ({Ww}×{Hh} source panels).

## Recommendation for final background
Recommended final: the **hybrid** background. The OSM-style (Hough-only) layer is
the cleanest but, on its own, fairly sparse (much of the frame is black); the
hybrid adds the faint long-edge detail back so the **city stays legible** as an
urban plan while still suppressing the V1 tree/shadow noodle noise and remaining
clearly secondary to the glowing movement. Use **OSM-style** instead if a more
minimal, purely structural look is wanted. The V1 Canny background is deprecated
for hero use. True OSM would require (a) installing `osmnx`, (b) network access,
and (c) georeferencing the plan (≥2 known lat/lon control points) to align OSM to
the local world frame.
"""
    txt = OUT_REPORT.read_text(encoding="utf-8") if OUT_REPORT.exists() else ""
    # Replace any prior V2 section to keep the report idempotent
    marker = "\n\n---\n\n# Background V2"
    if marker in txt:
        txt = txt.split(marker)[0]
    OUT_REPORT.write_text(txt + section, encoding="utf-8")
    print(f"[INFO] Report updated: {OUT_REPORT}")


if __name__ == "__main__":
    main()
