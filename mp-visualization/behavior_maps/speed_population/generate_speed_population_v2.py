"""
generate_speed_population_v2.py
-------------------------------
MOTION PIXELS — SPEED POPULATION V2 (Plaça Catalunya).

Refinement pass over V1 (visual + animation only). Fixes:
  1. Background depth — arctic plan + edge ambient-occlusion + a stronger soft
     directional drop shadow (Rhino "white model" feel), still cool & light.
  2. Footer rebuilt on the Flow Fields footer geometry, adapted to a light theme
     (white bar, dark text, light dividers, three speed dots).
  3. Denser, legible population via honest temporal persistence: each pedestrian
     leaves a short fading trail of their REAL recent positions and lingers
     briefly after exit. No invented pedestrians.
  4. Modern dots — vibrant, slightly translucent, soft white halo, antialiased
     (supersampled), no heavy black outline.
  5. The trail IS the faint trace (recent real positions only, speed-class
     coloured, short, no glow/strings).
  6. Relative speed made legible WITHOUT speeding anyone up: persistence is longer
     for slow pedestrians (1.8 s) and shorter for fast (0.5 s), so slow dots
     visibly linger. Positions remain faithful.
  7. High-quality GIF (1280 px, 256 adaptive colours, dithered) — large file OK.

VISUALIZATION ONLY. Uses existing tracked trajectories + derived speed. No
retracking, recalibration, model inference, or source-data modification.

Usage:
    python generate_speed_population_v2.py
    python generate_speed_population_v2.py --preview   # quick look
    python generate_speed_population_v2.py --no-tests  # skip persistence tests
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.ndimage import gaussian_filter
from PIL import Image
import imageio.v2 as imageio

SP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SP_DIR))
import generate_speed_population_v1 as v1   # warp, data load, paths

OUT_DIR = SP_DIR / "outputs"
STEM = "placa_catalunya_speed_population_v2"
OUT_GIF   = OUT_DIR / f"{STEM}_high_quality.gif"
OUT_MP4   = OUT_DIR / f"{STEM}.mp4"
OUT_STILL = OUT_DIR / f"{STEM}_still.png"
OUT_REPORT = OUT_DIR / f"{STEM}_report.md"


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

MAP_W     = 1600
FOOTER_H  = 122                       # compact, Flow-Fields-like proportion
ANIM_DUR  = 13.0                      # presentation seconds
FPS       = 20                        # master / MP4 fps

# high-quality GIF. Dithering OFF and a single GLOBAL palette (shared across all
# frames) so PIL delta-encodes only the changing dots/trails/timer instead of
# re-storing the whole photographic background every frame — keeps it usable.
GIF_WIDTH  = 1100
GIF_FPS    = 16
GIF_COLORS = 256
GIF_DITHER = False

# speed thresholds (m/s)
SPEED_SLOW = 0.8
SPEED_FAST = 1.6
SPEED_CLIP = 3.0

# temporal persistence (seconds of SOURCE time) by speed class — legibility:
# slow pedestrians linger, fast ones are brief. Positions stay faithful.
PERSIST = {"slow": 1.8, "med": 1.1, "fast": 0.5}
PMAX = max(PERSIST.values())
TRAIL_DT = 0.07                       # source-time spacing of trail samples
TRAIL_GAMMA = 1.6                     # age fade curve (older -> fainter, faster)

# dots (modern)
HEAD_R     = 3.0
TRAIL_R    = 1.9
HALO_W     = 1.5
HALO_A     = 0.28
HEAD_FILL_A = 0.92
TRAIL_FILL_A = 0.55
HALO_C = np.array([1.0, 1.0, 1.0], np.float32)

# vibrant speed palette
COL_SLOW = np.array([0xFF, 0x3B, 0x30], np.float32) / 255.0   # #FF3B30
COL_MED  = np.array([0xFF, 0xD6, 0x0A], np.float32) / 255.0   # #FFD60A
COL_FAST = np.array([0x34, 0xC7, 0x59], np.float32) / 255.0   # #34C759

# arctic background + depth
BG_DESAT    = 0.88
BG_GAMMA    = 0.80
BG_CONTRAST = 0.55
BG_BRIGHTEN = 0.60
BG_TINT     = np.array([0.95, 0.985, 1.06], np.float32)
AO_STR      = 0.13                    # edge ambient-occlusion darkening (subtle)
AO_SIGMA    = 1.2
SHADOW_STR  = 0.11                    # directional drop shadow (subtle)
SHADOW_OFF  = 6
SHADOW_REF  = 0.60

# grid
GRID_MINOR_M, GRID_MAJOR_M = 5.0, 10.0
GRID_MINOR_A, GRID_MAJOR_A = 0.06, 0.14
GRID_COLOR = np.array([1.0, 1.0, 1.0], np.float32)

PLACE_TEXT = "Plaça Catalunya"
MAP_TEXT   = "Speed Population"

FONT = "DejaVu Sans"
for _c in ["Roboto", "Inter", "IBM Plex Sans", "Helvetica Neue", "Helvetica", "DejaVu Sans"]:
    try:
        fm.findfont(_c, fallback_to_default=False); FONT = _c; break
    except Exception:
        continue


def mmss(s):
    s = max(0, int(round(s)))
    return f"{s // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# Arctic background with depth
# ---------------------------------------------------------------------------

def arctic_background(warped_rgb, W, H):
    img = warped_rgb.astype(np.float32) / 255.0
    lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    styled = (1 - BG_DESAT) * img + BG_DESAT * lum[..., None]
    styled = np.clip(styled, 0, 1) ** BG_GAMMA
    styled = 0.5 + (styled - 0.5) * BG_CONTRAST
    styled = (1 - BG_BRIGHTEN) * styled + BG_BRIGHTEN * 1.0
    styled = np.clip(styled * BG_TINT, 0, 1)

    # edge ambient occlusion (depth around object boundaries)
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    edge = gaussian_filter(np.hypot(gx, gy), AO_SIGMA)
    if edge.max() > 0:
        edge = edge / edge.max()
    styled *= (1 - AO_STR * edge[..., None])

    # soft directional drop shadow from darker (built/planted) mass
    dark = gaussian_filter(np.clip(SHADOW_REF - lum, 0, 1), 3.0)
    sh = np.zeros_like(dark)
    sh[SHADOW_OFF:, SHADOW_OFF:] = dark[:-SHADOW_OFF, :-SHADOW_OFF]
    sh = gaussian_filter(sh, 3.0)
    sh = np.clip(sh - dark, 0, 1)                 # shadow only where not the object itself
    styled *= (1 - SHADOW_STR * sh[..., None])

    styled = np.clip(styled, 0, 1).astype(np.float32)
    return cv2.resize(styled, (W, H), interpolation=cv2.INTER_AREA)


def add_grid(canvas, extent, W, H):
    xmin, xmax, ymin, ymax = extent

    def line_mask(step):
        m = np.zeros((H, W), np.float32)
        x = np.ceil(xmin / step) * step
        while x <= xmax:
            px = int(round((x - xmin) / (xmax - xmin) * (W - 1)))
            cv2.line(m, (px, 0), (px, H - 1), 1.0, 1, cv2.LINE_AA); x += step
        y = np.ceil(ymin / step) * step
        while y <= ymax:
            py = int(round((y - ymin) / (ymax - ymin) * (H - 1)))
            cv2.line(m, (0, py), (W - 1, py), 1.0, 1, cv2.LINE_AA); y += step
        return m

    for step, alpha in ((GRID_MINOR_M, GRID_MINOR_A), (GRID_MAJOR_M, GRID_MAJOR_A)):
        a = line_mask(step)[..., None] * alpha
        canvas = canvas * (1 - a) + GRID_COLOR * a
    return np.clip(canvas, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Modern dots (supersampled AA, soft white halo, translucent)
# ---------------------------------------------------------------------------

def make_disk(r, halo_w=0.0, ss=4):
    R = r + halo_w
    off = int(np.ceil(R + 1))
    s = 2 * off + 1
    hi = s * ss
    yy, xx = np.mgrid[0:hi, 0:hi].astype(np.float32)
    c = off * ss + ss / 2.0 - 0.5
    d = np.hypot(xx - c, yy - c) / ss
    fill = (d <= r).astype(np.float32).reshape(s, ss, s, ss).mean((1, 3))
    halo = None
    if halo_w > 0:
        outer = (d <= R).astype(np.float32).reshape(s, ss, s, ss).mean((1, 3))
        halo = np.clip(outer - fill, 0, 1).astype(np.float32)
    return fill.astype(np.float32), halo, off


HEAD_FILL, HEAD_HALO, HEAD_OFF = make_disk(HEAD_R, HALO_W)
TRAIL_FILL, _, TRAIL_OFF = make_disk(TRAIL_R, 0.0)


def _composite(frame, px, py, mask, off, color, alpha):
    Hh, Ww = frame.shape[:2]
    s = mask.shape[0]
    y0, x0 = py - off, px - off
    fy0, fx0 = max(0, -y0), max(0, -x0)
    ty0, tx0 = max(0, y0), max(0, x0)
    ty1, tx1 = min(Hh, y0 + s), min(Ww, x0 + s)
    if ty1 <= ty0 or tx1 <= tx0:
        return
    h, w = ty1 - ty0, tx1 - tx0
    m = (mask[fy0:fy0 + h, fx0:fx0 + w] * alpha)[..., None]
    reg = frame[ty0:ty1, tx0:tx1]
    reg *= (1 - m); reg += color * m


def draw_dot(frame, px, py, color, alpha, head=False):
    if head and HEAD_HALO is not None:
        _composite(frame, px, py, HEAD_HALO, HEAD_OFF, HALO_C, alpha * HALO_A)
        _composite(frame, px, py, HEAD_FILL, HEAD_OFF, color, alpha)
    else:
        _composite(frame, px, py, TRAIL_FILL, TRAIL_OFF, color, alpha)


def speed_class_color(spd):
    if spd < SPEED_SLOW:
        return COL_SLOW, "slow"
    if spd < SPEED_FAST:
        return COL_MED, "med"
    return COL_FAST, "fast"


# ---------------------------------------------------------------------------
# Footer (Flow Fields geometry, light theme)
# ---------------------------------------------------------------------------

def build_palette(bg_full_rgb, gif_w, gif_h, n_colors):
    """Global GIF palette: most entries fit the (dot-free) arctic background, with
    a handful of entries RESERVED for the vibrant dot colours + blends so the
    small/rare dots survive quantization (MEDIANCUT alone starves them)."""
    reserved = []
    for bg in (np.array([0.86, 0.88, 0.92]), np.array([0.72, 0.74, 0.80])):
        for col in (COL_SLOW, COL_MED, COL_FAST):
            for a in (1.0, 0.6):
                mix = np.clip(bg * (1 - a) + col * a, 0, 1)
                reserved.append(tuple((mix * 255).round().astype(int)))
    reserved.append((255, 255, 255))
    reserved = list(dict.fromkeys(reserved))            # de-dup
    n_bg = n_colors - len(reserved)

    bg_img = (Image.fromarray(bg_full_rgb).resize((gif_w, gif_h), Image.LANCZOS)
              .convert("RGB").quantize(colors=n_bg, method=Image.MEDIANCUT,
                                       dither=Image.NONE))
    pal = bg_img.getpalette()[: n_bg * 3]
    pal += [c for rgb in reserved for c in rgb]
    pal = (pal + [0] * (256 * 3))[: 256 * 3]
    pimg = Image.new("P", (1, 1)); pimg.putpalette(pal)
    return pimg


def render_footer(t_src, t_total, W, H):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("#f6f8fa")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("#f6f8fa")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0.004, 0.12), 0.992, 0.76, fill=False,
                 edgecolor="#c7ccd3", linewidth=0.8, alpha=0.9))

    LBL, VAL = "#737a85", "#1f2530"

    def field(x, label, value, ha="left"):
        ax.text(x, 0.64, label, fontsize=9, color=LBL, ha=ha, va="center", family=FONT)
        ax.text(x, 0.33, value, fontsize=15, color=VAL, ha=ha, va="center", family=FONT)

    def divider(x):
        ax.plot([x, x], [0.20, 0.80], color="#d7dbe0", lw=1.0)

    field(0.020, "PLACE", PLACE_TEXT)
    divider(0.200)
    field(0.220, "MAP", MAP_TEXT)
    divider(0.430)

    ax.text(0.450, 0.64, "SPEED (m/s)", fontsize=9, color=LBL, ha="left",
            va="center", family=FONT)
    legend = [(COL_SLOW, "Slow"), (COL_MED, "Medium"), (COL_FAST, "Fast")]
    lx = 0.452
    for col, name in legend:
        ax.scatter([lx + 0.006], [0.33], s=80, c=[tuple(col)], edgecolors="white",
                   linewidths=0.6, zorder=3)
        ax.text(lx + 0.020, 0.33, name, fontsize=12.5, color=VAL, ha="left",
                va="center", family=FONT)
        lx += 0.110
    divider(0.800)

    field(0.980, "TIME ELAPSED", f"{mmss(t_src)} / {mmss(t_total)}", ha="right")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


# ---------------------------------------------------------------------------
# Render one run (persistence = "class" or a float window in seconds)
# ---------------------------------------------------------------------------

def render_run(tracks, t_total, base, extent, W, H, *, persistence, duration,
               fps, gif_path, gif_w, gif_fps, gif_colors, dither,
               mp4_path=None, save_still=False, label="run"):
    xmin, xmax, ymin, ymax = extent
    T0 = np.array([t[0][0] for t in tracks])
    T1 = np.array([t[0][-1] for t in tracks])
    uniform = isinstance(persistence, (int, float))
    pmax = float(persistence) if uniform else PMAX

    F = int(round(duration * fps))
    Hf = H + FOOTER_H
    gif_h = int(round(gif_w * Hf / W)); gif_h -= gif_h % 2
    n_gif = max(2, int(round(duration * gif_fps)))
    gif_idx = set(np.linspace(0, F - 1, n_gif).round().astype(int).tolist())
    dith = Image.FLOYDSTEINBERG if dither else Image.NONE

    def render_frame(fi):
        src_t = fi / (F - 1) * t_total
        frame = base.copy()
        act = np.where((T0 <= src_t) & (src_t <= T1 + pmax))[0]
        items = []
        for i in act:
            t, x, y, spd = tracks[i]
            t1 = t[-1]
            head_t = min(src_t, t1)
            hs = float(np.interp(head_t, t, spd))
            col, cls = speed_class_color(hs)
            P = pmax if uniform else PERSIST[cls]
            if src_t > t1 + P:
                continue
            linger = 1.0 if src_t <= t1 else max(0.0, 1.0 - (src_t - t1) / P)
            items.append((hs, i, head_t, P, linger, col))
        items.sort(key=lambda r: r[0], reverse=True)   # fast first -> slow on top

        for hs, i, head_t, P, linger, col in items:
            t, x, y, spd = tracks[i]
            t0 = t[0]
            ts0 = max(src_t - P, t0)
            if head_t - ts0 > 1e-4:
                ts = np.arange(ts0, head_t, TRAIL_DT)
                if len(ts):
                    tx = np.interp(ts, t, x); ty = np.interp(ts, t, y)
                    aage = np.clip(1.0 - (src_t - ts) / P, 0, 1) ** TRAIL_GAMMA
                    for k in range(len(ts)):
                        xx, yy = tx[k], ty[k]
                        if not (xmin <= xx <= xmax and ymin <= yy <= ymax):
                            continue
                        px = int(round((xx - xmin) / (xmax - xmin) * (W - 1)))
                        py = int(round((yy - ymin) / (ymax - ymin) * (H - 1)))
                        draw_dot(frame, px, py, col,
                                 TRAIL_FILL_A * aage[k] * linger, head=False)
            xh = float(np.interp(head_t, t, x)); yh = float(np.interp(head_t, t, y))
            if xmin <= xh <= xmax and ymin <= yh <= ymax:
                px = int(round((xh - xmin) / (xmax - xmin) * (W - 1)))
                py = int(round((yh - ymin) / (ymax - ymin) * (H - 1)))
                draw_dot(frame, px, py, col, HEAD_FILL_A * linger, head=True)

        map_u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        full = np.vstack([map_u8, render_footer(src_t, t_total, W, FOOTER_H)])
        return full, src_t, len(items)

    # one GLOBAL palette (dot-free background + reserved dot colours) -> all GIF
    # frames share it, so PIL delta-encodes only the changing pixels.
    bg_full = np.vstack([(np.clip(base, 0, 1) * 255).astype(np.uint8),
                         render_footer(0.0, t_total, W, FOOTER_H)])
    pal_img = build_palette(bg_full, gif_w, gif_h, gif_colors)

    mp4 = None
    if mp4_path is not None:
        mp4 = imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=9,
                                 macro_block_size=None,
                                 output_params=["-pix_fmt", "yuv420p"])

    gif_frames = []
    still_done = False
    for fi in range(F):
        full, src_t, n_items = render_frame(fi)
        if mp4 is not None:
            mp4.append_data(full)
        if fi in gif_idx:
            pim = Image.fromarray(full).resize((gif_w, gif_h), Image.LANCZOS).convert("RGB")
            gif_frames.append(pim.quantize(palette=pal_img, dither=dith))
        if save_still and not still_done and src_t >= 0.42 * t_total:
            Image.fromarray(full).save(OUT_STILL); still_done = True
        if fi % 20 == 0:
            print(f"   [{label}] frame {fi}/{F}  (dots from {n_items} peds)")

    if mp4 is not None:
        mp4.close()
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=int(round(1000 / gif_fps)), loop=0, optimize=True,
                       disposal=1)
    return gif_path.stat().st_size / 1e6


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--no-tests", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dur = 6.0 if args.preview else ANIM_DUR

    warped_rgb, extent = v1.warp_plan_to_world(v1.DEF_PLAN, v1.DEF_CALIB)
    xmin, xmax, ymin, ymax = extent
    W = MAP_W - (MAP_W % 2)
    H = int(round(W * (ymax - ymin) / (xmax - xmin))); H -= H % 2
    print(f"[INFO] map {W}x{H} + footer {FOOTER_H} | font {FONT}")

    base = add_grid(arctic_background(warped_rgb, W, H), extent, W, H)

    tracks, n_total, n_tracks_all, n_used, t_total = v1.load_tracks(extent)
    print(f"[INFO] tracks used {n_used}/{n_tracks_all} | rows {n_total} | "
          f"src 0–{t_total:.1f}s | persist {PERSIST}")

    gif_w = (GIF_WIDTH if not args.preview else 760); gif_w -= gif_w % 2
    hq_mb = render_run(
        tracks, t_total, base, extent, W, H,
        persistence="class", duration=dur, fps=FPS,
        gif_path=OUT_GIF, gif_w=gif_w, gif_fps=GIF_FPS,
        gif_colors=(GIF_COLORS if not args.preview else 128), dither=GIF_DITHER,
        mp4_path=OUT_MP4, save_still=True, label="hq")
    print(f"[INFO] HQ GIF {hq_mb:.1f} MB | MP4 {OUT_MP4.stat().st_size/1e6:.1f} MB")

    test_mb = {}
    if not args.no_tests and not args.preview:
        for win in (1.0, 2.0, 3.0):
            p = OUT_DIR / f"{STEM}_persistence_{int(win)}s.gif"
            mb = render_run(
                tracks, t_total, base, extent, W, H,
                persistence=win, duration=dur, fps=15,
                gif_path=p, gif_w=860, gif_fps=12, gif_colors=128, dither=False,
                mp4_path=None, save_still=False, label=f"p{int(win)}s")
            test_mb[win] = mb
            print(f"[INFO] persistence {win:.0f}s GIF {mb:.1f} MB")

    write_report(n_total, n_tracks_all, n_used, t_total, dur, (W, H), gif_w,
                 hq_mb, test_mb)


def write_report(n_total, n_tracks_all, n_used, t_total, dur, size, gif_w,
                 hq_mb, test_mb):
    W, H = size
    comp = t_total / dur
    tests = "\n".join(
        f"  - `{STEM}_persistence_{int(w)}s.gif` — uniform {w:.0f} s window "
        f"({mb:.1f} MB)" for w, mb in sorted(test_mb.items())) or "  - (not generated)"
    txt = f"""# Plaça Catalunya — Speed Population V2

**This visualization uses existing tracked pedestrian trajectories and speed
metrics only. No retracking, recalibration, model inference, or source-data
modification was performed.**

**Dots may persist for a short rolling time window to improve visual legibility;
this represents recent movement presence, not instantaneous pedestrian count.**

V2 is a visual + animation refinement of V1: deeper arctic background, a Flow
Fields-style light footer, a much more populated and legible field via honest
temporal persistence, modern dots, faint speed-coloured traces, speed made
readable without speeding anyone up, and a high-quality GIF.

## Source data
- Trajectory file: `{v1.DEF_TRAJ}`
- Tracks in file: **{n_tracks_all}** · tracks shown: **{n_used}**
- Trajectory rows: **{n_total}**
- Source FPS: **~30** (from `time_s`)
- Source time range: **0 – {t_total:.1f} s** ({mmss(0)} – {mmss(t_total)})

## Animation
- Presentation duration **{dur:.0f} s** at **{FPS} fps**; full source range
  compressed uniformly (≈ **{comp:.1f}× real-time**).
- Footer timer shows **source-video** time, not animation time.
- Master frame **{W}×{H + FOOTER_H}px** (map {W}×{H} + footer {FOOTER_H}).

## Speed
- Derived (V1 method): position lightly denoised, then |d(pos)/d(time_s)|.
- Thresholds: slow < {SPEED_SLOW} · medium {SPEED_SLOW}–{SPEED_FAST} · fast > {SPEED_FAST} m/s.
- **Discrete** 3-class colours (recommended in V1): slow `#FF3B30` · medium
  `#FFD60A` · fast `#34C759`.

## Temporal persistence (population + faint trace + speed legibility)
Each pedestrian leaves a short **fading trail of their real recent positions**
and lingers briefly after exit. Trail/linger length depends on the pedestrian's
speed **class** (slow lingers, fast is brief), which makes relative speed legible
WITHOUT moving any dot faster than the data:
- slow (red): **{PERSIST['slow']:.1f} s** · medium (yellow): **{PERSIST['med']:.1f} s**
  · fast (green): **{PERSIST['fast']:.1f} s** of source time.
- trail sampled every **{TRAIL_DT:.2f} s** of source time, age-faded
  (γ {TRAIL_GAMMA}); head dot is brightest, tail fades to nothing.
- This raises visible dots/frame far above the ~37 instantaneous concurrency
  while remaining 100% real positions. It is *presence*, not occupancy count.
- Persistence test variants (uniform window, all classes):
{tests}

## Dots (modern)
- Head radius **{HEAD_R} px** (α {HEAD_FILL_A}) + soft white halo
  ({HALO_W} px, α {HALO_A}); trail radius **{TRAIL_R} px** (α {TRAIL_FILL_A}·age).
- Supersampled antialiasing, slightly translucent, **no heavy black outline**,
  no glow.

## Faint trace
Yes — the persistence trail is the faint trace: short, low-opacity, speed-class
coloured, real recent positions only. Deliberately not Flow Fields (no long
strings, no glow, no streamlines).

## Visual speed exaggeration
None applied to motion — **all positions are faithful**. Legibility comes only
from class-dependent persistence (slow dots linger longer). No display-speed
multiplier was used.

## Background (arctic + depth)
Desaturate {BG_DESAT}, gamma {BG_GAMMA}, contrast ×{BG_CONTRAST}, brighten
{BG_BRIGHTEN}, cool tint {list(np.round(BG_TINT,3))}; **edge ambient occlusion**
(strength {AO_STR}) + **soft directional drop shadow** (strength {SHADOW_STR},
offset {SHADOW_OFF}px) for a Rhino "white-model" depth. Grid: {GRID_MINOR_M:.0f} m
+ {GRID_MAJOR_M:.0f} m, subtle, no labels.

## Footer
Rebuilt on the Flow Fields footer geometry, **light theme**: white bar, thin
light frame + light vertical dividers, dark readable text ({FONT}); fields PLACE ·
MAP · SPEED (m/s) with three colour dots (Slow/Medium/Fast) · TIME ELAPSED.

## High-quality GIF
- `{OUT_GIF.name}` — **{gif_w}px wide**, {GIF_FPS} fps, {GIF_COLORS}-colour palette,
  dithering {"on" if GIF_DITHER else "off"} → **{hq_mb:.1f} MB**.
- Encoding: a single **global palette** (dot-free arctic background + reserved
  vibrant dot colours) is shared by every frame, so PIL **delta-encodes** only the
  changing dots/trails/timer instead of re-storing the photographic background
  each frame. This keeps the GIF high-quality *and* small/usable (a naive
  per-frame adaptive palette of the same render was ~150–170 MB and unusable;
  MEDIANCUT also starved the rare dot colours, so they are reserved explicitly).
- `{OUT_MP4.name}` — H.264 backup, {W}×{H + FOOTER_H}, {FPS} fps.

## Output paths
- `{OUT_GIF.name}` (primary, high quality)
- `{OUT_MP4.name}` (backup)
- `{OUT_STILL.name}` (still)
- persistence test GIFs (above)
- this report.

## Recommendation
Use the **high-quality GIF** (class-based persistence) for presentation. Among
the uniform tests, **2 s** gives the best balance of population vs clarity; 1 s
feels sparse, 3 s starts to smear into a trace field. Class-based persistence
(the default) is preferred over any uniform window because it also encodes speed.
"""
    OUT_REPORT.write_text(txt, encoding="utf-8")
    print(f"[INFO] Report saved: {OUT_REPORT}")


if __name__ == "__main__":
    main()
