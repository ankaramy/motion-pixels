"""
generate_speed_population_v1.py
-------------------------------
MOTION PIXELS — SPEED POPULATION map (Plaça Catalunya).

A NEW visual language, distinct from Flow Fields. Each moving dot is ONE tracked
pedestrian, plotted at their real world position and coloured by their real
instantaneous speed (red = slow, yellow = medium, green = fast). Pedestrians
enter, move, slow/accelerate and exit through a clean "arctic" architectural plan
with a subtle metric grid.

Key differences from Flow Fields:
  - individual pedestrians, not a collective field;
  - light / arctic cartographic aesthetic (no black, no glow, no strings, no
    trails, no heatmap);
  - speed colour scale instead of a density palette.

Real speed is preserved: dot positions are driven by the source video time axis
(`time_s`), the full tracked time range is compressed into the presentation
duration with a single uniform factor, so fast pedestrians visibly move faster
than slow ones. The footer timer shows source-video time, not animation time.

VISUALIZATION ONLY. A rendering of existing tracked pedestrian trajectories. No
retracking, recalibration, model inference, or source-data modification.

Usage:
    python generate_speed_population_v1.py
    python generate_speed_population_v1.py --preview     # faster, fewer frames
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from PIL import Image
import imageio.v2 as imageio

# Reuse the calibrated plan->world warp (no recalibration) and the dataset paths.
BEHAVIOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BEHAVIOR_DIR))
from generate_bottleneck_map_test import warp_plan_to_world
HERO_DIR = BEHAVIOR_DIR / "trace_dust_hero"
sys.path.insert(0, str(HERO_DIR))
import generate_trace_dust_hero_v3 as hero  # for DEF_TRAJ / DEF_PLAN / DEF_CALIB

DEF_TRAJ, DEF_PLAN, DEF_CALIB = hero.DEF_TRAJ, hero.DEF_PLAN, hero.DEF_CALIB

OUT_DIR = Path(__file__).resolve().parent / "outputs"
STEM = "placa_catalunya_speed_population_v1"
OUT_GIF      = OUT_DIR / f"{STEM}.gif"            # primary (= recommended variant)
OUT_MP4      = OUT_DIR / f"{STEM}.mp4"            # backup  (= recommended variant)
OUT_STILL    = OUT_DIR / f"{STEM}_still.png"
OUT_GIF_CONT = OUT_DIR / f"{STEM}_continuous.gif"
OUT_GIF_DISC = OUT_DIR / f"{STEM}_discrete.gif"
OUT_REPORT   = OUT_DIR / f"{STEM}_report.md"


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

MAP_W      = 1600                 # map width in px (height from world aspect)
FOOTER_H   = 150
ANIM_DUR   = 12.0                 # presentation seconds (within the 10-15 s brief)
FPS        = 20
GIF_WIDTH  = 660
GIF_COLORS = 48
GIF_FPS    = 12                   # GIF subsampled below MP4 fps to control size

# speed thresholds (m/s) — match the reference legend
SPEED_SLOW = 0.8                  # < SLOW         -> red
SPEED_FAST = 1.6                  # >= FAST        -> green   (between -> yellow)
SPEED_WIN_S = 0.4                 # +/- window (s) used to measure speed (denoise)
SPEED_CLIP  = 3.0                 # cap for continuous colour mapping

# dots
DOT_R      = 4.0                  # fill radius (px)
OUTLINE_W  = 1.2                  # outline thickness (px)
OUTLINE_A  = 0.55
OUTLINE_C  = np.array([0.18, 0.20, 0.24], np.float32)   # thin dark outline
FADE_FRAMES = 1.6                 # fade-in / fade-out length, in presentation frames

# speed palette
COL_SLOW = np.array([0xE5, 0x39, 0x35], np.float32) / 255.0
COL_MED  = np.array([0xFD, 0xD8, 0x35], np.float32) / 255.0
COL_FAST = np.array([0x43, 0xA0, 0x47], np.float32) / 255.0
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "speed_ryg", [tuple(COL_SLOW), tuple(COL_MED), tuple(COL_FAST)])

# arctic background styling
BG_DESAT    = 0.88                # pull toward grayscale
BG_CONTRAST = 0.55               # <1 lowers contrast around mid grey
BG_BRIGHTEN = 0.58               # blend toward white
BG_GAMMA    = 0.78               # <1 lifts midtones/shadows (brighter trees)
BG_TINT     = np.array([0.95, 0.985, 1.06], np.float32)   # cool blue-grey
SHADOW_STR  = 0.08               # subtle drop-shadow strength
SHADOW_OFF  = 4                  # shadow offset (px)

# grid
GRID_MINOR_M = 5.0
GRID_MAJOR_M = 10.0
GRID_MINOR_A = 0.07
GRID_MAJOR_A = 0.15
GRID_COLOR   = np.array([1.0, 1.0, 1.0], np.float32)

PLACE_TEXT = "Plaça Catalunya"
MAP_TEXT   = "Speed Population"

# pick a clean sans (avoid Arial if a nicer one is present)
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
# Arctic background + grid
# ---------------------------------------------------------------------------

def arctic_background(warped_rgb, W, H):
    """Desaturate, brighten, cool-tint and lower the contrast of the plan, add a
    very subtle drop shadow, and resize to the map canvas. Returns float [0,1]."""
    img = warped_rgb.astype(np.float32) / 255.0
    lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    styled = (1 - BG_DESAT) * img + BG_DESAT * lum[..., None]
    styled = np.clip(styled, 0, 1) ** BG_GAMMA          # lift shadows (brighter)
    styled = 0.5 + (styled - 0.5) * BG_CONTRAST
    styled = (1 - BG_BRIGHTEN) * styled + BG_BRIGHTEN * 1.0
    styled = np.clip(styled * BG_TINT, 0, 1)

    # subtle drop shadow from the darker (built) features of the original plan
    dark = np.clip(0.55 - lum, 0, 1)
    dark = gaussian_filter(dark, 3.0)
    sh = np.zeros_like(dark)
    sh[SHADOW_OFF:, SHADOW_OFF:] = dark[:-SHADOW_OFF, :-SHADOW_OFF]
    sh = gaussian_filter(sh, 2.0)
    styled *= (1 - SHADOW_STR * sh[..., None])

    styled = np.clip(styled, 0, 1).astype(np.float32)
    return cv2.resize(styled, (W, H), interpolation=cv2.INTER_AREA)


def add_grid(canvas, extent, W, H):
    """Composite a subtle metric grid (5 m minor + 10 m major) over the canvas."""
    xmin, xmax, ymin, ymax = extent

    def line_mask(step):
        m = np.zeros((H, W), np.float32)
        x = np.ceil(xmin / step) * step
        while x <= xmax:
            px = int(round((x - xmin) / (xmax - xmin) * (W - 1)))
            cv2.line(m, (px, 0), (px, H - 1), 1.0, 1, cv2.LINE_AA)
            x += step
        y = np.ceil(ymin / step) * step
        while y <= ymax:
            py = int(round((y - ymin) / (ymax - ymin) * (H - 1)))
            cv2.line(m, (0, py), (W - 1, py), 1.0, 1, cv2.LINE_AA)
            y += step
        return m

    for step, alpha in ((GRID_MINOR_M, GRID_MINOR_A), (GRID_MAJOR_M, GRID_MAJOR_A)):
        a = line_mask(step)[..., None] * alpha
        canvas = canvas * (1 - a) + GRID_COLOR * a
    return np.clip(canvas, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Tracks -> per-pedestrian position + real instantaneous speed
# ---------------------------------------------------------------------------

def load_tracks(extent):
    """Return list of (t, x, y, speed) per track plus dataset stats.

    Speed is measured from real world coordinates and the real `time_s` axis,
    over a +/- SPEED_WIN_S window (light denoising for known calibration jitter).
    Positions are kept exactly; only the *speed colour* is smoothed.
    """
    xmin, xmax, ymin, ymax = extent
    df = pd.read_csv(DEF_TRAJ, usecols=["frame", "time_s", "track_id",
                                        "world_x", "world_y"])
    n_total = len(df)
    n_tracks_all = int(df.track_id.nunique())
    t_total = float(df.time_s.max())

    tracks = []
    n_used = 0
    for _, g in df.sort_values(["track_id", "frame"]).groupby("track_id"):
        t = g.time_s.to_numpy(np.float64)
        x = g.world_x.to_numpy(np.float64)
        y = g.world_y.to_numpy(np.float64)
        # de-duplicate identical timestamps (keep strictly increasing time)
        keep = np.concatenate([[True], np.diff(t) > 1e-6])
        t, x, y = t[keep], x[keep], y[keep]
        if len(t) < 3:
            continue
        # keep tracks that touch the displayed plan window
        if not np.any((x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)):
            continue

        # speed via a smoothed-position central difference over ~SPEED_WIN_S
        dt_med = np.median(np.diff(t))
        sig = max(1.0, (SPEED_WIN_S / max(dt_med, 1e-3)) / 3.0)
        xs = gaussian_filter1d(x, sig, mode="nearest")
        ys = gaussian_filter1d(y, sig, mode="nearest")
        vx = np.gradient(xs, t)
        vy = np.gradient(ys, t)
        spd = np.clip(np.hypot(vx, vy), 0, SPEED_CLIP)

        tracks.append((t, x, y, spd))
        n_used += 1

    return tracks, n_total, n_tracks_all, n_used, t_total


def precompute_frames(tracks, t_total, F):
    """Sample every track onto the F presentation frames (uniform time
    compression). Returns X, Y (world), S (speed), A (alpha) arrays of shape
    (n_tracks, F); inactive/off-screen samples carry alpha 0."""
    src_t = np.linspace(0.0, t_total, F)
    fade_src = (FADE_FRAMES / max(F - 1, 1)) * t_total   # fade length in src-secs

    n = len(tracks)
    X = np.full((n, F), np.nan, np.float32)
    Y = np.full((n, F), np.nan, np.float32)
    S = np.zeros((n, F), np.float32)
    A = np.zeros((n, F), np.float32)

    for i, (t, x, y, spd) in enumerate(tracks):
        t0, t1 = t[0], t[-1]
        m = (src_t >= t0) & (src_t <= t1)
        if not m.any():
            continue
        st = src_t[m]
        X[i, m] = np.interp(st, t, x)
        Y[i, m] = np.interp(st, t, y)
        S[i, m] = np.interp(st, t, spd)
        a = np.minimum.reduce([(st - t0) / fade_src, (t1 - st) / fade_src,
                               np.ones_like(st)])
        A[i, m] = np.clip(a, 0, 1)
    return src_t, X, Y, S, A


# ---------------------------------------------------------------------------
# Dot compositing (no glow; thin outline; soft fade)
# ---------------------------------------------------------------------------

def make_dot_masks(radius, outline_w):
    off = int(np.ceil(radius + outline_w + 1))
    s = 2 * off + 1
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    d = np.hypot(xx - off, yy - off)
    fill = np.clip(radius + 0.5 - d, 0, 1)
    outer = np.clip(radius + outline_w + 0.5 - d, 0, 1)
    ring = np.clip(outer - fill, 0, 1)
    return fill.astype(np.float32), ring.astype(np.float32), off


FILL, RING, DOT_OFF = make_dot_masks(DOT_R, OUTLINE_W)
DOT_S = FILL.shape[0]


def composite_dot(frame, px, py, color, alpha):
    Hh, Ww = frame.shape[:2]
    y0, x0 = py - DOT_OFF, px - DOT_OFF
    y1, x1 = y0 + DOT_S, x0 + DOT_S
    fy0, fx0 = max(0, -y0), max(0, -x0)
    ty0, tx0 = max(0, y0), max(0, x0)
    ty1, tx1 = min(Hh, y1), min(Ww, x1)
    if ty1 <= ty0 or tx1 <= tx0:
        return
    h, w = ty1 - ty0, tx1 - tx0
    fill = FILL[fy0:fy0 + h, fx0:fx0 + w]
    ring = RING[fy0:fy0 + h, fx0:fx0 + w]
    reg = frame[ty0:ty1, tx0:tx1]
    r = (ring * alpha * OUTLINE_A)[..., None]
    reg *= (1 - r); reg += OUTLINE_C * r
    f = (fill * alpha)[..., None]
    reg *= (1 - f); reg += color * f


def speed_color(spd, discrete):
    if discrete:
        if spd < SPEED_SLOW:
            return COL_SLOW
        if spd < SPEED_FAST:
            return COL_MED
        return COL_FAST
    # continuous: anchor yellow at the slow/fast midpoint
    if spd < SPEED_SLOW:
        u = 0.5 * np.clip(spd / SPEED_SLOW, 0, 1)
    else:
        u = 0.5 + 0.5 * np.clip((spd - SPEED_SLOW) / (SPEED_FAST - SPEED_SLOW), 0, 1)
    return np.asarray(SPEED_CMAP(float(u))[:3], np.float32)


# ---------------------------------------------------------------------------
# Footer (light / arctic)
# ---------------------------------------------------------------------------

def render_footer(t_src, t_total, W, H):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("#f4f6f8")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("#f4f6f8")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # thin separator line at the top of the footer
    ax.plot([0.0, 1.0], [0.985, 0.985], color="#c9ced6", lw=1.0)

    def field(x, label, value, ha="left"):
        ax.text(x, 0.66, label, fontsize=8, color="#7b828c", ha=ha, va="center",
                family=FONT)
        ax.text(x, 0.34, value, fontsize=14, color="#20242b", ha=ha, va="center",
                family=FONT)

    field(0.022, "PLACE", PLACE_TEXT)
    field(0.210, "MAP", MAP_TEXT)

    # SPEED legend — three coloured dots
    ax.text(0.420, 0.66, "SPEED (m/s)", fontsize=8, color="#7b828c", ha="left",
            va="center", family=FONT)
    legend = [(COL_SLOW, "Slow", "< 0.8"),
              (COL_MED,  "Medium", "0.8 – 1.6"),
              (COL_FAST, "Fast", "> 1.6")]
    lx = 0.420
    for col, name, rng in legend:
        ax.scatter([lx + 0.006], [0.34], s=70, c=[tuple(col)],
                   edgecolors="#20242b", linewidths=0.5, zorder=3)
        ax.text(lx + 0.020, 0.40, name, fontsize=11, color="#20242b",
                ha="left", va="center", family=FONT)
        ax.text(lx + 0.020, 0.21, rng, fontsize=8, color="#7b828c",
                ha="left", va="center", family=FONT)
        lx += 0.135

    field(0.978, "TIME ELAPSED (SOURCE VIDEO)",
          f"{mmss(t_src)} / {mmss(t_total)}", ha="right")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


# ---------------------------------------------------------------------------
# Render one variant (continuous or discrete)
# ---------------------------------------------------------------------------

def render_variant(discrete, base, extent, src_t, X, Y, S, A,
                   W, H, gif_path, mp4_path, t_total, footers, save_still):
    xmin, xmax, ymin, ymax = extent
    F = len(src_t)
    gif_w = GIF_WIDTH - (GIF_WIDTH % 2)
    gif_h = int(round(gif_w * (H + FOOTER_H) / W)); gif_h -= gif_h % 2
    n_gif = max(2, int(round(F / FPS * GIF_FPS)))
    gif_idx = set(np.linspace(0, F - 1, n_gif).round().astype(int).tolist())
    gif_frames = []
    mp4 = imageio.get_writer(mp4_path, fps=FPS, codec="libx264", quality=8,
                             macro_block_size=None,
                             output_params=["-pix_fmt", "yuv420p"])

    still_done = False
    label = "discrete" if discrete else "continuous"
    for fi in range(F):
        frame = base.copy()
        act = np.where(A[:, fi] > 0.01)[0]
        # draw slowest last so reds read on top in crowded spots
        order = act[np.argsort(S[act, fi])[::-1]]
        for i in order:
            x, y = X[i, fi], Y[i, fi]
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                continue
            px = int(round((x - xmin) / (xmax - xmin) * (W - 1)))
            py = int(round((y - ymin) / (ymax - ymin) * (H - 1)))
            composite_dot(frame, px, py, speed_color(S[i, fi], discrete),
                          float(A[i, fi]))

        map_u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        full = np.vstack([map_u8, footers[fi]])

        mp4.append_data(full)
        if fi in gif_idx:
            pim = Image.fromarray(full).resize((gif_w, gif_h), Image.LANCZOS).convert("RGB")
            gif_frames.append(pim.quantize(colors=GIF_COLORS, method=Image.MEDIANCUT,
                                           dither=Image.NONE))
        if save_still and not still_done and src_t[fi] >= 0.40 * t_total:
            Image.fromarray(full).save(OUT_STILL); still_done = True
        if fi % 20 == 0:
            print(f"   [{label}] frame {fi}/{F}")

    mp4.close()
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=int(round(1000 / GIF_FPS)), loop=0, optimize=True,
                       disposal=2)
    return gif_path.stat().st_size / 1e6, mp4_path.stat().st_size / 1e6


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true",
                    help="shorter / fewer frames for a quick look")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dur = 6.0 if args.preview else ANIM_DUR
    F = int(round(dur * FPS))

    warped_rgb, extent = warp_plan_to_world(DEF_PLAN, DEF_CALIB)
    xmin, xmax, ymin, ymax = extent
    W = MAP_W - (MAP_W % 2)
    H = int(round(W * (ymax - ymin) / (xmax - xmin))); H -= H % 2

    print(f"[INFO] world extent X=[{xmin:.1f},{xmax:.1f}] Y=[{ymin:.1f},{ymax:.1f}] "
          f"| map {W}x{H} + footer {FOOTER_H} | font {FONT}")

    base = arctic_background(warped_rgb, W, H)
    base = add_grid(base, extent, W, H)

    tracks, n_total, n_tracks_all, n_used, t_total = load_tracks(extent)
    src_t, X, Y, S, A = precompute_frames(tracks, t_total, F)
    mean_conc = float((A > 0.01).sum(0).mean())
    print(f"[INFO] tracks used {n_used}/{n_tracks_all} | rows {n_total} | "
          f"src range 0–{t_total:.1f}s | mean dots/frame {mean_conc:.0f} | F {F}")

    # footers are identical between variants for a given frame -> render once
    footers = [render_footer(src_t[fi], t_total, W, FOOTER_H) for fi in range(F)]

    disc_gif_mb, disc_mp4_mb = render_variant(
        True, base, extent, src_t, X, Y, S, A, W, H,
        OUT_GIF_DISC, OUT_MP4, t_total, footers, save_still=True)
    cont_gif_mb, _ = render_variant(
        False, base, extent, src_t, X, Y, S, A, W, H,
        OUT_GIF_CONT, OUT_DIR / f"{STEM}_continuous.mp4", t_total, footers,
        save_still=False)

    # primary outputs = recommended (discrete) variant
    import shutil
    shutil.copyfile(OUT_GIF_DISC, OUT_GIF)

    print(f"[INFO] discrete GIF {disc_gif_mb:.1f} MB | continuous GIF "
          f"{cont_gif_mb:.1f} MB | MP4 {disc_mp4_mb:.1f} MB")
    write_report(n_total, n_tracks_all, n_used, t_total, F, (W, H), mean_conc,
                 disc_gif_mb, cont_gif_mb, disc_mp4_mb)


def write_report(n_total, n_tracks_all, n_used, t_total, F, size, mean_conc,
                 disc_gif_mb, cont_gif_mb, mp4_mb):
    W, H = size
    comp = t_total / ANIM_DUR
    txt = f"""# Plaça Catalunya — Speed Population V1

**This is a visualization-only rendering of existing tracked pedestrian
trajectories. No retracking, recalibration, model inference, or source-data
modification was performed.**

A new map in the **Speed Map** family, deliberately distinct from Flow Fields.
Every moving dot is **one tracked pedestrian**, plotted at their real world
position and coloured by their real instantaneous speed. Pedestrians enter, move,
slow down / speed up, and exit through a clean arctic architectural plan with a
subtle metric grid. No strings, no glow trails, no heatmap.

## Source data
- Trajectory file: `{DEF_TRAJ}`
- Tracks in file: **{n_tracks_all}** · tracks shown (≥3 pts, touch plan window): **{n_used}**
- Trajectory rows: **{n_total}**
- Source FPS: **~30** (from `time_s`; median Δt ≈ 0.0333 s)
- Source time range: **0 – {t_total:.1f} s** ({mmss(0)} – {mmss(t_total)})
- Mean pedestrians on screen per frame: **~{mean_conc:.0f}**

## Animation
- Presentation duration: **{ANIM_DUR:.0f} s** at **{FPS} fps** ({F} frames).
- Full source range is compressed uniformly (≈ **{comp:.1f}× real-time**); relative
  speed is preserved — fast pedestrians visibly move faster than slow ones.
- Map: **{W}×{H}px**; total frame **{W}×{H + FOOTER_H}px** with footer.
- Footer timer shows **source-video time**, not animation time.

## Speed calculation
- Per track, world position is lightly smoothed (Gaussian, σ ≈ {SPEED_WIN_S}/3 s in
  samples) to suppress known calibration jitter, then speed = |d(pos)/d(time_s)|
  via a central difference (effective ±{SPEED_WIN_S:.1f} s window).
- Positions themselves are **not** moved; only the speed *colour* is denoised.
- Speeds clipped to [0, {SPEED_CLIP:.0f}] m/s for colour mapping.

## Speed colour scale
- Thresholds: **slow < {SPEED_SLOW} m/s**, **medium {SPEED_SLOW}–{SPEED_FAST} m/s**,
  **fast > {SPEED_FAST} m/s**.
- Colours: slow `#E53935` · medium `#FDD835` · fast `#43A047`.
- Two variants produced:
  - **Discrete** (3 bins) — `{OUT_GIF_DISC.name}` ({disc_gif_mb:.1f} MB)
  - **Continuous** (red→yellow→green, yellow anchored at the {SPEED_SLOW}–{SPEED_FAST}
    midpoint) — `{OUT_GIF_CONT.name}` ({cont_gif_mb:.1f} MB)

## Dots
- Fill radius **{DOT_R} px**, thin dark outline ({OUTLINE_W} px, α {OUTLINE_A}) for
  readability; **no glow**.
- Soft fade-in / fade-out over **{FADE_FRAMES} frames** at each pedestrian's
  start/end (hides appearance/disappearance; no teleporting).
- In crowded spots, slower (red) dots are drawn on top so congestion reads.

## Grid
- Subtle metric grid: **{GRID_MINOR_M:.0f} m minor** (α {GRID_MINOR_A}) +
  **{GRID_MAJOR_M:.0f} m major** (α {GRID_MAJOR_A}), thin white lines, no labels.
- The 10 m major grid reads best as the dominant rhythm; the 5 m minor adds fine
  spatial texture without dominating.

## Background styling (arctic render)
- Desaturate {BG_DESAT} toward luminance, lower contrast ×{BG_CONTRAST}, brighten
  {BG_BRIGHTEN} toward white, cool blue-grey tint {list(np.round(BG_TINT,3))}.
- Subtle drop shadow (strength {SHADOW_STR}, offset {SHADOW_OFF} px) from the darker
  built features, for gentle depth. No black background, no neon, no glow.
- Plan warped from the existing calibrated `plan_points_px → world_points`
  homography (no recalibration).

## Output paths
- `{OUT_GIF.name}` — primary GIF (= discrete, recommended)
- `{OUT_MP4.name}` — primary MP4 backup (H.264, {W}×{H + FOOTER_H})
- `{OUT_STILL.name}` — still ({W}×{H + FOOTER_H})
- `{OUT_GIF_DISC.name}` — discrete 3-bin variant
- `{OUT_GIF_CONT.name}` — continuous variant
- this report.

## Recommendation: continuous vs discrete
**Discrete (3 bins) is recommended** as the default Speed Population render: the
three colours map directly onto the footer legend, congestion (red) vs free flow
(green) reads instantly at a glance, and it survives GIF colour quantization
cleanly. Use the **continuous** variant when subtle within-band speed gradients
(e.g. gradual deceleration approaching a bottleneck) are the point of the figure.

This establishes the distinct Speed Map visual language: light, cartographic,
architectural and analytical — clearly separate from the dark Flow Fields family.
"""
    OUT_REPORT.write_text(txt, encoding="utf-8")
    print(f"[INFO] Report saved: {OUT_REPORT}")


if __name__ == "__main__":
    main()
