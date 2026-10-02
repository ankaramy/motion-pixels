"""
generate_speed_population_v3.py
-------------------------------
MOTION PIXELS — SPEED POPULATION V3 (Plaça Catalunya).

A targeted modification of V2 (reuses V2's background / grid / dot helpers). V3:
  1. Footer matched much more closely to the Flow Fields footer (position logic,
     thin frame + dividers, compact grey-label / dark-value hierarchy, spacing).
  2. The arctic satellite plan now fills the WHOLE canvas; the legend is a
     bottom-aligned frosted-white HUD overlaid ON the image (no separate band).
  3. All dots move slower (presentation-time remapping): the full source range is
     stretched over a longer presentation, so motion is perceivable. The legend
     timer still shows real source-video time.
  4. Longer, slowly-fading traces behind each dot (test 3 / 5 / 7 s of source
     time) — "moving dots + a fading memory of where they came from". No glow,
     no thick strings; subtle.
  5. New, more elegant speed palettes (warm + cool), replacing red/yellow/green.
  6. GIF preserves dot + trace colours (global palette with reserved colours).

VISUALIZATION ONLY. Uses existing tracked trajectories + derived speed. No
retracking, recalibration, model inference, or source-data modification.

Usage:
    python generate_speed_population_v3.py
    python generate_speed_population_v3.py --preview
    python generate_speed_population_v3.py --no-tests
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import imageio.v2 as imageio

SP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SP_DIR))
import generate_speed_population_v2 as v2          # bg / grid / dots / load
v1 = v2.v1

OUT_DIR = SP_DIR / "outputs"
STEM = "placa_catalunya_speed_population_v3"

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

MAP_W    = 1600
ANIM_DUR = 22.0                 # longer => slower dots (was 13 s in V2)
FPS      = 16
GIF_FPS  = 16
GIF_WIDTH  = 1100
GIF_COLORS = 256

SPEED_SLOW, SPEED_FAST = 0.8, 1.6

TRACE_SECONDS = 5.0             # default trace length (source seconds)
TRACE_TESTS   = [3.0, 5.0, 7.0]
TRAIL_DT      = 0.14            # source-time spacing of trace samples
TRACE_GAMMA   = 1.9            # slow backward fade
TRAIL_FILL_A  = 0.50
HEAD_FILL_A   = 0.95
LINGER_OUT    = 0.8            # fade-out after a pedestrian exits (source s)

# elegant palettes [slow, medium, fast]
PALETTES = {
    "warm": [np.array([0xE8, 0x5D, 0x4F], np.float32) / 255,   # coral / vermilion
             np.array([0xF2, 0xB8, 0x4B], np.float32) / 255,   # warm amber
             np.array([0x00, 0xAF, 0xA6], np.float32) / 255],  # teal
    "cool": [np.array([0x8B, 0x5C, 0xF6], np.float32) / 255,   # violet
             np.array([0xF2, 0xB8, 0x4B], np.float32) / 255,   # amber
             np.array([0x00, 0xBC, 0xD4], np.float32) / 255],  # cyan
}

# HUD
HUD_H = 96
FONT = v2.FONT
PLACE_TEXT, MAP_TEXT = "Plaça Catalunya", "Speed Population"


def mmss(s):
    s = max(0, int(round(s)))
    return f"{s // 60:02d}:{s % 60:02d}"


def speed_color(spd, palette):
    if spd < SPEED_SLOW:
        return palette[0]
    if spd < SPEED_FAST:
        return palette[1]
    return palette[2]


# ---------------------------------------------------------------------------
# Global GIF palette (background + reserved dot/trace/HUD colours)
# ---------------------------------------------------------------------------

def build_palette(bg_full_rgb, gif_w, gif_h, n_colors, palette):
    reserved = []
    for bg in (np.array([0.88, 0.90, 0.94]), np.array([0.72, 0.74, 0.80])):
        for col in palette:
            for a in (1.0, 0.7, 0.45, 0.28):       # incl. faint trace blends
                reserved.append(tuple((np.clip(bg * (1 - a) + col * a, 0, 1)
                                       * 255).round().astype(int)))
    reserved += [(255, 255, 255), (27, 31, 39), (95, 102, 112), (154, 160, 168)]
    reserved = list(dict.fromkeys(reserved))
    n_bg = n_colors - len(reserved)

    bg_img = (Image.fromarray(bg_full_rgb).resize((gif_w, gif_h), Image.LANCZOS)
              .convert("RGB").quantize(colors=n_bg, method=Image.MEDIANCUT,
                                       dither=Image.NONE))
    pal = bg_img.getpalette()[: n_bg * 3]
    pal += [c for rgb in reserved for c in rgb]
    pal = (pal + [0] * (256 * 3))[: 256 * 3]
    pimg = Image.new("P", (1, 1)); pimg.putpalette(pal)
    return pimg


# ---------------------------------------------------------------------------
# HUD overlay (Flow Fields footer geometry, frosted light panel ON the image)
# ---------------------------------------------------------------------------

def render_hud(t_src, t_total, W, H, palette):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("none")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # frosted white panel, thin dark-grey border, rounded (HUD look)
    ax.add_patch(FancyBboxPatch((0.012, 0.16), 0.976, 0.68,
                 boxstyle="round,pad=0,rounding_size=0.06",
                 facecolor="white", alpha=0.72, edgecolor="none", zorder=1))
    ax.add_patch(FancyBboxPatch((0.012, 0.16), 0.976, 0.68,
                 boxstyle="round,pad=0,rounding_size=0.06",
                 facecolor="none", edgecolor="#8d939b", linewidth=1.0,
                 alpha=0.9, zorder=3))

    LBL, VAL = "#5f6670", "#1b1f27"

    def field(x, label, value, ha="left"):
        ax.text(x, 0.60, label, fontsize=9, color=LBL, ha=ha, va="center",
                family=FONT, zorder=4)
        ax.text(x, 0.37, value, fontsize=14, color=VAL, ha=ha, va="center",
                family=FONT, zorder=4)

    def divider(x):
        ax.plot([x, x], [0.30, 0.70], color="#c7ccd3", lw=1.0, zorder=4)

    field(0.030, "PLACE", PLACE_TEXT)
    divider(0.232)
    field(0.250, "MAP", MAP_TEXT)
    divider(0.470)

    ax.text(0.486, 0.60, "SPEED (m/s)", fontsize=9, color=LBL, ha="left",
            va="center", family=FONT, zorder=4)
    lx = 0.488
    for col, name in zip(palette, ("Slow", "Medium", "Fast")):
        ax.scatter([lx + 0.006], [0.37], s=78, c=[tuple(col)], edgecolors="white",
                   linewidths=0.6, zorder=5)
        ax.text(lx + 0.019, 0.37, name, fontsize=12, color=VAL, ha="left",
                va="center", family=FONT, zorder=4)
        lx += 0.100
    divider(0.812)

    field(0.972, "TIME ELAPSED", f"{mmss(t_src)} / {mmss(t_total)}", ha="right")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32) / 255.0
    plt.close(fig)
    return rgba


def composite_hud(frame, hud_rgba):
    h = hud_rgba.shape[0]
    a = hud_rgba[..., 3:4]
    reg = frame[-h:]
    reg *= (1 - a); reg += hud_rgba[..., :3] * a


# ---------------------------------------------------------------------------
# Render one run
# ---------------------------------------------------------------------------

def render_run(tracks, t_total, base, extent, W, H, *, palette, trace_s,
               duration, fps, gif_path, gif_w, gif_fps, gif_colors,
               mp4_path=None, still_path=None, label="run"):
    xmin, xmax, ymin, ymax = extent
    T0 = np.array([t[0][0] for t in tracks])
    T1 = np.array([t[0][-1] for t in tracks])

    F = int(round(duration * fps))
    gif_h = int(round(gif_w * H / W)); gif_h -= gif_h % 2
    n_gif = max(2, int(round(duration * gif_fps)))
    gif_idx = set(np.linspace(0, F - 1, n_gif).round().astype(int).tolist())

    hud_cache = {}

    def hud_for(src_t):
        key = f"{mmss(src_t)}"                      # timer only changes per second
        h = hud_cache.get(key)
        if h is None:
            h = render_hud(src_t, t_total, W, HUD_H, palette); hud_cache[key] = h
        return h

    def render_frame(fi):
        src_t = fi / (F - 1) * t_total
        frame = base.copy()
        act = np.where((T0 <= src_t) & (src_t <= T1 + LINGER_OUT))[0]
        items = []
        for i in act:
            t, x, y, spd = tracks[i]
            t1 = t[-1]
            head_t = min(src_t, t1)
            hs = float(np.interp(head_t, t, spd))
            linger = 1.0 if src_t <= t1 else max(0.0, 1.0 - (src_t - t1) / LINGER_OUT)
            items.append((hs, i, head_t, linger, speed_color(hs, palette)))
        items.sort(key=lambda r: r[0], reverse=True)   # fast first -> slow on top

        for hs, i, head_t, linger, col in items:
            t, x, y, spd = tracks[i]
            t0 = t[0]
            ts0 = max(head_t - trace_s, t0)
            if head_t - ts0 > 1e-4:
                ts = np.arange(ts0, head_t, TRAIL_DT)
                if len(ts):
                    tx = np.interp(ts, t, x); ty = np.interp(ts, t, y)
                    aage = np.clip(1.0 - (head_t - ts) / trace_s, 0, 1) ** TRACE_GAMMA
                    for k in range(len(ts)):
                        xx, yy = tx[k], ty[k]
                        if not (xmin <= xx <= xmax and ymin <= yy <= ymax):
                            continue
                        px = int(round((xx - xmin) / (xmax - xmin) * (W - 1)))
                        py = int(round((yy - ymin) / (ymax - ymin) * (H - 1)))
                        v2.draw_dot(frame, px, py, col,
                                    TRAIL_FILL_A * aage[k] * linger, head=False)
            xh = float(np.interp(head_t, t, x)); yh = float(np.interp(head_t, t, y))
            if xmin <= xh <= xmax and ymin <= yh <= ymax:
                px = int(round((xh - xmin) / (xmax - xmin) * (W - 1)))
                py = int(round((yh - ymin) / (ymax - ymin) * (H - 1)))
                v2.draw_dot(frame, px, py, col, HEAD_FILL_A * linger, head=True)

        composite_hud(frame, hud_for(src_t))
        return (np.clip(frame, 0, 1) * 255).astype(np.uint8), src_t, len(items)

    # background-with-HUD reference -> reserved-colour global palette
    bg_ref = base.copy(); composite_hud(bg_ref, render_hud(0.0, t_total, W, HUD_H, palette))
    bg_full = (np.clip(bg_ref, 0, 1) * 255).astype(np.uint8)
    pal_img = build_palette(bg_full, gif_w, gif_h, gif_colors, palette)

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
            gif_frames.append(pim.quantize(palette=pal_img, dither=Image.NONE))
        if still_path is not None and not still_done and src_t >= 0.42 * t_total:
            Image.fromarray(full).save(still_path); still_done = True
        if fi % 25 == 0:
            print(f"   [{label}] frame {fi}/{F}  ({n_items} peds)")

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
    dur = 8.0 if args.preview else ANIM_DUR

    warped_rgb, extent = v1.warp_plan_to_world(v1.DEF_PLAN, v1.DEF_CALIB)
    xmin, xmax, ymin, ymax = extent
    W = MAP_W - (MAP_W % 2)
    H = int(round(W * (ymax - ymin) / (xmax - xmin))); H -= H % 2
    base = v2.add_grid(v2.arctic_background(warped_rgb, W, H), extent, W, H)
    print(f"[INFO] full-frame map {W}x{H} (HUD overlay {HUD_H}) | font {FONT}")

    tracks, n_total, n_tracks_all, n_used, t_total = v1.load_tracks(extent)
    print(f"[INFO] tracks {n_used}/{n_tracks_all} | rows {n_total} | src 0–{t_total:.1f}s "
          f"| compression {t_total/dur:.1f}x (was ~{t_total/13:.1f}x in V2)")

    gif_w = (GIF_WIDTH if not args.preview else 820); gif_w -= gif_w % 2
    sizes = {}
    for name, palette in PALETTES.items():
        gif = OUT_DIR / f"{STEM}_palette_{name}.gif"
        mp4 = OUT_DIR / f"{STEM}_palette_{name}.mp4"
        still = OUT_DIR / f"{STEM}_palette_{name}_still.png"
        sizes[name] = render_run(
            tracks, t_total, base, extent, W, H, palette=palette,
            trace_s=TRACE_SECONDS, duration=dur, fps=FPS, gif_path=gif,
            gif_w=gif_w, gif_fps=GIF_FPS, gif_colors=GIF_COLORS,
            mp4_path=mp4, still_path=still, label=f"{name}")
        print(f"[INFO] palette {name}: GIF {sizes[name]:.1f} MB | MP4 "
              f"{mp4.stat().st_size/1e6:.1f} MB")

    trace_mb = {}
    if not args.no_tests and not args.preview:
        for ts in TRACE_TESTS:
            p = OUT_DIR / f"{STEM}_trace_{int(ts)}s.gif"
            trace_mb[ts] = render_run(
                tracks, t_total, base, extent, W, H, palette=PALETTES["warm"],
                trace_s=ts, duration=dur, fps=12, gif_path=p, gif_w=960,
                gif_fps=12, gif_colors=GIF_COLORS, label=f"trace{int(ts)}s")
            print(f"[INFO] trace {ts:.0f}s GIF {trace_mb[ts]:.1f} MB")

    write_report(n_total, n_tracks_all, n_used, t_total, dur, (W, H), gif_w,
                 sizes, trace_mb)


def write_report(n_total, n_tracks_all, n_used, t_total, dur, size, gif_w,
                 sizes, trace_mb):
    W, H = size
    comp = t_total / dur
    traces = "\n".join(
        f"  - `{STEM}_trace_{int(t)}s.gif` — {t:.0f} s trace ({mb:.1f} MB)"
        for t, mb in sorted(trace_mb.items())) or "  - (not generated)"
    txt = f"""# Plaça Catalunya — Speed Population V3

**This visualization uses existing tracked pedestrian trajectories and speed
metrics only. No retracking, recalibration, model inference, or source-data
modification was performed.**

**Dots persist as a short, slowly-fading trace of their recent real positions to
improve legibility; this represents recent movement presence, not instantaneous
pedestrian count. Dot motion is shown at a reduced presentation rate (see
"Presentation-time remapping"); the legend timer still shows real source time.**

V3 modifies V2: a Flow-Fields-matched legend rendered as a frosted **HUD overlay**
on a **full-frame** arctic plan, **slower** dot motion, **longer fading traces**,
and **new elegant palettes** (no red/yellow/green).

## Source data
- Trajectory file: `{v1.DEF_TRAJ}`
- Tracks shown: **{n_used}** / {n_tracks_all} · rows **{n_total}** · source FPS **~30**
- Source time range: **0 – {t_total:.1f} s** ({mmss(0)} – {mmss(t_total)})

## Presentation-time remapping (slower dots)
- Presentation **{dur:.0f} s** at **{FPS} fps**; the full source range is stretched
  uniformly over it → **≈ {comp:.1f}× real-time** (V2 was ≈ {t_total/13:.1f}×). Dots
  therefore move **~{(1 - 13/dur)*100:.0f}% slower** than V2 ({dur/13:.2f}× longer
  on screen). Relative speed is preserved; the legend timer shows **real
  source-video** time.

## Full-frame plan + HUD overlay
- The arctic satellite plan fills the entire **{W}×{H}** canvas (no separate
  footer band). The legend is a **bottom-aligned frosted-white HUD** drawn over
  the image: low-opacity white panel (α 0.72), thin dark-grey rounded border,
  dark readable text. It consumes **no extra image height**.
- Layout follows the Flow Fields footer: grey labels / dark values, thin
  dividers, PLACE · MAP · SPEED (m/s) with three dots · TIME ELAPSED, font {FONT}.

## Traces ("fading memory")
- Each dot leaves a trace of its **real** recent positions over the last
  **{TRACE_SECONDS:.0f} s** of source time (default), sampled every {TRAIL_DT:.2f} s,
  fading backward from the head (γ {TRACE_GAMMA}); short exit fade {LINGER_OUT:.1f} s.
- No glow, no thick strings — subtle. Trace-length tests (warm palette):
{traces}

## Dots
- Head radius {v2.HEAD_R} px (α {HEAD_FILL_A}) + soft white halo; trace radius
  {v2.TRAIL_R} px (α {TRAIL_FILL_A}·age). Supersampled AA, translucent, no heavy
  outline, no glow.

## Speed
- Thresholds slow < {SPEED_SLOW} · medium {SPEED_SLOW}–{SPEED_FAST} · fast > {SPEED_FAST} m/s
  (discrete classes).
- **Warm palette** — slow `#E85D4F` coral · medium `#F2B84B` amber · fast
  `#00AFA6` teal → `{STEM}_palette_warm.gif` ({sizes.get('warm',0):.1f} MB).
- **Cool palette** — slow `#8B5CF6` violet · medium `#F2B84B` amber · fast
  `#00BCD4` cyan → `{STEM}_palette_cool.gif` ({sizes.get('cool',0):.1f} MB).

## GIF quality
- {gif_w}px, {GIF_FPS} fps, {GIF_COLORS}-colour global palette with **reserved**
  dot/trace/HUD colours (incl. faint trace blends) so colours survive
  quantization; PIL delta-encodes only the moving content. Dot + trace colours
  preserved.

## Output paths
- `{STEM}_palette_warm.gif` / `.mp4` / `_still.png`
- `{STEM}_palette_cool.gif` / `.mp4` / `_still.png`
- trace-length test GIFs (above)
- this report.

## Recommendation
- **Palette: warm (coral → amber → teal).** It reads slow→fast most intuitively
  (warm danger/slow → cool calm/fast), is elegant and print-friendly, and the
  teal "fast" separates cleanly from the amber middle. The cool palette
  (violet→amber→cyan) is striking but violet "slow" vs cyan "fast" is a slightly
  less natural slow→fast cue.
- **Trace length: 5 s** is the best balance — 3 s feels thin, 7 s starts to
  crowd; 5 s populates the plaza while staying subtle.
"""
    OUT_REPORT = OUT_DIR / f"{STEM}_report.md"
    OUT_REPORT.write_text(txt, encoding="utf-8")
    print(f"[INFO] Report saved: {OUT_REPORT}")


if __name__ == "__main__":
    main()
