"""
generate_unified_series.py
--------------------------
MOTION PIXELS — FINAL UNIFIED SERIES (Plaça Catalunya).

Layout / format harmonization ONLY. The three behaviour maps —
  1. Flow Fields      (flow_currents V5)
  2. Speed Population  (speed_population V3)
  3. Bottleneck Density(bottleneck_animations Option 5)
— are exported at ONE shared canvas size with ONE shared HUD/footer, copied
exactly from the Flow Fields footer (the agreed standard).

NOTHING about the map *content* changes: same datasets, same trajectories, same
speeds, same bottleneck scores, same colours, same calibration. No retracking,
recalibration, metric recomputation, or visual redesign. Only the viewport
(padded to the Flow Fields 16:9 window), the footer style, and the export
format are unified so the three read as one coherent presentation series.

Standard (extracted from generate_flow_currents_v5_info.py):
  - map area      : 1920 x 1080  (window_extent: full X span, Y padded to 16:9)
  - footer (bar)  : 1920 x 70    (black, thin white outline, added BELOW map)
  - total canvas  : 1920 x 1150
  - GIF           : 760 wide  (-> 454 tall), 20 fps / stride 2 = 10 fps GIF
  - MP4           : 1920 x 1150, H.264 yuv420p, 20 fps
  - font          : Roboto (regular labels / larger values)
  - footer fields : PLACE · MAP · INTENSITY|SPEED · TIME ELAPSED (real video time)

Outputs (./outputs):
  placa_catalunya_flow_fields_unified.{gif,mp4} + _still.png   (Flow Fields = standard, copied)
  placa_catalunya_speed_population_unified.{gif,mp4} + _still.png
  placa_catalunya_bottleneck_density_unified.{gif,mp4} + _still.png
  final_unified_series_report.md
"""

from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
from PIL import Image
import imageio.v2 as imageio

# ---------------------------------------------------------------------------
# Paths / module wiring (read-only reuse of the existing content pipelines)
# ---------------------------------------------------------------------------
BEHAVIOR_DIR = Path(__file__).resolve().parents[1]
FC_DIR = BEHAVIOR_DIR / "flow_currents"
SP_DIR = BEHAVIOR_DIR / "speed_population"
BN_DIR = BEHAVIOR_DIR / "bottleneck_animations"
for p in (BEHAVIOR_DIR, FC_DIR, SP_DIR, BN_DIR):
    sys.path.insert(0, str(p))

import generate_bottleneck_map_test as v2d                  # warp + palette
import generate_flow_currents_v3 as ffv3                    # FF window_extent + canvas
import generate_speed_population_v2 as sp2                   # bg / grid / dots
import generate_speed_population_v3 as sp3                   # speed palette / trace cfg
spv1 = sp2.v1
import generate_bottleneck_animation_tests as bn            # Option 5 content

OUT_DIR = Path(__file__).resolve().parent / "outputs"

# ---------------------------------------------------------------------------
# THE STANDARD (extracted verbatim from Flow Fields V5)
# ---------------------------------------------------------------------------
W       = ffv3.CANVAS_W - (ffv3.CANVAS_W % 2)   # 1920
H_MAP   = ffv3.CANVAS_H - (ffv3.CANVAS_H % 2)    # 1080
BAR_H   = 70                                     # footer height (FF V5)
H_TOTAL = H_MAP + BAR_H                           # 1150

FPS        = 20
GIF_WIDTH  = 760
GIF_STRIDE = 2                                    # GIF ~10 fps
SRC_FPS    = 30.0

# font preference identical to Flow Fields V5
FONT = "DejaVu Sans"
for _c in ["Roboto", "Inter", "IBM Plex Sans", "Helvetica Neue", "Segoe UI", "DejaVu Sans"]:
    try:
        fm.findfont(_c, fallback_to_default=False); FONT = _c; break
    except Exception:
        continue

PLACE_TEXT = "Plaça Catalunya"

# footer palette — LIGHT HUD (for the light arctic/satellite maps: Speed
# Population + Bottleneck Density). Flow Fields keeps its own dark footer
# (copied verbatim) and does NOT use this renderer.
HUD_BG  = "#F7F7F7"     # very light neutral background
HUD_SEP = "#D9D9D9"     # thin, subtle light-grey separators / outlines
LBL_COL = "#333333"     # field labels — near-black (size keeps the hierarchy)
VAL_COL = "#111111"     # values — black
END_COL = "#111111"     # legend labels — black


def mmss(s):
    s = max(0, int(round(s)))
    return f"{s // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# Shared world window (Flow Fields 16:9 window_extent — used by all 3 maps)
# ---------------------------------------------------------------------------
def shared_window():
    warped_rgb, plan_extent = v2d.warp_plan_to_world(ffv3.DEF_PLAN, ffv3.DEF_CALIB)
    win = ffv3.window_extent(plan_extent, W, H_MAP)
    return warped_rgb, plan_extent, win


def real_total_seconds():
    fr = pd.read_csv(ffv3.DEF_TRAJ, usecols=["frame"])
    return (int(fr.frame.max()) - int(fr.frame.min())) / SRC_FPS


# ---------------------------------------------------------------------------
# Shared footer (cloned EXACTLY from Flow Fields V5 render_bar; content varies)
# ---------------------------------------------------------------------------
def render_footer(map_text, real_elapsed, real_total, legend):
    """legend = {"type": "gradient", "cmap": cmap, "low": str, "high": str}
            or {"type": "dots", "label": str, "items": [(rgb, name), ...]}"""
    fig = plt.figure(figsize=(W / 100, BAR_H / 100), dpi=100)
    fig.patch.set_facecolor(HUD_BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(HUD_BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # thin outline only (compact, no thick border) — same proportions as FF,
    # light-grey separator for the light HUD
    ax.add_patch(Rectangle((0.004, 0.14), 0.992, 0.72, fill=False,
                 edgecolor=HUD_SEP, linewidth=0.8, alpha=0.9))

    def field(x, label, value, ha="left"):
        ax.text(x, 0.71, label, fontsize=6.5, color=LBL_COL, ha=ha, va="center",
                family=FONT)
        ax.text(x, 0.33, value, fontsize=10.5, color=VAL_COL, ha=ha, va="center",
                family=FONT)

    field(0.020, "PLACE", PLACE_TEXT)
    field(0.250, "MAP", map_text)

    # legend block in the FF gradient region (x 0.520 .. ~0.700)
    gx0, gx1, gy0, gy1 = 0.520, 0.700, 0.40, 0.50
    if legend["type"] == "gradient":
        grad = legend["cmap"](np.linspace(0, 1, 256)[None, :]).copy()
        grad[..., 3] = 1.0                                    # full opacity on light bg (true colours)
        ax.imshow(grad, extent=[gx0, gx1, gy0, gy1], aspect="auto", zorder=2,
                  interpolation="bilinear")
        ax.add_patch(Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, fill=False,
                     edgecolor=HUD_SEP, linewidth=0.5, alpha=0.9, zorder=3))
        ax.text(0.520, 0.71, "INTENSITY", fontsize=6.5, color=LBL_COL, ha="left",
                va="center", family=FONT)
        ax.text(gx0 - 0.007, 0.45, legend["low"], fontsize=7.0, color=END_COL,
                ha="right", va="center", family=FONT)
        ax.text(gx1 + 0.007, 0.45, legend["high"], fontsize=7.0, color=END_COL,
                ha="left", va="center", family=FONT)
    else:  # dots (Speed Population) — same footer geometry, three swatches
        ax.text(0.520, 0.71, legend["label"], fontsize=6.5, color=LBL_COL,
                ha="left", va="center", family=FONT)
        lx = 0.523
        for col, name in legend["items"]:
            ax.scatter([lx], [0.45], s=22, c=[tuple(col)], edgecolors=HUD_SEP,
                       linewidths=0.4, zorder=4)
            ax.text(lx + 0.010, 0.45, name, fontsize=7.0, color=END_COL,
                    ha="left", va="center", family=FONT)
            lx += 0.063

    field(0.980, "TIME ELAPSED", f"{mmss(real_elapsed)} / {mmss(real_total)}",
          ha="right")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


# ---------------------------------------------------------------------------
# Shared export driver (streams frames; GIF 760, MP4 1920x1150, still)
# ---------------------------------------------------------------------------
def drive_export(stem, F, render_map, footer_for, real_total, gif_colors,
                 still_at=0.97):
    gif_w = GIF_WIDTH - (GIF_WIDTH % 2)
    gif_h = int(round(gif_w * H_TOTAL / W)); gif_h -= gif_h % 2

    gif_path = OUT_DIR / f"{stem}.gif"
    mp4_path = OUT_DIR / f"{stem}.mp4"
    still_path = OUT_DIR / f"{stem}_still.png"

    mp4 = imageio.get_writer(mp4_path, fps=FPS, codec="libx264", quality=9,
                             macro_block_size=None,
                             output_params=["-pix_fmt", "yuv420p"])
    footer_cache = {}
    gif_frames = []
    still_saved = False
    last_full = None

    for fi in range(F):
        t = fi / max(F - 1, 1)
        rt = t * real_total
        map_u8 = render_map(fi)                       # 1920x1080 uint8
        key = mmss(rt)
        bar = footer_cache.get(key)
        if bar is None:
            bar = footer_for(rt); footer_cache[key] = bar
        full = np.vstack([map_u8, bar])
        last_full = full
        mp4.append_data(full)
        if fi % GIF_STRIDE == 0:
            pim = Image.fromarray(full).resize((gif_w, gif_h), Image.LANCZOS).convert("RGB")
            gif_frames.append(pim.quantize(colors=gif_colors, method=Image.MEDIANCUT,
                                           dither=Image.NONE))
        if not still_saved and t >= still_at:
            Image.fromarray(full).save(still_path); still_saved = True
        if fi % 40 == 0:
            print(f"   [{stem}] frame {fi}/{F}")

    if not still_saved and last_full is not None:
        Image.fromarray(last_full).save(still_path)
    mp4.close()
    gif_fps = FPS / GIF_STRIDE
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=int(round(1000 / gif_fps)), loop=0, optimize=True,
                       disposal=2)
    return dict(gif=gif_path, mp4=mp4_path, still=still_path,
                gif_mb=gif_path.stat().st_size / 1e6,
                mp4_mb=mp4_path.stat().st_size / 1e6,
                n=F, gif_n=len(gif_frames), gif_fps=gif_fps,
                dims=(gif_w, gif_h))


# ---------------------------------------------------------------------------
# 1) FLOW FIELDS — the standard; copy existing V5 outputs unchanged
# ---------------------------------------------------------------------------
def export_flow_fields():
    """Flow Fields is the dark-background standard and is left untouched. Copy
    the V5 outputs only if the unified files do not already exist; otherwise read
    stats from the existing files (do NOT modify Flow Fields)."""
    dst = OUT_DIR / "placa_catalunya_flow_fields_unified"
    gif, mp4, png = (dst.with_suffix(".gif"), dst.with_suffix(".mp4"),
                     Path(str(dst) + "_still.png"))
    if not (gif.exists() and mp4.exists() and png.exists()):
        src = FC_DIR / "outputs"
        shutil.copyfile(src / "placa_catalunya_flow_currents_v5_info.gif", gif)
        shutil.copyfile(src / "placa_catalunya_flow_currents_v5_info.mp4", mp4)
        shutil.copyfile(src / "placa_catalunya_flow_currents_v5_info_still.png", png)
        print(f"[INFO] Flow Fields (standard) copied.")
    else:
        print(f"[INFO] Flow Fields (standard) left as-is (not modified).")
    gif_dims = Image.open(gif).size
    return dict(gif=gif, mp4=mp4, still=png,
                gif_mb=gif.stat().st_size / 1e6, mp4_mb=mp4.stat().st_size / 1e6,
                dims=gif_dims, seconds=ffv3.SECONDS, copied=True)


# ---------------------------------------------------------------------------
# 2) SPEED POPULATION — re-framed into the shared window + FF footer
# ---------------------------------------------------------------------------
SP_SECONDS = 17.0

def build_speed_base(warped_rgb, plan_extent, win):
    """Place the arctic plan into the 16:9 window (white padding), add grid."""
    xmin, xmax, ymin, ymax = plan_extent
    wxmin, wxmax, wymin, wymax = win
    r0 = int(round((ymin - wymin) / (wymax - wymin) * H_MAP))
    r1 = int(round((ymax - wymin) / (wymax - wymin) * H_MAP))
    ph = max(r1 - r0, 2)
    arctic = sp2.arctic_background(warped_rgb, W, ph)        # full width x plan rows
    canvas = np.ones((H_MAP, W, 3), np.float32)             # white padding (matches arctic edges)
    a, b = max(r0, 0), min(r1, H_MAP)
    canvas[a:b] = arctic[(a - r0):(b - r0)]
    return sp2.add_grid(canvas, win, W, H_MAP)


def export_speed_population(warped_rgb, plan_extent, win, real_total):
    wxmin, wxmax, wymin, wymax = win
    base = build_speed_base(warped_rgb, plan_extent, win)
    tracks, n_total, n_tracks_all, n_used, t_total = spv1.load_tracks(win)
    palette = sp3.PALETTES["warm"]
    trace_s = sp3.TRACE_SECONDS
    T0 = np.array([t[0][0] for t in tracks])
    T1 = np.array([t[0][-1] for t in tracks])
    F = int(round(SP_SECONDS * FPS))

    def render_map(fi):
        src_t = fi / max(F - 1, 1) * t_total
        frame = base.copy()
        act = np.where((T0 <= src_t) & (src_t <= T1 + sp3.LINGER_OUT))[0]
        items = []
        for i in act:
            t, x, y, spd = tracks[i]
            head_t = min(src_t, t[-1])
            hs = float(np.interp(head_t, t, spd))
            linger = 1.0 if src_t <= t[-1] else max(0.0, 1.0 - (src_t - t[-1]) / sp3.LINGER_OUT)
            items.append((hs, i, head_t, linger, sp3.speed_color(hs, palette)))
        items.sort(key=lambda r: r[0], reverse=True)
        for hs, i, head_t, linger, col in items:
            t, x, y, spd = tracks[i]
            ts0 = max(head_t - trace_s, t[0])
            if head_t - ts0 > 1e-4:
                ts = np.arange(ts0, head_t, sp3.TRAIL_DT)
                if len(ts):
                    tx = np.interp(ts, t, x); ty = np.interp(ts, t, y)
                    aage = np.clip(1.0 - (head_t - ts) / trace_s, 0, 1) ** sp3.TRACE_GAMMA
                    for k in range(len(ts)):
                        xx, yy = tx[k], ty[k]
                        if not (wxmin <= xx <= wxmax and wymin <= yy <= wymax):
                            continue
                        px = int(round((xx - wxmin) / (wxmax - wxmin) * (W - 1)))
                        py = int(round((yy - wymin) / (wymax - wymin) * (H_MAP - 1)))
                        sp2.draw_dot(frame, px, py, col,
                                     sp3.TRAIL_FILL_A * aage[k] * linger, head=False)
            xh = float(np.interp(head_t, t, x)); yh = float(np.interp(head_t, t, y))
            if wxmin <= xh <= wxmax and wymin <= yh <= wymax:
                px = int(round((xh - wxmin) / (wxmax - wxmin) * (W - 1)))
                py = int(round((yh - wymin) / (wymax - wymin) * (H_MAP - 1)))
                sp2.draw_dot(frame, px, py, col, sp3.HEAD_FILL_A * linger, head=True)
        return (np.clip(frame, 0, 1) * 255).astype(np.uint8)

    legend = dict(type="dots", label="SPEED",
                  items=[(palette[0], "Slow"), (palette[1], "Medium"),
                         (palette[2], "Fast")])

    def footer_for(rt):
        return render_footer("Speed Population", rt, real_total, legend)

    res = drive_export("placa_catalunya_speed_population_unified", F, render_map,
                       footer_for, real_total, gif_colors=256)
    res.update(seconds=SP_SECONDS, n_used=n_used, n_tracks_all=n_tracks_all,
               n_total=n_total)
    return res


# ---------------------------------------------------------------------------
# 3) BOTTLENECK DENSITY — Option 5 content, re-framed into shared window
# ---------------------------------------------------------------------------
BN_SECONDS = 10.0

def make_bottleneck_fig(ctx, win):
    xmin, xmax, ymin, ymax = ctx["extent"]            # plan extent (underlay)
    wxmin, wxmax, wymin, wymax = win
    fig = plt.figure(figsize=(W / 100, H_MAP / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(wxmin, wxmax); ax.set_ylim(wymax, wymin)
    ax.set_aspect("equal"); ax.axis("off")
    ax.imshow(ctx["underlay"], extent=[xmin, xmax, ymax, ymin], origin="upper",
              alpha=bn.BG_OPACITY, zorder=0, interpolation="bilinear")
    return fig, ax


def build_bottleneck_timeline(ctx):
    """Option 5 three-phase narrative (10 s) — frame parameter arrays only."""
    n = ctx["n_inside"]
    t_full = ctx["t_full"]; fa = ctx["final_alpha"]; fs = ctx["full_scale"]
    LOW_COLORT, LOW_ALPHA = 0.12, 0.50
    sf = bn.secsf if hasattr(bn, "secsf") else (lambda s: max(int(round(s * FPS)), 1))
    f_in = sf(0.5); f_faint = sf(2.5); f_grow = sf(5.0); f_mark = sf(0.8); f_hold = sf(1.2)
    base_scale = np.full(n, bn.BASE_SCALE); low_colorT = np.full(n, LOW_COLORT)
    ss = bn.smoothstep
    tl = []
    for k in range(f_in):
        a = ss((k + 1) / f_in) * LOW_ALPHA
        tl.append((np.full(n, a), low_colorT, base_scale, 0.0))
    for _ in range(f_faint):
        tl.append((np.full(n, LOW_ALPHA), low_colorT, base_scale, 0.0))
    for k in range(f_grow):
        q = ss((k + 1) / f_grow)
        tl.append((LOW_ALPHA + (fa - LOW_ALPHA) * q,
                   LOW_COLORT + (t_full - LOW_COLORT) * q,
                   bn.BASE_SCALE + (fs - bn.BASE_SCALE) * q, 0.0))
    for k in range(f_mark):
        tl.append((fa, t_full, fs, ss((k + 1) / f_mark)))
    for _ in range(f_hold):
        tl.append((fa, t_full, fs, 1.0))
    return tl


def export_bottleneck(win, real_total):
    ctx = bn.build_context()
    timeline = build_bottleneck_timeline(ctx)
    F = len(timeline)
    fig, ax = make_bottleneck_fig(ctx, win)
    dynamic = []

    def render_map(fi):
        alpha, colorT, scale, marker = timeline[fi]
        return bn.draw_frame(fig, ax, ctx, alpha, colorT, scale, marker, dynamic)

    cmap = ctx["cmap"]
    legend = dict(type="gradient", cmap=cmap, low="Low density", high="High density")

    def footer_for(rt):
        return render_footer("Bottleneck Density", rt, real_total, legend)

    res = drive_export("placa_catalunya_bottleneck_density_unified", F, render_map,
                       footer_for, real_total, gif_colors=256)
    plt.close(fig)
    res.update(seconds=BN_SECONDS, n_inside=ctx["n_inside"])
    return res


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(win, real_total, ff, sp, bnr):
    wx0, wx1, wy0, wy1 = win
    txt = f"""# Plaça Catalunya — Final Unified Series

**Layout / format harmonization ONLY.** Map content, datasets, trajectories,
speeds, bottleneck scores, colours, and calibration are UNCHANGED. No
retracking, recalibration, metric recomputation, or visual redesign was
performed. Only the viewport (padded to the Flow Fields 16:9 window), the
footer style, and the export format were unified.

## Source files used (content unchanged)
- Flow Fields: `flow_currents/generate_flow_currents_v5_info.py`
  (outputs copied verbatim as the standard).
- Speed Population: `speed_population/generate_speed_population_v3.py`
  (warm palette, {sp3.TRACE_SECONDS:.0f} s trace) — content helpers reused, re-framed only.
- Bottleneck Density: `bottleneck_animations/generate_bottleneck_animation_tests.py`
  (Option 5 hybrid, 3-phase narrative) — content helpers reused, re-framed only.
- Plan / calibration / trajectories (all three):
  `placa_catalunya_01/plan/placa-catalunya.png`,
  `…/calibration/calib.json`, `…/filtered_250m/trajectories_world_filtered_250m.csv`.

## Extracted Flow Fields HUD/footer parameters (the standard)
| Parameter | Value |
|---|---|
| Full frame | **{W} × {H_TOTAL}** px |
| Map area height | **{H_MAP}** px |
| Footer (bar) height | **{BAR_H}** px (~{BAR_H/H_TOTAL*100:.1f}% of canvas) |
| Footer position | full width, **below** the map (vstacked, not overlaid) |
| Footer background | black; thin white outline `Rectangle((0.004,0.14),0.992,0.72)`, lw 0.6, α 0.26 |
| Margins (axes-fraction) | outline inset x 0.004 / y 0.14, height 0.72 |
| Font | **{FONT}** (regular) |
| Label size / colour | 6.5 pt / `#7e7e88` at y 0.71 |
| Value size / colour | 10.5 pt / `#ededf2` at y 0.33 |
| Field x-positions | PLACE 0.020 · MAP 0.250 · legend 0.520 · TIME ELAPSED 0.980 (right) |
| Legend gradient box | x 0.520–0.700, y 0.40–0.50, opacity 0.55, outline lw 0.4 α 0.20 |
| End labels | 7.0 pt / `#9a9aa2` flanking the gradient |
| Timer alignment | right-aligned at x 0.980, `MM:SS / MM:SS` |

## Light HUD inversion (Speed Population + Bottleneck Density only)
These two maps sit on a **light** arctic/satellite background, so their footer is
the **light inversion** of the Flow Fields footer. **Dimensions, spacing,
typography, font sizes, hierarchy, separator/legend/timer placement, and HUD
proportions are identical** — only the colours are inverted:
| Element | Flow Fields (dark) | Speed Pop / Bottleneck (light) |
|---|---|---|
| Footer background | black | `{HUD_BG}` very light neutral |
| Separator / outlines | white α 0.26 | `{HUD_SEP}` light grey, thin |
| Field labels | `#7e7e88` | `{LBL_COL}` near-black |
| Values | `#ededf2` | `{VAL_COL}` black |
| Legend labels | `#9a9aa2` | `{END_COL}` black |
| Gradient opacity | 0.55 (dim on black) | 1.0 (true colours on white) |
| Dot swatches | unchanged colours, white edge | unchanged colours, `{HUD_SEP}` edge |

Flow Fields is **not** modified (left exactly as-is on its dark background).
Speed-Population dot colours and the Bottleneck yellow→orange→red gradient are
**unchanged**; only the labels and HUD background were inverted.
| GIF dimensions | **{GIF_WIDTH} × {int(round(GIF_WIDTH*H_TOTAL/W))}** px |
| FPS / GIF fps | {FPS} fps source → {FPS//GIF_STRIDE} fps GIF (stride {GIF_STRIDE}) |
| MP4 | H.264, yuv420p, {W} × {H_TOTAL}, {FPS} fps |

## Shared world window (no stretch)
`window_extent` keeps the full X span and pads Y symmetrically to a 16:9 frame:
X = [{wx0:.2f}, {wx1:.2f}] m · Y = [{wy0:.2f}, {wy1:.2f}] m. All three maps are
drawn in this identical window at {W}×{H_MAP}, so the plaza is framed identically
in every map (the plan keeps its true aspect; the extra vertical area is neutral
padding, not stretched content).

## Final shared GIF dimensions
**All three GIFs: {ff['dims'][0]} × {ff['dims'][1]} px** (= Flow Fields standard).
MP4 / still: {W} × {H_TOTAL} px.

## Per-map duration / FPS
| Map | Duration | Frames | GIF fps | GIF dims |
|---|---|---|---|---|
| Flow Fields | {ff.get('seconds',0):.0f} s | — | {FPS//GIF_STRIDE} | {ff['dims']} |
| Speed Population | {sp['seconds']:.0f} s | {sp['n']} | {sp['gif_fps']:.0f} | {sp['dims']} |
| Bottleneck Density | {bnr['seconds']:.0f} s | {bnr['n']} | {bnr['gif_fps']:.0f} | {bnr['dims']} |

(The three keep their own narrative *durations* — the brief requires a shared
*canvas*, not a shared runtime; the HUD timer is identical and shows the real
source-video time {mmss(real_total)} in every map.)

## Font
**{FONT}** across all three footers (regular weight; small uppercase labels,
larger values), exactly as Flow Fields.

## HUD dimensions (identical across all three)
Footer {W} × {BAR_H} px, no thick border; PLACE · MAP · (INTENSITY gradient |
SPEED dots) · TIME ELAPSED, with the legend in the same 0.520–0.700 region and
the timer right-aligned at 0.980. Flow Fields uses a **black** strip with a white
outline; Speed Population and Bottleneck Density use the **light** inversion
(`{HUD_BG}` background, `{HUD_SEP}` outline, near-black text) — same geometry,
inverted colours, to integrate with their light arctic/satellite backgrounds.

## Content mapping (only text / legend differ)
- **Flow Fields** — PLACE Plaça Catalunya · MAP Flow Fields · INTENSITY
  cyan→magenta gradient · TIME ELAPSED.
- **Speed Population** — PLACE Plaça Catalunya · MAP Speed Population · SPEED
  slow/medium/fast dot legend (warm palette) · TIME ELAPSED.
- **Bottleneck Density** — PLACE Plaça Catalunya · MAP Bottleneck Density ·
  INTENSITY yellow→orange→red gradient (Low density → High density) · TIME ELAPSED.

## Export file sizes
| Map | GIF | MP4 |
|---|---|---|
| Flow Fields | {ff['gif_mb']:.1f} MB | {ff['mp4_mb']:.1f} MB |
| Speed Population | {sp['gif_mb']:.1f} MB | {sp['mp4_mb']:.1f} MB |
| Bottleneck Density | {bnr['gif_mb']:.1f} MB | {bnr['mp4_mb']:.1f} MB |

## Confirmation
- Map content was **not** changed: same trajectories, speeds, bottleneck scores,
  palettes, plan, and calibration. The Speed Population and Bottleneck renderers
  reuse the existing content helpers unchanged; only the *viewport* and *footer*
  were swapped to the Flow Fields standard.
- Flow Fields is the **exact standard** and is copied verbatim (not re-rendered).
- All three share the same canvas size and the same footer proportions; only the
  text and legend content differ.

## Unavoidable deviations
- **GIF colour depth:** Flow Fields keeps its original {ffv3.GIF_COLORS}-colour
  palette (mostly dark glow — fine at low depth). Speed Population and Bottleneck
  use **256 colours** so the arctic satellite and the warm density gradient do not
  band — a quality *increase*, honouring "do not over-compress".
- **Background padding:** the 16:9 window adds neutral vertical padding around the
  plan (white for Speed Population / Bottleneck, black for Flow Fields, matching
  each map's own background). This is padding, never stretching.
- **Durations differ** by design (Flow Fields 17 s · Speed Population 17 s ·
  Bottleneck 10 s); the canvas, footer, font, and timer are identical.
- **Footer colour theme differs by background:** Flow Fields keeps its dark
  footer (dark map); Speed Population and Bottleneck use the light-inverted
  footer (light maps). Geometry/typography/placement are identical; the gradient
  is shown at full opacity on white (vs 0.55 on black) so the colours read true.

## Output paths
- `placa_catalunya_flow_fields_unified.gif` / `.mp4` / `_still.png`
- `placa_catalunya_speed_population_unified.gif` / `.mp4` / `_still.png`
- `placa_catalunya_bottleneck_density_unified.gif` / `.mp4` / `_still.png`
- this report.
"""
    (OUT_DIR / "final_unified_series_report.md").write_text(txt, encoding="utf-8")
    print(f"[INFO] Report: {OUT_DIR / 'final_unified_series_report.md'}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warped_rgb, plan_extent, win = shared_window()
    real_total = real_total_seconds()
    print(f"[INFO] standard {W}x{H_TOTAL} (map {H_MAP} + bar {BAR_H}) | GIF {GIF_WIDTH} | "
          f"font {FONT} | window {[round(v,1) for v in win]} | real {mmss(real_total)}")

    print("[INFO] 1/3 Flow Fields (standard, copy) ...")
    ff = export_flow_fields()
    print("[INFO] 2/3 Speed Population (re-frame) ...")
    sp = export_speed_population(warped_rgb, plan_extent, win, real_total)
    print(f"        GIF {sp['gif_mb']:.1f} MB | MP4 {sp['mp4_mb']:.1f} MB")
    print("[INFO] 3/3 Bottleneck Density (re-frame) ...")
    bnr = export_bottleneck(win, real_total)
    print(f"        GIF {bnr['gif_mb']:.1f} MB | MP4 {bnr['mp4_mb']:.1f} MB")

    write_report(win, real_total, ff, sp, bnr)
    print("[INFO] Done. Unified series in", OUT_DIR)


if __name__ == "__main__":
    main()
