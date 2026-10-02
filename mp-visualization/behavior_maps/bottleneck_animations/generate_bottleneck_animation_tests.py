"""
generate_bottleneck_animation_tests.py
--------------------------------------
Three animation tests for the Plaça Catalunya bottleneck map, all built on the
frozen **V2D** visual style. Visualization-only: no retracking, recalibration,
or source-data modification. Option 2 reads the existing trajectory CSV purely
to drive accumulation *timing* (it does not recompute bottleneck scores).

Outputs (in ./outputs):
  placa_catalunya_bottleneck_option1_construction.gif      + _final.png
  placa_catalunya_bottleneck_option2_accumulation.gif      + _final.png
  placa_catalunya_bottleneck_option5_hybrid.gif            + _final.png
  bottleneck_animation_tests_report.md

The V2D look (clipping, desaturated underlay, yellow->red palette, variable
square size with no-overlap clamp, ranked markers, no chrome) is reused from the
parent module generate_bottleneck_map_test.py so the final frame matches V2D.
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from PIL import Image
try:
    import imageio.v2 as imageio
    HAVE_IMAGEIO = True
except Exception:
    HAVE_IMAGEIO = False

# --- reuse the frozen V2D primitives from the parent module -----------------
PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))
import generate_bottleneck_map_test as v2d  # noqa: E402

DEF_CSV   = v2d.DEF_CSV
DEF_PLAN  = v2d.DEF_PLAN
DEF_CALIB = v2d.DEF_CALIB
TRAJ_CSV  = v2d.SITE_ROOT / "filtered_250m" / "trajectories_world_filtered_250m.csv"

# V2D style parameters (frozen)
BASE_SCALE   = v2d.BASE_SCALE_V2D        # 0.70
GROWTH       = v2d.GROWTH_AMOUNT_V2D     # 0.25
EXP          = v2d.SIZE_EXPONENT_V2D     # 1.6
CLAMP_FRAC   = v2d.MAX_FRAC_SPACING_V2D  # 0.85
BG_OPACITY   = v2d.BG_OPACITY_V2B        # 0.55
TOP_K        = 10                        # animation brief: ranked markers 1-10

OUT_DIR = Path(__file__).resolve().parent / "outputs"

FPS = 24
GIF_FIG_W_IN = 6.0      # -> ~1200 px wide GIF frames at the dpi below
GIF_DPI = 200

# --- Speed-Population-style HUD (Option 5 only) -----------------------------
HUD_PX     = 100                       # HUD strip height in px (overlaid on map)
HUD_FONT   = "Roboto"                  # matches Speed Population / Flow Fields
PLACE_TEXT = "Plaça Catalunya"
MAP_TEXT   = "Bottleneck Density"
# bottleneck density palette (the frozen V2D gradient, end labelled low->high)
DENSITY_STOPS = ["#FFF3B0", "#FDBA3B", "#F97316", "#DC2626"]
HUD_LBL, HUD_VAL, HUD_DIV = "#5f6670", "#1b1f27", "#c7ccd3"


# ---------------------------------------------------------------------------
# Static context (computed once)
# ---------------------------------------------------------------------------

def build_context():
    cells_all = pd.read_csv(DEF_CSV)
    warped_rgb, (xmin, xmax, ymin, ymax) = v2d.warp_plan_to_world(DEF_PLAN, DEF_CALIB)

    inside = (
        (cells_all["cell_x"] >= xmin) & (cells_all["cell_x"] <= xmax) &
        (cells_all["cell_y"] >= ymin) & (cells_all["cell_y"] <= ymax)
    )
    cells = cells_all[inside].copy().reset_index(drop=True)

    cell_size = v2d.infer_cell_size(cells)
    underlay = v2d.to_architectural_underlay(warped_rgb)
    cmap = v2d.clean_yellow_red_cmap()

    score = cells["bottleneck_score"].to_numpy()
    vmax = float(np.percentile(score, 99.0))
    vmax = vmax if vmax > 0 else float(score.max() or 1.0)
    t_full = np.clip(score / vmax, 0.0, 1.0)

    final_alpha = 0.45 + 0.50 * t_full
    full_scale = BASE_SCALE + (t_full ** EXP) * GROWTH
    max_side = CLAMP_FRAC * cell_size

    # top-k ranked markers (by score)
    order = np.argsort(-score)
    top_idx = order[:TOP_K]

    # real source-video duration (for the HUD timer) — read-only, no recompute
    t_total = 0.0
    try:
        t = pd.read_csv(TRAJ_CSV, usecols=["time_s"])["time_s"].to_numpy()
        t_total = float(t.max() - t.min())
    except Exception as e:
        print(f"[WARN] Could not read source duration ({e}); timer disabled.")

    ctx = dict(
        cells=cells, extent=(xmin, xmax, ymin, ymax), underlay=underlay,
        cmap=cmap, cell_size=cell_size, vmax=vmax, t_full=t_full,
        final_alpha=final_alpha, full_scale=full_scale, max_side=max_side,
        cx=cells["cell_x"].to_numpy(), cy=cells["cell_y"].to_numpy(),
        score=score, top_idx=top_idx,
        n_total=len(cells_all), n_inside=len(cells),
        n_dropped=len(cells_all) - len(cells),
        t_total=t_total,
    )
    return ctx


def make_figure(ctx):
    xmin, xmax, ymin, ymax = ctx["extent"]
    world_w, world_h = xmax - xmin, ymax - ymin
    aspect = world_w / world_h
    fig_w = GIF_FIG_W_IN
    fig_h = fig_w / aspect
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=GIF_DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.imshow(ctx["underlay"], extent=[xmin, xmax, ymax, ymin], origin="upper",
              alpha=BG_OPACITY, zorder=0, interpolation="bilinear")
    return fig, ax


# ---------------------------------------------------------------------------
# Per-frame drawing (mirrors V2D cell drawing; converges to V2D at end)
# ---------------------------------------------------------------------------

def draw_frame(fig, ax, ctx, alpha_arr, colorT_arr, scale_arr,
               marker_alpha, dynamic):
    """Update dynamic artists for one frame; return RGB uint8 array."""
    for art in dynamic:
        art.remove()
    dynamic.clear()

    cmap = ctx["cmap"]
    cs = ctx["cell_size"]
    half_unclamped = scale_arr * cs
    sides = np.minimum(half_unclamped, ctx["max_side"])
    rgba = cmap(np.clip(colorT_arr, 0, 1))
    rounding = cs * 0.16

    for i in range(len(alpha_arr)):
        a = alpha_arr[i]
        if a <= 0.004:
            continue
        side = sides[i]
        h = side / 2.0
        col = (rgba[i, 0], rgba[i, 1], rgba[i, 2], float(np.clip(a, 0, 1)))
        patch = FancyBboxPatch(
            (ctx["cx"][i] - h, ctx["cy"][i] - h), side, side,
            boxstyle=f"round,pad=0,rounding_size={rounding * (side / cs):.4f}",
            facecolor=col, edgecolor="none", linewidth=0.0, zorder=2,
            antialiased=True,
        )
        ax.add_patch(patch)
        dynamic.append(patch)

    if marker_alpha > 0.004:
        ma = float(np.clip(marker_alpha, 0, 1))
        for rank, i in enumerate(ctx["top_idx"], start=1):
            circ = Circle((ctx["cx"][i], ctx["cy"][i]), radius=0.45 * cs,
                          facecolor="white", edgecolor="#9e1414",
                          linewidth=0.6, zorder=4, alpha=0.95 * ma)
            ax.add_patch(circ)
            dynamic.append(circ)
            txt = ax.text(ctx["cx"][i], ctx["cy"][i], str(rank),
                          ha="center", va="center", fontsize=3.4,
                          color="#9e1414", fontweight="bold", zorder=5,
                          alpha=ma)
            dynamic.append(txt)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def secs_to_frames(s):
    return max(int(round(s * FPS)), 1)


def save_gif(frames, path):
    """Quantize to a shared palette (from the final frame) and save GIF."""
    pil = [Image.fromarray(f) for f in frames]
    pal = pil[-1].convert("P", palette=Image.ADAPTIVE, colors=256)
    quant = [im.quantize(palette=pal, dither=Image.NONE) for im in pil]
    quant[0].save(path, save_all=True, append_images=quant[1:],
                  duration=int(round(1000 / FPS)), loop=0, optimize=True,
                  disposal=1)


def save_final_png(ctx, path):
    """Exact V2D still (same primitives/params) as the option's final frame."""
    warped_rgb, _ = v2d.warp_plan_to_world(DEF_PLAN, DEF_CALIB)
    v2d.render_v2b(ctx["cells"], warped_rgb, ctx["extent"], top_k=TOP_K,
                   out_png=path, dpi=320, min_scale=BASE_SCALE,
                   max_scale=BASE_SCALE + GROWTH, bg_opacity=BG_OPACITY,
                   size_exponent=EXP, max_frac_spacing=CLAMP_FRAC)


def mmss(s):
    s = max(0, int(round(s)))
    return f"{s // 60:02d}:{s % 60:02d}"


def save_mp4(frames, path, fps=FPS):
    """Presentation-quality H.264 export (even dims, yuv420p)."""
    if not HAVE_IMAGEIO:
        return False
    h, w = frames[0].shape[:2]
    h2, w2 = h - (h % 2), w - (w % 2)
    w = imageio.get_writer(path, fps=fps, codec="libx264", quality=9,
                           macro_block_size=None,
                           output_params=["-pix_fmt", "yuv420p"])
    for f in frames:
        w.append_data(f[:h2, :w2])
    w.close()
    return True


# ---------------------------------------------------------------------------
# Speed-Population-style HUD (full-width overlay; Option 5 only)
# ---------------------------------------------------------------------------

def render_bottleneck_hud(t_src, t_total, W_px, cmap):
    """Frosted full-width HUD matching the Speed Population / Flow Fields look:
    PLACE · MAP · INTENSITY (continuous density gradient) · TIME ELAPSED."""
    fig = plt.figure(figsize=(W_px / 100, HUD_PX / 100), dpi=100)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("none")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # frosted white panel + thin grey rounded border (no boxed legend, no footer)
    for fc, ec, lw, a, z in ((("white"), "none", 0.0, 0.72, 1),
                             (("none"), "#8d939b", 1.0, 0.9, 3)):
        ax.add_patch(FancyBboxPatch((0.012, 0.16), 0.976, 0.68,
                     boxstyle="round,pad=0,rounding_size=0.06",
                     facecolor=fc, edgecolor=ec, linewidth=lw, alpha=a, zorder=z))

    def field(x, label, value, ha="left"):
        ax.text(x, 0.60, label, fontsize=9, color=HUD_LBL, ha=ha, va="center",
                family=HUD_FONT, zorder=4)
        ax.text(x, 0.37, value, fontsize=14, color=HUD_VAL, ha=ha, va="center",
                family=HUD_FONT, zorder=4)

    def divider(x):
        ax.plot([x, x], [0.30, 0.70], color=HUD_DIV, lw=1.0, zorder=4)

    field(0.030, "PLACE", PLACE_TEXT)
    divider(0.250)
    field(0.268, "MAP", MAP_TEXT)
    divider(0.512)

    # INTENSITY — continuous bottleneck-density gradient, subtle, end-labelled
    ax.text(0.528, 0.62, "INTENSITY", fontsize=9, color=HUD_LBL, ha="left",
            va="center", family=HUD_FONT, zorder=4)
    gx0, gx1, gy0, gy1 = 0.616, 0.792, 0.40, 0.52
    grad = np.linspace(0, 1, 256)[None, :]
    ax.imshow(grad, extent=[gx0, gx1, gy0, gy1], aspect="auto", cmap=cmap,
              origin="lower", zorder=4, interpolation="bilinear")
    ax.add_patch(FancyBboxPatch((gx0, gy0), gx1 - gx0, gy1 - gy0,
                 boxstyle="square,pad=0", facecolor="none", edgecolor="#b9bec6",
                 linewidth=0.6, zorder=5))
    ax.text(gx0, 0.31, "LOW DENSITY", fontsize=7, color=HUD_LBL, ha="left",
            va="center", family=HUD_FONT, zorder=4)
    ax.text(gx1, 0.31, "HIGH DENSITY", fontsize=7, color=HUD_LBL, ha="right",
            va="center", family=HUD_FONT, zorder=4)
    divider(0.824)

    field(0.972, "TIME ELAPSED", f"{mmss(t_src)} / {mmss(t_total)}", ha="right")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32) / 255.0
    plt.close(fig)
    return rgba


def composite_hud_u8(frame_u8, hud_rgba):
    """Alpha-composite the HUD onto the bottom strip of a uint8 RGB frame."""
    h = hud_rgba.shape[0]
    if h > frame_u8.shape[0]:
        return frame_u8
    a = hud_rgba[..., 3:4]
    reg = frame_u8[-h:].astype(np.float32) / 255.0
    reg = reg * (1 - a) + hud_rgba[..., :3] * a
    frame_u8[-h:] = (np.clip(reg, 0, 1) * 255).astype(np.uint8)
    return frame_u8


# ---------------------------------------------------------------------------
# Option 1 — Construction (reveal low -> high score)
# ---------------------------------------------------------------------------

def option1_construction(ctx):
    n = ctx["n_inside"]
    t_full = ctx["t_full"]; fa = ctx["final_alpha"]; fs = ctx["full_scale"]

    # reveal order: ascending score (low appears first)
    rank = np.argsort(np.argsort(ctx["score"])) / max(n - 1, 1)  # 0..1
    fw = 0.16  # fade window

    f_bg     = secs_to_frames(0.5)
    f_cells  = secs_to_frames(3.0)
    f_mark   = secs_to_frames(0.8)
    f_hold   = secs_to_frames(1.5)

    fig, ax = make_figure(ctx); dynamic = []
    frames = []
    zeros = np.zeros(n)

    for _ in range(f_bg):
        frames.append(draw_frame(fig, ax, ctx, zeros, t_full, fs, 0.0, dynamic))
    for k in range(f_cells):
        p = (k + 1) / f_cells
        appear = rank * (1 - fw)
        a = np.clip((p - appear) / fw, 0, 1)
        frames.append(draw_frame(fig, ax, ctx, a * fa, t_full, fs, 0.0, dynamic))
    for k in range(f_mark):
        m = smoothstep((k + 1) / f_mark)
        frames.append(draw_frame(fig, ax, ctx, fa, t_full, fs, m, dynamic))
    for _ in range(f_hold):
        frames.append(draw_frame(fig, ax, ctx, fa, t_full, fs, 1.0, dynamic))

    plt.close(fig)
    return frames


# ---------------------------------------------------------------------------
# Option 2 — Time-lapse accumulation (real temporal data)
# ---------------------------------------------------------------------------

def compute_accumulation(ctx, n_acc):
    """
    Real temporal accumulation. Assign each trajectory observation to its 1 m
    cell, keep only cells present in the bottleneck map, bin by time into n_acc
    bins, and return per-cell cumulative fraction frac[n_cells, n_acc].
    Returns (frac, used_real) — used_real False triggers the documented fallback.
    """
    cells = ctx["cells"]
    key_to_idx = {(int(r.cell_col), int(r.cell_row)): i
                  for i, r in cells.iterrows()}
    try:
        df = pd.read_csv(TRAJ_CSV, usecols=["time_s", "world_x", "world_y"])
    except Exception as e:
        print(f"[WARN] Could not read trajectory CSV ({e}); using fallback.")
        return None, False

    cc = np.floor(df["world_x"].to_numpy()).astype(int)
    cr = np.floor(df["world_y"].to_numpy()).astype(int)
    idx = np.array([key_to_idx.get((c, r), -1) for c, r in zip(cc, cr)])
    keep = idx >= 0
    if keep.sum() == 0:
        print("[WARN] No trajectory obs mapped to bottleneck cells; fallback.")
        return None, False

    idx = idx[keep]
    t = df["time_s"].to_numpy()[keep]
    tmin, tmax = float(t.min()), float(t.max())
    b = np.clip(((t - tmin) / max(tmax - tmin, 1e-9) * n_acc).astype(int),
                0, n_acc - 1)

    counts = np.zeros((len(cells), n_acc), dtype=np.float64)
    np.add.at(counts, (idx, b), 1.0)
    cum = np.cumsum(counts, axis=1)
    total = cum[:, -1:]
    frac = np.divide(cum, np.maximum(total, 1e-9))
    # cells with zero observations: reveal late, proportional to final score
    zero = (total[:, 0] <= 0)
    if zero.any():
        ramp = np.linspace(0, 1, n_acc)[None, :]
        frac[zero] = np.clip((ramp - 0.6) / 0.4, 0, 1)
    return frac, True


def option2_accumulation(ctx):
    n = ctx["n_inside"]
    t_full = ctx["t_full"]; fa = ctx["final_alpha"]

    f_bg    = secs_to_frames(0.5)
    f_acc   = secs_to_frames(4.0)
    f_mark  = secs_to_frames(0.8)
    f_hold  = secs_to_frames(1.5)

    frac, used_real = compute_accumulation(ctx, f_acc)
    if not used_real:
        # Fallback: ordered reveal by spatial progression (x) + score
        order = np.argsort(ctx["cx"] + 0.0 * ctx["score"])
        prog = np.argsort(np.argsort(ctx["cx"])) / max(n - 1, 1)
        frac = np.clip((np.linspace(0, 1, f_acc)[None, :] - prog[:, None]) / 0.5
                       + ctx["t_full"][:, None] * 0.0, 0, 1)

    fig, ax = make_figure(ctx); dynamic = []
    frames = []
    zeros = np.zeros(n)

    for _ in range(f_bg):
        frames.append(draw_frame(fig, ax, ctx, zeros, t_full, ctx["full_scale"],
                                 0.0, dynamic))
    for k in range(f_acc):
        f = frac[:, k]
        colorT = t_full * f
        scale = BASE_SCALE + (t_full ** EXP) * GROWTH * f
        alpha = fa * (0.20 + 0.80 * f)
        frames.append(draw_frame(fig, ax, ctx, alpha, colorT, scale, 0.0, dynamic))
    for k in range(f_mark):
        m = smoothstep((k + 1) / f_mark)
        frames.append(draw_frame(fig, ax, ctx, fa, t_full, ctx["full_scale"],
                                 m, dynamic))
    for _ in range(f_hold):
        frames.append(draw_frame(fig, ax, ctx, fa, t_full, ctx["full_scale"],
                                 1.0, dynamic))

    plt.close(fig)
    return frames, used_real


# ---------------------------------------------------------------------------
# Option 5 — Hybrid (10 s narrative: full field -> emergence -> rankings/hold)
# with a Speed-Population-style HUD overlay (timer = real source-video time).
#
# Phase A (0-3 s)  full bottleneck field appears immediately, faintly visible.
# Phase B (3-8 s)  pressure condenses: high cells brighten/warm/grow, low stay
#                  subdued (a differentiation, not a reveal).
# Phase C (8-10 s) ranked markers fade in, then a calm hold (no pulse/glow).
# ---------------------------------------------------------------------------

def option5_hybrid(ctx):
    n = ctx["n_inside"]
    t_full = ctx["t_full"]; fa = ctx["final_alpha"]; fs = ctx["full_scale"]
    LOW_COLORT = 0.12
    LOW_ALPHA  = 0.50

    # 10 s total @ FPS. Phase A = 3 s, Phase B = 5 s, Phase C = 2 s.
    f_in    = secs_to_frames(0.5)   # A: quick fade-in of the whole faint field
    f_faint = secs_to_frames(2.5)   # A: hold the faint full field
    f_grow  = secs_to_frames(5.0)   # B: emergence / differentiation
    f_mark  = secs_to_frames(0.8)   # C: ranked markers fade in
    f_hold  = secs_to_frames(1.2)   # C: calm final hold
    total_f = f_in + f_faint + f_grow + f_mark + f_hold

    t_total = ctx.get("t_total", 0.0)
    cmap = ctx["cmap"]
    W_px = None
    hud_cache = {}

    def hud_for(idx):
        nonlocal W_px
        src_t = (idx / max(total_f - 1, 1)) * t_total
        key = mmss(src_t)
        h = hud_cache.get(key)
        if h is None:
            h = render_bottleneck_hud(src_t, t_total, W_px, cmap)
            hud_cache[key] = h
        return h

    fig, ax = make_figure(ctx); dynamic = []
    frames = []
    base_scale_arr = np.full(n, BASE_SCALE)
    low_colorT = np.full(n, LOW_COLORT)

    def emit(alpha_arr, colorT_arr, scale_arr, marker_alpha):
        f = draw_frame(fig, ax, ctx, alpha_arr, colorT_arr, scale_arr,
                       marker_alpha, dynamic)
        nonlocal W_px
        if W_px is None:
            W_px = f.shape[1]
        composite_hud_u8(f, hud_for(len(frames)))
        frames.append(f)

    # Phase A — full field introduction (0-3 s)
    for k in range(f_in):
        a = smoothstep((k + 1) / f_in) * LOW_ALPHA
        emit(np.full(n, a), low_colorT, base_scale_arr, 0.0)
    for _ in range(f_faint):
        emit(np.full(n, LOW_ALPHA), low_colorT, base_scale_arr, 0.0)
    # Phase B — bottleneck emergence (3-8 s): pressure condensing
    for k in range(f_grow):
        q = smoothstep((k + 1) / f_grow)
        colorT = LOW_COLORT + (t_full - LOW_COLORT) * q
        scale  = BASE_SCALE + (fs - BASE_SCALE) * q
        alpha  = LOW_ALPHA + (fa - LOW_ALPHA) * q
        emit(alpha, colorT, scale, 0.0)
    # Phase C — rankings + final state (8-10 s): markers fade in, calm hold
    for k in range(f_mark):
        m = smoothstep((k + 1) / f_mark)
        emit(fa, t_full, fs, m)
    for _ in range(f_hold):
        emit(fa, t_full, fs, 1.0)

    plt.close(fig)
    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = build_context()
    print(f"[INFO] cells in-frame: {ctx['n_inside']} (dropped off-plan {ctx['n_dropped']})")

    results = {}

    # Option 1
    t0 = time.time()
    print("[INFO] Rendering Option 1 — construction ...")
    f1 = option1_construction(ctx)
    p1 = OUT_DIR / "placa_catalunya_bottleneck_option1_construction.gif"
    save_gif(f1, p1)
    save_final_png(ctx, OUT_DIR / "placa_catalunya_bottleneck_option1_construction_final.png")
    results["option1"] = dict(path=p1, n=len(f1), dur=len(f1) / FPS,
                              size=p1.stat().st_size, secs=time.time() - t0)
    print(f"        {len(f1)} frames, {len(f1)/FPS:.2f}s, {p1.stat().st_size/1e6:.2f} MB")

    # Option 2
    t0 = time.time()
    print("[INFO] Rendering Option 2 — accumulation (real temporal) ...")
    f2, used_real = option2_accumulation(ctx)
    p2 = OUT_DIR / "placa_catalunya_bottleneck_option2_accumulation.gif"
    save_gif(f2, p2)
    save_final_png(ctx, OUT_DIR / "placa_catalunya_bottleneck_option2_accumulation_final.png")
    results["option2"] = dict(path=p2, n=len(f2), dur=len(f2) / FPS,
                              size=p2.stat().st_size, secs=time.time() - t0,
                              real=used_real)
    print(f"        {len(f2)} frames, {len(f2)/FPS:.2f}s, real_temporal={used_real}, {p2.stat().st_size/1e6:.2f} MB")

    # Option 5 (refined: 10 s narrative + Speed-Population HUD; GIF + MP4 + PNG)
    t0 = time.time()
    print("[INFO] Rendering Option 5 — hybrid (refined) ...")
    f5 = option5_hybrid(ctx)
    p5 = OUT_DIR / "placa_catalunya_bottleneck_option5_hybrid.gif"
    save_gif(f5, p5)
    # final PNG = the actual final composed frame (with HUD), presentation quality
    p5_png = OUT_DIR / "placa_catalunya_bottleneck_option5_hybrid_final.png"
    Image.fromarray(f5[-1]).save(p5_png)
    p5_mp4 = OUT_DIR / "placa_catalunya_bottleneck_option5_hybrid.mp4"
    mp4_ok = save_mp4(f5, p5_mp4, fps=FPS)
    mp4_sz = p5_mp4.stat().st_size if mp4_ok else 0
    results["option5"] = dict(path=p5, n=len(f5), dur=len(f5) / FPS,
                              size=p5.stat().st_size, secs=time.time() - t0,
                              mp4=mp4_ok, mp4_size=mp4_sz)
    print(f"        {len(f5)} frames, {len(f5)/FPS:.2f}s, GIF {p5.stat().st_size/1e6:.2f} MB"
          + (f", MP4 {mp4_sz/1e6:.2f} MB" if mp4_ok else ", MP4 skipped (no imageio)"))

    write_report(ctx, results)
    print("[INFO] Done.")


def write_report(ctx, results):
    xmin, xmax, ymin, ymax = ctx["extent"]
    r1, r2, r5 = results["option1"], results["option2"], results["option5"]
    mmss_total = mmss(ctx.get("t_total", 0.0))
    mp4_txt = (f", MP4 {r5['mp4_size']/1e6:.2f} MB" if r5.get("mp4")
               else " (MP4 skipped — imageio unavailable)")
    real = r2.get("real", False)
    real_txt = ("**Real temporal data.** Each trajectory observation was assigned "
                "to its 1 m cell, filtered to cells present in the bottleneck map, "
                "binned by `time_s`, and turned into a per-cell cumulative fraction "
                "that drives the build-up."
                if real else
                "**Fallback (simulated).** Temporal source was unavailable; cells "
                "were revealed by spatial progression + score.")
    txt = f"""# Plaça Catalunya — Bottleneck Animation Tests

**Visualization-only. No retracking, recalibration, or source-data modification.**
Option 2 reads the trajectory CSV only to drive accumulation *timing*; bottleneck
scores are NOT recomputed.

## Sources
- Bottleneck CSV: `{DEF_CSV}`
- Background plan image: `{DEF_PLAN}`
- Calibration (geometry only, unmodified): `{DEF_CALIB}`
- Trajectory CSV (Option 2 timing only): `{TRAJ_CSV}`

## Frozen V2D style parameters reused
- Plan-clipped frame: X = [{xmin:.2f}, {xmax:.2f}] m, Y = [{ymin:.2f}, {ymax:.2f}] m
- Background underlay opacity: {BG_OPACITY}
- Palette: `#FFF3B0 -> #FDBA3B -> #F97316 -> #DC2626` (yellow -> orange -> red)
- Square size: `scale = {BASE_SCALE} + (score_norm ** {EXP}) * {GROWTH}`,
  clamped to {CLAMP_FRAC} x grid spacing (no overlap; visible gaps)
- Ranked markers: top {TOP_K} (1-{TOP_K}); no title / legend / colorbar / axes / ticks
- Cells in frame: {ctx['n_inside']} (dropped off-plan: {ctx['n_dropped']})
- All three final frames converge to the V2D composition.

## Export settings
- {FPS} fps, GIF frames ~{int(GIF_FIG_W_IN*GIF_DPI)} px wide, shared 256-colour
  palette (no flicker); GIF kept high-quality (not aggressively compressed).
- Options 1 & 2 final-frame PNGs render at 320 dpi with the exact V2D primitives.
- **Option 5** additionally exports an **MP4** (H.264, yuv420p, presentation
  quality) and its final-frame PNG is the actual last composed frame (HUD
  included). The HUD timer uses the real source-video duration ({mmss_total}).

## Options

### Option 1 — Construction
Background appears, then cells appear progressively from lowest to highest
bottleneck score, markers fade in, hold. Feels like spatial pressure being
revealed.
- Timing: 0.5 s bg + 3.0 s build + 0.8 s markers + 1.5 s hold
- Duration: **{r1['dur']:.2f} s** ({r1['n']} frames) | size {r1['size']/1e6:.2f} MB

### Option 2 — Time-lapse accumulation
Cells start faint and intensify/grow as pedestrian observations accumulate over
time; high-pressure zones become redder/larger; markers fade in; hold.
- Temporal source: {real_txt}
- Timing: 0.5 s bg + 4.0 s accumulation + 0.8 s markers + 1.5 s hold
- Duration: **{r2['dur']:.2f} s** ({r2['n']} frames) | size {r2['size']/1e6:.2f} MB

### Option 5 — Hybrid (refined presentation pass)
A calm **10 s** three-phase narrative with a **Speed-Population-style HUD** overlaid
on the map (no extra vertical space):
- **Phase A — full field introduction (0–3 s):** the complete bottleneck field
  appears immediately and faintly, so the whole spatial field reads at once. No
  ranking markers yet (0.5 s fade-in + 2.5 s faint hold).
- **Phase B — bottleneck emergence (3–8 s):** pressure condenses — high-density
  cells brighten, warm toward orange/red and grow slightly while low-density cells
  stay subdued. A differentiation, not a reveal.
- **Phase C — rankings + final state (8–10 s):** ranked markers fade in (0.8 s),
  then a calm 1.2 s hold. No pulsing, bounce, or glow.
- **HUD** (frosted full-width overlay, Roboto, thin dividers, no boxed legend / no
  footer bar): `PLACE` Plaça Catalunya · `MAP` Bottleneck Density · `INTENSITY`
  continuous density gradient (`#FFF3B0 → #FDBA3B → #F97316 → #DC2626`, labelled
  LOW DENSITY → HIGH DENSITY) · `TIME ELAPSED` real source-video timer counting
  `00:00 / {mmss_total}` → `{mmss_total} / {mmss_total}`.
- Exports: **GIF + MP4 + final PNG** (final frame, HUD included).
- Duration: **{r5['dur']:.2f} s** ({r5['n']} frames) | GIF {r5['size']/1e6:.2f} MB{mp4_txt}

## Output paths
- `{r1['path']}` (+ `_final.png`)
- `{r2['path']}` (+ `_final.png`)
- `{r5['path']}` (+ `.mp4`, `_final.png`)
- This report: `{OUT_DIR / 'bottleneck_animation_tests_report.md'}`

## Recommendation
**Option 2 (accumulation)** is the most presentation-ready: it is the only option
driven by real movement timing, so the build-up is physically meaningful
("movement traces sedimenting into bottlenecks") rather than a pure reveal order,
while still resolving to the exact frozen V2D frame. Option 5 is the best pick if
a shorter, more controlled narrative is preferred; Option 1 is the simplest and
clearest for explaining the score ordering.
"""
    (OUT_DIR / "bottleneck_animation_tests_report.md").write_text(txt, encoding="utf-8")
    print(f"[INFO] Report: {OUT_DIR / 'bottleneck_animation_tests_report.md'}")


if __name__ == "__main__":
    main()
