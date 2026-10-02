#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motion Pixels - logo glitch/reconstruction title animation.

"data becoming architecture" -- a clean bold Gugi logo breaks into precise
square pixels + horizontal slice tearing (B/W only, no RGB/VHS), holds the
glitch state, then reabsorbs perfectly back to the clean logo.

Designed as a SEAMLESS LOOP: glitch intensity is 0 at both the first and last
frame, so clean -> glitch -> clean wraps with no visible seam.

Outputs (1920x1080, 60 fps):
    motion_pixels_logo_animation.mp4   (H.264)
    motion_pixels_logo_animation.gif   (presentation-optimized loop)

Tooling: NumPy + Pillow, encoded via the ffmpeg bundled with imageio-ffmpeg.
No web / HTML / CSS / JS.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio

HERE      = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(HERE, "Gugi-Regular.ttf")
MP4_OUT   = os.path.join(HERE, "motion_pixels_logo_animation.mp4")
GIF_OUT   = os.path.join(HERE, "motion_pixels_logo_animation.gif")
CLEAN_PNG = os.path.join(HERE, "clean_logo.png")
GLITCH_PNG= os.path.join(HERE, "full_glitch_preview.png")

W, H    = 1920, 1080
FPS     = 60
GSEED   = 11
PREVIEW = os.environ.get("MP_PREVIEW", "0") == "1"

# ---- timeline (frames @ 60fps) -- 420 frames = 7.0 s, seamless loop ----
F0      = 30     # 0.0-0.5  clean hold (loop rest) + micro-flicker
F1      = 120    # 0.5-2.0  pixel formation begins (edges scatter, mild slice)
F2      = 210    # 2.0-3.5  transformation -> full glitch
F3      = 300    # 3.5-5.0  full glitch hold + subtle twitch
F4      = 390    # 5.0-6.5  reconstruction -> clean
NF      = 420    # 6.5-7.0  clean hold (== opening)

# ----------------------------------------------------------------------------
def smoothstep(a, b, x):
    if a == b:
        return float(x >= b)
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# ----------------------------------------------------------------------------
# CLEAN LOGO (bold two-line staggered Gugi), supersampled for crisp edges
# ----------------------------------------------------------------------------
def build_clean(scale=3):
    w, h = W * scale, H * scale
    fs = int(250 * scale)
    sw = max(1, int(fs * 0.028))
    font = ImageFont.truetype(FONT_PATH, fs)
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)

    def measure(t):
        b = d.textbbox((0, 0), t, font=font, stroke_width=sw)
        return b, b[2]-b[0], b[3]-b[1]

    b1, w1, h1 = measure("Motion")
    b2, w2, h2 = measure("Pixels")
    line_gap = int(fs * 0.06)
    stagger  = int(fs * 0.62)
    block_w = max(w1, stagger + w2)
    block_h = h1 + line_gap + h2
    x0 = (w - block_w) // 2
    y0 = (h - block_h) // 2
    d.text((x0 - b1[0], y0 - b1[1]), "Motion", font=font, fill=255,
           stroke_width=sw, stroke_fill=255)
    d.text((x0 + stagger - b2[0], y0 + h1 + line_gap - b2[1]), "Pixels",
           font=font, fill=255, stroke_width=sw, stroke_fill=255)
    arr = np.asarray(img, dtype=np.uint8)
    arr = np.asarray(Image.fromarray(arr).resize((W, H), Image.LANCZOS),
                     dtype=np.float32) / 255.0
    return arr                                  # coverage 0..1 (1 = black ink)

C = build_clean(3)
Image.fromarray(((1.0 - C) * 255).astype(np.uint8)).save(CLEAN_PNG)

# binary ink + bbox
INK = C > 0.5
ys, xs = np.where(INK)
XMIN, XMAX = xs.min(), xs.max()
YMIN, YMAX = ys.min(), ys.max()
CX = 0.5 * (XMIN + XMAX)
CY = 0.5 * (YMIN + YMAX)
HALFW = max(1.0, 0.5 * (XMAX - XMIN))
HALFH = max(1.0, 0.5 * (YMAX - YMIN))
BW = XMAX - XMIN
BH = YMAX - YMIN

# edge pixels (ink minus eroded ink)
def erode(mask, k=2):
    e = mask.copy()
    e[:, :-k] &= mask[:, k:]
    e[:, k:]  &= mask[:, :-k]
    e[:-k, :] &= mask[k:, :]
    e[k:, :]  &= mask[:-k, :]
    return e

EDGE = INK & ~erode(INK, 2)
ey, ex = np.where(EDGE)

# region emphasis weight on edges: left (M), right (S), bottom (P / lower line)
wx = ex.astype(np.float32); wy = ey.astype(np.float32)
W_left   = np.exp(-(wx - XMIN) / (0.18 * BW))
W_right  = np.exp(-(XMAX - wx) / (0.18 * BW))
W_bottom = np.exp(-(YMAX - wy) / (0.16 * BH))
EW = 0.18 + 1.0 * W_left + 1.0 * W_right + 0.9 * W_bottom
EW = EW / EW.sum()

# outward drift direction per edge pixel (left/right + slight down)
dxv = (wx - CX) / HALFW
dyv = (wy - CY) / HALFH + 0.25
dnorm = np.maximum(1e-3, np.hypot(dxv, dyv))
DIRX = dxv / dnorm
DIRY = dyv / dnorm

# ----------------------------------------------------------------------------
# STATIC GLITCH STRUCTURE (seeded once -> temporally smooth / designed)
# ----------------------------------------------------------------------------
grng = np.random.default_rng(GSEED)

# thin horizontal scanlines across the logo (ragged-edge displacement, readable)
bands = []
yb = YMIN - 16
while yb < YMAX + 16:
    bh = int(grng.integers(2, 6))
    n = grng.standard_normal()
    n = np.sign(n) * min(1.0, abs(n) / 2.6)        # mostly small, few larger
    bands.append((yb, min(H, yb + bh), float(n)))
    yb += bh
# a few bands flagged for occasional "twitch" jumps in the hold
twitch_bands = set(grng.choice(len(bands), size=max(3, len(bands)//8),
                               replace=False).tolist())

# designed rectangular block offsets
blocks = []
for _ in range(7):
    bx = int(grng.integers(XMIN, XMAX - 40))
    by = int(grng.integers(YMIN, YMAX - 16))
    bw = int(grng.integers(30, 130))
    bh = int(grng.integers(8, 26))
    nn = float(np.sign(grng.standard_normal()) *
               grng.uniform(0.4, 1.0))
    blocks.append((bx, by, bw, bh, nn))

# horizontal tearing streaks anchored to right/left edges
n_streak = 90
si = grng.choice(len(ex), size=n_streak, p=EW)
streaks = []
for i in si:
    sx, sy = int(ex[i]), int(ey[i])
    sdir = 1 if sx >= CX else -1
    slen = int(grng.integers(20, 160))
    sthk = int(grng.integers(1, 4))
    streaks.append((sx, sy, sdir, slen, sthk))

# ----------------------------------------------------------------------------
# PER-FRAME GLITCH COMPOSITION
# ----------------------------------------------------------------------------
def stamp_blocks(buf, xc, yc, sizes):
    """Filled black squares of given per-particle size at (xc,yc)."""
    xi = np.round(xc).astype(np.int32)
    yi = np.round(yc).astype(np.int32)
    for s in np.unique(sizes):
        sel = sizes == s
        s = int(s)
        xx = xi[sel]; yy = yi[sel]
        for dy in range(s):
            for dx in range(s):
                X = xx + dx; Y = yy + dy
                m = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
                buf[Y[m], X[m]] = 1.0

def render_glitch(f):
    # ---- intensity envelope (0 at both ends -> seamless) ----
    if f <= F0:
        s = 0.0
    elif f <= F1:
        s = 0.40 * smoothstep(F0, F1, f)          # formation
    elif f <= F2:
        s = 0.40 + 0.60 * smoothstep(F1, F2, f)   # morph -> full
    elif f <= F3:
        s = 1.0                                    # full hold
    elif f <= F4:
        s = 1.0 - smoothstep(F3, F4, f)           # reconstruct
    else:
        s = 0.0                                    # clean hold

    # layer gates (designed progression)
    g_scatter = s                                  # edges scatter from the start
    g_slice   = smoothstep(0.15, 1.0, s)
    g_heavy   = smoothstep(0.40, 1.0, s)           # blocks + streaks + tearing
    in_hold   = (F2 < f <= F3)

    rng = np.random.default_rng(1000 + f)
    Cg = C.copy()

    # 1) horizontal slice displacement (smooth, amplitude follows s)
    if g_slice > 0:
        amp = 22.0 * g_slice
        for i, (y0, y1, n) in enumerate(bands):
            off = amp * n
            if in_hold and i in twitch_bands:
                off += rng.integers(-3, 4)         # tiny 1-3px twitch jumps
            off = int(round(off))
            if off:
                Cg[y0:y1] = np.roll(Cg[y0:y1], off, axis=1)

    # 2) designed block offsets
    if g_heavy > 0:
        for (bx, by, bw, bh, nn) in blocks:
            off = int(round(24 * g_heavy * nn))
            if off:
                seg = Cg[by:by+bh, bx:bx+bw]
                Cg[by:by+bh, bx:bx+bw] = np.roll(seg, off, axis=1)

    # 3) edge pixel scatter -- square pixels detach & hover outward
    if g_scatter > 0:
        n_part = int(2200 * g_scatter)
        if n_part > 0:
            idx = rng.choice(len(ex), size=n_part, p=EW)
            dist = rng.random(n_part) ** 1.7 * (80.0 * g_scatter)
            px = ex[idx] + DIRX[idx] * dist + rng.normal(0, 3.5, n_part)
            py = ey[idx] + DIRY[idx] * dist + rng.normal(0, 2.5, n_part)
            far = dist / (80.0 * g_scatter + 1e-6)
            sizes = np.clip(np.round(10 - 7 * far +
                                     rng.integers(-1, 2, n_part)), 3, 10)
            stamp_blocks(Cg, px, py, sizes)
            # designed square "holes" punched at edges (pure white, B/W only)
            if g_heavy > 0.4:
                cidx = rng.choice(len(ex), size=int(550 * g_heavy), p=EW)
                cs = np.full(len(cidx), 5)
                buf0 = np.zeros_like(Cg)
                stamp_blocks(buf0, ex[cidx] - 2, ey[cidx] - 2, cs)
                Cg[buf0 > 0] = 0.0

    # 4) horizontal tearing streaks
    if g_heavy > 0:
        for (sx, sy, sdir, slen, sthk) in streaks:
            L = int(slen * g_heavy)
            if L <= 0:
                continue
            x_a = sx if sdir > 0 else sx - L
            x_b = sx + L if sdir > 0 else sx
            x_a = max(0, x_a); x_b = min(W, x_b)
            Cg[sy:sy+sthk, x_a:x_b] = 1.0

    gray = (1.0 - np.clip(Cg, 0.0, 1.0)) * 255.0

    # 5) micro-flicker during the clean->formation lead-in (0 at exact ends)
    if 0 < f < F1:
        win = np.sin(np.pi * f / F1)                    # 0 at f=0 and f=F1
        if win > 1e-6:
            flick = 1.0 - 0.05 * win * rng.random()
            gray = 255.0 - (255.0 - gray) * flick

    return np.clip(gray, 0, 255).astype(np.uint8)

def frame_rgb(f):
    g = render_glitch(f)
    return np.stack([g, g, g], axis=2)

# ----------------------------------------------------------------------------
def main():
    if PREVIEW:
        pdir = os.path.join(HERE, "_preview"); os.makedirs(pdir, exist_ok=True)
        for f in [0, 60, 100, 150, 180, 210, 255, 300, 345, 390, 419]:
            Image.fromarray(frame_rgb(f)).save(os.path.join(pdir, f"f{f:03d}.png"))
        Image.fromarray(frame_rgb(255)).save(GLITCH_PNG)
        print("preview written")
        return

    writer = imageio.get_writer(
        MP4_OUT, fps=FPS, codec="libx264", macro_block_size=8,
        ffmpeg_params=["-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p"],
        ffmpeg_log_level="error")
    gif_frames = []
    for f in range(NF):
        rgb = frame_rgb(f)
        writer.append_data(rgb)
        if f % 2 == 0:                              # GIF @ 30fps
            small = Image.fromarray(rgb).resize((960, 540), Image.LANCZOS)
            gif_frames.append(small)
        if f % 60 == 0:
            print(f"  frame {f:3d}/{NF}")
    writer.close()
    print(f"[done] {MP4_OUT}")

    # presentation GIF: seamless loop, palette-reduced
    gif_frames[0].save(
        GIF_OUT, save_all=True, append_images=gif_frames[1:],
        duration=int(1000/30), loop=0, optimize=True, disposal=2)
    print(f"[done] {GIF_OUT}")
    Image.fromarray(frame_rgb(255)).save(GLITCH_PNG)

if __name__ == "__main__":
    main()
