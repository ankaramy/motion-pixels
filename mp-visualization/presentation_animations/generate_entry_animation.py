#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motion Pixels - Noumena-inspired computational entry animation.

Pipeline (no Manim / MoviePy required):
    PIL  -> text mask + clean typography
    NumPy-> particle simulation, flow-field trajectories, ink buffers
    imageio + bundled imageio-ffmpeg -> H.264 MP4

Sequence:
    1. Empty white field, sparse points appear
    2. Points move along curved (flow-field) trajectories, leaving trails
    3. Data field expands (more points / trails)
    4. Trails dissolve into square pixels, drift inward
    5. Pixels organise into "Motion Pixels", then sharpen to clean Gugi type
    6. Refined glitch (horizontal displacement, block offsets, scanlines)
    7. Inversion: title -> pixels, white bg -> black, polarity reverses to white title
    8. Final hold: black bg, crisp white title (2 s)

Output: 1920x1080, 60 fps, ~11.5 s.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
HERE     = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(HERE, "Gugi-Regular.ttf")
MP4_OUT  = os.path.join(HERE, "motion_pixels_entry_animation.mp4")
PNG_OUT  = os.path.join(HERE, "motion_pixels_final_black.png")

W, H   = 1920, 1080
FPS    = 60
SEED   = 7
PREVIEW = os.environ.get("MP_PREVIEW", "0") == "1"   # render a sparse subset of frames

rng = np.random.default_rng(SEED)

# Phase boundaries (cumulative frame indices) -- total 690 frames = 11.5 s
F_APPEAR = 72     # 0.0 -> 1.2   sparse points appear
F_TRACES = 192    # 1.2 -> 3.2   curved trajectories + trails
F_EXPAND = 270    # 3.2 -> 4.5   data field expands
F_PIXEL  = 366    # 4.5 -> 6.1   trails dissolve into pixels, drift inward
F_TYPO   = 456    # 6.1 -> 7.6   pixels form + sharpen into typography
F_GLITCH = 492    # 7.6 -> 8.2   refined glitch
F_INVERT = 570    # 8.2 -> 9.5   inversion / polarity reversal
F_HOLD   = 690    # 9.5 -> 11.5  final black hold
N_FRAMES = F_HOLD

# Colours (grayscale ink model: 0 = white, 1 = black)
INK_MAX_DARK = 0.92

# ----------------------------------------------------------------------------
# EASING
# ----------------------------------------------------------------------------
def smoothstep(a, b, x):
    if a == b:
        return 1.0 if x >= b else 0.0
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def ease_in_out(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# ----------------------------------------------------------------------------
# TEXT  ->  clean image + sampled pixel targets
# ----------------------------------------------------------------------------
TEXT = "Motion Pixels"

def fit_font(target_w):
    """Find a Gugi size so the text spans ~target_w px."""
    size = 200
    f = ImageFont.truetype(FONT_PATH, size)
    bb = f.getbbox(TEXT)
    w = bb[2] - bb[0]
    size = int(size * target_w / w)
    return ImageFont.truetype(FONT_PATH, size)

FONT = fit_font(int(W * 0.58))

def render_text_mask():
    """Grayscale mask (0..255, glyph=255) of centred title."""
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    bb = d.textbbox((0, 0), TEXT, font=FONT)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x = (W - tw) / 2 - bb[0]
    y = (H - th) / 2 - bb[1]
    d.text((x, y), TEXT, font=FONT, fill=255)
    return np.asarray(img, dtype=np.float32) / 255.0

TEXT_MASK = render_text_mask()                       # 0..1 glyph coverage

# clean black-on-white title (the "sharp" image), grayscale 0..255.
# Pure values so the inverted final hold is crisp white-on-black.
CLEAN = ((1.0 - TEXT_MASK) * 255.0).astype(np.float32)

def sample_targets(cell=9, cap=3000):
    """Grid-sample glyph interior -> square-pixel target positions."""
    ys, xs = [], []
    half = cell // 2
    for gy in range(half, H, cell):
        row = TEXT_MASK[gy]
        for gx in range(half, W, cell):
            if row[gx] > 0.5:
                xs.append(gx)
                ys.append(gy)
    pts = np.stack([np.array(xs, np.float32), np.array(ys, np.float32)], axis=1)
    if len(pts) > cap:
        idx = rng.choice(len(pts), cap, replace=False)
        pts = pts[idx]
    return pts

TARGETS = sample_targets(cell=7, cap=3200)
N = len(TARGETS)
print(f"[setup] font px={FONT.size}  targets/particles N={N}")

# ----------------------------------------------------------------------------
# PARTICLE STATE
# ----------------------------------------------------------------------------
# spawn positions: scattered across the field
px = rng.uniform(0, W, N).astype(np.float32)
py = rng.uniform(0, H, N).astype(np.float32)

# one-to-one (shuffled) mapping particle -> text target
order = rng.permutation(N)
tgt = TARGETS[order]
tx, ty = tgt[:, 0].copy(), tgt[:, 1].copy()

# per-particle traits
base_dark = rng.uniform(0.40, 0.88, N).astype(np.float32)   # gray .. near-black
speed_var = rng.uniform(0.6, 1.4, N).astype(np.float32)
# births spread across appear+traces+expand so the field grows progressively
birth = rng.uniform(0, F_EXPAND, N).astype(np.float32)
birth[rng.permutation(N)[: N // 5]] = rng.uniform(0, F_APPEAR, N // 5)  # an early wave

# flow-field parameters (smooth, organic curved drift)
k1, k2, k3, k4 = 0.0042, 0.0035, 0.0038, 0.0046
p1, p2, p3, p4 = rng.uniform(0, 2 * np.pi, 4)

def flow(x, y):
    vx = np.sin(k1 * y + p1) + 0.7 * np.cos(k2 * x + p2)
    vy = np.cos(k3 * x + p3) + 0.7 * np.sin(k4 * y + p4)
    return vx, vy

trail = np.zeros((H, W), dtype=np.float32)   # persistent ink buffer

# ----------------------------------------------------------------------------
# RENDER HELPERS
# ----------------------------------------------------------------------------
def stamp_squares(buf, xs, ys, vals, size):
    """Max-blend filled squares of given size at integer positions."""
    size = max(1, int(round(size)))
    xi = np.round(xs).astype(np.int32)
    yi = np.round(ys).astype(np.int32)
    for dy in range(size):
        yy = yi + dy
        for dx in range(size):
            xx = xi + dx
            m = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            np.maximum.at(buf, (yy[m], xx[m]), vals[m])

def apply_glitch(gray, intensity):
    """Subtle: horizontal band displacement + block offsets + scanline fragments.
    No RGB split, no VHS. Operates mainly over the central title band."""
    out = gray.copy()
    if intensity <= 0:
        return out
    y0, y1 = int(H * 0.30), int(H * 0.70)        # title band
    # horizontal band displacement
    n_bands = int(3 + 5 * intensity)
    for _ in range(n_bands):
        by = rng.integers(y0, y1 - 8)
        bh = int(rng.integers(2, 14))
        sh = int(rng.integers(-1, 2) * rng.integers(4, int(6 + 34 * intensity)))
        if sh:
            out[by:by + bh] = np.roll(out[by:by + bh], sh, axis=1)
    # small rectangular block offsets
    n_blk = int(2 + 4 * intensity)
    for _ in range(n_blk):
        bx = rng.integers(int(W * 0.20), int(W * 0.75))
        by = rng.integers(y0, y1 - 20)
        bw = int(rng.integers(20, 90))
        bh = int(rng.integers(6, 22))
        sx = int(rng.integers(-22, 23) * intensity)
        if sx:
            seg = out[by:by + bh, bx:bx + bw]
            out[by:by + bh, bx:bx + bw] = np.roll(seg, sx, axis=1)
    # faint scanline fragments
    n_scan = int(2 + 4 * intensity)
    for _ in range(n_scan):
        sy = rng.integers(y0, y1)
        x0 = rng.integers(int(W * 0.20), int(W * 0.60))
        xw = rng.integers(60, 320)
        out[sy, x0:x0 + xw] = np.clip(out[sy, x0:x0 + xw] - 30 * intensity, 0, 255)
    return out

# inversion block-dissolve thresholds (per block)
DBLK = 14
gh, gw = (H + DBLK - 1) // DBLK, (W + DBLK - 1) // DBLK
DISS = rng.random((gh, gw)).astype(np.float32)
DISS_FULL = np.kron(DISS, np.ones((DBLK, DBLK), np.float32))[:H, :W]

def gray_to_rgb(gray):
    g = np.clip(gray, 0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=2)

# ----------------------------------------------------------------------------
# MAIN RENDER LOOP
# ----------------------------------------------------------------------------
def render_frame(f):
    global trail, px, py

    # ---- image-based late phases (glitch / inversion / hold) ----
    if f >= F_TYPO:
        if f < F_GLITCH:                                   # 6. refined glitch
            t = (f - F_TYPO) / max(1, (F_GLITCH - F_TYPO))
            inten = np.sin(np.pi * t) * 0.85               # ramp up then settle
            return gray_to_rgb(apply_glitch(CLEAN, inten))
        if f < F_INVERT:                                   # 7. inversion
            p = (f - F_GLITCH) / max(1, (F_INVERT - F_GLITCH))
            p = ease_in_out(p)
            B = 255.0 - CLEAN                              # white title on black
            flip = (DISS_FULL < p)
            out = np.where(flip, B, CLEAN)
            # brief pixel jitter while polarity is reversing
            if 0.08 < p < 0.75:
                j = int(6 * np.sin(np.pi * p))
                if j:
                    band = slice(int(H * 0.33), int(H * 0.67))
                    out[band] = np.roll(out[band], rng.integers(-j, j + 1), axis=1)
            return gray_to_rgb(out)
        # 8. final hold: crisp white title on black
        return gray_to_rgb(255.0 - CLEAN)

    # ---- particle phases (1-5) ----
    # global pacing
    speed_ramp = smoothstep(0, F_TRACES, f)                # slow confident start
    m = ease_in_out(smoothstep(F_EXPAND, F_TYPO, f))       # wander -> target blend
    # marker geometry: small soft points -> square pixels
    msize = 2.5 + 5.0 * m
    # effective darkness: gray field -> crisp near-black pixels
    eff = base_dark * (1.0 - m) + INK_MAX_DARK * m

    # advance flow-field wander
    vx, vy = flow(px, py)
    step = 1.9 * speed_ramp * speed_var
    px += vx * step
    py += vy * step
    px %= W
    py %= H

    # blended render position (pull inward to the title = "spatial force")
    rxp = (1.0 - m) * px + m * tx
    ryp = (1.0 - m) * py + m * ty

    # trail decay schedule: long trails early, short as pixels form
    if f < F_EXPAND:
        decay = 0.90
    elif f < F_PIXEL:
        decay = 0.90 - 0.34 * smoothstep(F_EXPAND, F_PIXEL, f)   # -> 0.56
    else:
        decay = 0.52
    trail *= decay

    # alive + alpha (fade-in from birth)
    alpha = np.clip((f - birth) / 22.0, 0.0, 1.0)
    alive = alpha > 0.001
    if alive.any():
        vals = (eff * alpha)[alive]
        stamp_squares(trail, rxp[alive], ryp[alive], vals, msize)

    gray = (1.0 - np.clip(trail, 0, 1)) * 255.0

    # 5b. sharpen: crossfade pixel field -> clean Gugi type at end of typography
    cf = smoothstep(F_TYPO - 34, F_TYPO, f)
    if cf > 0:
        gray = gray * (1.0 - cf) + CLEAN * cf

    return gray_to_rgb(gray)

# ----------------------------------------------------------------------------
# DRIVE
# ----------------------------------------------------------------------------
def main():
    if PREVIEW:
        # render representative frames to PNGs for a quick visual check
        prev_dir = os.path.join(HERE, "_preview")
        os.makedirs(prev_dir, exist_ok=True)
        marks = [10, 40, 72, 130, 230, 300, 340, 400, 440, 456,
                 474, 500, 530, 560, 600, 689]
        for f in range(N_FRAMES):
            frame = render_frame(f)            # keep state evolving
            if f in marks:
                Image.fromarray(frame).save(os.path.join(prev_dir, f"f{f:03d}.png"))
        print(f"[preview] wrote frames to {prev_dir}")
        return

    writer = imageio.get_writer(
        MP4_OUT, fps=FPS, codec="libx264", quality=None,
        macro_block_size=8, ffmpeg_params=["-crf", "16", "-preset", "slow",
                                           "-pix_fmt", "yuv420p"],
        ffmpeg_log_level="error",
    )
    last = None
    for f in range(N_FRAMES):
        last = render_frame(f)
        writer.append_data(last)
        if f % 60 == 0:
            print(f"  frame {f:3d}/{N_FRAMES}")
    writer.close()

    # final black-background still
    final = gray_to_rgb(255.0 - CLEAN)
    imageio.imwrite(PNG_OUT, final)
    print(f"[done] {MP4_OUT}")
    print(f"[done] {PNG_OUT}")

if __name__ == "__main__":
    main()
