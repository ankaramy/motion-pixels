#!/usr/bin/env python3
"""
Motion Pixels – Barcelona Study Sites Animation
1920 × 1080  |  30 fps  |  17 seconds
"""
import os, sys, math, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────
OUT_DIR = r"C:\Users\OWNER\Desktop\mp-presentation\motion_pixels_barcelona_location_animation"
MP4_OUT = os.path.join(OUT_DIR, "barcelona_motion_pixels.mp4")
PNG_OUT = os.path.join(OUT_DIR, "barcelona_motion_pixels_final.png")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# TIMING  (frames at 30 fps)
# ─────────────────────────────────────────────────────────────────────
FPS        = 30
TOTAL_SECS = 17
N_FRAMES   = TOTAL_SECS * FPS          # 510

F_BLANK    = int(1.5  * FPS)           # 45  – blank intro
F_ZOOM_S   = int(6.0  * FPS)           # 180 – zoom starts
F_ZOOM_E   = int(9.0  * FPS)           # 270 – zoom ends / dots start
F_DOT_S    = int(9.0  * FPS)           # 270
F_DOT_E    = int(12.0 * FPS)           # 360 – all dots visible
# 360-510 (5 s) → all dots pulse, hold on last frame

# ─────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────
C_BG    = '#F4F4F4'
C_ST    = '#1A1A1A'      # street lines
C_CORE  = '#3B0764'      # dot core   (violet-950)
C_R1    = '#7C3AED'      # pulse ring 1 (violet-600)
C_R2    = '#C4B5FD'      # pulse ring 2 (violet-300, outer glow)

# ─────────────────────────────────────────────────────────────────────
# STUDY SITES  (lat, lon)
# ─────────────────────────────────────────────────────────────────────
SITE_NAMES = [
    "Esplanade Espanya",
    "Placa Espanya",
    "Placa Montjuic",
    "Montjuic Stairs",
    "Placa Catalunya",
    "Placa MACBA",
    "Passeig Colom",
]
SITE_LATLON = np.array([
    [41.372841, 2.151004],
    [41.374637, 2.149926],
    [41.370603, 2.151927],
    [41.369084, 2.152622],
    [41.387070, 2.170018],
    [41.383005, 2.167093],
    [41.379613, 2.181737],
])

# ─────────────────────────────────────────────────────────────────────
# PROJECTION  (WGS-84 → Web Mercator / EPSG:3857)
# ─────────────────────────────────────────────────────────────────────
def merc(lat, lon):
    R = 6378137.0
    x = math.radians(lon) * R
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * R
    return x, y

def merc_arr(latlon):
    return np.array([merc(r[0], r[1]) for r in latlon])

SITE_XY = merc_arr(SITE_LATLON)   # (7, 2)

# ─────────────────────────────────────────────────────────────────────
# VIEW WINDOWS  [xmin, xmax, ymin, ymax]  in Mercator metres
# ─────────────────────────────────────────────────────────────────────
CANVAS_AR = 1920 / 1080

def make_window(lat_s, lat_n, lon_w, lon_e, target_ar=CANVAS_AR):
    """Return (xmin,xmax,ymin,ymax) padded to target aspect ratio."""
    sw = merc(lat_s, lon_w)
    ne = merc(lat_n, lon_e)
    cx = (sw[0] + ne[0]) / 2
    cy = (sw[1] + ne[1]) / 2
    dx = ne[0] - sw[0]
    dy = ne[1] - sw[1]
    if dx / dy < target_ar:
        dx = dy * target_ar
    else:
        dy = dx / target_ar
    return (cx - dx/2, cx + dx/2, cy - dy/2, cy + dy/2)

BCN_WIN = make_window(41.27, 41.51, 2.00, 2.30)   # full-city view
STU_WIN = make_window(41.362, 41.393, 2.143, 2.188, target_ar=CANVAS_AR)

# ─────────────────────────────────────────────────────────────────────
# JITTER  – push overlapping dots apart (in Mercator metres)
# ─────────────────────────────────────────────────────────────────────
def spread_dots(xy, min_m=300, iters=150):
    xy = xy.astype(float).copy()
    n  = len(xy)
    for _ in range(iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                v = xy[i] - xy[j]
                d = np.linalg.norm(v)
                if 0 < d < min_m:
                    push = (v / d) * ((min_m - d) / 2 + 15)
                    xy[i] += push
                    xy[j] -= push
                    moved = True
        if not moved:
            break
    return xy

SITE_XY_J = spread_dots(SITE_XY.copy())

# ─────────────────────────────────────────────────────────────────────
# STREET NETWORK
# ─────────────────────────────────────────────────────────────────────
def load_streets():
    try:
        import osmnx as ox
        ox.settings.use_cache = True
        ox.settings.log_console = False
        print("Downloading Barcelona street network (cached after first run)...")
        kw = dict(network_type='drive', simplify=True)
        # osmnx 2.x: bbox=(left, bottom, right, top) i.e. (lon_w,lat_s,lon_e,lat_n)
        try:
            G = ox.graph_from_bbox(bbox=(2.00, 41.27, 2.30, 41.51), **kw)
        except Exception:
            G = ox.graph_from_bbox(north=41.51, south=41.27,
                                   east=2.30, west=2.00, **kw)
        G = ox.project_graph(G, to_crs='EPSG:3857')
        segs = []
        for u, v, data in G.edges(data=True):
            if 'geometry' in data:
                xs, ys = data['geometry'].xy
                for k in range(len(xs) - 1):
                    segs.append(((xs[k], ys[k]), (xs[k+1], ys[k+1])))
            else:
                xu, yu = G.nodes[u]['x'], G.nodes[u]['y']
                xv, yv = G.nodes[v]['x'], G.nodes[v]['y']
                segs.append(((xu, yu), (xv, yv)))
        print(f"  Loaded {len(segs):,} segments.")
        return segs
    except Exception as e:
        print(f"  OSMnx failed ({e}) – using procedural grid fallback.")
        return _procedural_grid()


def _procedural_grid():
    """Eixample-like grid + Barcelona diagonals as fallback."""
    x0, x1, y0, y1 = BCN_WIN
    w, h = x1 - x0, y1 - y0
    segs = []
    step = 260
    for xi in np.arange(x0, x1 + step, step):
        segs.append(((xi, y0), (xi, y1)))
    for yi in np.arange(y0, y1 + step, step):
        segs.append(((x0, yi), (x1, yi)))
    # La Diagonal
    segs.append(((x0 + 0.05*w, y0 + 0.60*h), (x0 + 0.88*w, y0 + 0.83*h)))
    # Av. Meridiana
    segs.append(((x0 + 0.55*w, y0), (x0 + 0.65*w, y0 + h)))
    return segs

# ─────────────────────────────────────────────────────────────────────
# EASING HELPERS
# ─────────────────────────────────────────────────────────────────────
def smoothstep(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3 - 2 * t)

def lerp(a, b, t):
    return a + (b - a) * t

# ─────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────
def main():
    segs = load_streets()
    N    = len(segs)

    # Sort radially from Barcelona centre → "outward scan" draw feel
    cx = (BCN_WIN[0] + BCN_WIN[1]) / 2
    cy = (BCN_WIN[2] + BCN_WIN[3]) / 2
    radii = [math.hypot((s[0][0]+s[1][0])/2 - cx,
                        (s[0][1]+s[1][1])/2 - cy) for s in segs]
    segs  = [segs[i] for i in np.argsort(radii)]

    # ── Build figure ──────────────────────────────────────────────────
    DPI = 96
    fig = plt.figure(figsize=(1920/DPI, 1080/DPI), dpi=DPI)
    fig.patch.set_facecolor(C_BG)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(C_BG)
    ax.axis('off')
    ax.set_xlim(BCN_WIN[0], BCN_WIN[1])
    ax.set_ylim(BCN_WIN[2], BCN_WIN[3])

    lc = LineCollection([], linewidths=0.38, colors=C_ST, alpha=0.70, zorder=2)
    ax.add_collection(lc)

    nS = len(SITE_XY_J)
    xs = list(SITE_XY_J[:, 0])
    ys = list(SITE_XY_J[:, 1])

    core,  = ax.plot([], [], 'o', color=C_CORE, ms=10,  mew=0, zorder=12)
    ring1, = ax.plot([], [], 'o', color=C_R1,   ms=10,  mew=0, zorder=11, alpha=0)
    ring2, = ax.plot([], [], 'o', color=C_R2,   ms=10,  mew=0, zorder=10, alpha=0)

    def draw_frame(f):
        # ── 1. Street reveal ──────────────────────────────────────────
        if f < F_BLANK:
            n = 0
        else:
            t = smoothstep((f - F_BLANK) / max(F_ZOOM_E - F_BLANK, 1))
            n = min(int(t * N), N)
        lc.set_segments(segs[:n])

        # ── 2. Camera zoom ────────────────────────────────────────────
        if f < F_ZOOM_S:
            xl = (BCN_WIN[0], BCN_WIN[1])
            yl = (BCN_WIN[2], BCN_WIN[3])
        elif f <= F_ZOOM_E:
            t  = smoothstep((f - F_ZOOM_S) / (F_ZOOM_E - F_ZOOM_S))
            xl = (lerp(BCN_WIN[0], STU_WIN[0], t),
                  lerp(BCN_WIN[1], STU_WIN[1], t))
            yl = (lerp(BCN_WIN[2], STU_WIN[2], t),
                  lerp(BCN_WIN[3], STU_WIN[3], t))
        else:
            xl = (STU_WIN[0], STU_WIN[1])
            yl = (STU_WIN[2], STU_WIN[3])
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)

        # ── 3. Dot visibility ─────────────────────────────────────────
        if f < F_DOT_S:
            nv = 0
        elif f < F_DOT_E:
            interval = (F_DOT_E - F_DOT_S) / nS
            nv = min(int((f - F_DOT_S) / interval) + 1, nS)
        else:
            nv = nS

        if nv > 0:
            core.set_data(xs[:nv], ys[:nv])
        else:
            core.set_data([], [])

        # ── 4. Pulse rings ────────────────────────────────────────────
        if nv > 0:
            # continuous phase from when dots started appearing
            phi  = ((f - F_DOT_S) / FPS * 1.4) % 1.0
            phi2 = (phi + 0.50) % 1.0               # staggered second ring

            ring1.set_data(xs[:nv], ys[:nv])
            ring1.set_markersize(10 + phi  * 22)
            ring1.set_alpha(0.60 * (1 - phi))

            ring2.set_data(xs[:nv], ys[:nv])
            ring2.set_markersize(10 + phi2 * 40)
            ring2.set_alpha(0.30 * (1 - phi2))
        else:
            ring1.set_data([], []);  ring1.set_alpha(0)
            ring2.set_data([], []);  ring2.set_alpha(0)

    # ── Export MP4 via imageio-ffmpeg ─────────────────────────────────
    import imageio
    import imageio.plugins.ffmpeg

    ffmpeg_exe = imageio.plugins.ffmpeg.get_exe()
    print(f"Using ffmpeg: {ffmpeg_exe}")

    writer = imageio.get_writer(
        MP4_OUT,
        fps=FPS,
        quality=9,
        codec='libx264',
        pixelformat='yuv420p',
        ffmpeg_params=['-crf', '16', '-preset', 'slow'],
        ffmpeg_log_level='error',
    )

    print(f"Rendering {N_FRAMES} frames @ {FPS} fps...")
    for f in range(N_FRAMES):
        if f % (FPS * 2) == 0:
            pct = f / N_FRAMES * 100
            print(f"  [{pct:5.1f}%]  frame {f:4d}/{N_FRAMES}  t={f/FPS:.1f}s")
        draw_frame(f)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(buf)

    writer.close()
    print(f"MP4 saved: {MP4_OUT}")

    # ── Final still PNG ───────────────────────────────────────────────
    draw_frame(N_FRAMES - 1)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    import imageio
    imageio.imwrite(PNG_OUT, buf)
    print(f"PNG saved: {PNG_OUT}")

    plt.close(fig)
    print("All done.")


if __name__ == '__main__':
    main()
