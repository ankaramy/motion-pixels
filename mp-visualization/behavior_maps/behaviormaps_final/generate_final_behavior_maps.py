"""
generate_final_behavior_maps.py
-------------------------------
MOTION PIXELS — FINAL BEHAVIOR MAPS BATCH PIPELINE.

Productionises the three finalized Plaça Catalunya behaviour-map families across
all 7 recordings, using the exact accepted recipes (see
FINAL_BEHAVIOR_MAP_RECIPES.md) and the unified canvas / HUD standard
(final_unified_series). VISUALIZATION-ONLY: no retracking, recalibration, model
inference, or source-data modification. Bottleneck scores are read from
bottleneck_cells.csv and never recomputed.

Families:
  1. flow_fields        — dark wireframe + cyan→magenta strings/dust  (DARK HUD)
  2. speed_population    — arctic satellite + speed-coloured dots       (LIGHT HUD)
  3. bottleneck_density  — arctic underlay + yellow→red square cells    (LIGHT HUD)

Shared canvas: map 1920×1080 + footer 70 = 1920×1150; GIF 760×454 @ 10 fps
(20 fps source / stride 2); MP4 1920×1150 H.264; still 1920×1150. Font Roboto.
World window = 16:9 fit of each plan (full plan always visible; neutral padding,
no stretch).

Modes:
  python generate_final_behavior_maps.py --scout         # discover + manifest only (default)
  python generate_final_behavior_maps.py --smoke         # placa_catalunya only (all families)
  python generate_final_behavior_maps.py --all           # all 7 recordings
  optional: --families flow_fields,speed_population,bottleneck_density
            --recordings placa_catalunya_01,red_bridge_combined_01
"""

from pathlib import Path
import argparse
import json
import sys
import time
import traceback

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
from PIL import Image
import imageio.v2 as imageio

# ---------------------------------------------------------------------------
# Module wiring (read-only reuse of the accepted content pipelines)
# ---------------------------------------------------------------------------
BEHAVIOR_DIR = Path(__file__).resolve().parents[1]
FC_DIR  = BEHAVIOR_DIR / "flow_currents"
HERO_DIR = BEHAVIOR_DIR / "trace_dust_hero"
SP_DIR  = BEHAVIOR_DIR / "speed_population"
BN_DIR  = BEHAVIOR_DIR / "bottleneck_animations"
for p in (BEHAVIOR_DIR, FC_DIR, HERO_DIR, SP_DIR, BN_DIR):
    sys.path.insert(0, str(p))

import generate_bottleneck_map_test as v2d          # warp + palette + cell prims
import generate_flow_currents_v3 as ffv3            # flow pipeline (V5 params applied here)
import generate_trace_dust_hero_v3 as h3            # plan styling / density helpers
import generate_speed_population_v2 as sp2          # arctic bg / grid / dots
import generate_speed_population_v3 as sp3          # warm palette / trace cfg
spv1 = sp2.v1
import generate_bottleneck_animation_tests as bn    # Option 5 hybrid content

import os
# Per-recording tracking data lives outside Git under MP_DATA_ROOT (docs/DATA_AVAILABILITY.md).
DATA_ROOT = Path(os.environ.get("MP_DATA_ROOT", Path(__file__).resolve().parents[3] / "mp-data" / "external"))
OUT_ROOT  = Path(__file__).resolve().parent
FAM_DIRS = {
    "flow_fields":       OUT_ROOT / "flow_fields",
    "speed_population":  OUT_ROOT / "speed_population",
    "bottleneck_density":OUT_ROOT / "bottleneck_density",
}
MANIFEST_DIR = OUT_ROOT / "manifest"
REPORTS_DIR  = OUT_ROOT / "reports"

# ---------------------------------------------------------------------------
# Shared standard (unified canvas + HUD)
# ---------------------------------------------------------------------------
W, H_MAP, BAR_H = 1920, 1080, 70
H_TOTAL = H_MAP + BAR_H
FPS, GIF_WIDTH, GIF_STRIDE, SRC_FPS = 20, 760, 2, 30.0

FF_SECONDS, SP_SECONDS, BN_SECONDS = 17.0, 17.0, 10.0

# particle scaling reference (Plaça Catalunya): 465 flow paths -> N_A 4200 / N_B 6500
REF_K_A, REF_K_B = 4200 / 465.0, 6500 / 465.0

FONT = "DejaVu Sans"
for _c in ["Roboto", "Inter", "IBM Plex Sans", "Helvetica Neue", "Segoe UI", "DejaVu Sans"]:
    try:
        fm.findfont(_c, fallback_to_default=False); FONT = _c; break
    except Exception:
        continue

# HUD themes — identical geometry, inverted colours
DARK  = dict(bg="black",   sep="white",   sep_lw=0.6, sep_a=0.26,
             lbl="#7e7e88", val="#ededf2", end="#9a9aa2",
             grad_a=0.55, box_lw=0.4, box_a=0.20, dot_edge="white")
LIGHT = dict(bg="#F7F7F7", sep="#D9D9D9", sep_lw=0.8, sep_a=0.9,
             lbl="#333333", val="#111111", end="#111111",
             grad_a=1.0,  box_lw=0.5, box_a=0.9,  dot_edge="#D9D9D9")

PLACE_NAMES = {
    "placa_catalunya_01":     "Plaça Catalunya",
    "placa_espanya_01":       "Plaça Espanya",
    "placa_montjuic_01":      "Plaça Montjuïc",
    "esplanade_espanya_01":   "Esplanade Espanya",
    "red_bridge_combined_01": "Red Bridge",
    "stairs_montjuic_01":     "Stairs Montjuïc I",
    "stairs_montjuic_02":     "Stairs Montjuïc II",
}


def place_label(rec_id):
    return PLACE_NAMES.get(rec_id, rec_id.replace("_", " ").title())


def mmss(s):
    s = max(0, int(round(s)))
    return f"{s // 60:02d}:{s % 60:02d}"


class SkipMap(Exception):
    pass


# ---------------------------------------------------------------------------
# Discovery / manifest
# ---------------------------------------------------------------------------
def discover_recordings():
    recs = []
    for d in sorted(DATA_ROOT.iterdir()):
        if not d.is_dir():
            continue
        calib = d / "calibration" / "calib.json"
        traj  = d / "filtered_250m" / "trajectories_world_filtered_250m.csv"
        plans = list((d / "plan").glob("*.png")) if (d / "plan").is_dir() else []
        if calib.exists() and traj.exists() and plans:
            recs.append(d.name)
    return recs


def build_manifest(rec_ids):
    rows = []
    for rid in rec_ids:
        d = DATA_ROOT / rid
        plan = next(iter((d / "plan").glob("*.png")), None)
        calib = d / "calibration" / "calib.json"
        traj  = d / "filtered_250m" / "trajectories_world_filtered_250m.csv"
        bott  = d / "filtered_250m" / "bottlenecks" / "bottleneck_cells.csv"
        flowc = d / "filtered_250m" / "flow_fields" / "flow_field_cells.csv"
        speed = d / "filtered_250m" / "metrics" / "speed_per_observation.csv"

        missing = []
        for label, pth in [("plan", plan), ("calib", calib), ("trajectories", traj)]:
            if pth is None or not Path(pth).exists():
                missing.append(label)
        n_rows = n_tracks = 0
        t_total = 0.0
        if traj.exists():
            df = pd.read_csv(traj, usecols=["frame", "time_s", "track_id"])
            n_rows = int(len(df)); n_tracks = int(df.track_id.nunique())
            t_total = (int(df.frame.max()) - int(df.frame.min())) / SRC_FPS

        can_flow = traj.exists() and plan is not None and calib.exists()
        can_speed = can_flow
        can_bott = bott.exists() and plan is not None and calib.exists()

        rows.append(dict(
            recording_id=rid, place=place_label(rid),
            plan=str(plan) if plan else "", calib=str(calib),
            trajectories=str(traj),
            bottleneck=str(bott) if bott.exists() else "",
            flow_field=str(flowc) if flowc.exists() else "",
            metrics_speed=str(speed) if speed.exists() else "",
            rows=n_rows, tracks=n_tracks,
            time_total_s=round(t_total, 1), time_total_mmss=mmss(t_total),
            can_flow_fields=can_flow, can_speed_population=can_speed,
            can_bottleneck_density=can_bott,
            missing=";".join(missing) if missing else "",
        ))
    return rows


def write_manifest(rows):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(MANIFEST_DIR / "recording_manifest.csv", index=False)
    (MANIFEST_DIR / "recording_manifest.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Manifest: {MANIFEST_DIR / 'recording_manifest.csv'}")


# ---------------------------------------------------------------------------
# Per-recording geometry context
# ---------------------------------------------------------------------------
def window_extent_fit(plan_extent, W, H):
    """16:9 window that CONTAINS the whole plan (pad the smaller dim; no stretch,
    no crop). Reduces to Flow Fields' window_extent for wider-than-16:9 plans."""
    xmin, xmax, ymin, ymax = plan_extent
    pw, ph = xmax - xmin, ymax - ymin
    target = W / H
    if pw / ph >= target:
        ww = pw; hh = ww / target          # wider: pad height
    else:
        hh = ph; ww = hh * target           # narrower: pad width
    xc, yc = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    return (xc - ww / 2, xc + ww / 2, yc - hh / 2, yc + hh / 2)


def rec_context(rid):
    d = DATA_ROOT / rid
    plan = next(iter((d / "plan").glob("*.png")))
    calib = d / "calibration" / "calib.json"
    traj = d / "filtered_250m" / "trajectories_world_filtered_250m.csv"
    bott = d / "filtered_250m" / "bottlenecks" / "bottleneck_cells.csv"
    warped, plan_extent = v2d.warp_plan_to_world(plan, calib)
    win = window_extent_fit(plan_extent, W, H_MAP)
    fr = pd.read_csv(traj, usecols=["frame"])
    real_total = (int(fr.frame.max()) - int(fr.frame.min())) / SRC_FPS
    return dict(rid=rid, place=place_label(rid), plan=str(plan), calib=str(calib),
                traj=str(traj), bottleneck=str(bott), warped=warped,
                plan_extent=plan_extent, win=win, real_total=real_total)


# ---------------------------------------------------------------------------
# Shared HUD footer (dark/light theme; identical geometry)
# ---------------------------------------------------------------------------
def render_footer(theme, place, map_text, rt, real_total, legend):
    th = DARK if theme == "dark" else LIGHT
    fig = plt.figure(figsize=(W / 100, BAR_H / 100), dpi=100)
    fig.patch.set_facecolor(th["bg"])
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(th["bg"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.add_patch(Rectangle((0.004, 0.14), 0.992, 0.72, fill=False,
                 edgecolor=th["sep"], linewidth=th["sep_lw"], alpha=th["sep_a"]))

    def field(x, label, value, ha="left"):
        ax.text(x, 0.71, label, fontsize=6.5, color=th["lbl"], ha=ha, va="center", family=FONT)
        ax.text(x, 0.33, value, fontsize=10.5, color=th["val"], ha=ha, va="center", family=FONT)

    field(0.020, "PLACE", place)
    field(0.250, "MAP", map_text)

    gx0, gx1, gy0, gy1 = 0.520, 0.700, 0.40, 0.50
    if legend["type"] == "gradient":
        grad = legend["cmap"](np.linspace(0, 1, 256)[None, :]).copy()
        grad[..., 3] = th["grad_a"]
        ax.imshow(grad, extent=[gx0, gx1, gy0, gy1], aspect="auto", zorder=2,
                  interpolation="bilinear")
        ax.add_patch(Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, fill=False,
                     edgecolor=th["sep"], linewidth=th["box_lw"], alpha=th["box_a"], zorder=3))
        ax.text(0.520, 0.71, "INTENSITY", fontsize=6.5, color=th["lbl"], ha="left",
                va="center", family=FONT)
        ax.text(gx0 - 0.007, 0.45, legend["low"], fontsize=7.0, color=th["end"],
                ha="right", va="center", family=FONT)
        ax.text(gx1 + 0.007, 0.45, legend["high"], fontsize=7.0, color=th["end"],
                ha="left", va="center", family=FONT)
    else:  # dots
        ax.text(0.520, 0.71, legend["label"], fontsize=6.5, color=th["lbl"],
                ha="left", va="center", family=FONT)
        lx = 0.523
        for col, name in legend["items"]:
            ax.scatter([lx], [0.45], s=22, c=[tuple(col)], edgecolors=th["dot_edge"],
                       linewidths=0.4, zorder=4)
            ax.text(lx + 0.010, 0.45, name, fontsize=7.0, color=th["end"],
                    ha="left", va="center", family=FONT)
            lx += 0.063

    field(0.980, "TIME ELAPSED", f"{mmss(rt)} / {mmss(real_total)}", ha="right")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


# ---------------------------------------------------------------------------
# Shared export driver (streams; GIF 760, MP4 1920x1150, still)
# ---------------------------------------------------------------------------
def drive_export(stem, family_dir, F, render_map, footer_for, real_total,
                 gif_colors, still_at=0.97):
    family_dir.mkdir(parents=True, exist_ok=True)
    gif_w = GIF_WIDTH - (GIF_WIDTH % 2)
    gif_h = int(round(gif_w * H_TOTAL / W)); gif_h -= gif_h % 2
    gif_path = family_dir / f"{stem}.gif"
    mp4_path = family_dir / f"{stem}.mp4"
    still_path = family_dir / f"{stem}_still.png"

    mp4 = imageio.get_writer(mp4_path, fps=FPS, codec="libx264", quality=9,
                             macro_block_size=None, output_params=["-pix_fmt", "yuv420p"])
    footer_cache = {}
    gif_frames = []
    still_saved = False
    last_full = None
    for fi in range(F):
        t = fi / max(F - 1, 1)
        rt = t * real_total
        map_u8 = render_map(fi)
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
    if not still_saved and last_full is not None:
        Image.fromarray(last_full).save(still_path)
    mp4.close()
    gif_fps = FPS / GIF_STRIDE
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=int(round(1000 / gif_fps)), loop=0, optimize=True, disposal=2)
    return dict(gif=gif_path, mp4=mp4_path, still=still_path,
                gif_mb=gif_path.stat().st_size / 1e6,
                mp4_mb=mp4_path.stat().st_size / 1e6,
                dims=(gif_w, gif_h), n=F)


# ---------------------------------------------------------------------------
# Generalised plan placement (handles letterbox AND pillarbox, no stretch)
# ---------------------------------------------------------------------------
def _place_rows_cols(plan_extent, win):
    xmin, xmax, ymin, ymax = plan_extent
    wxmin, wxmax, wymin, wymax = win
    c0 = int(round((xmin - wxmin) / (wxmax - wxmin) * W))
    c1 = int(round((xmax - wxmin) / (wxmax - wxmin) * W))
    r0 = int(round((ymin - wymin) / (wymax - wymin) * H_MAP))
    r1 = int(round((ymax - wymin) / (wymax - wymin) * H_MAP))
    return r0, r1, c0, c1


def place_into_window(img_small, plan_extent, win, fill):
    """Place a plan-sized image into the W×H_MAP window at true world rows/cols,
    padding the rest with `fill` (scalar or 3-vector)."""
    r0, r1, c0, c1 = _place_rows_cols(plan_extent, win)
    ph, pw = max(r1 - r0, 2), max(c1 - c0, 2)
    resized = cv2.resize(img_small, (pw, ph), interpolation=cv2.INTER_AREA)
    canvas = np.empty((H_MAP, W, 3), np.float32)
    canvas[:] = np.asarray(fill, np.float32)
    a, b = max(r0, 0), min(r1, H_MAP)
    c, d = max(c0, 0), min(c1, W)
    canvas[a:b, c:d] = resized[(a - r0):(b - r0), (c - c0):(d - c0)]
    return canvas


# ---------------------------------------------------------------------------
# 1) FLOW FIELDS
# ---------------------------------------------------------------------------
def build_flow_fields(ctx):
    ffv3.DEF_TRAJ, ffv3.DEF_PLAN, ffv3.DEF_CALIB = ctx["traj"], ctx["plan"], ctx["calib"]
    warped, plan_extent, win = ctx["warped"], ctx["plan_extent"], ctx["win"]
    wxmin, wxmax, wymin, wymax = win

    polylines, dens, n_total, n_tracks = ffv3.load_geometry(plan_extent)
    rng = np.random.default_rng(7)
    seg_xy, seg_rgb, seg_appear = ffv3.build_segments(polylines, dens, rng)
    samples, lengths, colors = ffv3.build_flow(polylines, dens)
    n_flow = len(samples)
    if n_flow == 0:
        raise SkipMap("no flow paths (no trajectories >= MIN_LEN_M inside plan)")

    # architectural plan placed into the window (letterbox/pillarbox, black pad)
    major, secondary, n_lines = h3.architectural_layers(warped)
    plan_small = (ffv3.PLAN_MAJOR_OP * np.dstack([major] * 3) * h3.MAJOR_TINT +
                  ffv3.PLAN_SECONDARY_OP * np.dstack([secondary] * 3) * h3.SECONDARY_TINT)
    plan_small = np.clip(plan_small, 0, 1).astype(np.float32)
    plan_img = place_into_window(plan_small, plan_extent, win, fill=0.0)

    na = int(np.clip(round(REF_K_A * n_flow), 800, ffv3.N_A))
    nb = int(np.clip(round(REF_K_B * n_flow), 1200, ffv3.N_B))
    pprob = lengths / lengths.sum()
    tidA = rng.choice(n_flow, size=na, p=pprob); ph0A = rng.random(na)
    tidB = rng.choice(n_flow, size=nb, p=pprob); ph0B = rng.random(nb)

    F = int(round(FF_SECONDS * FPS))
    PLAN_FADE = DRAW_START = 1.0 / FF_SECONDS
    DRAW_END = DUST_START = 9.0 / FF_SECONDS
    GLOW_FULL_T = 15.5 / FF_SECONDS
    state = {"trail": np.zeros((H_MAP, W, 3), np.float32), "trace_full": None}

    def render_map(fi):
        t = fi / (F - 1)
        frame = plan_img * np.clip(t / PLAN_FADE, 0, 1)
        movement = np.zeros((H_MAP, W, 3), np.float32)
        if t >= DRAW_START:
            if t >= DRAW_END:
                if state["trace_full"] is None:
                    state["trace_full"] = ffv3.render_segment_buffer(
                        seg_xy, seg_rgb, np.ones(len(seg_appear), bool), win, W, H_MAP)
                traces = state["trace_full"]
            else:
                local = (t - DRAW_START) / (DRAW_END - DRAW_START)
                appear_local = (seg_appear - ffv3.DRAW_START) / (ffv3.DRAW_END - ffv3.DRAW_START)
                traces = ffv3.render_segment_buffer(seg_xy, seg_rgb, appear_local <= local, win, W, H_MAP)
            movement += ffv3.TRACE_CORE_GAIN * traces
        if t >= DUST_START:
            pg = np.clip((t - DUST_START) / (0.85 - DUST_START), 0, 1)
            inst = np.zeros((H_MAP, W, 3), np.float32)
            for tid, ph0, blur, gain in ((tidA, ph0A, ffv3.A_BLUR, ffv3.A_GAIN),
                                         (tidB, ph0B, ffv3.B_BLUR, ffv3.B_GAIN)):
                phase = (ph0 + ffv3.TARGET_SPEED_MS * (fi / FPS) / lengths[tid]) % 1.0
                env = np.clip(phase / ffv3.FADE_IN, 0, 1) * np.clip((1 - phase) / ffv3.FADE_OUT, 0, 1)
                idx = np.clip((phase * ffv3.K_SAMPLES).astype(np.int32), 0, ffv3.K_SAMPLES - 1)
                pos = samples[tid, idx]; col = colors[tid, idx] * env[:, None]
                px = np.clip(((pos[:, 0] - wxmin) / (wxmax - wxmin) * (W - 1)).astype(np.int32), 0, W - 1)
                py = np.clip(((pos[:, 1] - wymin) / (wymax - wymin) * (H_MAP - 1)).astype(np.int32), 0, H_MAP - 1)
                acc = ffv3.splat(W, H_MAP, px, py, col, gain * pg)
                for c in range(3):
                    acc[..., c] = cv2.GaussianBlur(acc[..., c], (0, 0), blur)
                inst += acc
            state["trail"] = state["trail"] * ffv3.TRAIL_DECAY + inst
            movement += ffv3.PARTICLE_CORE_GAIN * state["trail"]
        if t < DRAW_START:
            gs = 0.0
        elif t < DUST_START:
            gs = ffv3.lerp(*ffv3.GLOW_DRAW, (t - DRAW_START) / (DUST_START - DRAW_START))
        else:
            gs = ffv3.lerp(*ffv3.GLOW_DUST, (t - DUST_START) / (GLOW_FULL_T - DUST_START))
        out = frame + movement + ffv3.bloom(movement, gs)
        out = out / (1.0 + ffv3.ROLLOFF * out)
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)

    legend = dict(type="gradient", cmap=ffv3.PAL_CMAP, low="Low flow", high="High flow")

    def footer_for(rt):
        return render_footer("dark", ctx["place"], "Flow Fields", rt, ctx["real_total"], legend)

    meta = dict(n_tracks=n_tracks, n_rows=n_total, n_flow=n_flow, n_lines=n_lines,
                N_A=na, N_B=nb, seconds=FF_SECONDS, gif_colors=128,
                note=(f"particles scaled to {na}/{nb} for {n_flow} flow paths"
                      if (na < ffv3.N_A or nb < ffv3.N_B) else "full particle counts"))
    return F, render_map, footer_for, meta, 128


# ---------------------------------------------------------------------------
# 2) SPEED POPULATION
# ---------------------------------------------------------------------------
def build_speed_population(ctx):
    spv1.DEF_TRAJ, spv1.DEF_PLAN, spv1.DEF_CALIB = ctx["traj"], ctx["plan"], ctx["calib"]
    warped, plan_extent, win = ctx["warped"], ctx["plan_extent"], ctx["win"]
    wxmin, wxmax, wymin, wymax = win

    r0, r1, c0, c1 = _place_rows_cols(plan_extent, win)
    arctic = sp2.arctic_background(warped, max(c1 - c0, 2), max(r1 - r0, 2))
    canvas = np.ones((H_MAP, W, 3), np.float32)            # white padding
    a, b = max(r0, 0), min(r1, H_MAP); c, d = max(c0, 0), min(c1, W)
    canvas[a:b, c:d] = arctic[(a - r0):(b - r0), (c - c0):(d - c0)]
    base = sp2.add_grid(canvas, win, W, H_MAP)

    tracks, n_total, n_tracks_all, n_used, t_total = spv1.load_tracks(win)
    if n_used == 0:
        raise SkipMap("no tracks inside window")
    palette = sp3.PALETTES["warm"]
    trace_s = sp3.TRACE_SECONDS
    T0 = np.array([t[0][0] for t in tracks]); T1 = np.array([t[0][-1] for t in tracks])
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
                        sp2.draw_dot(frame, px, py, col, sp3.TRAIL_FILL_A * aage[k] * linger, head=False)
            xh = float(np.interp(head_t, t, x)); yh = float(np.interp(head_t, t, y))
            if wxmin <= xh <= wxmax and wymin <= yh <= wymax:
                px = int(round((xh - wxmin) / (wxmax - wxmin) * (W - 1)))
                py = int(round((yh - wymin) / (wymax - wymin) * (H_MAP - 1)))
                sp2.draw_dot(frame, px, py, col, sp3.HEAD_FILL_A * linger, head=True)
        return (np.clip(frame, 0, 1) * 255).astype(np.uint8)

    legend = dict(type="dots", label="SPEED",
                  items=[(palette[0], "Slow"), (palette[1], "Medium"), (palette[2], "Fast")])

    def footer_for(rt):
        return render_footer("light", ctx["place"], "Speed Population", rt, ctx["real_total"], legend)

    meta = dict(n_tracks=n_tracks_all, n_used=n_used, n_rows=n_total,
                seconds=SP_SECONDS, trace_s=trace_s, gif_colors=256,
                note=f"persistence trace {trace_s:.0f} s; {n_used} tracks shown")
    return F, render_map, footer_for, meta, 256


# ---------------------------------------------------------------------------
# 3) BOTTLENECK DENSITY
# ---------------------------------------------------------------------------
def make_bottleneck_fig(ctxb, win):
    xmin, xmax, ymin, ymax = ctxb["extent"]
    wxmin, wxmax, wymin, wymax = win
    fig = plt.figure(figsize=(W / 100, H_MAP / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(wxmin, wxmax); ax.set_ylim(wymax, wymin)
    ax.set_aspect("equal"); ax.axis("off")
    ax.imshow(ctxb["underlay"], extent=[xmin, xmax, ymax, ymin], origin="upper",
              alpha=bn.BG_OPACITY, zorder=0, interpolation="bilinear")
    return fig, ax


def bottleneck_timeline(ctxb):
    n = ctxb["n_inside"]
    t_full = ctxb["t_full"]; fa = ctxb["final_alpha"]; fs = ctxb["full_scale"]
    LOW_COLORT, LOW_ALPHA = 0.12, 0.50
    sf = lambda s: max(int(round(s * FPS)), 1)
    f_in, f_faint, f_grow, f_mark, f_hold = sf(0.5), sf(2.5), sf(5.0), sf(0.8), sf(1.2)
    base_scale = np.full(n, bn.BASE_SCALE); low_colorT = np.full(n, LOW_COLORT)
    ss = bn.smoothstep
    tl = []
    for k in range(f_in):
        tl.append((np.full(n, ss((k + 1) / f_in) * LOW_ALPHA), low_colorT, base_scale, 0.0))
    for _ in range(f_faint):
        tl.append((np.full(n, LOW_ALPHA), low_colorT, base_scale, 0.0))
    for k in range(f_grow):
        q = ss((k + 1) / f_grow)
        tl.append((LOW_ALPHA + (fa - LOW_ALPHA) * q, LOW_COLORT + (t_full - LOW_COLORT) * q,
                   bn.BASE_SCALE + (fs - bn.BASE_SCALE) * q, 0.0))
    for k in range(f_mark):
        tl.append((fa, t_full, fs, ss((k + 1) / f_mark)))
    for _ in range(f_hold):
        tl.append((fa, t_full, fs, 1.0))
    return tl


def build_bottleneck(ctx):
    if not ctx["bottleneck"] or not Path(ctx["bottleneck"]).exists():
        raise SkipMap("bottleneck_cells.csv missing")
    bn.DEF_CSV, bn.DEF_PLAN, bn.DEF_CALIB = ctx["bottleneck"], ctx["plan"], ctx["calib"]
    bn.TRAJ_CSV = Path(ctx["traj"])
    ctxb = bn.build_context()
    if ctxb["n_inside"] == 0:
        raise SkipMap("no bottleneck cells inside plan window")
    timeline = bottleneck_timeline(ctxb)
    F = len(timeline)
    fig, ax = make_bottleneck_fig(ctxb, ctx["win"])
    dynamic = []

    def render_map(fi):
        a, c, s, m = timeline[fi]
        return bn.draw_frame(fig, ax, ctxb, a, c, s, m, dynamic)

    legend = dict(type="gradient", cmap=ctxb["cmap"], low="Low density", high="High density")

    def footer_for(rt):
        return render_footer("light", ctx["place"], "Bottleneck Density", rt, ctx["real_total"], legend)

    meta = dict(n_inside=ctxb["n_inside"], n_dropped=ctxb["n_dropped"],
                seconds=BN_SECONDS, gif_colors=256,
                note=f"{ctxb['n_inside']} cells in-frame (dropped {ctxb['n_dropped']})")
    return F, render_map, footer_for, meta, 256, fig


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
BUILDERS = {
    "flow_fields": build_flow_fields,
    "speed_population": build_speed_population,
    "bottleneck_density": build_bottleneck,
}


def generate_one(rid, family, ctx):
    fig = None
    builder = BUILDERS[family]
    out = builder(ctx)
    if family == "bottleneck_density":
        F, render_map, footer_for, meta, gif_colors, fig = out
    else:
        F, render_map, footer_for, meta, gif_colors = out
    stem = f"{rid}_{family}"
    res = drive_export(stem, FAM_DIRS[family], F, render_map, footer_for,
                       ctx["real_total"], gif_colors=gif_colors)
    if fig is not None:
        plt.close(fig)
    res.update(meta)
    return res


def run_batch(rec_ids, families):
    results = {}
    for rid in rec_ids:
        print(f"\n[REC] {rid}")
        try:
            ctx = rec_context(rid)
        except Exception as e:
            print(f"  [FAIL] context: {e}")
            results[rid] = {f: dict(status="fail", error=f"context: {e}") for f in families}
            continue
        results[rid] = {}
        for fam in families:
            t0 = time.time()
            try:
                res = generate_one(rid, fam, ctx)
                res["status"] = "ok"; res["secs"] = round(time.time() - t0, 1)
                results[rid][fam] = res
                print(f"  [ok] {fam}: GIF {res['gif_mb']:.1f}MB MP4 {res['mp4_mb']:.1f}MB "
                      f"({res['secs']:.0f}s) — {res.get('note','')}")
            except SkipMap as e:
                results[rid][fam] = dict(status="skip", error=str(e))
                print(f"  [skip] {fam}: {e}")
            except Exception as e:
                results[rid][fam] = dict(status="fail", error=str(e))
                print(f"  [FAIL] {fam}: {e}")
                traceback.print_exc()
    return results


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def _fmt(res):
    if res.get("status") != "ok":
        return f"{res.get('status','-')}" + (f" ({res['error']})" if res.get("error") else "")
    return f"ok · GIF {res['gif_mb']:.1f} MB · MP4 {res['mp4_mb']:.1f} MB · {res.get('note','')}"


def write_reports(manifest_rows, results, families):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mrec = {r["recording_id"]: r for r in manifest_rows}
    fam_titles = {"flow_fields": "Flow Fields", "speed_population": "Speed Population",
                  "bottleneck_density": "Bottleneck Density"}

    # per-family reports
    for fam in families:
        lines = [f"# {fam_titles[fam]} — Batch Report", "",
                 "Visualization-only. No retracking, recalibration, score recomputation, "
                 "or source-data modification.", "",
                 f"Shared canvas {W}×{H_TOTAL} (map {H_MAP} + HUD {BAR_H}); GIF "
                 f"{GIF_WIDTH}×{int(round(GIF_WIDTH*H_TOTAL/W))} @ {FPS//GIF_STRIDE} fps; "
                 f"MP4 {W}×{H_TOTAL} @ {FPS} fps; font {FONT}.", "",
                 "| Recording | Status | Tracks | Time | Output / notes |",
                 "|---|---|---|---|---|"]
        for rid in results:
            r = results[rid].get(fam, {})
            m = mrec.get(rid, {})
            lines.append(f"| {rid} | {r.get('status','-')} | {m.get('tracks','-')} | "
                         f"{m.get('time_total_mmss','-')} | {_fmt(r)} |")
        (REPORTS_DIR / f"{fam}_batch_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # global report
    g = [f"# Motion Pixels — Final Behavior Maps Batch Report", "",
         "**Visualization-only.** No retracking, recalibration, bottleneck-score "
         "recomputation, or source-data modification. Bottleneck scores read from "
         "`bottleneck_cells.csv`. See `FINAL_BEHAVIOR_MAP_RECIPES.md` for the exact "
         "recipe extracted for each family.", "",
         "## Shared dimensions & HUD",
         f"- Map **{W}×{H_MAP}**, HUD footer **{BAR_H}** → canvas **{W}×{H_TOTAL}**.",
         f"- GIF **{GIF_WIDTH}×{int(round(GIF_WIDTH*H_TOTAL/W))}** @ **{FPS//GIF_STRIDE} fps** "
         f"(stride {GIF_STRIDE}); MP4 **{W}×{H_TOTAL}** H.264 @ {FPS} fps; still PNG {W}×{H_TOTAL}.",
         f"- Font **{FONT}**; HUD geometry identical across families. Flow Fields = DARK "
         "footer; Speed Population & Bottleneck Density = LIGHT footer. Timer = real "
         "source-video time. World window = 16:9 fit of each plan (full plan visible; "
         "neutral padding; no stretch).",
         f"- Durations: Flow Fields {FF_SECONDS:.0f} s · Speed Population {SP_SECONDS:.0f} s "
         f"· Bottleneck Density {BN_SECONDS:.0f} s.", "",
         "## Recordings discovered", ""]
    g.append("| Recording | Place | Tracks | Rows | Time | Flow | Speed | Bottleneck |")
    g.append("|---|---|---|---|---|---|---|---|")
    for r in manifest_rows:
        g.append(f"| {r['recording_id']} | {r['place']} | {r['tracks']} | {r['rows']} | "
                 f"{r['time_total_mmss']} | {r['can_flow_fields']} | {r['can_speed_population']} | "
                 f"{r['can_bottleneck_density']} |")
    g += ["", "## Results per recording", ""]
    for rid in results:
        g.append(f"### {rid} — {place_label(rid)}")
        for fam in families:
            g.append(f"- **{fam_titles[fam]}**: {_fmt(results[rid].get(fam, {}))}")
        g.append("")
    # adjustments + confirmation
    g += ["## Adjustments for sparse recordings",
          "- **Flow Fields:** particle counts scale with available flow paths "
          "(Plaça Catalunya reference = 465 paths → N_A 4200 / N_B 6500; ≈9/14 per "
          "path). Sparser recordings get proportionally fewer particles (clamped, "
          "never faked); per-recording counts in the table notes.",
          "- **Speed Population:** same 5 s fading-trace persistence window for all; "
          "no duplicated pedestrians.",
          "- **Pillarbox/letterbox:** plans narrower than 16:9 (placa_montjuic, "
          "red_bridge, stairs_montjuic_01/02) are padded on the sides; wider plans "
          "padded top/bottom. The full plan is always shown; nothing is stretched or "
          "cropped.", "",
          "## Confirmation",
          "- No source data modified; all CSVs/plans/calibrations read-only.",
          "- Bottleneck scores not recomputed.",
          "- All families share the same canvas size and HUD geometry.", "",
          "## Output locations",
          f"- `behaviormaps_final/flow_fields/` · `…/speed_population/` · `…/bottleneck_density/`",
          f"- Manifest: `behaviormaps_final/manifest/` · Reports: `behaviormaps_final/reports/`"]
    (REPORTS_DIR / "FINAL_BEHAVIOR_MAPS_BATCH_REPORT.md").write_text("\n".join(g) + "\n", encoding="utf-8")
    print(f"[INFO] Reports written to {REPORTS_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scout", action="store_true", help="discover + manifest only (default)")
    ap.add_argument("--smoke", action="store_true", help="placa_catalunya_01 only")
    ap.add_argument("--all", action="store_true", help="all recordings")
    ap.add_argument("--families", default="flow_fields,speed_population,bottleneck_density")
    ap.add_argument("--recordings", default="")
    args = ap.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    rec_ids = discover_recordings()
    print(f"[INFO] discovered {len(rec_ids)} recordings: {rec_ids}")
    manifest_rows = build_manifest(rec_ids)
    write_manifest(manifest_rows)

    if args.recordings:
        targets = [r.strip() for r in args.recordings.split(",") if r.strip()]
    elif args.smoke:
        targets = ["placa_catalunya_01"]
    elif args.all:
        targets = rec_ids
    else:  # scout (default)
        print("\n[SCOUT] manifest written; planned generation:")
        for r in manifest_rows:
            ok = [f for f in families if r.get(f"can_{f}")]
            print(f"  {r['recording_id']:26s} tracks={r['tracks']:5d} time={r['time_total_mmss']} -> {ok}")
        print("\n[SCOUT] dry-run only. Re-run with --smoke (placa_catalunya) then --all.")
        return

    print(f"[INFO] generating {families} for {targets}")
    results = run_batch(targets, families)
    write_reports(manifest_rows, results, families)
    print("\n[DONE]")


if __name__ == "__main__":
    main()
