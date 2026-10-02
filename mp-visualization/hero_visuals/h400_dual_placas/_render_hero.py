"""Render the H400 dual-plaza hero visual (background + transparent).
Uses cached frozen-model predictions in placa_H400_best_per_track.pkl (no recompute).
"""
from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(r"C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-visualization\hero_visuals\h400_dual_placas")
SITES = {
    "placa_catalunya_01": {
        "title": "PLAÇA  CATALUNYA",
        "plan": r"C:\Users\OWNER\Desktop\new_datasets\placa_catalunya_01\plan\placa-catalunya.png",
        "calib": r"C:\Users\OWNER\Desktop\new_datasets\placa_catalunya_01\calibration\calib.json",
        "min_sep": 5.0, "predlen_min": 5.0, "gtlen_min": 10.0,
        "overlay_max": 0.7, "cap": 3,
        "pick": ["3941", "264", "5477"],
    },
    "placa_espanya_01": {
        "title": "PLAÇA  ESPANYA",
        "plan": r"C:\Users\OWNER\Desktop\new_datasets\placa_espanya_01\plan\placa-espanya.png",
        "calib": r"C:\Users\OWNER\Desktop\new_datasets\placa_espanya_01\calibration\calib.json",
        "min_sep": 15.0, "predlen_min": 6.0, "gtlen_min": 20.0,
        "overlay_max": 0.9, "cap": 3,
        "pick": ["6268", "4394", "122"],
    },
}
MAGENTA = "#ff1ad9"
WHITE = "#ffffff"

def world_to_plan_affine(calib):
    c = json.load(open(calib))
    W = np.array(c["world_points"]); P = np.array(c["plan_points_px"])
    A = np.column_stack([W, np.ones(len(W))])
    M, *_ = np.linalg.lstsq(A, P, rcond=None)
    return M  # (3,2)

def w2p(pts, M):
    pts = np.asarray(pts, float)
    return np.column_stack([pts, np.ones(len(pts))]) @ M

def traj_quality(p):
    """net-direction error (deg), GT straightness, early pred-vs-GT overlay (m)."""
    gt = np.asarray(p["gt"]); pr = np.asarray(p["pred"])
    v = pr[-1] - pr[0]; g = gt[-1] - gt[0]
    if np.linalg.norm(v) < 1e-6 or np.linalg.norm(g) < 1e-6:
        netdir = 999.0
    else:
        netdir = abs(np.degrees(np.arctan2(v[1], v[0]) - np.arctan2(g[1], g[0])
                                + np.pi) % (2 * np.pi) - np.pi)
    pl = np.linalg.norm(np.diff(gt, axis=0), axis=1).sum()
    straight = float(np.linalg.norm(gt[-1] - gt[0]) / pl) if pl > 1e-6 else 0.0
    n = min(len(pr), len(gt)); q = max(2, n // 8)
    overlay = float(np.linalg.norm(pr[1:q] - gt[1:q], axis=1).mean())
    return netdir, straight, overlay

def select(packs, cfg):
    if cfg.get("pick"):
        by_id = {p["trajectory_id"].split("__")[1]: p for p in packs}
        return [by_id[i] for i in cfg["pick"] if i in by_id]
    cand = []
    for p in packs:
        netdir, straight, overlay = traj_quality(p)
        if (netdir <= 20 and p["gt_len"] >= cfg["gtlen_min"]
                and p["pred_len"] >= cfg["predlen_min"] and overlay <= cfg["overlay_max"]):
            score = p["pred_len"] + 0.10 * p["gt_len"] + 8.0 * straight - 4.0 * overlay
            cand.append((score, p))
    cand.sort(key=lambda t: t[0], reverse=True)
    chosen = []
    for _, p in cand:
        s = np.array(p["obs"][-1])
        if all(np.linalg.norm(s - np.array(q["obs"][-1])) >= cfg["min_sep"] for q in chosen):
            chosen.append(p)
        if len(chosen) >= cfg["cap"]:
            break
    return chosen

def darken_plan(path, factor=0.30):
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).astype(np.float32) * factor
    return np.clip(arr, 0, 255).astype(np.uint8)

def panel_crop(site_key, cfg):
    M = world_to_plan_affine(cfg["calib"])
    chosen = select(SITE_PACKS[site_key], cfg)
    allpx = np.vstack([np.vstack([w2p(p["obs"], M), w2p(p["gt"], M), w2p(p["pred"], M)])
                       for p in chosen])
    x0, y0 = allpx.min(0); x1, y1 = allpx.max(0)
    spanx, spany = x1 - x0, y1 - y0
    mg = 0.14 * max(spanx, spany)
    bx0, bx1, by0, by1 = x0 - mg, x1 + mg, y0 - mg, y1 + mg
    return M, chosen, (bx0, bx1, by0, by1)

def draw_panel(ax, site_key, cfg, with_bg):
    M, chosen, (bx0, bx1, by0, by1) = panel_crop(site_key, cfg)
    seg = [(w2p(p["obs"], M), w2p(p["gt"], M), w2p(p["pred"], M)) for p in chosen]

    if with_bg:
        plan = darken_plan(cfg["plan"], 0.34)
        ax.imshow(plan, alpha=0.62, zorder=0, interpolation="bilinear")

    for obs, gt, pr in seg:
        # ground-truth future: white dotted
        ax.plot(gt[:, 0], gt[:, 1], color=WHITE, lw=2.4, ls=(0, (1, 2.6)),
                alpha=0.78, zorder=4, solid_capstyle="round")
        # history: bold white solid
        ax.plot(obs[:, 0], obs[:, 1], color=WHITE, lw=5.2, alpha=0.97,
                zorder=6, solid_capstyle="round")
        # prediction H400: magenta with glow
        ax.plot(pr[:, 0], pr[:, 1], color=MAGENTA, lw=13, alpha=0.18,
                zorder=7, solid_capstyle="round")
        ax.plot(pr[:, 0], pr[:, 1], color=MAGENTA, lw=5.6, alpha=0.98,
                zorder=8, solid_capstyle="round")
        ax.scatter([pr[-1, 0]], [pr[-1, 1]], s=70, color=MAGENTA, zorder=9,
                   edgecolors="none")
        # prediction start: white X
        ax.scatter([obs[-1, 0]], [obs[-1, 1]], s=190, marker="x", color=WHITE,
                   linewidths=3.2, zorder=10)

    ax.set_xlim(bx0, bx1); ax.set_ylim(by1, by0)  # invert y (image space)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    # minimal site label
    ax.text(0.5, 0.045, cfg["title"], transform=ax.transAxes, ha="center",
            va="center", color=WHITE, fontsize=21, fontweight="light",
            family="DejaVu Sans", alpha=0.82)
    return len(chosen), chosen

def build(with_bg, fname):
    # panel aspect ratios drive layout so both fill equally (no letterbox / dead gap)
    aspects, crops = {}, {}
    for k, cfg in SITES.items():
        _, _, (bx0, bx1, by0, by1) = panel_crop(k, cfg)
        aspects[k] = (bx1 - bx0) / (by1 - by0)
    PH = 8.0            # panel height (in)
    LM, RM, GAP = 0.35, 0.35, 0.9
    BM, TM = 1.15, 0.45  # bottom (legend) / top margins
    widths = {k: PH * a for k, a in aspects.items()}
    figw = LM + sum(widths.values()) + GAP + RM
    figh = BM + PH + TM
    fig = plt.figure(figsize=(figw, figh), dpi=200)
    if with_bg:
        fig.patch.set_facecolor("black")
    else:
        fig.patch.set_alpha(0.0)
    counts = {}
    xcur = LM
    for k, cfg in SITES.items():
        w = widths[k]
        ax = fig.add_axes([xcur / figw, BM / figh, w / figw, PH / figh])
        if not with_bg:
            ax.patch.set_alpha(0.0)
        n, _ = draw_panel(ax, k, cfg, with_bg)
        counts[k] = n
        xcur += w + GAP
    # minimal legend, lower-center
    handles = [
        Line2D([0], [0], color=WHITE, lw=4.5, label="history"),
        Line2D([0], [0], color=WHITE, lw=2.2, ls=(0, (1, 2.6)), label="actual future"),
        Line2D([0], [0], color=MAGENTA, lw=4.5, label="predicted (H400)"),
        Line2D([0], [0], color=WHITE, lw=0, marker="x", markersize=10,
               markeredgewidth=2.5, label="prediction start"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                     fontsize=14, labelcolor=WHITE, handlelength=2.6,
                     columnspacing=2.6, bbox_to_anchor=(0.5, 0.003))
    fig.savefig(OUT / fname, dpi=200, facecolor=(fig.get_facecolor() if with_bg else "none"),
                transparent=not with_bg)
    plt.close(fig)
    print("wrote", fname, "px=", int(figw * 200), counts)
    return counts

if __name__ == "__main__":
    bundle = pickle.load(open(OUT / "placa_H400_best_per_track.pkl", "rb"))
    SITE_PACKS = bundle["windows"]
    c1 = build(True, "hero_H400_placa_catalunya_placa_espanya_background.png")
    c2 = build(False, "hero_H400_placa_catalunya_placa_espanya_transparent.png")
    # save selection detail
    detail = {}
    for k, cfg in SITES.items():
        ch = select(SITE_PACKS[k], cfg)
        detail[k] = [{"trajectory_id": p["trajectory_id"], "ade": p["ade"],
                      "fde": p["fde"], "gt_len": p["gt_len"], "pred_len": p["pred_len"],
                      "len_ratio": p["len_ratio"], "angular_err_deg": p["angular_err_deg"],
                      "start_idx": p["start_idx"]} for p in ch]
    json.dump(detail, open(OUT / "selected_windows.json", "w"), indent=2)
    print("counts bg", c1, "transparent", c2)
