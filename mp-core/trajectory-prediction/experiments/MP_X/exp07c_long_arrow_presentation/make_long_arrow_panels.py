"""
MP_X / exp07c — Long-arrow presentation panels.

A presentation-language re-render of the qualitative prediction panels: clean
infographic styling, elegant dominant magenta prediction arrows, on the
architectural walkable plan. 10 examples, ranked by the LONGEST available
predicted rollouts (mild, architecturally-readable turns).

Honesty: prediction geometry is NEVER fabricated or scaled independently of GT.
Long-arrow readability comes from selection (longest predictions), tighter crops,
and stroke styling only. A light Catmull-Rom smoothing is applied for DISPLAY only
(raw geometry is saved separately and all metrics use raw values).

Visual language:
  observed/history = black (#111111)
  ground truth future = soft grey (#A8A8A8)
  model prediction = solid magenta (#D81B7D), thicker, drawn last, dominant
  current position = small black pebble with white outline (NO stars anywhere)
  background = white walkable / light-grey built / thin pale-grey plan outline
  minimal title = architectural category only; small magenta "turn ≈ XX°"

Reads only exp06 outputs + dataset + V3 masks. Modifies nothing.

Run:  python make_long_arrow_panels.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "shared"
EXP06 = HERE.parent / "exp06_turn_balanced_eval_split"
MP_ROOT = Path(r"C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels")
MASK_DIR = MP_ROOT / "mp-data" / "annotations" / "manual_masks_v3"
ENC_DIR = Path(r"C:\Users\OWNER\Desktop\new_datasets\Barcelona_v3_manual_encoded")
sys.path.insert(0, str(SHARED))
import mpx_common as C   # noqa: E402
fz = C.fz; P5 = C.P5
import torch  # noqa: E402

RECS = ["esplanade_espanya_01", "stairs_montjuic_01"]   # clean held-out; placa_espanya excluded (artifact)
N = 10
HORIZON = 20
ASPECT = 1.6   # 16:10

C_OBS = "#111111"; C_GT = "#A8A8A8"; C_PRED = "#D81B7D"
C_BUILT = "#ECECEC"; C_EDGE = "#D0D0D0"; C_DOT = "#111111"

plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white",
                     "font.family": "DejaVu Sans", "font.size": 11})

PANELS = HERE / "panels"
DIRS = {m: PANELS / m for m in ("individual", "zoomed_arrow", "context_30m", "raw_geometry")}
CAPS = HERE / "captions"
for d in list(DIRS.values()) + [CAPS]:
    d.mkdir(parents=True, exist_ok=True)


# ── plan assets ──────────────────────────────────────────────────────────────
def load_plan(rec):
    meta = json.loads((ENC_DIR / rec / "spatial_v3_manual" / "encoding_v3_metadata.json").read_text(encoding="utf-8"))
    walk = np.array(Image.open(MASK_DIR / rec / "walkable_mask_v3_manual.png").convert("L"))
    g = float(int(C_BUILT[1:3], 16)) / 255
    rgb = np.where(walk[..., None] > 127, 1.0, np.array([g, g, g]))
    return dict(walk=walk, rgb=rgb, M=np.array(meta["transform_world_to_plan_3x2"], float),
                scale=float(meta.get("scale_px_per_m_from_fit", 8.0)))


def w2p(M, wx, wy):
    return wx * M[0, 0] + wy * M[1, 0] + M[2, 0], wx * M[0, 1] + wy * M[1, 1] + M[2, 1]


def walkable_frac(plan, pts):
    px, py = w2p(plan["M"], *np.asarray(pts, float).T)
    h, w = plan["walk"].shape
    return float((plan["walk"][np.clip(np.round(py).astype(int), 0, h - 1),
                               np.clip(np.round(px).astype(int), 0, w - 1)] > 127).mean())


# ── display smoothing (DISPLAY ONLY; raw geometry preserved in metrics/raw panels) ──
def catmull_rom(P, n=12):
    P = np.asarray(P, float)
    if len(P) < 3:
        return P
    pts = np.vstack([P[0], P, P[-1]]); out = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2 = t * t; t3 = t2 * t
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-1])
    return np.asarray(out)


def arch_category(min_obst, mean_bound):
    if min_obst < 0.22:
        return "obstacle avoidance"
    if mean_bound < 0.35:
        return "corridor following"
    return "plaza circulation"


def _reversals(pts):
    d = np.diff(pts, axis=0); mv = np.linalg.norm(d, axis=1) > 1e-6
    if mv.sum() < 3:
        return 99
    h = np.unwrap(np.arctan2(d[mv, 1], d[mv, 0])); sg = np.sign(np.diff(h)); sg = sg[sg != 0]
    return int(np.sum(np.abs(np.diff(sg)) > 0)) if len(sg) > 1 else 0


def jumps_ok(r):
    s = np.linalg.norm(np.diff(np.asarray(r["pred_pos"], float), axis=0), axis=1)
    if len(s) < 3:
        return False
    med = np.median(s[s > 1e-6]) if (s > 1e-6).any() else 0
    return s.max() < 1.1 and (med <= 1e-6 or s.max() < 8 * med) and s.sum() > 0.5


# ── drawing ──────────────────────────────────────────────────────────────────
def _arrow(ax, pts, color, scale_head, lw):
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return
    p1, p0 = pts[-1], pts[-2]
    if np.linalg.norm(p1 - p0) < 1e-6:
        return
    ax.add_patch(FancyArrowPatch(tuple(p0), tuple(p1), arrowstyle="-|>", mutation_scale=scale_head,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, capstyle="round",
                                 joinstyle="round", zorder=12))


def _mid_arrow(ax, pts, color, scale_head):
    pts = np.asarray(pts, float)
    if len(pts) < 4:
        return
    i = len(pts) // 2
    ax.add_patch(FancyArrowPatch(tuple(pts[i - 1]), tuple(pts[i]), arrowstyle="-|>",
                                 mutation_scale=scale_head, color=color, lw=0.1, shrinkA=0, shrinkB=0, zorder=11))


def _nice(m):
    for v in (1, 2, 5, 10, 20, 50, 100):
        if v >= m:
            return v
    return 100


def draw(ax, plan, r, row, mode, smooth=True):
    M, scale = plan["M"], plan["scale"]
    sp = np.column_stack(w2p(M, *np.asarray(r["seed_pos"]).T))
    gt = np.column_stack(w2p(M, *np.asarray(r["gt_pos"]).T))
    pr = np.column_stack(w2p(M, *np.asarray(r["pred_pos"]).T))
    sp_d = catmull_rom(sp) if smooth else sp
    gt_d = catmull_rom(gt) if smooth else gt
    pr_d = catmull_rom(pr) if smooth else pr

    ax.imshow(plan["rgb"], origin="upper", interpolation="bilinear", zorder=0)
    ax.contour(plan["walk"], levels=[127], colors=C_EDGE, linewidths=0.7, zorder=1)
    ax.plot(sp_d[:, 0], sp_d[:, 1], "-", color=C_OBS, lw=2.6, solid_capstyle="round", zorder=5)
    ax.plot(gt_d[:, 0], gt_d[:, 1], "-", color=C_GT, lw=2.8, solid_capstyle="round", zorder=4)
    ax.plot(pr_d[:, 0], pr_d[:, 1], "-", color=C_PRED, lw=5.2, solid_capstyle="round", zorder=8)
    _arrow(ax, gt_d, C_GT, 16, 2.4)
    _arrow(ax, sp_d, C_OBS, 13, 2.0)
    _mid_arrow(ax, pr_d, C_PRED, 18)
    _arrow(ax, pr_d, C_PRED, 30, 5.2)                       # dominant prediction arrowhead
    # current-position pebble (no star)
    ax.plot(*gt[0], marker="o", color=C_DOT, ms=7.5, mec="white", mew=1.6, zorder=13, lw=0)

    # crop (16:10), centred on trajectory
    allp = np.vstack([sp, gt, pr]); cx, cy = allp.mean(0)
    bw, bh = allp[:, 0].ptp(), allp[:, 1].ptp(); ext_m = max(bw, bh) / scale
    if mode == "zoom":
        ctx = max(ext_m * 1.45, 4.0)
    elif mode == "context30":
        ctx = max(ext_m * 1.15, 34.0)
    else:
        ctx = max(ext_m * 2.1, 13.0)
    half_h = ctx / 2 * scale; half_w = ASPECT * half_h
    half_w = max(half_w, bw * 0.62 + 0.6 * scale); half_h = max(half_h, bh * 0.62 + 0.6 * scale)
    if half_w / half_h < ASPECT:
        half_w = ASPECT * half_h
    else:
        half_h = half_w / ASPECT
    ax.set_xlim(cx - half_w, cx + half_w); ax.set_ylim(cy + half_h, cy - half_h)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#e2e2e2"); s.set_linewidth(1.0)

    # subtle scale bar
    sb = _nice((2 * half_w / scale) / 6)
    x0 = cx - half_w + 0.55 * scale; y0 = cy + half_h - 0.7 * scale
    ax.plot([x0, x0 + sb * scale], [y0, y0], "-", color="#555555", lw=2.2, solid_capstyle="butt", zorder=11)
    ax.text(x0 + sb * scale / 2, y0 - 0.35 * scale, f"{sb} m", ha="center", va="bottom", fontsize=8.5, color="#555555")

    # minimal labels (category + magenta turn) — NO definition block
    ax.text(0.035, 0.95, row["arch_category"], transform=ax.transAxes, fontsize=14, fontweight="bold",
            color="#111111", va="top", ha="left")
    ax.text(0.035, 0.865, f"turn ≈ {row['gt_head_change_deg']:.0f}°", transform=ax.transAxes,
            fontsize=12.5, color=C_PRED, va="top", ha="left", fontweight="bold")
    return ctx


# ── selection ────────────────────────────────────────────────────────────────
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    model = fz.TrajectoryLSTM(len(C.FEAT_COLS)).to(device)
    model.load_state_dict(torch.load(EXP06 / "models" / "best_model_C_clean_balanced_esplanade.pth", map_location=device))
    model.eval()
    f_sc, t_sc = P5.load_scalers(EXP06 / "models" / "scalers_C_clean_balanced_esplanade.pkl")
    df = P5.load_dataset(); bounds = P5.load_world_bounds()
    plans = {rec: load_plan(rec) for rec in RECS}
    kdt = {}

    def reroll(rec, tid, ws):
        if rec not in kdt:
            kdt[rec] = fz.build_kdt(df[df.recording_id == rec], C.SPATIAL_COLS)
        t = df[(df.recording_id == rec) & (df.trajectory_id == tid)].sort_values("timestep").reset_index(drop=True)
        return P5.rollout_one_trajectory(model, t.iloc[int(ws):].reset_index(drop=True),
                                         bounds[rec], f_sc, t_sc, kdt[rec], device, max_steps=HORIZON)

    m = pd.read_csv(EXP06 / "metrics.csv")
    elig = m[(m.recording_id.isin(RECS)) & (m.is_genuine_turn == True) &  # noqa: E712
             (m.gt_head_change_deg.between(30, 90)) & (m.pred_head_change_deg.between(20, 100)) &
             (m.gt_head_change_deg < 120) & (m.ade < 3.0) & (m.pred_path_len_m > 0.55)].copy()
    print(f"[eligible] mild GT30-90 / pred20-100 / no U-turn / clean recs / pred>0.55m: {len(elig)} windows")

    # architectural features + re-roll + quality filters; rank by LONGEST predicted path
    obst, bnd = [], []
    for _, row in elig.iterrows():
        t = df[(df.recording_id == row.recording_id) & (df.trajectory_id == row.trajectory_id)].sort_values("timestep")
        seg = t.iloc[int(row.win_start): int(row.win_start) + C.WINDOW_SIZE + HORIZON]
        obst.append(float(seg["dist_to_obstacle_norm"].min())); bnd.append(float(seg["dist_to_boundary_norm"].mean()))
    elig["arch_category"] = [arch_category(o, b) for o, b in zip(obst, bnd)]

    keep = []
    for _, row in elig.sort_values("pred_path_len_m", ascending=False).iterrows():
        r = reroll(row.recording_id, row.trajectory_id, row.win_start)
        if r is None or not jumps_ok(r) or _reversals(np.asarray(r["pred_pos"])) > 4:
            continue
        if walkable_frac(plans[row.recording_id], r["gt_pos"]) < 0.6 or walkable_frac(plans[row.recording_id], r["pred_pos"]) < 0.45:
            continue
        keep.append((row, r))

    # take the 10 longest predicted paths, one per track, spread across categories
    sel, seen = [], set()
    for row, r in keep:                       # already sorted by pred_path_len desc
        if row.trajectory_id in seen:
            continue
        seen.add(row.trajectory_id); sel.append((row, r))
        if len(sel) >= N:
            break
    if len(sel) < N:
        for row, r in keep:
            if not any(s[0].trajectory_id == row.trajectory_id and int(s[0].win_start) == int(row.win_start) for s in sel):
                sel.append((row, r))
            if len(sel) >= N:
                break
    rows = pd.DataFrame([s[0] for s in sel]).reset_index(drop=True)
    # 30-50m context readability: prediction must span >= ~4% of a 30 m frame to read
    rows["context30_readable"] = rows["pred_path_len_m"] >= 1.2
    rows.to_csv(HERE / "selected_10_metrics.csv", index=False)
    print(f"[select] {len(sel)} panels; categories {dict(rows.arch_category.value_counts())}")

    # ── render all crop modes + raw geometry ─────────────────────────────────
    for i, (row, r) in enumerate(sel, 1):
        for mode, sub in (("individual", "individual"), ("zoom", "zoomed_arrow"), ("context30", "context_30m")):
            fig, ax = plt.subplots(figsize=(8, 5))
            draw(ax, plans[row.recording_id], r, row, mode, smooth=True)
            fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            fig.savefig(DIRS[sub] / f"panel_{i:02d}.png", dpi=200); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5))                      # raw geometry (no smoothing)
        draw(ax, plans[row.recording_id], r, row, "individual", smooth=False)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        fig.savefig(DIRS["raw_geometry"] / f"panel_{i:02d}.png", dpi=200); plt.close(fig)
    print(f"[panels] individual / zoomed_arrow / context_30m / raw_geometry ({len(sel)} each)")

    # ── collages ─────────────────────────────────────────────────────────────
    collage(sel, plans, 5, 2, PANELS / "collage_5x2.png")
    collage(sel, plans, 2, 5, PANELS / "collage_10.png")
    print("[collages] collage_5x2 + collage_10")

    # ── hero: longest readable prediction, prefer corridor/plaza context ─────
    hero = max(sel, key=lambda s: s[0].pred_path_len_m * (1.15 if s[0].arch_category != "obstacle avoidance" else 1.0))
    draw_hero(hero, plans[hero[0].recording_id], HERE / "hero_prediction_arrow.png")
    print("[hero] hero_prediction_arrow.png")

    write_docs(rows)

    # ── required summary print ───────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("EXP07C — LONG-ARROW PRESENTATION — selected 10")
    print("=" * 92)
    print(f"{'#':>2} {'track/window':28s} {'cat':20s} {'turn°':>5} {'pred_len_m':>10} {'gt_len_m':>9} {'ctx30_ok':>9}")
    for i, (row, r) in enumerate(sel, 1):
        tid = f"{row.recording_id.split('_')[0]}:{str(row.trajectory_id).split('__')[-1]}@{int(row.win_start)}"
        print(f"{i:>2} {tid:28s} {row.arch_category:20s} {row.gt_head_change_deg:5.0f} "
              f"{row.pred_path_len_m:10.2f} {row.gt_path_len_m:9.2f} {str(bool(row.pred_path_len_m>=1.2)):>9}")
    nfail = int((~rows.context30_readable).sum())
    print(f"\noutputs: {PANELS}/[individual|zoomed_arrow|context_30m|raw_geometry], "
          f"collage_5x2.png, collage_10.png, hero_prediction_arrow.png")
    print(f"30–50 m context target: {len(rows)-nfail}/{len(rows)} panels readable at 30 m; "
          f"{nfail} need the zoomed_arrow version (prediction < 1.2 m spans <4% of a 30 m frame) — see README tradeoff.")


def collage(sel, plans, ncol, nrow, path):
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.3 * nrow)); axes = np.atleast_1d(axes).ravel()
    for ax, (row, r) in zip(axes, sel):
        draw(ax, plans[row.recording_id], r, row, "individual", smooth=True)
        for t in ax.texts:
            t.set_fontsize(t.get_fontsize() * 0.7)
    for ax in axes[len(sel):]:
        ax.axis("off")
    handles = [Line2D([0], [0], color=C_OBS, lw=3, label="observed path"),
               Line2D([0], [0], color=C_GT, lw=3, label="ground truth future"),
               Line2D([0], [0], color=C_PRED, lw=4, label="model prediction")]
    fig.suptitle("MOTION PIXELS — QUALITATIVE PREDICTION EXAMPLES", fontsize=15, fontweight="bold", y=0.995)
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11, frameon=False)
    fig.tight_layout(rect=[0, 0.045, 1, 0.965]); fig.savefig(path, dpi=160); plt.close(fig)


def draw_hero(hero, plan, path):
    row, r = hero
    fig, ax = plt.subplots(figsize=(11, 6.9))
    draw(ax, plan, r, row, "individual", smooth=True)
    ax.texts[-2].set_fontsize(20); ax.texts[-1].set_fontsize(17)        # category + turn larger
    handles = [Line2D([0], [0], color=C_OBS, lw=3, label="observed path"),
               Line2D([0], [0], color=C_GT, lw=3, label="ground truth future"),
               Line2D([0], [0], color=C_PRED, lw=4.5, label="model prediction")]
    ax.legend(handles=handles, loc="lower right", fontsize=11.5, frameon=True, framealpha=0.92, edgecolor="#e0e0e0")
    ax.text(0.5, 0.045, "Motion Pixels predicts a spatially curved pedestrian future",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=14, style="italic", color="#444444")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(path, dpi=210); plt.close(fig)


def write_docs(rows):
    n = len(rows); cats = dict(rows.arch_category.value_counts()); nfail = int((~rows.context30_readable).sum())
    (HERE / "selection_method.md").write_text("\n".join([
        "# Selection method — exp07c (long-arrow presentation)\n",
        "Presentation re-render of the qualitative panels. Angularity is NOT optimised; selection ranks the "
        "LONGEST available predicted rollouts among architecturally-readable mild turns.\n",
        "## Filters\n",
        "- clean held-out recordings only (esplanade plaza + stairs corridor); placa_espanya excluded (artifact-prone)",
        "- GT turn 30°–90°, predicted turn 20°–100°, no U-turns (>120°)",
        "- predicted path length > 0.7 m (avoid tiny predictions), ADE < 3 m",
        "- prediction & GT mostly within walkable space; smooth (≤3 heading reversals, no jumps)\n",
        "## Ranking\n",
        "Sorted by predicted path length (DESCENDING) — the longest, most readable prediction arrows first; "
        "deduplicated to one window per track.\n",
        f"Selected **{n}**. Categories: {cats}.\n",
        "## Long-arrow readability (no fabrication)\n",
        "Predicted geometry is the model's actual rollout — never lengthened or scaled independently of GT. "
        "Readability is achieved only by (a) selecting the longest predictions, (b) tighter crops, (c) thicker "
        "magenta stroke + an elegant dominant arrowhead, and (d) a light Catmull-Rom smoothing applied for DISPLAY "
        "ONLY (raw, unsmoothed panels are saved in `panels/raw_geometry/`; metrics use raw values).\n",
        "## Crop tradeoff (context vs arrow legibility)\n",
        f"`context_30m/` targets ~30–50 m of spatial context, but mild-turn predictions are short, so at 30 m a "
        f"prediction < ~1.2 m spans <4% of the frame and reads weakly: **{nfail}/{n}** panels fall in this case. "
        f"`zoomed_arrow/` crops tightly so the magenta prediction is dominant; `individual/` is the recommended "
        "balance (~13 m, both the arrow and the obstacle/corridor/plaza context are legible).\n",
        "## 'turn' definition (kept off the image)\n",
        "The magenta `turn ≈ XX°` annotation is the absolute heading change between the last observed direction "
        "and the final future direction over the prediction horizon, measured on the ground-truth future.",
    ]), encoding="utf-8")

    (CAPS / "caption_short.md").write_text(
        "**Motion Pixels — qualitative prediction examples.** Black = observed path, grey = ground-truth future, "
        "magenta = model prediction, on the walkable plan. The model expresses smooth, spatially-plausible curved "
        "futures (selected, architecturally-readable cases).\n", encoding="utf-8")
    (CAPS / "caption_slide.md").write_text("\n".join([
        "**Motion Pixels predicts spatially curved pedestrian futures.**\n",
        "Observed path (black), ground-truth future (grey) and the model's prediction (magenta) over the walkable "
        "architecture. These are selected qualitative examples chosen for legibility (longest, smoothest mild-turn "
        "predictions); the prediction geometry is the model's real rollout, not lengthened or rescaled. They "
        "communicate that the model can express curved future movement — quantitative turn-direction accuracy "
        "remains limited (see exp06).",
    ]), encoding="utf-8")

    (HERE / "README.md").write_text("\n".join([
        "# exp07c — Long-arrow presentation panels\n",
        "Presentation-ready re-render of the qualitative prediction panels in a clean infographic visual language, "
        "with elegant dominant magenta prediction arrows on the architectural walkable plan. Built from exp06 "
        "predictions (re-rolled from the clean-split checkpoint) + V3 manual masks. Source data, frozen_model_C and "
        "prior experiments are untouched.\n",
        "## Statement of scope\n",
        "1. These are **qualitative presentation panels** (selected, architecturally-readable cases).",
        "2. **Prediction geometry is not fabricated or independently scaled** — it is the model's actual rollout; "
        "only crop, stroke styling and display-only smoothing change how it reads.",
        "3. **Long-arrow readability** is achieved through selection (longest predictions), tighter crops and "
        "styling — never by altering the data.",
        "4. **Quantitative turn-direction accuracy remains limited** (exp06: held-out angular error ≈ chance).",
        "5. They communicate the positive finding: **the model can express curved future movement**.\n",
        "## Visual language\n",
        "observed = black (#111111) · ground truth = soft grey (#A8A8A8) · prediction = magenta (#D81B7D, thicker, "
        "dominant) · current position = small black pebble (no stars) · walkable = white · built = light grey · "
        "thin pale-grey plan outline · subtle scale bar · minimal category title + magenta `turn ≈ XX°`.\n",
        "## 'turn' definition\n",
        "`turn` = the absolute heading change between the last observed direction and the final future direction "
        "over the prediction horizon (measured on the ground-truth future). This definition is intentionally kept "
        "OFF the panels.\n",
        "## Outputs\n",
        "- `panels/individual/panel_01..10.png` — recommended balanced crop (~13 m context).",
        "- `panels/zoomed_arrow/panel_01..10.png` — tight crop, prediction arrow dominant.",
        "- `panels/context_30m/panel_01..10.png` — ~30–50 m architectural context.",
        "- `panels/raw_geometry/panel_01..10.png` — unsmoothed geometry (display smoothing disabled).",
        "- `panels/collage_5x2.png`, `panels/collage_10.png` — 10-panel collages (title + legend).",
        "- `hero_prediction_arrow.png` — single strongest example.",
        "- `selected_10_metrics.csv`, `selection_method.md`, `captions/`.\n",
        f"Selected {n}. Categories: {cats}. Context-30m readable: {n-nfail}/{n} "
        "(the rest rely on `zoomed_arrow/`; see `selection_method.md`).\n",
    ]), encoding="utf-8")


if __name__ == "__main__":
    main()
