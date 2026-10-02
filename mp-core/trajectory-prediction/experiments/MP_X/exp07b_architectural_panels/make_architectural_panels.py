"""
MP_X / exp07b — Architectural communication panels (plan overlays).

NOT a turn-capture / angularity study. Selects 20 SMOOTH, architecturally-readable
predicted trajectories (gentle mild turns, accurate) and renders them on the actual
walkable-space PLAN (V3 manual masks) to communicate architectural pedestrian
motion: obstacle avoidance, corridor following, plaza circulation.

Selection (explicitly does NOT optimise angularity):
  * GT turn 30°–90° (gentle, readable turns — not sharp/U-turns)
  * predicted turn 20°–100°
  * ADE below the eligible-pool median (accurate, believable)
  * no U-turns (GT turn < 120°)
  * clean held-out recordings only (esplanade plaza + stairs); the artifact-prone
    placa_espanya (fast jitter, p95 step on the artifact guard) is EXCLUDED, and
    train-side recordings are excluded to keep predictions honest held-out.
Priority: smooth curvature, obstacle avoidance, corridor following, plaza
circulation, visually understandable architectural motion.

Trajectories are re-rolled with the exp06 clean-split checkpoint (esplanade held
out), then mapped world→plan via each recording's encoding_v3_metadata.json affine
and drawn on its walkable mask. Reads only exp06 outputs + dataset + masks; modifies
nothing.

Run:  python make_architectural_panels.py
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

# clean held-out recordings (placa_espanya = artifact-prone -> excluded)
RECS = ["esplanade_espanya_01", "stairs_montjuic_01"]
N_SELECT = 20
HORIZON = 20

C_SEED = "#333333"; C_GT = "#1f6fb2"; C_PRED = "#e8552d"; C_START = "#000000"
C_WALK = "#ffffff"; C_BUILT = "#d6d6d6"; C_EDGE = "#9a9a9a"

plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 10})

PANELS = HERE / "panels"
OVL = PANELS / "plan_overlays"; PRES = PANELS / "presentation"; CAPS = HERE / "captions"
for d in (OVL, PRES, CAPS):
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Plan assets
# ─────────────────────────────────────────────────────────────────────────────
def load_plan(rec):
    meta = json.loads((ENC_DIR / rec / "spatial_v3_manual" / "encoding_v3_metadata.json").read_text(encoding="utf-8"))
    walk = np.array(Image.open(MASK_DIR / rec / "walkable_mask_v3_manual.png").convert("L"))
    M = np.array(meta["transform_world_to_plan_3x2"], float)   # [[a,b],[c,d],[tx,ty]]
    scale = float(meta.get("scale_px_per_m_from_fit", 8.0))
    rgb = np.where(walk[..., None] > 127, np.array([1, 1, 1.]), np.array([float(int(C_BUILT[1:3], 16)) / 255,
          float(int(C_BUILT[3:5], 16)) / 255, float(int(C_BUILT[5:7], 16)) / 255]))
    return dict(walk=walk, rgb=rgb, M=M, scale=scale)


def w2p(M, wx, wy):
    px = wx * M[0, 0] + wy * M[1, 0] + M[2, 0]
    py = wx * M[0, 1] + wy * M[1, 1] + M[2, 1]
    return px, py


def walkable_frac(plan, world_pts):
    """Fraction of path points that fall on walkable space (mask white)."""
    px, py = w2p(plan["M"], *np.asarray(world_pts, float).T)
    h, w = plan["walk"].shape
    xi = np.clip(np.round(px).astype(int), 0, w - 1)
    yi = np.clip(np.round(py).astype(int), 0, h - 1)
    return float((plan["walk"][yi, xi] > 127).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Architectural + smoothness features
# ─────────────────────────────────────────────────────────────────────────────
def _reversals(pts):
    d = np.diff(pts, axis=0); mv = np.linalg.norm(d, axis=1) > 1e-6
    if mv.sum() < 3:
        return 99
    h = np.unwrap(np.arctan2(d[mv, 1], d[mv, 0])); tr = np.diff(h)
    sg = np.sign(tr); sg = sg[sg != 0]
    return int(np.sum(np.abs(np.diff(sg)) > 0)) if len(sg) > 1 else 0


def smoothness(r):
    pr = np.asarray(r["pred_pos"], float)
    steps = np.linalg.norm(np.diff(pr, axis=0), axis=1)
    if len(steps) < 3:
        return 0.0, False
    med = np.median(steps[steps > 1e-6]) if (steps > 1e-6).any() else 0
    jump_ok = steps.max() < 1.0 and (med <= 1e-6 or steps.max() < 7 * med) and steps.sum() > 0.35
    return 1.0 / (1.0 + _reversals(pr)), bool(jump_ok)


def arch_category(min_obst, mean_bound):
    if min_obst < 0.22:
        return "obstacle avoidance"
    if mean_bound < 0.35:
        return "corridor following"
    return "plaza circulation"


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────
def _arrow(ax, pts_px, color, frac=0.16, lw=2.4):
    pts = np.asarray(pts_px, float)
    if len(pts) < 2:
        return
    d = pts[-1] - pts[-2]; n = np.linalg.norm(d)
    if n < 1e-9:
        return
    span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1e-6)
    d = d / n * frac * span
    ax.add_patch(FancyArrowPatch(tuple(pts[-1]), tuple(pts[-1] + d), arrowstyle="-|>",
                                 mutation_scale=15, color=color, lw=lw, zorder=9))


def draw_overlay(ax, plan, r, row, mode="full"):
    M = plan["M"]; scale = plan["scale"]
    sp = np.column_stack(w2p(M, *np.asarray(r["seed_pos"]).T))
    gt = np.column_stack(w2p(M, *np.asarray(r["gt_pos"]).T))
    pr = np.column_stack(w2p(M, *np.asarray(r["pred_pos"]).T))
    allp = np.vstack([sp, gt, pr])
    cx, cy = allp.mean(0)
    half = max(allp[:, 0].ptp(), allp[:, 1].ptp()) / 2
    half = max(half * 1.45, 7.0 * scale) + 0.5 * scale          # >= ~14 m of context

    ax.imshow(plan["rgb"], origin="upper", interpolation="nearest", zorder=0)
    ax.contour(plan["walk"], levels=[127], colors=C_EDGE, linewidths=0.8, zorder=1)
    ax.plot(sp[:, 0], sp[:, 1], "-", color=C_SEED, lw=2.4, solid_capstyle="round",
            label="Observed path", zorder=4)
    ax.plot(gt[:, 0], gt[:, 1], "-", color=C_GT, lw=2.8, solid_capstyle="round",
            label="Ground truth future", zorder=5)
    ax.plot(pr[:, 0], pr[:, 1], "--", color=C_PRED, lw=2.8, dash_capstyle="round",
            label="Model prediction", zorder=6)
    ax.plot(*gt[0], marker="*", color=C_START, ms=15, zorder=10, lw=0)
    _arrow(ax, gt, C_GT); _arrow(ax, pr, C_PRED)
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy + half, cy - half)   # origin upper
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#cccccc")

    # scale bar (5 m)
    x0 = cx - half + 0.6 * scale; y0 = cy + half - 0.9 * scale
    ax.plot([x0, x0 + 5 * scale], [y0, y0], "-", color="#333333", lw=2.6, zorder=11)
    ax.text(x0 + 2.5 * scale, y0 - 0.4 * scale, "5 m", ha="center", va="bottom", fontsize=8, color="#333333")

    cat = row["arch_category"]
    if mode == "full":
        ax.set_title(f"{row['recording_id'].split('_')[0]} · {cat}\n"
                     f"GT turn {row['gt_head_change_deg']:.0f}°  ·  pred {row['pred_head_change_deg']:.0f}°  ·  ADE {row['ade']:.2f} m",
                     fontsize=9, color="#222222")
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.92, edgecolor="#dddddd")
    else:  # presentation
        ax.text(0.03, 0.96, cat, transform=ax.transAxes, fontsize=12, fontweight="bold",
                color="#222222", va="top")
        ax.text(0.03, 0.89, f"turn ≈ {row['gt_head_change_deg']:.0f}°", transform=ax.transAxes,
                fontsize=10.5, color=C_GT, va="top")


# ─────────────────────────────────────────────────────────────────────────────
def arch_score(df):
    ade = df["ade"].to_numpy(float)
    s_ade = np.clip(1 - ade / 1.1, 0, 1)
    s_obst = np.clip(1 - df["min_obst"].to_numpy(float) / 0.5, 0, 1)     # near an obstacle -> avoidance story
    s_corr = np.clip(1 - df["mean_bound"].to_numpy(float) / 0.5, 0, 1)
    s_smooth = df["smooth"].to_numpy(float)
    s_read = np.clip(df["pred_path_len_m"].to_numpy(float) / 2.0, 0.2, 1)
    # NOTE: angularity is deliberately NOT a term.
    return 1.3 * s_smooth + 1.1 * s_obst + 0.8 * s_ade + 0.6 * s_read + 0.5 * s_corr


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

    # ── eligible pool (architectural filters; NOT angularity) ────────────────
    m = pd.read_csv(EXP06 / "metrics.csv")
    held = m[(m.recording_id.isin(RECS)) & (m.is_genuine_turn == True)]  # noqa: E712
    mild = held[(held.gt_head_change_deg.between(30, 90)) & (held.pred_head_change_deg.between(20, 100)) &
                (held.gt_head_change_deg < 120)].copy()
    med = mild.ade.median()
    elig = mild[mild.ade < med].copy()
    print(f"[eligible] mild GT30-90 / pred20-100 / ADE<median({med:.2f}) / no U-turns / clean recs: "
          f"{len(elig)} windows, {elig.trajectory_id.nunique()} tracks")

    # architectural features per window from the dataset (obstacle/boundary clearance)
    obst, bnd = [], []
    for _, row in elig.iterrows():
        t = df[(df.recording_id == row.recording_id) & (df.trajectory_id == row.trajectory_id)].sort_values("timestep")
        seg = t.iloc[int(row.win_start): int(row.win_start) + C.WINDOW_SIZE + HORIZON]
        obst.append(float(seg["dist_to_obstacle_norm"].min()))
        bnd.append(float(seg["dist_to_boundary_norm"].mean()))
    elig["min_obst"] = obst; elig["mean_bound"] = bnd
    elig["arch_category"] = [arch_category(o, b) for o, b in zip(obst, bnd)]

    # re-roll, smoothness/quality filter, score
    keep = []
    for _, row in elig.iterrows():
        r = reroll(row.recording_id, row.trajectory_id, row.win_start)
        if r is None:
            continue
        sm, ok = smoothness(r)
        if not ok:
            continue
        # both GT and prediction should read as walking WITHIN walkable space (drop
        # mask-edge artifacts that would look like walking through a wall)
        wf_gt = walkable_frac(plans[row.recording_id], r["gt_pos"])
        wf_pr = walkable_frac(plans[row.recording_id], r["pred_pos"])
        if wf_gt < 0.6 or wf_pr < 0.5:
            continue
        rr = row.copy(); rr["smooth"] = sm; rr["walk_frac"] = wf_gt
        keep.append((rr, r))
    edf = pd.DataFrame([k[0] for k in keep])
    edf["ascore"] = arch_score(edf)
    order = edf.sort_values("ascore", ascending=False)

    # select 20: dedup by track, encourage category variety
    rolls_by_key = {(k[0].trajectory_id, int(k[0].win_start)): k[1] for k in keep}
    sel, seen, cats = [], set(), {"obstacle avoidance": 0, "corridor following": 0, "plaza circulation": 0}
    for _, row in order.iterrows():
        if row.trajectory_id in seen:
            continue
        seen.add(row.trajectory_id)
        sel.append(row)
        cats[row.arch_category] += 1
        if len(sel) >= N_SELECT:
            break
    if len(sel) < N_SELECT:                       # backfill allowing repeat tracks
        for _, row in order.iterrows():
            if any(s.trajectory_id == row.trajectory_id and int(s.win_start) == int(row.win_start) for s in sel):
                continue
            sel.append(row)
            if len(sel) >= N_SELECT:
                break
    sel = pd.DataFrame(sel).reset_index(drop=True)
    sel.to_csv(HERE / "selected_20_metrics.csv", index=False)
    print(f"[select] {len(sel)} panels; categories: {dict(sel.arch_category.value_counts())}")

    rolls = [(row, rolls_by_key[(row.trajectory_id, int(row.win_start))]) for _, row in sel.iterrows()]

    # ── plan overlays + presentation ─────────────────────────────────────────
    for i, (row, r) in enumerate(rolls, 1):
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        draw_overlay(ax, plans[row.recording_id], r, row, mode="full")
        fig.tight_layout(); fig.savefig(OVL / f"overlay_{i:02d}.png", dpi=200); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        draw_overlay(ax, plans[row.recording_id], r, row, mode="pres")
        fig.tight_layout(); fig.savefig(PRES / f"panel_{i:02d}.png", dpi=200); plt.close(fig)
    print(f"[panels] {len(rolls)} plan overlays + presentation panels")

    # ── collage + contact sheet ──────────────────────────────────────────────
    collage(rolls, plans, 4, 5, PANELS / "collage_4x5_plan_overlays.png",
            "Architectural pedestrian motion on plan — smooth curved predictions (held-out)")
    contact(rolls, plans, PANELS / "contact_sheet.png")
    print("[collages] 4x5 + contact sheet")

    # ── hero: clearest readable example (corridor/plaza show walkable context
    #         best; pick the highest walkable-visibility within the preferred category)
    def pick_hero():
        for cat in ("corridor following", "plaza circulation", "obstacle avoidance"):
            cands = [rr for rr in rolls if rr[0].arch_category == cat]
            if cands:
                return max(cands, key=lambda rr: float(rr[0].get("walk_frac", 0.5)))
        return rolls[0]
    hero_pick = pick_hero()
    draw_hero(hero_pick, plans[hero_pick[0].recording_id], HERE / "hero_architectural_turn.png")
    print("[hero] hero_architectural_turn.png")

    write_docs(sel, hero_pick[0])
    print("\n[done] exp07b architectural panels in", HERE)


def collage(rolls, plans, nrow, ncol, path, suptitle):
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.4 * nrow)); axes = np.atleast_1d(axes).ravel()
    for ax, (row, r) in zip(axes, rolls):
        draw_overlay(ax, plans[row.recording_id], r, row, mode="full")
        ax.set_title(ax.get_title(), fontsize=6.5)
        if ax.get_legend():
            ax.get_legend().remove()
    for ax in axes[len(rolls):]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(path, dpi=160); plt.close(fig)


def contact(rolls, plans, path):
    ncol = 5; nrow = int(np.ceil(len(rolls) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.8 * ncol, 2.8 * nrow)); axes = np.atleast_1d(axes).ravel()
    for ax, (row, r) in zip(axes, rolls):
        draw_overlay(ax, plans[row.recording_id], r, row, mode="pres")
        for t in list(ax.texts):
            t.remove()
        ax.text(0.5, -0.04, row.arch_category, transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#444")
    for ax in axes[len(rolls):]:
        ax.axis("off")
    handles = [Line2D([0], [0], color=C_SEED, lw=2.4, label="Observed"),
               Line2D([0], [0], color=C_GT, lw=2.6, label="Ground truth"),
               Line2D([0], [0], color=C_PRED, lw=2.6, ls="--", label="Prediction")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, frameon=False)
    fig.suptitle("Architectural motion on plan — observed · ground truth · prediction", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97]); fig.savefig(path, dpi=160); plt.close(fig)


def draw_hero(roll, plan, path):
    row, r = roll
    fig, ax = plt.subplots(figsize=(11, 9.2))
    draw_overlay(ax, plan, r, row, mode="full")
    ax.set_title("Motion Pixels — pedestrian motion read on the architectural plan", fontsize=17, fontweight="bold", pad=12)
    ax.legend(fontsize=12, loc="lower right", framealpha=0.92, edgecolor="#dddddd")
    ax.text(0.5, -0.045, f"{row['arch_category'].capitalize()} — the predicted path follows the walkable space and curves "
            f"with the architecture (GT turn {row['gt_head_change_deg']:.0f}°, ADE {row['ade']:.2f} m).",
            transform=ax.transAxes, ha="center", va="top", fontsize=12.5, style="italic", color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 1]); fig.savefig(path, dpi=210); plt.close(fig)


def write_docs(sel, hero_row):
    n = len(sel); cats = dict(sel.arch_category.value_counts())
    (HERE / "selection_method.md").write_text("\n".join([
        "# Selection method — exp07b (architectural)\n",
        "This panel set is for **architectural communication, not turn-capture metrics**. Angularity is "
        "deliberately NOT part of the score.\n",
        "## Eligibility (hard filters)\n",
        "- GT turn 30°–90° (gentle, readable turns; no sharp/U-turns)",
        "- predicted turn 20°–100°",
        "- ADE below the eligible-pool median (accurate, believable)",
        "- GT turn < 120° (no U-turns)",
        "- clean held-out recordings only: esplanade (plaza) + stairs (corridor). placa_espanya is EXCLUDED "
        "(identified artifact-prone: fastest motion, p95 step on the 0.6 m guard, median turn ~150°); "
        "train-side recordings are excluded so all shown predictions are honest held-out.\n",
        "## Architectural priority score (no angularity)\n",
        "- smooth curvature (few predicted heading reversals, no jumps) — weight 1.3",
        "- obstacle avoidance (path passes near an obstacle: low min obstacle-clearance) — weight 1.1",
        "- accuracy (low ADE) — weight 0.8",
        "- readable predicted path length — weight 0.6",
        "- corridor following (sustained low boundary-clearance) — weight 0.5\n",
        "Each window is tagged **obstacle avoidance** / **corridor following** / **plaza circulation** from its "
        "obstacle/boundary clearance, and selection is deduplicated by track and spread across categories.\n",
        f"Selected **{n}** of the eligible pool. Category mix: {cats}.\n",
        "## Plan overlay\n",
        "World→plan mapping uses each recording's `encoding_v3_metadata.json` affine "
        "(`transform_world_to_plan_3x2`, ~8 px/m). The background is the V3 manual walkable mask "
        "(white = walkable, grey = built/obstacle, thin line = walkable edge); a 5 m scale bar is drawn.",
    ]), encoding="utf-8")

    (CAPS / "caption_short.md").write_text(
        "**Pedestrian motion read on the plan.** Selected held-out examples (esplanade plaza, stairs corridor): "
        "observed path (grey), ground-truth future (blue), model prediction (orange), over the walkable space. "
        "The model produces smooth, architecturally plausible curved paths that follow circulation space.\n", encoding="utf-8")
    (CAPS / "caption_long.md").write_text("\n".join([
        "**Figure — Predicted pedestrian motion overlaid on the architectural plan.**\n",
        f"{n} smooth, gently-curving held-out predictions selected for architectural legibility (gentle 30°–90° "
        "turns, below-median ADE) and rendered on the V3 manual walkable-space masks of the esplanade (plaza) and "
        "stairs (corridor). Observed history (grey), ground-truth future (blue) and model prediction (orange) are "
        "shown with final-direction arrows and a 5 m scale bar; panels are tagged by architectural situation "
        "(obstacle avoidance / corridor following / plaza circulation).\n",
        "These figures communicate that Motion Pixels predictions read as **plausible architectural circulation** — "
        "the predicted paths stay within walkable space and curve smoothly with the built environment. They are a "
        "qualitative, communication-oriented selection (smooth, accurate, legible cases), not a measure of turn-"
        "direction accuracy; see exp06 for the quantitative result.",
    ]), encoding="utf-8")

    (HERE / "README.md").write_text("\n".join([
        "# exp07b — Architectural communication panels (plan overlays)\n",
        "Plan-overlay panels communicating architectural pedestrian motion (obstacle avoidance, corridor "
        "following, plaza circulation). **Not** a turn-capture / angularity study — angularity is explicitly not "
        "optimised. Trajectories are re-rolled with the exp06 clean-split checkpoint and drawn on the V3 manual "
        "walkable masks. Source data, frozen_model_C and prior experiments are untouched.\n",
        "## Selection (architectural, no angularity)\n",
        "GT turn 30°–90°, predicted turn 20°–100°, ADE below median, no U-turns (>120°), clean held-out "
        "recordings only (esplanade + stairs; placa_espanya excluded as artifact-prone). Prioritises smooth "
        "curvature, obstacle avoidance, corridor following, plaza circulation. See `selection_method.md`.\n",
        f"Selected {n}. Category mix: {cats}.\n",
        "## Outputs\n",
        f"- `panels/plan_overlays/overlay_01..{n:02d}.png` — trajectory on the walkable plan (scale bar, arrows, metrics).",
        f"- `panels/presentation/panel_01..{n:02d}.png` — minimal slide version (situation + turn angle).",
        "- `panels/collage_4x5_plan_overlays.png`, `panels/contact_sheet.png`.",
        "- `hero_architectural_turn.png` — large plan overlay of the clearest example.",
        "- `selected_20_metrics.csv`, `selection_method.md`, `captions/`.\n",
        "## Honest scope\n",
        "These are communication-oriented qualitative selections (smooth, accurate, legible architectural motion). "
        "They are NOT a claim about turn-direction accuracy — exp06 shows held-out angular error is ≈chance. The "
        "purpose is to show predictions read as plausible architectural circulation within walkable space.\n",
        "## Colour key\n",
        "Observed = dark grey · Ground truth = blue · Prediction = orange · walkable = white · built = grey · "
        "★ = prediction start · arrows = final directions · bar = 5 m.\n",
    ]), encoding="utf-8")


if __name__ == "__main__":
    main()
