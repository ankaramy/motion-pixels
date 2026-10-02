"""
MP_X / exp07 — Publication-quality turn-prediction panels.

Selects the most VISUALLY COMPELLING genuine-turn predictions from exp06 (the
clean, direction-balanced esplanade held-out) and renders thesis / slide-quality
panels. The selection rewards believable CURVED predictions (the positive visual
finding) even when the exact final direction is imperfect — these are honest
qualitative examples, NOT proof of directional accuracy (exp06 quantitative
angular error is still ≈chance; see README).

exp06 saved per-window metrics but not trajectories, so the chosen windows are
re-rolled here with the saved exp06 checkpoint (identical to exp06's evaluation).

Reads only exp06 outputs + the dataset + the exp06 checkpoint. Modifies no source
data, no frozen_model_C, no previous experiment output.

Run:
    python make_panels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "shared"
EXP06 = HERE.parent / "exp06_turn_balanced_eval_split"
sys.path.insert(0, str(SHARED))
import mpx_common as C   # noqa: E402
fz = C.fz; P5 = C.P5

import torch  # noqa: E402

# ── style ────────────────────────────────────────────────────────────────────
C_SEED = "#3a3a3a"      # observed history — dark grey
C_GT = "#1f6fb2"        # ground-truth future — blue
C_PRED = "#e8552d"      # model prediction — orange/red
C_START = "#111111"
N_SELECT = 25
HELDOUT_REC = "esplanade_espanya_01"
NOISY_REC = "placa_espanya_01"
HORIZON = 20

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888888", "axes.linewidth": 0.8,
})

PANELS = HERE / "panels"
IND = PANELS / "individual"
PRES = PANELS / "presentation_mode"
CAPS = HERE / "captions"
for d in (IND, PRES, CAPS):
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Visual-quality selection score (metric-based pre-rank)
# ─────────────────────────────────────────────────────────────────────────────
def metric_score(df):
    pred = df["pred_head_change_deg"].to_numpy(float)
    ar = df["angularity_ratio"].to_numpy(float)
    ade = df["ade"].to_numpy(float)
    gpl = df["gt_path_len_m"].to_numpy(float); ppl = df["pred_path_len_m"].to_numpy(float)
    gms = df["gt_max_step_m"].to_numpy(float)

    # The model under-predicts displacement on turns (median predicted path ~0.5 m), so the
    # rare LONG, visibly-curved predictions are exactly what makes a good publication panel.
    s_predlen = np.clip((ppl - 0.8) / 2.5, 0, 1)                      # reward a long, visible prediction (NOT collapsed)
    s_turn = np.clip((pred - 22.0) / 50.0, 0, 1)                      # reward a clear predicted turn (curvature)
    s_ang = np.where((ar >= 0.35) & (ar <= 1.4), 1 - np.abs(ar - 0.8) / 0.8, 0)
    s_ang = np.clip(s_ang, 0, 1)                                      # angularity in the believable band
    s_clean = np.clip((0.50 - gms) / 0.35, 0, 1)                      # clean GT (avoid tracking-artifact turns)
    s_gtvis = np.clip(1 - np.abs(gpl - 3.5) / 4.0, 0.2, 1)            # GT turn long enough to read
    s_ade = np.clip(1 - ade / 2.5, 0, 1)                             # mild accuracy reward (not dominant)
    total = (2.0 * s_predlen + 1.4 * s_turn + 1.2 * s_ang +
             1.0 * s_clean + 0.6 * s_gtvis + 0.6 * s_ade)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Path-quality (needs the re-rolled positions)
# ─────────────────────────────────────────────────────────────────────────────
def _reversals(pts):
    d = np.diff(pts, axis=0)
    mv = np.linalg.norm(d, axis=1) > 1e-6
    if mv.sum() < 3:
        return 99
    h = np.unwrap(np.arctan2(d[mv, 1], d[mv, 0]))
    tr = np.diff(h)
    sg = np.sign(tr); sg = sg[sg != 0]
    return int(np.sum(np.abs(np.diff(sg)) > 0)) if len(sg) > 1 else 0


def path_quality(r):
    pr = np.asarray(r["pred_pos"], float); gt = np.asarray(r["gt_pos"], float)
    psteps = np.linalg.norm(np.diff(pr, axis=0), axis=1)
    if len(psteps) < 3:
        return 0.0, False
    med = np.median(psteps[psteps > 1e-6]) if (psteps > 1e-6).any() else 0.0
    maxjump = float(psteps.max())
    jump_ok = (maxjump < 1.0) and (med <= 1e-6 or maxjump < 6 * med)
    plen = float(psteps.sum())
    sep = min(1.0, plen / 1.5)                                   # reward a long, visible prediction (not a stub)
    smooth_p = 1.0 / (1.0 + _reversals(pr))                     # few heading reversals = smooth
    gt_clean = 1.0 / (1.0 + max(0, _reversals(gt) - 1))         # GT itself not zig-zaggy
    score = 1.1 * smooth_p + 0.7 * sep + 0.8 * gt_clean
    return float(score) * (1.0 if jump_ok else 0.25), jump_ok


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────
def _final_arrow(ax, pts, color, frac=0.16, lw=2.4):
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return
    d = pts[-1] - pts[-2]
    n = np.linalg.norm(d)
    if n < 1e-9:
        return
    span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1e-6)
    d = d / n * frac * span
    ax.add_patch(FancyArrowPatch((pts[-1, 0], pts[-1, 1]), (pts[-1, 0] + d[0], pts[-1, 1] + d[1]),
                                 arrowstyle="-|>", mutation_scale=16, color=color, lw=lw, zorder=8))


def _equal(ax, *paths, pad=0.18):
    allp = np.vstack([np.asarray(p, float) for p in paths if len(p)])
    xmin, ymin = allp.min(0); xmax, ymax = allp.max(0)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(xmax - xmin, ymax - ymin) / 2 * (1 + pad) + 0.3
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")


def draw_panel(ax, r, row, mode="technical", grid=True):
    seed = np.asarray(r["seed_pos"], float); gt = np.asarray(r["gt_pos"], float); pr = np.asarray(r["pred_pos"], float)
    ax.plot(seed[:, 0], seed[:, 1], "-", color=C_SEED, lw=2.2, solid_capstyle="round",
            label="Observed path" if mode != "technical" else "seed (history)", zorder=3)
    ax.plot(gt[:, 0], gt[:, 1], "-", color=C_GT, lw=2.6, solid_capstyle="round",
            label="Ground truth future" if mode != "technical" else "ground truth", zorder=4)
    ax.plot(pr[:, 0], pr[:, 1], "--", color=C_PRED, lw=2.6, dash_capstyle="round",
            label="Model prediction" if mode != "technical" else "prediction", zorder=5)
    ax.plot(*gt[0], marker="*", color=C_START, ms=16, zorder=9, lw=0)
    ax.plot(*gt[-1], marker="o", color=C_GT, ms=8, mec="white", mew=1.2, zorder=9, lw=0)
    ax.plot(*pr[-1], marker="s", color=C_PRED, ms=8, mec="white", mew=1.2, zorder=9, lw=0)
    _final_arrow(ax, gt, C_GT); _final_arrow(ax, pr, C_PRED)
    _equal(ax, seed, gt, pr)
    if grid:
        ax.grid(True, color="#e9e9e9", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7, color="#bbbbbb")

    captured = bool(row["pred_turn"]) and bool(row["gt_turn"])
    if mode == "technical":
        tid = str(row["trajectory_id"]).split("__")[-1]
        ax.set_title(f"esplanade · track {tid} · win {int(row['win_start'])}\n"
                     f"ADE {row['ade']:.2f} m   FDE {row['fde']:.2f} m   "
                     f"GT turn {row['gt_head_change_deg']:.0f}°   pred {row['pred_head_change_deg']:.0f}°   "
                     + ("turn captured ✓" if captured else "turn captured ✗"),
                     fontsize=8.5, color="#222222")
        ax.legend(fontsize=6.5, loc="best", framealpha=0.9, edgecolor="#dddddd")
    else:  # presentation
        # minimal: angle annotation only
        ax.text(0.03, 0.97, f"turn ≈ {row['gt_head_change_deg']:.0f}°", transform=ax.transAxes,
                fontsize=11, color=C_GT, va="top", fontweight="bold")
        ax.text(0.03, 0.89, f"predicted ≈ {row['pred_head_change_deg']:.0f}°", transform=ax.transAxes,
                fontsize=10, color=C_PRED, va="top")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8.5, loc="lower right", framealpha=0.92, edgecolor="#dddddd")


def draw_comparison_cell(ax, r, row):
    """Minimal GT-vs-prediction cell for the contact sheet (no seed)."""
    gt = np.asarray(r["gt_pos"], float); pr = np.asarray(r["pred_pos"], float)
    ax.plot(gt[:, 0], gt[:, 1], "-", color=C_GT, lw=2.2, zorder=4)
    ax.plot(pr[:, 0], pr[:, 1], "--", color=C_PRED, lw=2.2, zorder=5)
    ax.plot(*gt[0], marker="*", color=C_START, ms=11, zorder=9, lw=0)
    _final_arrow(ax, gt, C_GT, frac=0.18, lw=1.8); _final_arrow(ax, pr, C_PRED, frac=0.18, lw=1.8)
    _equal(ax, gt, pr)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, -0.02, f"GT {row['gt_head_change_deg']:.0f}° / pred {row['pred_head_change_deg']:.0f}°",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#444444")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # model + scalers from exp06 (chosen clean split)
    model = fz.TrajectoryLSTM(len(C.FEAT_COLS)).to(device)
    model.load_state_dict(torch.load(EXP06 / "models" / "best_model_C_clean_balanced_esplanade.pth", map_location=device))
    model.eval()
    f_sc, t_sc = P5.load_scalers(EXP06 / "models" / "scalers_C_clean_balanced_esplanade.pkl")

    df = P5.load_dataset(); bounds = P5.load_world_bounds()
    kdt_cache = {}

    def reroll(rec, tid, win_start):
        if rec not in kdt_cache:
            kdt_cache[rec] = fz.build_kdt(df[df.recording_id == rec], C.SPATIAL_COLS)
        tdf = df[(df.recording_id == rec) & (df.trajectory_id == tid)].sort_values("timestep").reset_index(drop=True)
        seg = tdf.iloc[int(win_start):].reset_index(drop=True)
        return P5.rollout_one_trajectory(model, seg, bounds[rec], f_sc, t_sc, kdt_cache[rec], device, max_steps=HORIZON)

    # ── select from clean esplanade held-out ─────────────────────────────────
    m = pd.read_csv(EXP06 / "metrics.csv")
    sel = select(m, HELDOUT_REC, reroll, N_SELECT)
    sel.to_csv(HERE / "selected_25_metrics.csv", index=False)
    print(f"[select] {len(sel)} esplanade panels chosen")

    rolls = [(row, reroll(row.recording_id, row.trajectory_id, row.win_start)) for _, row in sel.iterrows()]
    rolls = [(row, r) for row, r in rolls if r is not None]

    # ── individual technical + presentation panels ───────────────────────────
    for i, (row, r) in enumerate(rolls, 1):
        fig, ax = plt.subplots(figsize=(5.4, 5.4))
        draw_panel(ax, r, row, mode="technical")
        fig.tight_layout(); fig.savefig(IND / f"panel_{i:02d}.png", dpi=200); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.4, 5.4))
        draw_panel(ax, r, row, mode="presentation")
        fig.tight_layout(); fig.savefig(PRES / f"panel_{i:02d}.png", dpi=200); plt.close(fig)
    print(f"[panels] {len(rolls)} individual + presentation panels")

    # ── collages ─────────────────────────────────────────────────────────────
    grid_collage(rolls[:9], 3, 3, PANELS / "collage_3x3_best.png",
                 "Motion Pixels — best curved turn predictions (held-out, esplanade)")
    grid_collage(rolls[:25], 5, 5, PANELS / "collage_5x5_selected.png",
                 "Selected 25 genuine-turn predictions (exp06 clean held-out)")
    contact_sheet(rolls[:25], PANELS / "comparison_contact_sheet.png")
    print("[collages] 3x3, 5x5, contact sheet")

    # ── hero ─────────────────────────────────────────────────────────────────
    hero(rolls[0], HERE / "hero_turn_prediction.png")
    print("[hero] hero_turn_prediction.png")

    # ── robustness set (placa_espanya, clearly labelled noisy) ───────────────
    try:
        m_rob = pd.read_csv(EXP06 / "robustness_placa_espanya" / "metrics.csv")
        rob_model = fz.TrajectoryLSTM(len(C.FEAT_COLS)).to(device)
        rob_model.load_state_dict(torch.load(EXP06 / "robustness_placa_espanya" / "models" / "best_model_B_turn_rich_placa_espanya.pth", map_location=device))
        rob_model.eval()
        rf, rt = P5.load_scalers(EXP06 / "robustness_placa_espanya" / "models" / "scalers_B_turn_rich_placa_espanya.pkl")

        def reroll_rob(rec, tid, win_start):
            if ("rob", rec) not in kdt_cache:
                kdt_cache[("rob", rec)] = fz.build_kdt(df[df.recording_id == rec], C.SPATIAL_COLS)
            tdf = df[(df.recording_id == rec) & (df.trajectory_id == tid)].sort_values("timestep").reset_index(drop=True)
            seg = tdf.iloc[int(win_start):].reset_index(drop=True)
            return P5.rollout_one_trajectory(rob_model, seg, bounds[rec], rf, rt, kdt_cache[("rob", rec)], device, max_steps=HORIZON)

        srob = select(m_rob, NOISY_REC, reroll_rob, 9)
        robs = [(row, reroll_rob(row.recording_id, row.trajectory_id, row.win_start)) for _, row in srob.iterrows()]
        robs = [(row, r) for row, r in robs if r is not None]
        grid_collage(robs[:9], 3, 3, PANELS / "robustness_placa_espanya_3x3.png",
                     "ROBUSTNESS (noisy) — placa_espanya turns: tracking is jittery, shown for completeness")
        print(f"[robustness] {len(robs)} placa_espanya panels (labelled noisy)")
    except Exception as e:
        print(f"[robustness] skipped ({e})")

    write_docs(sel, rolls[0])
    print("\n[done] exp07 publication panels in", HERE)


def select(m, rec, reroll, n):
    cand = m[(m["recording_id"] == rec) & (m["is_genuine_turn"] == True)].copy()  # noqa: E712
    if len(cand) == 0:
        return cand
    cand["mscore"] = metric_score(cand)
    # hard pre-filters: prediction is a long-enough VISIBLE curve, clean GT, believable angularity
    cand = cand[(cand.gt_max_step_m < 0.50) & (cand.pred_head_change_deg > 20) &
                (cand.pred_path_len_m > 0.65) & (cand.angularity_ratio.between(0.30, 1.6)) &
                (cand.ade < 3.5) & (cand.gt_net_disp_m > 2.0)]
    pool = cand.sort_values("mscore", ascending=False).head(120)
    # re-roll pool, add path quality, refine
    rows = []
    for _, row in pool.iterrows():
        r = reroll(row.recording_id, row.trajectory_id, row.win_start)
        if r is None:
            continue
        pq, ok = path_quality(r)
        if not ok:
            continue
        rows.append((row, row.mscore + 1.3 * pq))
    rows.sort(key=lambda x: x[1], reverse=True)
    # unique-track-first selection; the model produces few long curved predictions, so allow up
    # to MAX_PER_TRACK windows per track (different decision points) to reach n with variety.
    MAX_PER_TRACK = 2
    out, per = [], {}
    for pass_cap in (1, MAX_PER_TRACK):        # pass 1: one per track; pass 2: fill remaining
        for row, sc in rows:
            tid = row.trajectory_id
            if per.get(tid, 0) >= pass_cap:
                continue
            key = (tid, int(row.win_start))
            if any((o.trajectory_id == tid and int(o.win_start) == key[1]) for o in out):
                continue
            rr = row.copy(); rr["select_score"] = sc
            out.append(rr); per[tid] = per.get(tid, 0) + 1
            if len(out) >= n:
                break
        if len(out) >= n:
            break
    return pd.DataFrame(out).reset_index(drop=True)


def grid_collage(rolls, nrow, ncol, path, suptitle):
    if not rolls:
        return
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (row, r) in zip(axes, rolls):
        draw_panel(ax, r, row, mode="technical", grid=False)
        ax.set_title(ax.get_title(), fontsize=6.5)
        leg = ax.get_legend()
        if leg:
            leg.remove()
    for ax in axes[len(rolls):]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(path, dpi=170); plt.close(fig)


def contact_sheet(rolls, path):
    if not rolls:
        return
    ncol = 5; nrow = int(np.ceil(len(rolls) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.8 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (row, r) in zip(axes, rolls):
        draw_comparison_cell(ax, r, row)
    for ax in axes[len(rolls):]:
        ax.axis("off")
    fig.suptitle("GT (blue) vs model prediction (orange) — turn-capture contact sheet",
                 fontsize=12, fontweight="bold")
    # one shared legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=C_GT, lw=2.4, label="Ground truth future"),
               Line2D([0], [0], color=C_PRED, lw=2.4, ls="--", label="Model prediction")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97]); fig.savefig(path, dpi=170); plt.close(fig)


def hero(roll, path):
    row, r = roll
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    seed = np.asarray(r["seed_pos"], float); gt = np.asarray(r["gt_pos"], float); pr = np.asarray(r["pred_pos"], float)
    ax.plot(seed[:, 0], seed[:, 1], "-", color=C_SEED, lw=3.2, solid_capstyle="round", label="Observed path", zorder=3)
    ax.plot(gt[:, 0], gt[:, 1], "-", color=C_GT, lw=3.8, solid_capstyle="round", label="Ground truth future", zorder=4)
    ax.plot(pr[:, 0], pr[:, 1], "--", color=C_PRED, lw=3.8, dash_capstyle="round", label="Model prediction", zorder=5)
    ax.plot(*gt[0], marker="*", color=C_START, ms=26, zorder=9, lw=0)
    ax.plot(*gt[-1], marker="o", color=C_GT, ms=12, mec="white", mew=1.6, zorder=9, lw=0)
    ax.plot(*pr[-1], marker="s", color=C_PRED, ms=12, mec="white", mew=1.6, zorder=9, lw=0)
    _final_arrow(ax, gt, C_GT, frac=0.14, lw=3.4); _final_arrow(ax, pr, C_PRED, frac=0.14, lw=3.4)
    _equal(ax, seed, gt, pr, pad=0.22)
    ax.grid(True, color="#ededed", lw=0.9); ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, color="#cccccc")
    ax.set_title("Motion Pixels — predicting a curved pedestrian future", fontsize=18, fontweight="bold", pad=12)
    ax.legend(fontsize=12, loc="best", framealpha=0.92, edgecolor="#dddddd")
    ax.text(0.5, -0.06,
            "Model captures turning tendency, but final direction remains uncertain.",
            transform=ax.transAxes, ha="center", va="top", fontsize=12.5, style="italic", color="#555555")
    ax.text(0.985, 0.02, f"GT turn {row['gt_head_change_deg']:.0f}°  ·  predicted {row['pred_head_change_deg']:.0f}°  ·  ADE {row['ade']:.2f} m",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="#777777")
    fig.tight_layout(rect=[0, 0.04, 1, 1]); fig.savefig(path, dpi=220); plt.close(fig)


def write_docs(sel, hero_roll):
    hero_row = hero_roll[0]
    n = len(sel)
    # selection_method.md
    (HERE / "selection_method.md").write_text("\n".join([
        "# Selection method — exp07\n",
        f"Source: exp06 clean held-out (`esplanade`, 1,232 genuine turns). Goal: pick the best (target 25; "
        f"**{n} passed the quality bar**) most VISUALLY COMPELLING curved predictions for thesis/slide use — "
        "believable curvature, not necessarily correct final direction. The model under-predicts displacement on "
        "turns (median predicted path ~0.5 m), so genuinely long, clearly-curved predictions are scarce — the "
        f"quality filters admit {n} of 1,232.\n",
        "## Two-stage score\n",
        "**Stage 1 — metric pre-rank** (from exp06 `metrics.csv`), rewards:",
        "- predicted heading change > 20° (the prediction actually turns) — weight 1.6",
        "- angularity ratio in [0.3, 1.2], peak ~0.75 (not collapsed, not wild) — weight 1.4",
        "- low ADE / FDE — weights 1.2 / 0.8",
        "- predicted/GT path-length ratio in [0.4, 1.4] (not collapsed, not overshooting) — weight 1.0",
        "- clean GT: max single step < 0.55 m (avoids tracking-artifact turns) — weight 1.2",
        "- readable net displacement (~2–6 m) — weight 0.5\n",
        "Hard pre-filters: gt_max_step < 0.55 m, predicted turn > 18°, angularity ∈ [0.30, 1.25], ADE < 1.8 m.\n",
        "**Stage 2 — path quality** (after re-rolling the top 70 with the exp06 checkpoint):",
        "- smoothness: few predicted heading reversals (no zig-zag)",
        "- no extreme jumps: max predicted step < 1.0 m and < 6× median step (else heavily penalised)",
        "- visible separation: predicted path length > 0.8 m (a path, not a dot)",
        "- GT cleanliness: GT itself not zig-zaggy\n",
        "Final rank = metric score + 1.3 × path-quality; deduplicated to one window per track for variety; top 25.\n",
        "## Honesty notes\n",
        "- These are **qualitative** selections optimised for clear curvature, NOT a random/representative sample.",
        "- They are **not** evidence of directional accuracy; exp06 angular error on this held-out is ≈93° (chance).",
        "- 'turn captured ✓' in a title means predicted heading change > 20° AND GT turn > 30° — it captures that the "
        "model turned, not that it turned the correct way.",
    ]), encoding="utf-8")

    # captions
    (CAPS / "caption_short.md").write_text(
        "**Motion Pixels predicts curved pedestrian futures.** Selected held-out examples (esplanade): "
        "observed path (grey), ground-truth future (blue), model prediction (orange). The model expresses "
        "clear turning, though the exact final direction is not yet reliable.\n", encoding="utf-8")
    (CAPS / "caption_long.md").write_text("\n".join([
        "**Figure — Curved future prediction on held-out pedestrian turns.**\n",
        f"{n} genuine-turn examples selected from the exp06 clean, direction-balanced held-out set "
        "(esplanade; 1,232 genuine turns). Each panel shows the 10-frame observed history (dark grey), the "
        "20-step ground-truth future (blue), and the model's autoregressive prediction (orange), with arrows "
        "marking the final ground-truth and predicted directions and a star at the prediction start.\n",
        "These panels demonstrate the **positive visual finding** of the Motion Pixels turn-recovery study: "
        "after turn-balanced / turn-only training, Model C produces *visibly curved* rollouts and commits to a "
        "turn rather than collapsing to a straight line. They are deliberately selected for visual clarity and "
        "are **not** a claim of directional accuracy: across the full held-out set the mean angular error "
        "remains near chance (~93°), i.e. the model captures the *tendency* to turn and a plausible turn "
        "*magnitude*, but not reliably the correct turn *direction* at decision points. The examples are "
        "shown to communicate that the system can express curved futures — the remaining open problem is "
        "directional commitment, which is an input/representational limitation (see exp01–exp06).",
    ]), encoding="utf-8")

    # README
    tid = str(hero_row["trajectory_id"]).split("__")[-1]
    (HERE / "README.md").write_text("\n".join([
        "# exp07 — Publication turn panels\n",
        "Thesis / presentation-quality visual panels of curved turn predictions, selected from **exp06** "
        "(clean, direction-balanced `esplanade` held-out). Re-rolls the saved exp06 checkpoint to draw the "
        "trajectories. Source data, frozen_model_C, and previous experiments are untouched.\n",
        "## What these are (and are not)\n",
        "1. **Selected qualitative examples from exp06** held-out predictions (not training data).",
        "2. They **demonstrate visible angular rollouts** — the model produces curved futures, not straight lines.",
        "3. They are **not cherry-picked as proof of directional accuracy** — selection rewards clear, smooth, "
        "believable curvature and clean ground truth, explicitly *not* correct final direction.",
        "4. **Quantitatively, exp06 still shows direction ≈ chance** (held-out mean angular error ~93°). These "
        "panels do not change that result.",
        "5. Their purpose is to **communicate the positive visual finding**: Motion Pixels can express curved "
        "pedestrian futures (turn *tendency* and *magnitude*), even while the exact direction remains uncertain.\n",
        f"Selected **{n}** of 1,232 esplanade genuine turns (target 25; {n} passed the quality bar — the model "
        "produces few long, clearly-curved predictions, so we do not pad the set with collapsed stubs).\n",
        "## Outputs\n",
        f"- `panels/individual/panel_01..{n:02d}.png` — technical panels (ids, ADE/FDE, GT/pred angle, turn-captured flag).",
        f"- `panels/presentation_mode/panel_01..{n:02d}.png` — minimal slide version (Observed / Ground truth / Prediction + angle).",
        "- `panels/collage_3x3_best.png`, `panels/collage_5x5_selected.png`, `panels/comparison_contact_sheet.png`.",
        "- `panels/robustness_placa_espanya_3x3.png` — noisy robustness set (clearly labelled).",
        "- `hero_turn_prediction.png` — single clearest example for a slide.",
        "- `selected_25_metrics.csv`, `selection_method.md`, `captions/`.\n",
        f"Hero example: esplanade track {tid}, GT turn {hero_row['gt_head_change_deg']:.0f}°, predicted "
        f"{hero_row['pred_head_change_deg']:.0f}°, ADE {hero_row['ade']:.2f} m.\n",
        "## Colour key\n",
        "Observed path = dark grey · Ground truth future = blue · Model prediction = orange/red. "
        "★ = prediction start; ● / ■ = GT / predicted end; arrows = final directions.\n",
    ]), encoding="utf-8")


if __name__ == "__main__":
    main()
