"""
generate_ready_animations.py
============================
Batch of Motion Pixels trajectory animations (GIF + preview PNG) for a curated set of tracks,
grouped into best / worst / stress_test categories with category-specific prediction colours.

VISUALIZATION-ONLY. No training, retraining, loss/encoder/dataset/model changes. Trajectories and
predictions are recovered by deterministic replay of the frozen MODEL_XC_B_CURV_LIGHT rollout
(same machinery the template + plots_wassim + stress-test scripts already use). It reuses the
official template's drawing/animation code verbatim (square canvas, equal aspect, dotted grid,
quiet axes, black history, grey dotted GT, simultaneous GT+prediction reveal, moving arrowhead,
Horizon/Track annotation) and only swaps the prediction colour per category.

Outputs:
  ready_animations/best/        best_H{H}_track{t}.gif
  ready_animations/worst/       worst_H{H}_track{t}.gif
  ready_animations/stress_test/ stress_H{H}_track{t}.gif
  ready_animations/previews/    *_preview.png   (also mirrored next to each gif)
  ready_animations/READY_ANIMATIONS_REPORT.md
Run:  python generate_ready_animations.py
"""
from __future__ import annotations
import sys, shutil
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TEMPLATE_DIR = HERE.parent / "motion_pixels_animation_test"
sys.path.insert(0, str(TEMPLATE_DIR))
import generate_d3_style_animation as T   # official template (drawing + deterministic rollout)

L = T.L; XCC = T.XCC; OBS = XCC.OBS
DEVICE = T.DEVICE
plt = T.plt; manim = T.manim

# --------------------------------------------------------------------------------------
# Batch configuration
# --------------------------------------------------------------------------------------
COLORS = {"best": "#7C3AED", "worst": "#84CC16", "stress": "#2563EB"}   # purple / lime / blue
PRES = {"best", "worst"}                                                # presentation horizons
STRESS = {"stress"}

# (horizon, track_id) per category
BATCH = {
    "best": [(20, "1261"), (20, "6268"), (60, "4394"), (60, "6268"), (100, "4703"),
             (200, "3084"), (200, "1178"), (400, "112"), (400, "2039")],
    "worst": [(20, "1647"), (20, "3016"), (60, "1822"), (200, "6268"), (400, "540")],
    "stress": [(800, "2039"), (800, "1178"), (800, "540"),
               (1000, "1969"), (1000, "5350"),
               (1200, "750"), (1200, "3")],
}
PREFIX = {"best": "best", "worst": "worst", "stress": "stress"}
FOLDER = {"best": HERE / "best", "worst": HERE / "worst", "stress": HERE / "stress_test"}
PREVIEWS = HERE / "previews"
MIN_GT_NET = {20: 1.0, 60: 2.0, 100: 3.0, 200: 5.0, 400: 8.0}

# --------------------------------------------------------------------------------------
# Shared frozen state
# --------------------------------------------------------------------------------------
print("[init] loading data + frozen model …")
DF = XCC.load_df()
SPLIT = XCC.load_split()
WB = L.recording_world_bounds(DF)
LENS = DF.groupby("trajectory_id").size()
TEST = set(SPLIT[SPLIT.split == "test"].trajectory_id)
MODEL, F_SC, T_SC = T.load_xc_model()
INV = pd.read_csv(T.INVENTORY, dtype={"track_id": str})
_PRES_CACHE = {}        # H -> (seeds, start, gt, wba, meta)


def resolve_recording(track, need):
    """Unique TEST recording for this track id with >= need frames (longest if several)."""
    cands = [(rec, LENS.get(f"{rec}__{track}", 0)) for rec in XCC.RECS
             if f"{rec}__{track}" in TEST and LENS.get(f"{rec}__{track}", 0) >= need]
    if not cands:
        return None
    return max(cands, key=lambda c: c[1])[0]


def pres_windows(H):
    if H not in _PRES_CACHE:
        need = OBS + H
        te = [t for t in SPLIT[SPLIT.split == "test"].trajectory_id if LENS.get(t, 0) >= need]
        _PRES_CACHE[H] = L.build_eval_windows(DF, te, WB, n_steps=H, stride=2)
    return _PRES_CACHE[H]


def rollout(seeds, start, wba, H):
    return L.rollout_world_batch(MODEL, seeds, start, wba, F_SC, T_SC, H, DEVICE)


def hist_from(tid, start_idx):
    g = DF[DF.trajectory_id == tid].sort_values("timestep")
    s = g.iloc[start_idx:start_idx + OBS]
    return np.column_stack([s.world_x.to_numpy(float), s.world_y.to_numpy(float)])


def get_presentation(H, track):
    """Recover (hist, gt, pred, recording, method) for a presentation window.
    Inventory window if available (matches the published figures); otherwise the worst-ADE,
    non-artifact window of that track (for the worst category)."""
    rec = resolve_recording(track, OBS + H)
    if rec is None:
        return None
    tid = f"{rec}__{track}"
    seeds, start, gt, wba, meta = pres_windows(H)
    # inventory window?
    row = INV[(INV.horizon == f"H{H}") & (INV.track_id == track)]
    wid, method = None, None
    if len(row) and 0 <= int(row.iloc[0].window_id) < len(meta):
        cand = int(row.iloc[0].window_id)
        if meta[cand]["trajectory_id"] == tid:
            wid, method = cand, "inventory"
    if wid is None:                                   # worst-ADE window of this track
        idxs = np.array([i for i, m in enumerate(meta) if m["trajectory_id"] == tid])
        if len(idxs) == 0:
            return None
        pp = rollout(seeds[idxs], start[idxs], {k: v[idxs] for k, v in wba.items()}, H)
        g = gt[idxs]
        ade = np.linalg.norm(pp[:, 1:] - g[:, 1:], axis=2).mean(1)
        gt_net = np.linalg.norm(g[:, -1] - g[:, 0], axis=1)
        max_step = np.linalg.norm(np.diff(g, axis=1), axis=2).max(1)
        ok = (gt_net >= MIN_GT_NET[H]) & (max_step <= 2.0)
        pool = np.where(ok)[0] if ok.any() else np.arange(len(idxs))
        bl = pool[int(np.argmax(ade[pool]))]
        wid, method = int(idxs[bl]), "worst_ade"
    sl = slice(wid, wid + 1)
    pred = rollout(seeds[sl], start[sl], {k: v[sl] for k, v in wba.items()}, H)[0]
    hist = hist_from(meta[wid]["trajectory_id"], meta[wid]["start_idx"])
    return hist, gt[wid], pred, rec, method


def get_stress(H, track):
    """Recover (hist, gt, pred, recording, method) for a long-horizon extrapolation, seeded from
    the track's first 10 frames (start_idx=0). GT only where real data exists; pred extrapolates."""
    rec = resolve_recording(track, OBS + 2)
    if rec is None:
        return None
    g = DF[DF.trajectory_id == f"{rec}__{track}"].sort_values("timestep")
    feat = g[L.FEAT_COLS].to_numpy(np.float32)
    wxy = np.column_stack([g.world_x.to_numpy(float), g.world_y.to_numpy(float)])
    wb = WB[rec]
    wba = {k: np.array([wb[k]], float) for k in ("xmin", "xrng", "ymin", "yrng")}
    pred = rollout(feat[:OBS][None], wxy[OBS - 1][None], wba, H)[0]
    hist = wxy[:OBS]
    gt = wxy[OBS - 1: OBS - 1 + H + 1]                # real future only, clipped to H
    return hist, gt, pred, rec, "stress_seed0"


# --------------------------------------------------------------------------------------
# Animation (reuses the template drawing code; only prediction colour changes)
# --------------------------------------------------------------------------------------
def animate(hist, gt, pred, H, track, color, out_gif, out_png):
    T.CFG["c_pred"] = color
    T.CFG["horizon"] = H
    T.CFG["track"] = str(track)
    fig, ax = T.build_axes(hist, gt, pred)
    art = T.make_artists(ax)
    art["annot"].set_text(f"Horizon: H{H}\nTrack {track}")     # only template annotation; no metrics
    plan = T.build_timeline()

    def frame(i):
        name, frac = plan[i]
        if name == "empty":
            for k in ("hist", "gt", "pred"):
                art[k].set_data([], [])
            art["sep"].set_data([], []); art["metric"].set_text("")
        T.render(art, hist, gt, pred, name, frac, metric_text="")   # metrics omitted
        return tuple(art[k] for k in ("hist", "gt", "pred", "sep", "arrow", "metric"))

    anim = manim.FuncAnimation(fig, frame, frames=len(plan), interval=1000 / T.CFG["fps"], blit=False)
    anim.save(out_gif, writer="pillow", fps=T.CFG["fps"])
    # final-frame preview
    for k in ("hist", "gt", "pred"):
        art[k].set_data([], [])
    T.render(art, hist, gt, pred, "hold", 1.0, metric_text="")
    fig.savefig(out_png, dpi=170, facecolor=T.CFG["page_bg"]); plt.close(fig)


# --------------------------------------------------------------------------------------
# Main batch
# --------------------------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for d in (*FOLDER.values(), PREVIEWS):
        d.mkdir(parents=True, exist_ok=True)
    dur = len(T.build_timeline()) / T.CFG["fps"]
    done, missing, log = [], [], []

    for cat, items in BATCH.items():
        color = COLORS[cat]
        for H, track in items:
            case = get_stress(H, track) if cat == "stress" else get_presentation(H, track)
            if case is None:
                missing.append((cat, H, track)); print(f"[miss] {cat} H{H} track {track}")
                continue
            hist, gt, pred, rec, method = case
            stem = f"{PREFIX[cat]}_H{H}_track{track}"
            gif = FOLDER[cat] / f"{stem}.gif"
            png_cat = FOLDER[cat] / f"{stem}_preview.png"
            animate(hist, gt, pred, H, track, color, gif, png_cat)
            shutil.copy(png_cat, PREVIEWS / f"{stem}_preview.png")
            done.append((cat, H, track, rec, method, gif))
            log.append(dict(category=cat, horizon=f"H{H}", track=track, recording=rec,
                            window=method, gif=str(gif.relative_to(HERE)),
                            gt_pts=len(gt), pred_pts=len(pred)))
            print(f"[ok] {stem}  ({rec}, {method})  gt={len(gt)} pred={len(pred)}")

    write_report(done, missing, log, dur)
    print(f"\n[done] {len(done)} animations · {len(done)} previews · {len(missing)} missing")
    print(f"[dir]  {HERE}")


def write_report(done, missing, log, dur):
    def block(cat):
        rows = [r for r in log if r["category"] == cat]
        lines = [f"- **H{r['horizon'][1:]} · track {r['track']}** — {r['recording']} "
                 f"(window: {r['window']}; GT {r['gt_pts']} pts, pred {r['pred_pts']} pts)" for r in rows]
        return "\n".join(lines) if lines else "_(none)_"

    md = f"""# Ready Animations — Batch Report

**Total animations generated:** {len(done)}  ·  **previews:** {len(done)}  ·  **missing:** {len(missing)}
**Each:** square canvas, equal aspect, dotted grid, quiet axes, black history, grey dotted GT,
moving prediction arrowhead, simultaneous GT+prediction reveal, ~{dur:.1f}s @ {T.CFG['fps']} fps,
GIF + final-frame preview PNG. Annotation = `Horizon: H{{h}}` / `Track {{id}}` only (metrics omitted).

> **No training or model modification occurred.** Trajectories + predictions are a deterministic
> replay of the frozen `MODEL_XC_B_CURV_LIGHT` rollout (the same machinery used by the template,
> `generate_plots_wassim.py`, and `generate_long_horizon_stress_test.py`). No new models, losses,
> datasets, or encoders were touched.

## Colour mapping (prediction line + arrowhead)
| Category | Colour |
|---|---|
| best | `#7C3AED` purple |
| worst | `#84CC16` lime green |
| stress_test | `#2563EB` blue |
History `#111111`, GT dotted `#666666`, seed dot `#111111`, grid `#e4e4e4`, axes `#cfcfcf` — unchanged.

## Best animations (purple) — {len([d for d in done if d[0]=='best'])}
{block('best')}

## Worst animations (lime green) — {len([d for d in done if d[0]=='worst'])}
{block('worst')}

## Stress-test animations (blue) — {len([d for d in done if d[0]=='stress'])}
{block('stress')}

## Source files used
- Trajectory machinery / deterministic rollout: `mp-visualization/motion_pixels_animation_test/generate_d3_style_animation.py`
  (template) → `MODEL_XC/xc_common.py` → `xr_common.py` → `model_x_lib.py`
- Master dataset (via `model_x_lib.CONFIG`): `…/Barcelona_v3_manual_master_dataset/master_dataset.csv`
- Test split: `…/experiments/MODEL_X/splits/model_x_track_split.csv`
- Frozen checkpoint: `…/MODEL_XC/checkpoints/MODEL_XC_B_CURV_LIGHT/model_best.pth`
- Best-window lookup: `…/presentation_visuals/selected_visuals_inventory.csv`

## Window-selection method
- **Presentation (best/worst, H20–H400):** if the (horizon, track) exists in the inventory, its
  published `window_id` is used (matches the Plots_Wassim figures). Otherwise — for worst tracks not
  in the inventory — the **worst-ADE, non-artifact window** of that track is chosen deterministically
  (filters: GT net ≥ horizon minimum, max GT per-step ≤ 2.0 m to exclude tracking teleports).
- **Stress (H800–H1200):** each track is seeded from its first 10 frames (start_idx = 0) and the
  frozen model is rolled forward H steps — one continuous blue prediction line, no validated/
  extrapolated split. GT is drawn only where real future data exists (never invented beyond the
  recorded track). These are **exploratory extrapolations beyond the validated range** (≤ H400);
  the context lives here in the report, not as on-frame text.

## Assumptions
- **Stress horizons:** the brief listed "H100: 1969, 5350" but, per the note and the stress-test
  horizons, these were interpreted as **H1000**. Final stress horizons: H800, H1000, H1200.
- **Recordings** were resolved automatically as the unique TEST-split recording for each track id
  with sufficient length (e.g. worst 1647/3016/6268 → placa_espanya, 1822 → esplanade_espanya;
  stress 2039/1178/1969 → red_bridge, 540 → stairs_montjuic, 5350 → placa_catalunya, 750 →
  esplanade_espanya, 3 → placa_espanya).
- Track 540 appears here under *worst* (H400) per the brief, although it is also a curated *best*
  H400 track; it is rendered in lime green from its inventory window as requested.
- Metrics caption omitted (only the template Horizon/Track annotation is shown), per the brief's
  "no extra text beyond the template annotation".

## Missing / failed tracks
{("None — all requested tracks rendered." if not missing else chr(10).join(f'- {c} H{h} track {t}' for c,h,t in missing))}
"""
    (HERE / "READY_ANIMATIONS_REPORT.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
