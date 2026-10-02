"""
generate_plots_wassim.py
========================
Clean scientific / architectural re-plotting of selected pedestrian-prediction cases for
MODEL_XC_B_CURV_LIGHT only. White background, simple light axes, breathing space, no plan
overlay, no feet, no glow.

VISUALIZATION-ONLY. No training, retraining, loss tuning, dataset/encoder changes. The frozen
MODEL_XC_B_CURV_LIGHT checkpoint is *replayed* deterministically (same window construction as
presentation_visuals/make_presentation_visuals.py: OBS=10, stride=2, test split).

Per horizon (H20/H60/H100/H200/H400):
  - 3 BEST tracks: user-provided, matched to their window in selected_visuals_inventory.csv
  - 3 WORST tracks: auto-selected (highest ADE/RMSE, lowest R2) among readable, non-stationary,
    non-artifact windows, excluding the best tracks
  - individual plots + one 2x3 best/worst collage
Plus a cross-horizon accuracy summary (CSV + Markdown).

Output root: $MP_PLOTS_OUT (default: ./outputs next to this script)
Run:  python generate_plots_wassim.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Project wiring (reused read-only)
# --------------------------------------------------------------------------------------
import os
TP = Path(__file__).resolve().parents[2] / "mp-core" / "trajectory-prediction"
sys.path.insert(0, str(TP / "MODEL_XC"))
import xc_common as XCC          # noqa: E402  (pulls xr_common + model_x_lib)
L = XCC.L

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INVENTORY = TP / "presentation_visuals" / "selected_visuals_inventory.csv"
CKPT = TP / "MODEL_XC" / "checkpoints" / "MODEL_XC_B_CURV_LIGHT" / "model_best.pth"
SCALERS = TP / "MODEL_XC" / "checkpoints" / "MODEL_XC_B_CURV_LIGHT" / "scalers.json"

OUT_ROOT = Path(os.environ.get("MP_PLOTS_OUT", Path(__file__).resolve().parent / "outputs"))
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Case configuration
# --------------------------------------------------------------------------------------
HORIZONS = [20, 60, 100, 200, 400]
STRIDE = 2
MODEL_NAME = "MODEL_XC_B_CURV_LIGHT"

# user-provided best tracks (order preserved)
BEST_TRACKS = {
    20:  ["1261", "6268", "4394"],
    60:  ["4394", "1846", "6268"],
    100: ["4394", "4703", "1846"],
    200: ["4152", "3084", "1178"],
    400: ["112", "2039", "540"],
}

# minimum GT net displacement (m) to count a window as readable / non-stationary
MIN_GT_NET = {20: 1.0, 60: 2.0, 100: 3.0, 200: 5.0, 400: 8.0}

SITE = {
    "placa_espanya_01": "Plaça Espanya",
    "placa_catalunya_01": "Plaça Catalunya",
    "esplanade_espanya_01": "Esplanade Espanya",
    "stairs_montjuic_01": "Stairs Montjuïc",
    "red_bridge_combined_01": "Red Bridge",
}

# clean palette
C_HIST = "#444444"     # dark grey solid
C_GT = "#1f77b4"       # blue solid + markers
C_PRED = "#ff7f0e"     # orange dashed
C_SEP = "#000000"      # black separation point
C_GRID = "#cccccc"


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def load_xc_model():
    s = json.loads(Path(SCALERS).read_text())
    f_sc = L.ColumnScaler.from_dict(s["feature_scaler"])
    t_sc = L.ColumnScaler.from_dict(s["target_scaler"])
    m = L.TrajectoryLSTM(L.N_FEAT).to(DEVICE)
    m.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    m.eval()
    return m, f_sc, t_sc


# --------------------------------------------------------------------------------------
# Metrics (per window), computed over the H predicted future steps (anchor excluded)
# --------------------------------------------------------------------------------------
def per_window_metrics(pred, gt):
    """pred,gt : (M, H+1, 2) sharing pred[:,0]==gt[:,0] (separation point).
    Returns ADE (M,), RMSE (M,), R2 (M,)."""
    p = pred[:, 1:, :]
    g = gt[:, 1:, :]
    err = p - g                                   # (M,H,2)
    d = np.linalg.norm(err, axis=2)               # (M,H)
    ade = d.mean(axis=1)
    rmse = np.sqrt((d ** 2).mean(axis=1))
    ss_res = (err ** 2).sum(axis=(1, 2))
    gmean = g.mean(axis=1, keepdims=True)
    ss_tot = ((g - gmean) ** 2).sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / ss_tot
    r2 = np.where(ss_tot > 1e-9, r2, np.nan)
    return ade, rmse, r2


# --------------------------------------------------------------------------------------
# Rollout all test windows for a horizon (frozen XC replay)
# --------------------------------------------------------------------------------------
def rollout_horizon(df, split, wb, lens, model, f_sc, t_sc, H):
    need = XCC.OBS + H
    te_ids = [t for t in split[split.split == "test"].trajectory_id if lens.get(t, 0) >= need]
    seeds, start, gt, wba, meta = L.build_eval_windows(df, te_ids, wb, n_steps=H, stride=STRIDE)
    M = len(seeds)
    pred = np.empty((M, H + 1, 2))
    for i in range(0, M, 16384):
        sl = slice(i, i + 16384)
        pred[sl] = L.rollout_world_batch(
            model, seeds[sl], start[sl], {k: v[sl] for k, v in wba.items()},
            f_sc, t_sc, H, DEVICE)
    return seeds, start, gt, pred, meta


# --------------------------------------------------------------------------------------
# Plot one window on an axis (clean style)
# --------------------------------------------------------------------------------------
def pad_limits(pts, pad_frac=0.30, pad_min=1.0):
    xmin, ymin = pts.min(0); xmax, ymax = pts.max(0)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    xspan, yspan = xmax - xmin, ymax - ymin
    pad = max(pad_frac * max(xspan, yspan), pad_min)
    w = max(xspan + 2 * pad, 0.55 * (max(xspan, yspan) + 2 * pad))
    h = max(yspan + 2 * pad, 0.55 * (max(xspan, yspan) + 2 * pad))
    return (cx - w / 2, cx + w / 2), (cy - h / 2, cy + h / 2)


def draw_case(ax, hist, gt, pred, title):
    me = max(1, len(gt) // 25)                    # thin GT markers on long horizons
    ax.plot(hist[:, 0], hist[:, 1], "-", color=C_HIST, lw=1.5, label="history", zorder=3)
    ax.plot(gt[:, 0], gt[:, 1], "-o", color=C_GT, lw=1.4, ms=3.0, markevery=me,
            markerfacecolor=C_GT, markeredgecolor="none", label="GT future", zorder=4)
    ax.plot(pred[:, 0], pred[:, 1], "--", color=C_PRED, lw=1.6, label="prediction (XC)", zorder=5)
    ax.plot(gt[0, 0], gt[0, 1], "*", color=C_SEP, ms=9, label="start", zorder=6)

    allpts = np.vstack([hist, gt, pred])
    xl, yl = pad_limits(allpts)
    ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=C_GRID, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#bbbbbb"); s.set_linewidth(0.8)
    ax.tick_params(labelsize=7, color="#bbbbbb", length=3)
    ax.set_xlabel("x (m)", fontsize=7, color="#666666")
    ax.set_ylabel("y (m)", fontsize=7, color="#666666")
    ax.set_title(title, fontsize=8.5, loc="left", color="#222222")


def title_for(site, track, H, ade, rmse, r2):
    return f"{site}   ·   track {track}   ·   H{H}\nADE {ade:.2f} m   RMSE {rmse:.2f} m   R² {r2:.2f}"


# --------------------------------------------------------------------------------------
# Worst-window auto selection
# --------------------------------------------------------------------------------------
def select_worst(gt, pred, meta, ade, rmse, r2, H, exclude_tracks, n=3):
    rec = np.array([m["recording_id"] for m in meta])
    tid = np.array([str(m["trajectory_id"]).split("__")[-1] for m in meta])

    enr = XCC.enrich_full(pred, gt, rec, np.array([m["trajectory_id"] for m in meta]), H)
    gt_net = enr.nd_gt.to_numpy()
    gt_tort = enr.gt_tort.to_numpy()

    # GT per-step displacement — used to reject tracking artifacts (ID-switch teleports /
    # sustained drift), which are NOT genuine model failures.
    gstep = np.linalg.norm(np.diff(gt, axis=1), axis=2)        # (M,H)
    max_step = gstep.max(axis=1)
    mean_step = gstep.mean(axis=1)

    # readability / validity filters: no near-stationary, no jitter artifacts, no teleports,
    # plausible pedestrian pace, finite metrics
    valid = (gt_net >= MIN_GT_NET[H]) & (gt_tort < 3.5) \
        & (max_step <= 2.0) & (mean_step <= 0.75) \
        & np.isfinite(ade) & np.isfinite(rmse) & np.isfinite(r2)
    for t in exclude_tracks:
        valid &= (tid != t)
    idx = np.where(valid)[0]

    # badness: worst ADE + worst RMSE + worst (lowest) R2, min-max normalised within candidates
    def mm(a):
        lo, hi = np.nanmin(a), np.nanmax(a)
        return np.zeros_like(a) if hi - lo < 1e-9 else (a - lo) / (hi - lo)
    badness = mm(ade[idx]) + mm(rmse[idx]) + (1.0 - mm(r2[idx]))
    order = idx[np.argsort(-badness)]

    chosen, seen = [], set()
    for i in order:
        if tid[i] in seen:
            continue
        chosen.append(int(i)); seen.add(tid[i])
        if len(chosen) == n:
            break
    return chosen


# --------------------------------------------------------------------------------------
# Build observed-history path for a window
# --------------------------------------------------------------------------------------
def hist_path(df_by, meta_i):
    g = df_by[meta_i["trajectory_id"]]
    s = g.iloc[meta_i["start_idx"]: meta_i["start_idx"] + XCC.OBS]
    return np.column_stack([s.world_x.to_numpy(float), s.world_y.to_numpy(float)])


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    inv = pd.read_csv(INVENTORY, dtype={"track_id": str})
    df = XCC.load_df(); split = XCC.load_split(); wb = L.recording_world_bounds(df)
    lens = df.groupby("trajectory_id").size()
    model, f_sc, t_sc = load_xc_model()

    summary_rows = []
    for H in HORIZONS:
        Hdir = OUT_ROOT / f"H{H}"; Hdir.mkdir(parents=True, exist_ok=True)
        for old in Hdir.glob("*.png"):      # clear stale plots from previous selections
            old.unlink()
        seeds, start, gt, pred, meta = rollout_horizon(df, split, wb, lens, model, f_sc, t_sc, H)
        ade, rmse, r2 = per_window_metrics(pred, gt)
        tid_all = np.array([str(m["trajectory_id"]).split("__")[-1] for m in meta])

        # ---- best windows: user tracks -> inventory window_id (global build index) ----
        inv_H = inv[inv.horizon == f"H{H}"]
        best_idx = []
        for trk in BEST_TRACKS[H]:
            row = inv_H[inv_H.track_id == trk]
            if len(row) == 0:
                raise SystemExit(f"best track {trk} not found in inventory for H{H}")
            wid = int(row.iloc[0].window_id)
            assert tid_all[wid] == trk, (H, trk, wid, tid_all[wid])
            best_idx.append(wid)

        # ---- worst windows: auto ----
        worst_idx = select_worst(gt, pred, meta, ade, rmse, r2, H,
                                 exclude_tracks=set(BEST_TRACKS[H]), n=3)

        # df grouped for the tracks we plot (history reconstruction)
        plot_tids = {meta[i]["trajectory_id"] for i in best_idx + worst_idx}
        df_by = {t: g.sort_values("timestep")
                 for t, g in df[df.trajectory_id.isin(plot_tids)].groupby("trajectory_id")}

        def case(i):
            m = meta[i]
            site = SITE.get(m["recording_id"], m["recording_id"])
            trk = str(m["trajectory_id"]).split("__")[-1]
            return dict(i=i, site=site, track=trk, hist=hist_path(df_by, m),
                        gt=gt[i], pred=pred[i],
                        ade=float(ade[i]), rmse=float(rmse[i]), r2=float(r2[i]))

        best = [case(i) for i in best_idx]
        worst = [case(i) for i in worst_idx]

        # ---- individual plots ----
        for rank, c in enumerate(best, 1):
            fig, ax = plt.subplots(figsize=(6.0, 5.2))
            draw_case(ax, c["hist"], c["gt"], c["pred"],
                      title_for(c["site"], c["track"], H, c["ade"], c["rmse"], c["r2"]))
            ax.legend(fontsize=6.5, loc="best", framealpha=0.9, edgecolor="#dddddd")
            fig.tight_layout()
            fig.savefig(Hdir / f"best_{rank:02d}_track{c['track']}.png", dpi=200,
                        facecolor="white"); plt.close(fig)
        for rank, c in enumerate(worst, 1):
            fig, ax = plt.subplots(figsize=(6.0, 5.2))
            draw_case(ax, c["hist"], c["gt"], c["pred"],
                      title_for(c["site"], c["track"], H, c["ade"], c["rmse"], c["r2"]))
            ax.legend(fontsize=6.5, loc="best", framealpha=0.9, edgecolor="#dddddd")
            fig.tight_layout()
            fig.savefig(Hdir / f"worst_{rank:02d}_track{c['track']}.png", dpi=200,
                        facecolor="white"); plt.close(fig)

        # ---- 2x3 collage: top=best, bottom=worst ----
        fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.5))
        fig.patch.set_facecolor("white")
        for k, ax in enumerate(axes[0]):
            c = best[k]
            draw_case(ax, c["hist"], c["gt"], c["pred"],
                      title_for(c["site"], c["track"], H, c["ade"], c["rmse"], c["r2"]))
            if k == 0:
                ax.legend(fontsize=6.5, loc="best", framealpha=0.9, edgecolor="#dddddd")
        for k, ax in enumerate(axes[1]):
            c = worst[k]
            draw_case(ax, c["hist"], c["gt"], c["pred"],
                      title_for(c["site"], c["track"], H, c["ade"], c["rmse"], c["r2"]))
        fig.suptitle(f"MODEL_XC_B_CURV_LIGHT  —  H{H}   ·   top: best (user)   ·   bottom: worst (auto)",
                     fontsize=13, color="#222222")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(Hdir / f"H{H}_best_worst_collage.png", dpi=200, facecolor="white")
        plt.close(fig)

        # ---- summary aggregation over the selected 6 ----
        six = best + worst
        m_ade = float(np.mean([c["ade"] for c in six]))
        m_rmse = float(np.mean([c["rmse"] for c in six]))
        m_r2 = float(np.mean([c["r2"] for c in six]))
        summary_rows.append(dict(
            horizon=f"H{H}", mean_ade_m=round(m_ade, 4), mean_rmse_m=round(m_rmse, 4),
            mean_r2=round(m_r2, 4),
            best_tracks=";".join(c["track"] for c in best),
            worst_tracks=";".join(c["track"] for c in worst)))
        print(f"[H{H}] best={[c['track'] for c in best]} worst={[c['track'] for c in worst]}  "
              f"meanADE={m_ade:.3f} meanRMSE={m_rmse:.3f} meanR2={m_r2:.3f}  (M={len(meta)})")

    # ---- cross-horizon normalized accuracy score ----
    sdf = pd.DataFrame(summary_rows)

    def mmcol(a, invert):
        a = a.to_numpy(float); lo, hi = a.min(), a.max()
        n = np.full_like(a, 0.5) if hi - lo < 1e-9 else (a - lo) / (hi - lo)
        return 1.0 - n if invert else n
    ade_n = mmcol(sdf.mean_ade_m, invert=True)     # lower ADE -> higher
    rmse_n = mmcol(sdf.mean_rmse_m, invert=True)    # lower RMSE -> higher
    r2_n = mmcol(sdf.mean_r2, invert=False)         # higher R2 -> higher
    sdf["accuracy_score"] = np.round((ade_n + rmse_n + r2_n) / 3.0, 4)

    sdf = sdf[["horizon", "mean_ade_m", "mean_rmse_m", "mean_r2", "accuracy_score",
               "best_tracks", "worst_tracks"]]
    sdf.to_csv(OUT_ROOT / "summary_accuracy_scores.csv", index=False)

    def md_table(frame):
        cols = list(frame.columns)
        head = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = ["| " + " | ".join(str(v) for v in row) + " |"
                for row in frame.itertuples(index=False, name=None)]
        return "\n".join([head, sep, *rows])

    md = ["# MODEL_XC_B_CURV_LIGHT — Accuracy Summary (selected 6 tracks per horizon)", "",
          "Metrics computed over the H predicted future steps (separation point excluded).",
          "`accuracy_score` = mean of min-max-normalised, cross-horizon (ADE↓, RMSE↓, R²↑); 1 = best.",
          "", md_table(sdf), "",
          "Per-horizon means cover 3 user-provided best + 3 auto-selected worst tracks.",
          "Note: at long horizons the worst-case R² is large-negative (catastrophic divergence),",
          "so the 6-track mean R² is pulled down by the worst trio — this is expected."]
    (OUT_ROOT / "summary_accuracy_scores.md").write_text("\n".join(md), encoding="utf-8")

    print("\n", sdf.to_string(index=False))
    print(f"\n[ok] outputs -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
