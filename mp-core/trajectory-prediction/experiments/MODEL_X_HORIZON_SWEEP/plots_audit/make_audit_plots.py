"""
make_audit_plots.py — PHASE 1 plotting audit for the MODEL_X horizon sweep.

NO retraining, NO weight changes, NO dataset changes. This only re-runs inference
(deterministic, model.eval()) on selected test windows to regenerate trajectory plots
with (a) guaranteed equal metric scaling [ax.set_aspect("equal", adjustable="box")]
and (b) full visible metadata.

For each horizon (H20,H60,H100,H200,H400) and each category:
  - longest_gt      : largest GT path length
  - largest_pred    : largest predicted path length
  - median_ade      : windows nearest the median ADE
  - worst_collapse  : smallest pred/GT length ratio among non-trivial GT (>= median GT len)
Selection uses the authoritative per_window_metrics.csv (window_idx -> trajectory_id,start_idx),
so it does NOT cherry-pick lowest ADE.

Outputs per category: individual annotated panels + a contact-sheet grid.
Writes plot_inventory.csv across all horizons.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent          # .../MODEL_X_HORIZON_SWEEP/plots_audit
SWEEP = HERE.parent                              # .../MODEL_X_HORIZON_SWEEP
MODELX = SWEEP.parent / "MODEL_X"                # .../experiments/MODEL_X
sys.path.insert(0, str(MODELX))
import model_x_lib as L  # noqa: E402

HORIZONS = [20, 60, 100, 200, 400]
APPROX_DIST = {20: 1.0, 60: 3.0, 100: 5.0, 200: 10.0, 400: 20.0}
N_PER = 9
CATEGORIES = ["longest_gt", "largest_pred", "median_ade", "worst_collapse"]
CSV = L.CONFIG["dataset"]["csv_path"]
OBS = L.WINDOW_SIZE

_warnings: list[str] = []


def select(metr: pd.DataFrame, cat: str) -> pd.DataFrame:
    if cat == "longest_gt":
        return metr.nlargest(N_PER, "gt_path_len")
    if cat == "largest_pred":
        return metr.nlargest(N_PER, "pred_path_len")
    if cat == "median_ade":
        med = metr.ade.median()
        return metr.iloc[(metr.ade - med).abs().argsort()[:N_PER]]
    if cat == "worst_collapse":
        floor = metr.gt_path_len.median()
        pool = metr[metr.gt_path_len >= floor]
        if len(pool) < N_PER:
            pool = metr
        return pool.nsmallest(N_PER, "len_ratio")
    raise ValueError(cat)


def reconstruct(df_by_tid, tid, start_idx, H):
    """Build (seed_feat (10,F), start_world (2,), gt_path (H+1,2), obs_path (10,2))."""
    g = df_by_tid[tid]
    seg = g.iloc[start_idx: start_idx + OBS]
    seed = seg[L.FEAT_COLS].to_numpy(np.float32)
    obs = np.column_stack([seg.world_x.to_numpy(float), seg.world_y.to_numpy(float)])
    a = start_idx + OBS - 1
    fut = g.iloc[a: a + H + 1]
    gt = np.column_stack([fut.world_x.to_numpy(float), fut.world_y.to_numpy(float)])
    start_world = gt[0].copy()
    return seed, start_world, gt, obs


def panel(ax, obs, gt, pred, meta):
    ax.plot(obs[:, 0], obs[:, 1], "-o", color="black", ms=2.5, lw=1.3, label="observed")
    ax.plot(gt[:, 0], gt[:, 1], "-o", color="#7f7f7f", ms=2, lw=2.0, label="GT future")
    ax.plot(pred[:, 0], pred[:, 1], "--o", color="#e0218a", ms=2, lw=1.6, label="prediction")
    ax.plot(obs[-1, 0], obs[-1, 1], "o", color="#1f77b4", ms=6, label="current")
    # EQUAL METRIC SCALING (box) — preserves geometry
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=6); ax.grid(alpha=.25, lw=.4)
    txt = (f"{meta['recording']}  track {meta['track_short']}  win {meta['window_id']}\n"
           f"H{meta['H']} (~{meta['dist']}m)  ADE {meta['ade']:.2f}  FDE {meta['fde']:.2f}\n"
           f"GTlen {meta['gt_len']:.2f}m  predlen {meta['pred_len']:.2f}m  "
           f"ratio {meta['ratio']:.2f}")
    ax.set_title(txt, fontsize=6.5, loc="left")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[audit] device={device}")
    cols = ["recording_id", "trajectory_id", "timestep", "world_x", "world_y"] + L.FEAT_COLS
    df = pd.read_csv(CSV, usecols=lambda c: c in set(cols))
    wb = L.recording_world_bounds(df)

    inv_rows = []
    counts = {}
    for H in HORIZONS:
        Hdir = SWEEP / f"H{H}"
        mfile = Hdir / "per_window_metrics.csv"
        ckpt = Hdir / "model_best.pth"
        scl = Hdir / "scalers.json"
        if not (mfile.exists() and ckpt.exists() and scl.exists()):
            _warnings.append(f"H{H}: missing {[p.name for p in (mfile,ckpt,scl) if not p.exists()]}")
            print(f"[audit] SKIP H{H} (missing files)"); continue

        import json
        sc = json.loads(scl.read_text())
        f_sc = L.ColumnScaler.from_dict(sc["feature_scaler"])
        t_sc = L.ColumnScaler.from_dict(sc["target_scaler"])
        model = L.TrajectoryLSTM(L.N_FEAT).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()
        metr = pd.read_csv(mfile)

        needed_tids = set()
        picks = {}
        for cat in CATEGORIES:
            sel = select(metr, cat)
            picks[cat] = sel
            needed_tids.update(sel.trajectory_id.tolist())
        df_by_tid = {t: g.sort_values("timestep")
                     for t, g in df[df.trajectory_id.isin(needed_tids)].groupby("trajectory_id")}

        for cat in CATEGORIES:
            sel = picks[cat]
            outdir = HERE / f"H{H}" / cat
            seeds, starts, gts, obss, metas = [], [], [], [], []
            wbk = {"xmin": [], "xrng": [], "ymin": [], "yrng": []}
            for _, r in sel.iterrows():
                tid = r.trajectory_id
                seed, sw, gt, obs = reconstruct(df_by_tid, tid, int(r.start_idx), H)
                rec = r.recording_id
                seeds.append(seed); starts.append(sw); gts.append(gt); obss.append(obs)
                for k in wbk:
                    wbk[k].append(wb[rec][k])
                metas.append({"recording": rec, "track_short": str(tid).split("__")[-1],
                              "window_id": int(r.window_idx), "H": H, "dist": APPROX_DIST[H],
                              "ade": float(r.ade), "fde": float(r.fde),
                              "gt_len": float(r.gt_path_len), "pred_len": float(r.pred_path_len),
                              "ratio": float(r.len_ratio), "trajectory_id": tid})
            if not seeds:
                _warnings.append(f"H{H}/{cat}: no windows selected"); continue
            seeds = np.asarray(seeds, np.float32)
            starts = np.asarray(starts, np.float64)
            wb_arr = {k: np.asarray(v, np.float64) for k, v in wbk.items()}
            preds = L.rollout_world_batch(model, seeds, starts, wb_arr, f_sc, t_sc, H, device)

            # contact-sheet grid
            ncol = 3; nrow = int(np.ceil(len(metas) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(13, 4.2 * nrow))
            axes = np.atleast_1d(axes).flat
            for i, ax in enumerate(axes):
                if i < len(metas):
                    panel(ax, obss[i], gts[i], preds[i], metas[i])
                    if i == 0:
                        ax.legend(fontsize=6, loc="best")
                else:
                    ax.axis("off")
            fig.suptitle(f"MODEL_X H{H} (~{APPROX_DIST[H]} m) — {cat}  "
                         f"[equal metric scaling, adjustable=box]", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            grid_path = outdir / f"_grid_{cat}.png"
            fig.savefig(grid_path, dpi=140); plt.close(fig)

            # individual annotated panels (unique plot_path per inventory row)
            for i, m in enumerate(metas):
                fig, ax = plt.subplots(figsize=(5.2, 5.2))
                panel(ax, obss[i], gts[i], preds[i], m)
                ax.legend(fontsize=7, loc="best")
                fig.tight_layout()
                pth = outdir / (f"{cat}_{i:02d}_{m['recording']}_t{m['track_short']}"
                                f"_w{m['window_id']}.png")
                fig.savefig(pth, dpi=130); plt.close(fig)
                inv_rows.append({"horizon": f"H{H}", "category": cat,
                                 "recording": m["recording"], "track_id": m["track_short"],
                                 "window_id": m["window_id"], "ADE": round(m["ade"], 4),
                                 "FDE": round(m["fde"], 4), "gt_length": round(m["gt_len"], 4),
                                 "pred_length": round(m["pred_len"], 4),
                                 "pred_gt_ratio": round(m["ratio"], 4),
                                 "plot_path": str(pth.relative_to(SWEEP))})
            counts[(H, cat)] = len(metas)
            print(f"[audit] H{H}/{cat}: {len(metas)} panels")

    inv = pd.DataFrame(inv_rows)
    inv.to_csv(HERE / "plot_inventory.csv", index=False)
    print(f"[audit] inventory rows: {len(inv)} -> plot_inventory.csv")
    if _warnings:
        print("[audit] WARNINGS:")
        for w in _warnings:
            print("  -", w)
    # persist counts + warnings for the report
    import json
    (HERE / "_audit_meta.json").write_text(json.dumps(
        {"counts": {f"H{h}/{c}": n for (h, c), n in counts.items()},
         "warnings": _warnings, "total_plots": len(inv)}, indent=2))


if __name__ == "__main__":
    main()
