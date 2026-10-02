"""
build_recording_audit.py — PHASE 2 per-recording audit of the MODEL_X horizon sweep.

NO retraining / NO weight changes / NO dataset changes. Re-runs deterministic inference
(model.eval()) over ALL test windows of each horizon to compute jitter-aware length metrics
that the stored per_window_metrics.csv does not contain (predicted net displacement, smoothed
path lengths, cosine similarity of net-displacement vectors).

Per Phase-1 finding (cumulative GT length inflated by tracking jitter), every length question
is answered three ways: cumulative / net-displacement / smoothed (moving-avg) path length.

Outputs (all inside recording_audit/):
  per_recording_horizon_metrics.csv
  dataset_composition_by_recording.csv
  red_bridge_vs_others.csv
  recording_audit_window_inventory.csv
  figures/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent           # .../MODEL_X_HORIZON_SWEEP/recording_audit
SWEEP = HERE.parent
MODELX = SWEEP.parent / "MODEL_X"
FIG = HERE / "figures"
sys.path.insert(0, str(MODELX))
import model_x_lib as L  # noqa: E402

HORIZONS = [20, 60, 100, 200, 400]
APPROX = {20: 1.0, 60: 3.0, 100: 5.0, 200: 10.0, 400: 20.0}
CSV = L.CONFIG["dataset"]["csv_path"]
OBS = L.WINDOW_SIZE
CHUNK = 16384
SMOOTH_W = 5            # moving-average window (frames) for jitter-smoothed length
MOVE_EPS = 0.5         # m; "moving" window = GT net displacement > this (length-ratio subset)
PRED_DIR_EPS = 0.05    # m; below this predicted net disp has no defined direction
RED = "red_bridge_combined_01"

# Red-Bridge thresholds (documented): relative difference vs non-RB mean
REL_TOL = 0.15         # within +/-15% = "similar"
SHARE_TOL = 0.15       # window-share within +/-15% of mean non-RB share = "similar"


def smooth_paths(a: np.ndarray, w: int = SMOOTH_W) -> np.ndarray:
    """Centered moving average along axis=1 (edge-padded), fully vectorised. a:(M,P,2)."""
    M, P, D = a.shape
    if P < 3:
        return a.copy()
    if w % 2 == 0:
        w += 1
    w = min(w, P if P % 2 == 1 else P - 1)
    pad = w // 2
    ap = np.concatenate([np.repeat(a[:, :1], pad, axis=1), a,
                         np.repeat(a[:, -1:], pad, axis=1)], axis=1)
    cs = np.cumsum(ap, axis=1)
    cs = np.concatenate([np.zeros((M, 1, D)), cs], axis=1)
    return (cs[:, w:] - cs[:, :-w]) / w


def path_len(p: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(p, axis=1), axis=2).sum(axis=1)


def net_disp(p: np.ndarray) -> np.ndarray:
    return np.linalg.norm(p[:, -1] - p[:, 0], axis=1)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[audit2] device={device}")
    cols = ["recording_id", "trajectory_id", "timestep", "world_x", "world_y"] + L.FEAT_COLS
    df = pd.read_csv(CSV, usecols=lambda c: c in set(cols))
    wb = L.recording_world_bounds(df)
    Lens = df.groupby("trajectory_id").size()
    split = pd.read_csv(MODELX / "splits" / "model_x_track_split.csv")

    recs = sorted(df.recording_id.unique())
    win_rows = []          # enriched per-window inventory across horizons
    agg_rows = []          # per (horizon, recording)

    for H in HORIZONS:
        Hdir = SWEEP / f"H{H}"
        ckpt, scl = Hdir / "model_best.pth", Hdir / "scalers.json"
        if not (ckpt.exists() and scl.exists()):
            print(f"[audit2] SKIP H{H} (missing model/scalers)"); continue
        sc = json.loads(scl.read_text())
        f_sc = L.ColumnScaler.from_dict(sc["feature_scaler"])
        t_sc = L.ColumnScaler.from_dict(sc["target_scaler"])
        model = L.TrajectoryLSTM(L.N_FEAT).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()

        need = OBS + H
        te_ids = [t for t in split[split.split == "test"].trajectory_id if Lens.get(t, 0) >= need]
        seeds, start, gt, wba, meta = L.build_eval_windows(df, te_ids, wb, n_steps=H, stride=1)
        M = len(seeds)
        preds = np.empty((M, H + 1, 2))
        for i in range(0, M, CHUNK):
            sl = slice(i, i + CHUNK)
            preds[sl] = L.rollout_world_batch(model, seeds[sl], start[sl],
                                              {k: v[sl] for k, v in wba.items()},
                                              f_sc, t_sc, H, device)
        # ── vectorised metrics ──
        disp = np.linalg.norm(preds - gt, axis=2)              # (M,H+1)
        ade = disp[:, 1:].mean(axis=1); fde = disp[:, -1]
        cum_gt, cum_pred = path_len(gt), path_len(preds)
        nd_gt, nd_pred = net_disp(gt), net_disp(preds)
        sm_gt, sm_pred = path_len(smooth_paths(gt)), path_len(smooth_paths(preds))
        cum_ratio = cum_pred / np.maximum(cum_gt, 1e-9)
        nd_ratio = nd_pred / np.maximum(nd_gt, 1e-9)
        sm_ratio = sm_pred / np.maximum(sm_gt, 1e-9)
        # cosine of net-displacement vectors (GT vs pred)
        v_gt = gt[:, -1] - gt[:, 0]; v_pr = preds[:, -1] - preds[:, 0]
        n_gt = np.linalg.norm(v_gt, axis=1); n_pr = np.linalg.norm(v_pr, axis=1)
        dot = (v_gt * v_pr).sum(axis=1)
        cos = np.where((n_gt > MOVE_EPS) & (n_pr > PRED_DIR_EPS),
                       dot / np.maximum(n_gt * n_pr, 1e-9), np.nan)
        head_err = np.degrees(np.arccos(np.clip(cos, -1, 1)))   # net-disp heading error

        rec_arr = np.array([m["recording_id"] for m in meta])
        tid_arr = np.array([m["trajectory_id"] for m in meta])
        wdf = pd.DataFrame({
            "horizon": H, "recording": rec_arr, "trajectory_id": tid_arr,
            "window_id": np.arange(M),
            "ade": ade, "fde": fde,
            "cum_gt": cum_gt, "cum_pred": cum_pred, "cum_ratio": cum_ratio,
            "nd_gt": nd_gt, "nd_pred": nd_pred, "nd_ratio": nd_ratio,
            "sm_gt": sm_gt, "sm_pred": sm_pred, "sm_ratio": sm_ratio,
            "cosine": cos, "heading_err_deg": head_err,
            "moving": nd_gt > MOVE_EPS,
        })
        win_rows.append(wdf)
        print(f"[audit2] H{H}: {M} windows, {wdf.recording.nunique()} recordings")

        total_w = M
        for rec in recs:
            r = wdf[wdf.recording == rec]
            if len(r) == 0:
                continue
            mv = r[r.moving]                  # moving subset for length ratios/collapse
            cv = r.dropna(subset=["cosine"])  # valid-direction subset
            def md(s): return float(s.median()) if len(s) else float("nan")
            def mn(s): return float(s.mean()) if len(s) else float("nan")
            agg_rows.append({
                "horizon": H, "approx_dist_m": APPROX[H], "recording": rec,
                "n_windows": int(len(r)), "n_tracks": int(r.trajectory_id.nunique()),
                "pct_of_horizon_windows": round(100 * len(r) / total_w, 2),
                "ade_mean": mn(r.ade), "ade_med": md(r.ade),
                "fde_mean": mn(r.fde), "fde_med": md(r.fde),
                "cum_gt_med": md(r.cum_gt), "cum_pred_med": md(r.cum_pred),
                "cum_ratio_mean": mn(mv.cum_ratio), "cum_ratio_med": md(mv.cum_ratio),
                "nd_gt_med": md(r.nd_gt), "nd_pred_med": md(r.nd_pred),
                "nd_ratio_mean": mn(mv.nd_ratio), "nd_ratio_med": md(mv.nd_ratio),
                "sm_gt_med": md(r.sm_gt), "sm_pred_med": md(r.sm_pred),
                "sm_ratio_mean": mn(mv.sm_ratio), "sm_ratio_med": md(mv.sm_ratio),
                "cosine_mean": mn(cv.cosine), "cosine_med": md(cv.cosine),
                "heading_err_med_deg": md(cv.heading_err_deg),
                "pct_cos_gt_075": round(100 * (cv.cosine > 0.75).mean(), 2) if len(cv) else float("nan"),
                "pct_cos_gt_050": round(100 * (cv.cosine > 0.50).mean(), 2) if len(cv) else float("nan"),
                "collapse_cum_lt05": round(100 * (mv.cum_ratio < 0.5).mean(), 2) if len(mv) else float("nan"),
                "collapse_nd_lt05": round(100 * (mv.nd_ratio < 0.5).mean(), 2) if len(mv) else float("nan"),
                "collapse_sm_lt05": round(100 * (mv.sm_ratio < 0.5).mean(), 2) if len(mv) else float("nan"),
                "n_moving": int(len(mv)),
            })

    inv = pd.concat(win_rows, ignore_index=True)
    inv.to_csv(HERE / "recording_audit_window_inventory.csv", index=False)
    per = pd.DataFrame(agg_rows)
    per.to_csv(HERE / "per_recording_horizon_metrics.csv", index=False)

    # ── dataset composition (full master) ──
    comp = []
    tot_rows = len(df); tot_tracks = df.trajectory_id.nunique()
    tot_win = len(inv)
    for rec in recs:
        g = df[df.recording_id == rec]
        rw = inv[inv.recording == rec]
        comp.append({"recording": rec,
                     "rows": int(len(g)), "pct_rows": round(100*len(g)/tot_rows, 2),
                     "tracks": int(g.trajectory_id.nunique()),
                     "pct_tracks": round(100*g.trajectory_id.nunique()/tot_tracks, 2),
                     "total_rollout_windows_all_horizons": int(len(rw)),
                     "pct_rollout_windows": round(100*len(rw)/tot_win, 2)})
    comp = pd.DataFrame(comp)
    comp.to_csv(HERE / "dataset_composition_by_recording.csv", index=False)

    # ── Red Bridge vs others ──
    rb_metrics = {  # metric_key: (column, higher_is_better)
        "ADE_median": ("ade_med", False), "FDE_median": ("fde_med", False),
        "cumulative_pred_gt_median": ("cum_ratio_med", True),
        "net_disp_pred_gt_median": ("nd_ratio_med", True),
        "smoothed_pred_gt_median": ("sm_ratio_med", True),
        "cosine_similarity_median": ("cosine_med", True),
        "collapse_rate_net_disp_lt05": ("collapse_nd_lt05", False),
        "pct_of_horizon_windows": ("pct_of_horizon_windows", None),
    }
    rb_rows = []
    for H in HORIZONS:
        sub = per[per.horizon == H]
        if len(sub) == 0:
            continue
        rb = sub[sub.recording == RED]
        others = sub[sub.recording != RED]
        if len(rb) == 0:
            continue
        rb = rb.iloc[0]
        for mname, (col, hib) in rb_metrics.items():
            rbv = float(rb[col]); nonrb = float(others[col].mean())
            diff = rbv - nonrb
            if mname == "pct_of_horizon_windows":
                share_tol = SHARE_TOL * nonrb
                interp = ("overrepresented" if rbv > nonrb + share_tol
                          else "underrepresented" if rbv < nonrb - share_tol else "similar")
            else:
                rel = diff / nonrb if nonrb else 0.0
                if abs(rel) <= REL_TOL:
                    interp = "similar"
                elif hib:   # higher better
                    interp = "better_than_average" if diff > 0 else "worse_than_average"
                else:       # lower better
                    interp = "better_than_average" if diff < 0 else "worse_than_average"
            rb_rows.append({"horizon": f"H{H}", "metric": mname,
                            "red_bridge_value": round(rbv, 4),
                            "non_red_bridge_mean": round(nonrb, 4),
                            "difference": round(diff, 4), "interpretation": interp})
    rbdf = pd.DataFrame(rb_rows)
    rbdf.to_csv(HERE / "red_bridge_vs_others.csv", index=False)

    # ── figures ──
    FIG.mkdir(exist_ok=True)
    Hx = HORIZONS
    colors = {r: c for r, c in zip(recs, plt.cm.tab10(np.linspace(0, 1, len(recs))))}

    def line_fig(col, title, ylab, fname, hline=None):
        fig, ax = plt.subplots(figsize=(8, 5))
        for rec in recs:
            ys = [per[(per.horizon == H) & (per.recording == rec)][col].values[0]
                  if len(per[(per.horizon == H) & (per.recording == rec)]) else np.nan for H in Hx]
            ax.plot(Hx, ys, "-o", color=colors[rec], label=rec, lw=1.6, ms=4)
        if hline is not None:
            ax.axhline(hline, ls="--", c="k", alpha=.4)
        ax.set_xlabel("horizon (steps)"); ax.set_ylabel(ylab); ax.set_title(title)
        ax.set_xticks(Hx); ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIG / fname, dpi=130); plt.close(fig)

    line_fig("ade_med", "Per-recording ADE (median) by horizon", "ADE median (m)",
             "per_recording_ADE_by_horizon.png")
    line_fig("fde_med", "Per-recording FDE (median) by horizon", "FDE median (m)",
             "per_recording_FDE_by_horizon.png")
    line_fig("cum_ratio_med", "Per-recording CUMULATIVE pred/GT ratio (median, moving windows)",
             "cum pred/GT", "per_recording_cumulative_ratio_by_horizon.png", hline=1.0)
    line_fig("nd_ratio_med", "Per-recording NET-DISPLACEMENT pred/GT ratio (median, moving windows)",
             "net-disp pred/GT", "per_recording_net_displacement_ratio_by_horizon.png", hline=1.0)
    line_fig("sm_ratio_med", "Per-recording SMOOTHED pred/GT ratio (median, moving windows)",
             "smoothed pred/GT", "per_recording_smoothed_ratio_by_horizon.png", hline=1.0)
    line_fig("cosine_med", "Per-recording cosine similarity (median) GT vs pred net-disp",
             "cosine similarity", "per_recording_cosine_similarity_by_horizon.png", hline=0.0)
    line_fig("collapse_nd_lt05", "Collapse rate (net-disp ratio < 0.5) by recording & horizon",
             "% windows collapsed", "collapse_rate_by_recording_and_horizon.png")

    # window composition stacked bar
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(Hx))
    for rec in recs:
        vals = [per[(per.horizon == H) & (per.recording == rec)]["pct_of_horizon_windows"].values[0]
                if len(per[(per.horizon == H) & (per.recording == rec)]) else 0 for H in Hx]
        ax.bar([str(h) for h in Hx], vals, bottom=bottom, label=rec, color=colors[rec])
        bottom += np.array(vals)
    ax.set_xlabel("horizon"); ax.set_ylabel("% of horizon test windows")
    ax.set_title("Test-window composition by recording (per horizon)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(FIG / "horizon_window_composition_by_recording.png", dpi=130); plt.close(fig)

    # red bridge vs others summary (3 ratio types + cosine, median over horizons)
    fig, ax = plt.subplots(figsize=(9, 5))
    summ_metrics = [("cum_ratio_med", "cum"), ("nd_ratio_med", "net-disp"),
                    ("sm_ratio_med", "smoothed"), ("cosine_med", "cosine")]
    x = np.arange(len(summ_metrics)); w = 0.38
    rb_vals = [per[(per.recording == RED)][c].mean() for c, _ in summ_metrics]
    ot_vals = [per[(per.recording != RED)][c].mean() for c, _ in summ_metrics]
    ax.bar(x - w/2, rb_vals, w, label="Red Bridge (mean over horizons)", color="#d62728")
    ax.bar(x + w/2, ot_vals, w, label="non-Red-Bridge mean", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in summ_metrics])
    ax.axhline(1.0, ls="--", c="k", alpha=.4); ax.set_ylabel("ratio / cosine")
    ax.set_title("Red Bridge vs others — length-ratio & direction summary")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG / "red_bridge_vs_others_summary.png", dpi=130); plt.close(fig)

    print("[audit2] CSVs + figures written.")
    print(per.groupby("recording")[["ade_med", "cum_ratio_med", "nd_ratio_med",
                                     "sm_ratio_med", "cosine_med"]].mean().round(3).to_string())
    (HERE / "_meta.json").write_text(json.dumps(
        {"smooth_w": SMOOTH_W, "move_eps_m": MOVE_EPS, "rel_tol": REL_TOL,
         "share_tol": SHARE_TOL, "total_windows": int(len(inv))}, indent=2))


if __name__ == "__main__":
    main()
