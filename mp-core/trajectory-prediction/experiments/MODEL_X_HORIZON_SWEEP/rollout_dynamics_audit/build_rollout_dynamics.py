"""
build_rollout_dynamics.py — PHASE 3 rollout magnitude/direction dynamics audit.

Diagnosis only. NO retraining / NO weight changes / NO dataset changes. Per-timestep
rollout paths are reconstructed by DETERMINISTIC replay of the frozen H*/model_best.pth
(model.eval(), no dropout) — byte-identical to the stored predictions (ADE consistency
checked against per_window_metrics.csv). Per-step paths were not saved by the sweep, so
replay is the only way to get t=1..H granularity.

Primary magnitude metric = NET DISPLACEMENT growth (jitter-immune), per Phase 1/2.
Outputs in rollout_dynamics_audit/ (CSVs + figures/).
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

HERE = Path(__file__).resolve().parent
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
SMOOTH_W = 5
STEP_EPS = 1e-3        # m; min GT step for a defined step ratio
NDISP_EPS = 0.2        # m; min GT net disp for defined direction
PRED_DIR_EPS = 0.05
DECAY_FRAC = 0.80      # decay = pred step < 80% GT step
DECAY_RUN = 5          # ... for >=5 consecutive steps
MILESTONES = [1, 5, 10, 20, 50, 100, 200]
RECS = ["placa_catalunya_01", "placa_espanya_01", "esplanade_espanya_01",
        "stairs_montjuic_01", "red_bridge_combined_01"]
FOCUS = {"placa_espanya_01": "espanya", "stairs_montjuic_01": "stairs"}


def smooth_paths(a, w=SMOOTH_W):
    M, P, D = a.shape
    if P < 3:
        return a.copy()
    if w % 2 == 0:
        w += 1
    w = min(w, P if P % 2 == 1 else P - 1)
    pad = w // 2
    ap = np.concatenate([np.repeat(a[:, :1], pad, 1), a, np.repeat(a[:, -1:], pad, 1)], axis=1)
    cs = np.cumsum(ap, axis=1)
    cs = np.concatenate([np.zeros((M, 1, D)), cs], axis=1)
    return (cs[:, w:] - cs[:, :-w]) / w


def consecutive_decay_onset(below):
    """below: (M,T) bool. Return t_decay (1-based start of first >=DECAY_RUN run) or -1."""
    M, T = below.shape
    cc = np.zeros((M, T), dtype=np.int32)
    cc[:, 0] = below[:, 0]
    for t in range(1, T):
        cc[:, t] = np.where(below[:, t], cc[:, t - 1] + 1, 0)
    reached = cc >= DECAY_RUN
    has = reached.any(axis=1)
    first = np.argmax(reached, axis=1)             # index where run hits DECAY_RUN
    t_decay = np.where(has, first - DECAY_RUN + 2, -1)   # 1-based start step
    return t_decay


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[phase3] device={device}")
    cols = ["recording_id", "trajectory_id", "timestep", "world_x", "world_y"] + L.FEAT_COLS
    df = pd.read_csv(CSV, usecols=lambda c: c in set(cols))
    wb = L.recording_world_bounds(df)
    Lens = df.groupby("trajectory_id").size()
    split = pd.read_csv(MODELX / "splits" / "model_x_track_split.csv")

    step_rows, nd_rows, dir_rows, decay_rows, milestone_rows = [], [], [], [], []
    decay_hist = {}     # horizon -> array of t_decay (windows with decay)
    focus_curves = {}   # rec_tag -> dict(horizon -> per-t arrays)

    for H in HORIZONS:
        Hdir = SWEEP / f"H{H}"
        ckpt, scl = Hdir / "model_best.pth", Hdir / "scalers.json"
        if not (ckpt.exists() and scl.exists()):
            print(f"[phase3] SKIP H{H}"); continue
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
        rec_arr = np.array([m["recording_id"] for m in meta])

        # consistency check vs stored per_window_metrics
        ade_now = np.linalg.norm(preds[:, 1:] - gt[:, 1:], axis=2).mean(axis=1).mean()
        try:
            stored = pd.read_csv(Hdir / "per_window_metrics.csv")["ade"].mean()
            print(f"[phase3] H{H} ADE replay={ade_now:.4f} stored={stored:.4f} "
                  f"(Δ={abs(ade_now-stored):.2e})")
        except Exception:
            pass

        # per-step arrays
        gt_step = np.linalg.norm(np.diff(gt, axis=1), axis=2)        # (M,H)
        pr_step = np.linalg.norm(np.diff(preds, axis=1), axis=2)
        gt_sm = smooth_paths(gt); pr_sm = smooth_paths(preds)
        gt_step_sm = np.linalg.norm(np.diff(gt_sm, axis=1), axis=2)
        pr_step_sm = np.linalg.norm(np.diff(pr_sm, axis=1), axis=2)
        step_ratio = np.where(gt_step > STEP_EPS, pr_step / np.maximum(gt_step, 1e-9), np.nan)

        # net displacement from start at each t (t=0..H)
        gv = gt - gt[:, :1]; pv = preds - preds[:, :1]
        gt_nd = np.linalg.norm(gv, axis=2); pr_nd = np.linalg.norm(pv, axis=2)   # (M,H+1)
        nd_ratio = np.where(gt_nd > NDISP_EPS, pr_nd / np.maximum(gt_nd, 1e-9), np.nan)
        dot = (gv * pv).sum(axis=2)
        ng, npv_ = gt_nd, pr_nd
        cos = np.where((ng > NDISP_EPS) & (npv_ > PRED_DIR_EPS),
                       dot / np.maximum(ng * npv_, 1e-9), np.nan)
        head_err = np.degrees(np.arccos(np.clip(cos, -1, 1)))

        # decay onset per window
        below = (pr_step < DECAY_FRAC * gt_step) & (gt_step > STEP_EPS)
        t_decay = consecutive_decay_onset(below)
        decay_hist[H] = t_decay[t_decay > 0]

        groups = {"ALL": np.ones(M, bool)}
        for r in RECS:
            groups[r] = (rec_arr == r)

        for gname, mask in groups.items():
            if mask.sum() == 0:
                continue
            nwin = int(mask.sum())
            # per-timestep step metrics (t=1..H -> index 0..H-1)
            for ti in range(H):
                t = ti + 1
                step_rows.append({
                    "horizon": H, "recording": gname, "t": t,
                    "gt_step_med": float(np.nanmedian(gt_step[mask, ti])),
                    "pred_step_med": float(np.nanmedian(pr_step[mask, ti])),
                    "gt_step_sm_med": float(np.nanmedian(gt_step_sm[mask, ti])),
                    "pred_step_sm_med": float(np.nanmedian(pr_step_sm[mask, ti])),
                    "step_ratio_med": float(np.nanmedian(step_ratio[mask, ti])),
                })
            # per-timestep net disp + direction (t=1..H -> index 1..H)
            for t in range(1, H + 1):
                nd_rows.append({
                    "horizon": H, "recording": gname, "t": t,
                    "gt_netdisp_med": float(np.nanmedian(gt_nd[mask, t])),
                    "pred_netdisp_med": float(np.nanmedian(pr_nd[mask, t])),
                    "netdisp_ratio_med": float(np.nanmedian(nd_ratio[mask, t])),
                })
                cv = cos[mask, t]
                dir_rows.append({
                    "horizon": H, "recording": gname, "t": t,
                    "cosine_med": float(np.nanmedian(cv)),
                    "heading_err_med_deg": float(np.nanmedian(head_err[mask, t])),
                    "pct_cos_gt075": float(np.nanmean(cv > 0.75) * 100),
                    "pct_cos_gt050": float(np.nanmean(cv > 0.50) * 100),
                })
            # decay aggregate
            td = t_decay[mask]; has = td > 0
            decay_rows.append({
                "horizon": H, "recording": gname, "n_windows": nwin,
                "pct_windows_with_decay": round(100 * has.mean(), 2),
                "t_decay_mean": float(td[has].mean()) if has.any() else float("nan"),
                "t_decay_median": float(np.median(td[has])) if has.any() else float("nan"),
            })
            # milestones (step ratio + nd ratio at fixed t)
            for ms in MILESTONES:
                if ms > H:
                    continue
                milestone_rows.append({
                    "horizon": H, "recording": gname, "step": ms,
                    "step_ratio_med": float(np.nanmedian(step_ratio[mask, ms - 1])),
                    "netdisp_ratio_med": float(np.nanmedian(nd_ratio[mask, ms])),
                    "cosine_med": float(np.nanmedian(cos[mask, ms])),
                })
            # focus curves storage
            if gname in FOCUS:
                focus_curves.setdefault(FOCUS[gname], {})[H] = {
                    "t": np.arange(1, H + 1),
                    "step_ratio": np.array([np.nanmedian(step_ratio[mask, ti]) for ti in range(H)]),
                    "nd_ratio": np.array([np.nanmedian(nd_ratio[mask, t]) for t in range(1, H + 1)]),
                    "cosine": np.array([np.nanmedian(cos[mask, t]) for t in range(1, H + 1)]),
                }
        print(f"[phase3] H{H}: {M} windows aggregated")

    step_df = pd.DataFrame(step_rows); step_df.to_csv(HERE / "step_length_by_timestep.csv", index=False)
    nd_df = pd.DataFrame(nd_rows); nd_df.to_csv(HERE / "net_displacement_growth_by_timestep.csv", index=False)
    dir_df = pd.DataFrame(dir_rows); dir_df.to_csv(HERE / "direction_metrics_by_timestep.csv", index=False)
    decay_df = pd.DataFrame(decay_rows); decay_df.to_csv(HERE / "decay_onset_by_recording.csv", index=False)
    ms_df = pd.DataFrame(milestone_rows); ms_df.to_csv(HERE / "step_ratio_milestones.csv", index=False)
    # focus CSVs
    for rec, tag in FOCUS.items():
        sub_step = step_df[step_df.recording == rec]
        sub_nd = nd_df[nd_df.recording == rec][["horizon", "t", "netdisp_ratio_med", "gt_netdisp_med", "pred_netdisp_med"]]
        sub_dir = dir_df[dir_df.recording == rec][["horizon", "t", "cosine_med", "heading_err_med_deg"]]
        m = sub_step.merge(sub_nd, on=["horizon", "t"]).merge(sub_dir, on=["horizon", "t"])
        m.to_csv(HERE / f"{tag}_focus_metrics.csv", index=False)

    # ───────── figures ─────────
    FIG.mkdir(exist_ok=True)
    hcol = {h: c for h, c in zip(HORIZONS, plt.cm.viridis(np.linspace(0, .9, len(HORIZONS))))}
    rcol = {r: c for r, c in zip(RECS, plt.cm.tab10(np.linspace(0, 1, len(RECS))))}

    def all_curve(df_, col, title, ylab, fname, hlines=()):
        fig, ax = plt.subplots(figsize=(9, 5))
        for H in HORIZONS:
            s = df_[(df_.recording == "ALL") & (df_.horizon == H)].sort_values("t")
            if len(s):
                ax.plot(s.t, s[col], color=hcol[H], lw=1.5, label=f"H{H} (~{APPROX[H]}m)")
        for y in hlines:
            ax.axhline(y, ls="--", c="k", alpha=.4)
        ax.set_xlabel("rollout timestep t"); ax.set_ylabel(ylab); ax.set_title(title)
        ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(FIG / fname, dpi=130); plt.close(fig)

    all_curve(step_df, "step_ratio_med", "Step-length ratio (pred/GT) vs rollout time — ALL",
              "median pred_step / gt_step", "overall_step_length_vs_time.png", hlines=(1.0, 0.8))
    all_curve(nd_df, "netdisp_ratio_med", "Net-displacement growth ratio (pred/GT) vs time — ALL",
              "median pred_netdisp / gt_netdisp", "overall_net_displacement_growth.png", hlines=(1.0,))
    all_curve(dir_df, "cosine_med", "Direction cosine similarity vs rollout time — ALL",
              "median cosine", "overall_cosine_similarity_vs_time.png", hlines=(1.0, 0.0))
    all_curve(dir_df, "heading_err_med_deg", "Heading error vs rollout time — ALL",
              "median heading error (deg)", "overall_heading_error_vs_time.png", hlines=(90,))

    def per_rec_curve(df_, col, H, title, ylab, fname, hlines=()):
        fig, ax = plt.subplots(figsize=(9, 5))
        for r in RECS:
            s = df_[(df_.recording == r) & (df_.horizon == H)].sort_values("t")
            if len(s):
                ax.plot(s.t, s[col], color=rcol[r], lw=1.5, label=r)
        for y in hlines:
            ax.axhline(y, ls="--", c="k", alpha=.4)
        ax.set_xlabel("rollout timestep t"); ax.set_ylabel(ylab)
        ax.set_title(f"{title} (H{H} ~{APPROX[H]}m)"); ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIG / fname, dpi=130); plt.close(fig)

    per_rec_curve(step_df, "step_ratio_med", 200, "Per-recording step-length ratio vs time",
                  "pred/GT step", "per_recording_step_length_curves.png", hlines=(1.0, 0.8))
    per_rec_curve(nd_df, "netdisp_ratio_med", 200, "Per-recording net-disp ratio vs time",
                  "pred/GT net-disp", "per_recording_net_displacement_curves.png", hlines=(1.0,))
    per_rec_curve(dir_df, "cosine_med", 200, "Per-recording cosine vs time",
                  "cosine", "per_recording_cosine_curves.png", hlines=(1.0, 0.0))

    def focus_fig(tag, fname):
        fc = focus_curves.get(tag, {})
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        for H in HORIZONS:
            if H not in fc:
                continue
            d = fc[H]
            ax[0].plot(d["t"], d["step_ratio"], color=hcol[H], label=f"H{H}")
            ax[1].plot(d["t"], d["nd_ratio"], color=hcol[H], label=f"H{H}")
            ax[2].plot(d["t"], d["cosine"], color=hcol[H], label=f"H{H}")
        ax[0].axhline(0.8, ls="--", c="k", alpha=.4); ax[0].axhline(1, ls="--", c="k", alpha=.3)
        ax[1].axhline(1, ls="--", c="k", alpha=.4); ax[2].axhline(0, ls="--", c="k", alpha=.4)
        for a, t in zip(ax, ["step-length ratio", "net-disp ratio", "cosine"]):
            a.set_title(t); a.set_xlabel("t"); a.grid(alpha=.3); a.legend(fontsize=7)
        fig.suptitle(f"{tag} rollout dynamics (step ratio / net-disp growth / direction)", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIG / fname, dpi=130); plt.close(fig)

    focus_fig("espanya", "espanya_rollout_dynamics.png")
    focus_fig("stairs", "stairs_rollout_dynamics.png")

    # decay onset histograms
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(18, 3.6))
    for ax, H in zip(np.atleast_1d(axes), HORIZONS):
        d = decay_hist.get(H, np.array([]))
        if len(d):
            ax.hist(d, bins=30, color="#d62728", alpha=.75)
            ax.axvline(np.median(d), ls="--", c="k", label=f"median {np.median(d):.0f}")
            ax.legend(fontsize=7)
        ax.set_title(f"H{H} t_decay"); ax.set_xlabel("decay onset step t"); ax.grid(alpha=.3)
    fig.suptitle("Decay-onset distribution (first step where pred<80% GT for >=5 steps)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(FIG / "decay_onset_histograms.png", dpi=130)
    plt.close(fig)

    # ── console summary for the report ──
    print("\n=== MILESTONE step-length ratio (ALL) ===")
    piv = ms_df[ms_df.recording == "ALL"].pivot(index="horizon", columns="step", values="step_ratio_med")
    print(piv.round(3).to_string())
    print("\n=== step ratio at t=1 vs end, cosine t=1 vs end (ALL) ===")
    for H in HORIZONS:
        s = step_df[(step_df.recording == "ALL") & (step_df.horizon == H)].sort_values("t")
        dC = dir_df[(dir_df.recording == "ALL") & (dir_df.horizon == H)].sort_values("t")
        if len(s):
            print(f"H{H}: step_ratio t1={s.step_ratio_med.iloc[0]:.3f} tEnd={s.step_ratio_med.iloc[-1]:.3f} | "
                  f"cosine t1={dC.cosine_med.iloc[0]:.3f} tEnd={dC.cosine_med.iloc[-1]:.3f}")
    print("\n=== decay onset (ALL) ===")
    print(decay_df[decay_df.recording == "ALL"][["horizon", "pct_windows_with_decay",
          "t_decay_median", "t_decay_mean"]].to_string(index=False))
    print("\n=== per-recording mean cosine & nd_ratio over time (H200) ===")
    for r in RECS:
        c = dir_df[(dir_df.recording == r) & (dir_df.horizon == 200)]
        n = nd_df[(nd_df.recording == r) & (nd_df.horizon == 200)]
        if len(c):
            print(f"{r:24s} cosine t1={c.sort_values('t').cosine_med.iloc[0]:.3f} "
                  f"tEnd={c.sort_values('t').cosine_med.iloc[-1]:.3f} | "
                  f"nd_ratio tEnd={n.sort_values('t').netdisp_ratio_med.iloc[-1]:.3f}")
    (HERE / "_meta.json").write_text(json.dumps(
        {"smooth_w": SMOOTH_W, "decay_frac": DECAY_FRAC, "decay_run": DECAY_RUN,
         "ndisp_eps": NDISP_EPS, "step_eps": STEP_EPS, "milestones": MILESTONES}, indent=2))
    print("\n[phase3] CSVs + figures written.")


if __name__ == "__main__":
    main()
