"""
build_curvature_audit.py — PHASE 5A trajectory-shape / curvature audit.

Measurement only. NO retraining / NO new predictions: the frozen checkpoints (MODEL_X +
the four MODEL_XR variants) are replayed deterministically (model.eval(), no dropout) — the
rollouts are byte-identical to those already generated. Per-window paths were never stored,
so replay is the only way to compute per-step curvature.

Curvature is MACRO-scale: each rollout is adaptively denoised (moving-avg ~ segment length)
and down-sampled to N_SEG=10 nodes inside curv_metrics, then turning is measured between macro
segments. This is the CORRECTED method — an earlier pass used light w=5 smoothing which left GT
heading jitter-dominated (~1992 deg over 100 steps, impossible). Macro numbers are physically sane.

Outputs in curvature_audit/ (CSVs + figures/).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
XRROOT = HERE.parent
sys.path.insert(0, str(XRROOT))
import xr_common as XC  # noqa
L = XC.L
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)

MODELX = XRROOT.parent / "experiments" / "MODEL_X"
MODELS = {
    "MODEL_X":      (MODELX / "models" / "best_by_val_ADE.pth", MODELX / "models" / "scalers.json"),
    "XR_A_CONTROL": (XRROOT / "checkpoints" / "MODEL_XR_A_CONTROL" / "model_best.pth", XRROOT / "checkpoints" / "MODEL_XR_A_CONTROL" / "scalers.json"),
    "XR_B_MAG":     (XRROOT / "checkpoints" / "MODEL_XR_B_MAG" / "model_best.pth", XRROOT / "checkpoints" / "MODEL_XR_B_MAG" / "scalers.json"),
    "XR_C_MAG_LIGHT_DIR": (XRROOT / "checkpoints" / "MODEL_XR_C_MAG_LIGHT_DIR" / "model_best.pth", XRROOT / "checkpoints" / "MODEL_XR_C_MAG_LIGHT_DIR" / "scalers.json"),
    "XR_D_MAG_STRONGER":  (XRROOT / "checkpoints" / "MODEL_XR_D_MAG_STRONGER" / "model_best.pth", XRROOT / "checkpoints" / "MODEL_XR_D_MAG_STRONGER" / "scalers.json"),
}
PRIMARY_H = 100                  # main shape-analysis horizon (~5 m)
VS_HORIZONS = [20, 60, 100, 200, 400]
OPEN = ["placa_catalunya_01", "placa_espanya_01", "esplanade_espanya_01"]
CONSTR = ["stairs_montjuic_01", "red_bridge_combined_01"]
TURN_DEG = 15.0
MIN_NET = 1.0                    # m; window must travel >=1m to have meaningful shape
MIN_GT_HEAD = 15.0              # deg; GT must turn >=15deg total for a defined shape_ratio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name):
    ck, sc = MODELS[name]
    s = json.loads(Path(sc).read_text())
    f = L.ColumnScaler.from_dict(s["feature_scaler"]); t = L.ColumnScaler.from_dict(s["target_scaler"])
    m = L.TrajectoryLSTM(L.N_FEAT).to(device)
    m.load_state_dict(torch.load(ck, map_location=device)); m.eval()
    return m, f, t


N_SEG = 10   # macro-shape resolution: every rollout is reduced to N_SEG coarse segments


def coarse_nodes(path, n_seg=N_SEG):
    """Heavily denoise (adaptive moving-avg ~ segment length) then downsample to n_seg+1
    evenly-spaced nodes incl. true endpoints. Removes per-frame tracking jitter so heading
    reflects real walking direction, not noise. path (M,P,2) -> (M,n_seg+1,2)."""
    P = path.shape[1]
    if P <= n_seg + 1:
        return path.copy()
    w = max(5, (P // n_seg) | 1)                 # window ~ one segment, forced odd
    sm = XC.smooth_paths(path, w=w)
    idx = np.linspace(0, P - 1, n_seg + 1).astype(int)
    return sm[:, idx, :]


def curv_metrics(path):
    """Macro-scale curvature on coarse-resampled (jitter-free) path. path (M,P,2) raw."""
    nodes = coarse_nodes(path)                                  # (M,n_seg+1,2)
    d = np.diff(nodes, axis=1)                                  # (M,n_seg,2)
    seg = np.linalg.norm(d, axis=2)
    h = np.arctan2(d[:, :, 1], d[:, :, 0])
    turns = np.arctan2(np.sin(np.diff(h, axis=1)), np.cos(np.diff(h, axis=1)))
    at = np.abs(np.degrees(turns))
    cum_head = at.sum(axis=1)
    path_len = seg.sum(axis=1)
    net = np.linalg.norm(nodes[:, -1] - nodes[:, 0], axis=1)
    tort = path_len / np.maximum(net, 1e-9)
    dens = cum_head / np.maximum(path_len, 1e-9)
    n_turns = (at > TURN_DEG).sum(axis=1)
    ang_acc = np.abs(np.degrees(np.arctan2(np.sin(np.diff(turns, axis=1)),
                                           np.cos(np.diff(turns, axis=1))))).mean(axis=1) if turns.shape[1] > 1 else np.zeros(len(path))
    return {"cum_head": cum_head, "mean_turn": at.mean(axis=1), "max_turn": at.max(axis=1),
            "tortuosity": tort, "curv_density": dens, "n_turns": n_turns,
            "ang_accel": ang_acc, "path_len": path_len, "net": net}


def cum_head_over_time(path):
    """(M,P) cumulative |heading change| vs t, using coarse nodes interpolated back to t."""
    nodes = coarse_nodes(path)
    d = np.diff(nodes, axis=1); h = np.arctan2(d[:, :, 1], d[:, :, 0])
    turns = np.abs(np.degrees(np.arctan2(np.sin(np.diff(h, axis=1)), np.cos(np.diff(h, axis=1)))))
    cs = np.concatenate([np.zeros((len(path), 1)), np.cumsum(turns, axis=1)], axis=1)  # (M,n_seg-1) per node-junction
    # map node junctions to timesteps and interpolate to 0..P-1
    P = path.shape[1]
    node_t = np.linspace(0, P - 1, N_SEG + 1)
    junction_t = node_t[1:-1]                                   # turns happen at interior nodes
    junction_t = np.concatenate([[0], junction_t])             # align with cs columns
    out = np.empty((len(path), P))
    for i in range(len(path)):
        out[i] = np.interp(np.arange(P), junction_t, cs[i])
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df = XC.load_df(); split = XC.load_split(); wb = L.recording_world_bounds(df)
    lens = df.groupby("trajectory_id").size()

    # ===== PRIMARY horizon: full curvature for all models + GT =====
    need = XC.OBS + PRIMARY_H
    te_ids = [t for t in split[split.split == "test"].trajectory_id if lens.get(t, 0) >= need]
    paths = {}; rec = tid = None; gt = None
    for name in MODELS:
        m, f, t = load_model(name)
        p, g, r, ti = XC.rollout_all(m, df, te_ids, wb, f, t, PRIMARY_H, device)
        paths[name] = p
        if gt is None:
            gt, rec, tid = g, r, ti
        print(f"[curv] rolled {name} @H{PRIMARY_H}: {len(p)} windows")
    M = len(gt)
    gtc = curv_metrics(gt)
    modc = {name: curv_metrics(paths[name]) for name in MODELS}

    # meaningful-shape mask (defined on GT)
    meaningful = (gtc["net"] > MIN_NET) & (gtc["cum_head"] > MIN_GT_HEAD)

    # ---- per-window curvature_metrics.csv (long form) ----
    rows = []
    for name in (["GT"] + list(MODELS)):
        c = gtc if name == "GT" else modc[name]
        for i in range(M):
            rows.append({"model": name, "recording": rec[i], "trajectory_id": tid[i], "window": i,
                         "cum_head_deg": c["cum_head"][i], "mean_turn_deg": c["mean_turn"][i],
                         "max_turn_deg": c["max_turn"][i], "tortuosity": c["tortuosity"][i],
                         "curv_density_deg_per_m": c["curv_density"][i], "n_turns": int(c["n_turns"][i]),
                         "ang_accel_deg": c["ang_accel"][i], "meaningful": bool(meaningful[i])})
    cm = pd.DataFrame(rows); cm.to_csv(HERE / "curvature_metrics.csv", index=False)

    # ---- shape preservation per model (vs GT, meaningful windows) ----
    def ratios(name, mask):
        c = modc[name]
        sh = c["cum_head"][mask] / np.maximum(gtc["cum_head"][mask], 1e-9)
        to = c["tortuosity"][mask] / np.maximum(gtc["tortuosity"][mask], 1e-9)
        tc = c["n_turns"][mask] / np.maximum(gtc["n_turns"][mask], 1e-9)
        return sh, to, tc
    sp_rows = []
    for name in MODELS:
        sh, to, tc = ratios(name, meaningful)
        sp_rows.append({"model": name, "n_meaningful": int(meaningful.sum()),
                        "shape_ratio_med": float(np.median(sh)), "shape_ratio_mean": float(np.mean(sh)),
                        "tortuosity_ratio_med": float(np.median(to)),
                        "turn_count_ratio_med": float(np.median(tc)),
                        "cum_head_med_deg": float(np.median(modc[name]["cum_head"][meaningful])),
                        "gt_cum_head_med_deg": float(np.median(gtc["cum_head"][meaningful]))})
    sp = pd.DataFrame(sp_rows); sp.to_csv(HERE / "shape_preservation_metrics.csv", index=False)

    # ---- per-recording ----
    pr_rows = []
    for r in XC.RECS:
        rmask = meaningful & (rec == r)
        if rmask.sum() == 0:
            continue
        for name in MODELS:
            sh, to, tc = ratios(name, rmask)
            pr_rows.append({"recording": r, "model": name, "n": int(rmask.sum()),
                            "shape_ratio_med": float(np.median(sh)),
                            "tortuosity_ratio_med": float(np.median(to)),
                            "turn_count_ratio_med": float(np.median(tc)),
                            "gt_cum_head_med": float(np.median(gtc["cum_head"][rmask])),
                            "pred_cum_head_med": float(np.median(modc[name]["cum_head"][rmask]))})
    pr = pd.DataFrame(pr_rows); pr.to_csv(HERE / "per_recording_curvature_metrics.csv", index=False)

    # ---- topology ----
    tp_rows = []
    for grp, recs in [("OPEN", OPEN), ("CONSTRAINED", CONSTR)]:
        gmask = meaningful & np.isin(rec, recs)
        if gmask.sum() == 0:
            continue
        for name in MODELS:
            sh, to, tc = ratios(name, gmask)
            tp_rows.append({"topology": grp, "model": name, "n": int(gmask.sum()),
                            "shape_ratio_med": float(np.median(sh)),
                            "tortuosity_ratio_med": float(np.median(to)),
                            "turn_count_ratio_med": float(np.median(tc))})
    tp = pd.DataFrame(tp_rows); tp.to_csv(HERE / "topology_curvature_metrics.csv", index=False)

    # ===== curvature over time (GT, MODEL_X, XR_B_MAG) @ PRIMARY_H =====
    ot_rows = []
    cot = {"GT": cum_head_over_time(gt)}
    for name in ["MODEL_X", "XR_B_MAG"]:
        cot[name] = cum_head_over_time(paths[name])
    for t in range(PRIMARY_H + 1):
        row = {"t": t}
        for name in cot:
            row[name] = float(np.median(cot[name][meaningful, t]))
        ot_rows.append(row)
    ot = pd.DataFrame(ot_rows); ot.to_csv(HERE / "curvature_over_time.csv", index=False)

    # ===== shape_ratio vs horizon (MODEL_X, XR_B_MAG, and C/D) =====
    vs_rows = []
    for H in VS_HORIZONS:
        need_h = XC.OBS + H
        ids_h = [t for t in split[split.split == "test"].trajectory_id if lens.get(t, 0) >= need_h]
        gh = None; gc = None; mm = None
        for name in ["MODEL_X", "XR_B_MAG", "XR_C_MAG_LIGHT_DIR", "XR_D_MAG_STRONGER"]:
            m, f, t = load_model(name)
            p, g, r, ti = XC.rollout_all(m, df, ids_h, wb, f, t, H, device)
            if gh is None:
                gh = curv_metrics(g)
                mm = (gh["net"] > MIN_NET) & (gh["cum_head"] > MIN_GT_HEAD)
            c = curv_metrics(p)
            sh = c["cum_head"][mm] / np.maximum(gh["cum_head"][mm], 1e-9)
            vs_rows.append({"horizon": H, "model": name, "shape_ratio_med": float(np.median(sh)),
                            "tortuosity_ratio_med": float(np.median(c["tortuosity"][mm] / np.maximum(gh["tortuosity"][mm], 1e-9))),
                            "n_meaningful": int(mm.sum())})
        print(f"[curv] vs-horizon H{H} done")
    vs = pd.DataFrame(vs_rows); vs.to_csv(HERE / "curvature_vs_horizon.csv", index=False)

    # ===================== FIGURES =====================
    order = list(MODELS); colr = {n: c for n, c in zip(order, ["#7f7f7f", "#9467bd", "#e0218a", "#2ca02c", "#d62728"])}

    # curvature distribution by model (+GT) — cum_head on meaningful
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [gtc["cum_head"][meaningful]] + [modc[n]["cum_head"][meaningful] for n in order]
    ax.boxplot(data, labels=["GT"] + order, showfliers=False)
    ax.set_ylabel("cumulative |heading change| (deg, smoothed)")
    ax.set_title(f"Curvature distribution by model (H{PRIMARY_H}, meaningful windows)")
    plt.xticks(rotation=15); ax.grid(alpha=.3, axis="y"); fig.tight_layout()
    fig.savefig(FIG / "curvature_distribution_by_model.png", dpi=130); plt.close(fig)

    def barfig(col, title, ylab, fname, ref1=True):
        fig, ax = plt.subplots(figsize=(8, 5))
        ys = [sp[sp.model == n][col].iloc[0] for n in order]
        ax.bar(order, ys, color=[colr[n] for n in order])
        if ref1: ax.axhline(1.0, ls="--", c="k", alpha=.5, label="GT (=1.0)")
        for i, y in enumerate(ys): ax.text(i, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(ylab); ax.set_title(title); ax.grid(alpha=.3, axis="y")
        if ref1: ax.legend()
        plt.xticks(rotation=15); fig.tight_layout(); fig.savefig(FIG / fname, dpi=130); plt.close(fig)
    barfig("shape_ratio_med", f"Shape ratio (pred/GT cumulative heading) — H{PRIMARY_H}", "shape ratio", "shape_ratio_by_model.png")
    barfig("tortuosity_ratio_med", f"Tortuosity ratio (pred/GT) — H{PRIMARY_H}", "tortuosity ratio", "tortuosity_ratio_by_model.png")
    barfig("turn_count_ratio_med", f"Turn-count ratio (pred/GT, >{int(TURN_DEG)}deg) — H{PRIMARY_H}", "turn-count ratio", "turn_count_ratio_by_model.png")

    # curvature vs horizon (shape ratio)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in ["MODEL_X", "XR_B_MAG", "XR_C_MAG_LIGHT_DIR", "XR_D_MAG_STRONGER"]:
        s = vs[vs.model == name].sort_values("horizon")
        ax.plot(s.horizon, s.shape_ratio_med, "-o", label=name, color=colr[name])
    ax.axhline(1.0, ls="--", c="k", alpha=.5); ax.set_xticks(VS_HORIZONS)
    ax.set_xlabel("horizon"); ax.set_ylabel("shape ratio (pred/GT cum heading)")
    ax.set_title("Shape ratio vs horizon"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "curvature_vs_horizon.png", dpi=130); plt.close(fig)

    # curvature growth over time
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, c in [("GT", "#000000"), ("MODEL_X", "#9467bd"), ("XR_B_MAG", "#e0218a")]:
        ax.plot(ot.t, ot[name], label=name, color=c, lw=1.8)
    ax.set_xlabel("rollout timestep t"); ax.set_ylabel("median cumulative |heading change| (deg)")
    ax.set_title(f"Curvature growth over time (H{PRIMARY_H}, meaningful windows)")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "curvature_growth_over_time.png", dpi=130); plt.close(fig)

    # open vs constrained
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(order)); w = 0.38
    op = [tp[(tp.topology == "OPEN") & (tp.model == n)].shape_ratio_med.iloc[0] for n in order]
    cn = [tp[(tp.topology == "CONSTRAINED") & (tp.model == n)].shape_ratio_med.iloc[0] for n in order]
    ax.bar(x - w/2, op, w, label="OPEN plazas", color="#1f77b4")
    ax.bar(x + w/2, cn, w, label="CONSTRAINED", color="#ff7f0e")
    ax.axhline(1.0, ls="--", c="k", alpha=.5)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=15); ax.set_ylabel("shape ratio")
    ax.set_title("Shape preservation: open vs constrained"); ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "open_vs_constrained_curvature.png", dpi=130); plt.close(fig)

    # espanya / stairs shape analysis (shape & tortuosity ratio bars per model)
    def site_fig(r, fname):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        sub = pr[pr.recording == r]
        for k, col in enumerate(["shape_ratio_med", "tortuosity_ratio_med"]):
            ys = [sub[sub.model == n][col].iloc[0] if len(sub[sub.model == n]) else np.nan for n in order]
            ax[k].bar(order, ys, color=[colr[n] for n in order]); ax[k].axhline(1.0, ls="--", c="k", alpha=.5)
            ax[k].set_title(col); ax[k].grid(alpha=.3, axis="y"); ax[k].tick_params(axis="x", rotation=20, labelsize=7)
        fig.suptitle(f"{r} — shape analysis (H{PRIMARY_H})"); fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(FIG / fname, dpi=130); plt.close(fig)
    site_fig("placa_espanya_01", "espanya_shape_analysis.png")
    site_fig("stairs_montjuic_01", "stairs_shape_analysis.png")

    # ===== example panels (GT vs MODEL_X vs B_MAG), per recording best/worst shape (B_MAG) =====
    df_by = {t: g.sort_values("timestep") for t, g in df[df.trajectory_id.isin(set(tid))].groupby("trajectory_id")}
    _, _, _, _, meta = L.build_eval_windows(df, te_ids, wb, n_steps=PRIMARY_H, stride=1)
    starts = np.array([m["start_idx"] for m in meta])
    shB = modc["XR_B_MAG"]["cum_head"] / np.maximum(gtc["cum_head"], 1e-9)
    inv_rows = []

    def obs_path(i):
        g = df_by[tid[i]]; s = g.iloc[starts[i]:starts[i] + XC.OBS]
        return np.column_stack([s.world_x.to_numpy(float), s.world_y.to_numpy(float)])

    def panel(ax, i, tag):
        ob = obs_path(i)
        ax.plot(ob[:, 0], ob[:, 1], "-", color="black", lw=1.3, label="observed")
        ax.plot(gt[i, :, 0], gt[i, :, 1], "-", color="#7f7f7f", lw=2.4, label="GT")
        ax.plot(paths["MODEL_X"][i, :, 0], paths["MODEL_X"][i, :, 1], "--", color="#9467bd", lw=1.4, label="MODEL_X")
        ax.plot(paths["XR_B_MAG"][i, :, 0], paths["XR_B_MAG"][i, :, 1], "-", color="#e0218a", lw=1.7, label="XR_B_MAG")
        ax.plot(ob[-1, 0], ob[-1, 1], "o", color="#333", ms=4)
        ax.set_aspect("equal", adjustable="box"); ax.tick_params(labelsize=6)
        ax.set_title(f"{rec[i].replace('_01','')} {tag} shapeR={shB[i]:.2f}\nGTcurv={gtc['cum_head'][i]:.0f}deg", fontsize=7)

    for which, fname, pick in [("best", "best_shape_examples.png", "best"), ("worst", "worst_shape_examples.png", "worst")]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8)); axf = axes.flat
        for k, r in enumerate(XC.RECS):
            rmask = meaningful & (rec == r)
            idxs = np.where(rmask)[0]
            if len(idxs) == 0:
                axf[k].axis("off"); continue
            err = np.abs(shB[idxs] - 1.0)
            sel = idxs[np.argmin(err)] if pick == "best" else idxs[np.argmin(shB[idxs])]  # worst = most straightened
            panel(axf[k], sel, which)
            if k == 0: axf[k].legend(fontsize=6)
            inv_rows.append({"category": which, "recording": r, "window": int(sel),
                             "trajectory_id": tid[sel], "shape_ratio_B_MAG": float(shB[sel]),
                             "gt_cum_head_deg": float(gtc["cum_head"][sel]),
                             "bmag_cum_head_deg": float(modc["XR_B_MAG"]["cum_head"][sel])})
        for k in range(len(XC.RECS), 6):
            axf[k].axis("off")
        fig.suptitle(f"{which.upper()} shape match per recording — GT vs MODEL_X vs XR_B_MAG (H{PRIMARY_H})", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIG / fname, dpi=140); plt.close(fig)
    pd.DataFrame(inv_rows).to_csv(HERE / "shape_example_inventory.csv", index=False)

    # ===== console summary =====
    print("\n=== shape preservation (H%d, %d meaningful windows) ===" % (PRIMARY_H, meaningful.sum()))
    print(sp[["model", "shape_ratio_med", "tortuosity_ratio_med", "turn_count_ratio_med",
              "cum_head_med_deg", "gt_cum_head_med_deg"]].to_string(index=False))
    print("\n=== topology shape_ratio ===")
    print(tp.pivot(index="model", columns="topology", values="shape_ratio_med").round(3).to_string())
    print("\n=== per-recording shape_ratio (B_MAG vs MODEL_X) ===")
    pp = pr[pr.model.isin(["MODEL_X", "XR_B_MAG"])].pivot(index="recording", columns="model", values="shape_ratio_med")
    print(pp.round(3).to_string())
    (HERE / "_meta.json").write_text(json.dumps(
        {"primary_h": PRIMARY_H, "smooth_w": XC.SMOOTH_W, "turn_deg": TURN_DEG,
         "min_net_m": MIN_NET, "min_gt_head_deg": MIN_GT_HEAD, "n_meaningful": int(meaningful.sum())}, indent=2))
    print("\n[curv] done.")


if __name__ == "__main__":
    main()
