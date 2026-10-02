"""
make_presentation_visuals.py — PRESENTATION visual selection (no training, no model/data changes).

Selects the 6 most architecturally-READABLE prediction cases per horizon (H20/60/100/200/400)
across MODEL_X, MODEL_XR_B_MAG, MODEL_XC_B_CURV_LIGHT — by a visual-interest score (length,
direction, moderate curvature, magnitude realism, spatial extent, recording diversity), NOT by
lowest error. Plots get generous padding, equal metric scaling, visible grid; never clipped tight.

Outputs presentation_visuals/H*/individual/*.png, H*/H*_collage.png, selected_visuals_inventory.csv.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
TP = HERE.parent
sys.path.insert(0, str(TP / "MODEL_XC"))
import xc_common as XCC      # noqa (brings xr_common + model_x_lib)
L = XCC.L
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "MODEL_X":   (TP / "experiments" / "MODEL_X" / "models" / "best_by_val_ADE.pth",
                  TP / "experiments" / "MODEL_X" / "models" / "scalers.json"),
    "XR_B_MAG":  (TP / "MODEL_XR" / "checkpoints" / "MODEL_XR_B_MAG" / "model_best.pth",
                  TP / "MODEL_XR" / "checkpoints" / "MODEL_XR_B_MAG" / "scalers.json"),
    "XC_CURV":   (TP / "MODEL_XC" / "checkpoints" / "MODEL_XC_B_CURV_LIGHT" / "model_best.pth",
                  TP / "MODEL_XC" / "checkpoints" / "MODEL_XC_B_CURV_LIGHT" / "scalers.json"),
}
HORIZONS = [20, 60, 100, 200, 400]
STRIDE = 2
MIN_GT_NET = {20: 1.0, 60: 2.0, 100: 3.0, 200: 5.0, 400: 8.0}
N_SEL = 6
PAD_FRAC = 0.25; PAD_MIN = 1.5
COL = {"hist": "#222222", "gt": "#1f77b4", "MODEL_X": "#ff7f0e", "XR_B_MAG": "#777777", "XC_CURV": "#e0218a"}


def load_model(name):
    ck, sc = MODELS[name]
    s = json.loads(Path(sc).read_text())
    f = L.ColumnScaler.from_dict(s["feature_scaler"]); t = L.ColumnScaler.from_dict(s["target_scaler"])
    m = L.TrajectoryLSTM(L.N_FEAT).to(device)
    m.load_state_dict(torch.load(ck, map_location=device)); m.eval()
    return m, f, t


def minmax(a):
    a = np.asarray(a, float); lo, hi = np.nanmin(a), np.nanmax(a)
    return np.zeros_like(a) if hi - lo < 1e-9 else (a - lo) / (hi - lo)


def limits(pts):
    """Landscape-friendly padded window (equal-aspect safe). Generous metric padding; short axis
    expanded to >=50% of long axis so neither axis is a razor-thin sliver. Never tight."""
    xmin, ymin = pts.min(0); xmax, ymax = pts.max(0)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    xspan, yspan = xmax - xmin, ymax - ymin
    pad = max(PAD_FRAC * max(xspan, yspan), PAD_MIN)   # uniform metric padding
    w, h = xspan + 2 * pad, yspan + 2 * pad
    longside = max(w, h)
    w = max(w, 0.5 * longside); h = max(h, 0.5 * longside)   # no sliver; keeps landscape feel
    return (cx - w / 2, cx + w / 2), (cy - h / 2, cy + h / 2)


def draw(ax, w, legend=False, small=False):
    ax.plot(w["hist"][:, 0], w["hist"][:, 1], "-", color=COL["hist"], lw=1.8, label="history", zorder=3)
    ax.plot(w["gt"][:, 0], w["gt"][:, 1], "-o", color=COL["gt"], ms=2.5, lw=2.0, label="GT future", zorder=4)
    if "MODEL_X" in w["preds"]:
        p = w["preds"]["MODEL_X"]; ax.plot(p[:, 0], p[:, 1], "--", color=COL["MODEL_X"], lw=1.5, label="MODEL_X", zorder=4)
    pr = w["preds"]["XR_B_MAG"]; ax.plot(pr[:, 0], pr[:, 1], "--", color=COL["XR_B_MAG"], lw=1.6, label="XR_B_MAG (mag)", zorder=4)
    pc = w["preds"]["XC_CURV"]; ax.plot(pc[:, 0], pc[:, 1], "-", color=COL["XC_CURV"], lw=2.6, label="XC_CURV", zorder=5)
    if len(pc) >= 2:
        ax.annotate("", xy=pc[-1], xytext=pc[-2], arrowprops=dict(arrowstyle="-|>", color=COL["XC_CURV"], lw=2.2), zorder=6)
    ax.plot(w["hist"][-1, 0], w["hist"][-1, 1], "o", color="black", ms=6, zorder=7)
    allpts = np.vstack([w["hist"], w["gt"], pr, pc] + ([w["preds"]["MODEL_X"]] if "MODEL_X" in w["preds"] else []))
    xl, yl = limits(allpts); ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.set_aspect("equal", adjustable="box"); ax.grid(True, alpha=.35, lw=.6)
    ax.tick_params(labelsize=7 if not small else 6)
    t = f"{w['recording'].replace('_01','')}  trk {w['track']}  H{w['H']}\nndR {w['xc_net_ratio']:.2f}"
    if not np.isnan(w["xc_shape"]):
        t += f"  shapeR {w['xc_shape']:.2f}"
    ax.set_title(t, fontsize=8.5 if not small else 7.5, loc="left")
    if legend:
        ax.legend(fontsize=6.5, loc="best", framealpha=.85)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df = XCC.load_df(); split = XCC.load_split(); wb = L.recording_world_bounds(df)
    lens = df.groupby("trajectory_id").size()
    models = {n: load_model(n) for n in MODELS}
    inv_rows, missing = [], []

    for H in HORIZONS:
        need = XCC.OBS + H
        te_ids = [t for t in split[split.split == "test"].trajectory_id if lens.get(t, 0) >= need]
        seeds, start, gt, wba, meta = L.build_eval_windows(df, te_ids, wb, n_steps=H, stride=STRIDE)
        M = len(seeds)
        preds = {}
        for n, (m, f, t) in models.items():
            pp = np.empty((M, H + 1, 2))
            for i in range(0, M, 16384):
                sl = slice(i, i + 16384)
                pp[sl] = L.rollout_world_batch(m, seeds[sl], start[sl], {k: v[sl] for k, v in wba.items()}, f, t, H, device)
            preds[n] = pp
        rec = np.array([mm["recording_id"] for mm in meta]); tid = np.array([mm["trajectory_id"] for mm in meta])
        starts = np.array([mm["start_idx"] for mm in meta])
        dC = XCC.enrich_full(preds["XC_CURV"], gt, rec, tid, H)
        dR = XCC.enrich_full(preds["XR_B_MAG"], gt, rec, tid, H)

        gt_net = dC.nd_gt.to_numpy(); xc_net = dC.nd_pred.to_numpy()
        xc_ratio = dC.nd_ratio.to_numpy(); xc_cos = dC.cosine.to_numpy()
        xc_shape = dC.shape_ratio.to_numpy(); xr_shape = dR.shape_ratio.to_numpy()
        gt_cum = dC.gt_cum_head.to_numpy(); gt_tort = dC.gt_tort.to_numpy()
        # bbox span over gt + both preds
        def span_arr():
            stk = np.concatenate([gt, preds["XC_CURV"], preds["XR_B_MAG"]], axis=1)
            mn = stk.min(1); mx = stk.max(1)
            return np.maximum(mx[:, 0] - mn[:, 0], mx[:, 1] - mn[:, 1])
        bbox = span_arr()

        # hard filters for readability — REQUIRE alignment, REJECT over-curved "snakes"
        shape_def = np.where(np.isfinite(xc_shape), xc_shape, 1.0)   # treat straightish as shape=1
        ok = (gt_net >= MIN_GT_NET[H]) & (xc_net >= 0.5 * MIN_GT_NET[H]) & (bbox >= 0.8 * MIN_GT_NET[H]) \
            & (xc_ratio >= 0.7) & (xc_ratio <= 1.4) & (gt_tort < 3.5) & (xc_cos >= 0.6) \
            & (gt_cum >= 18.0) & (gt_cum <= 200.0) & (shape_def <= 1.8) & np.isfinite(xc_ratio)
        idx = np.where(ok)[0]
        if len(idx) == 0:
            missing.append(f"H{H}: no windows passed readability filters"); continue

        # visual score: alignment-first, prefer shape near 1.0, penalise over-curvature
        s_gtnet = minmax(gt_net[idx]); s_xcnet = minmax(xc_net[idx]); s_bbox = minmax(bbox[idx])
        s_cos = np.clip((xc_cos[idx] - 0.6) / 0.4, 0, 1)                       # 0.6->0, 1.0->1
        s_curv = np.clip(1 - np.abs(gt_cum[idx] - 55) / 110, 0, 1)            # mild: prefer moderate GT curve
        s_ratio = 1 - np.clip(np.abs(xc_ratio[idx] - 1.0) - 0.2, 0, 0.6) / 0.6
        sh = np.where(np.isfinite(xc_shape[idx]), xc_shape[idx], 1.0)
        s_shapeband = 1 - np.clip(np.abs(sh - 1.0) - 0.3, 0, 1.0) / 1.0       # full in [0.7,1.3], fades, 0 by >=2.3
        score = (0.14*s_gtnet + 0.08*s_xcnet + 0.28*s_cos + 0.16*s_curv +
                 0.12*s_ratio + 0.10*s_bbox + 0.12*s_shapeband)
        order = idx[np.argsort(-score)]
        scoremap = {idx[k]: score[k] for k in range(len(idx))}

        # select 6 with recording diversity (<=2 per recording) AND <=1 window per track
        chosen, per_rec, seen_tracks = [], {}, set()
        for i in order:
            r = rec[i]; per_rec[r] = per_rec.get(r, 0)
            if per_rec[r] < 2 and tid[i] not in seen_tracks:
                chosen.append(i); per_rec[r] += 1; seen_tracks.add(tid[i])
            if len(chosen) == N_SEL:
                break
        for i in order:  # backfill if <6 (still avoid duplicate tracks)
            if len(chosen) == N_SEL:
                break
            if i not in chosen and tid[i] not in seen_tracks:
                chosen.append(i); seen_tracks.add(tid[i])

        # build window dicts
        df_by = {t: g.sort_values("timestep") for t, g in df[df.trajectory_id.isin(set(tid[chosen]))].groupby("trajectory_id")}
        def obs_path(i):
            g = df_by[tid[i]]; s = g.iloc[starts[i]:starts[i] + XCC.OBS]
            return np.column_stack([s.world_x.to_numpy(float), s.world_y.to_numpy(float)])
        wins = []
        for i in chosen:
            wins.append({"recording": rec[i], "track": str(tid[i]).split("__")[-1], "H": H,
                         "hist": obs_path(i), "gt": gt[i],
                         "preds": {n: preds[n][i] for n in MODELS},
                         "xc_net_ratio": float(xc_ratio[i]), "xc_shape": float(xc_shape[i]),
                         "i": int(i)})

        Hdir = HERE / f"H{H}"; indiv = Hdir / "individual"; indiv.mkdir(parents=True, exist_ok=True)
        collage_path = Hdir / f"H{H}_collage.png"
        # individuals (landscape)
        for rank, (i, w) in enumerate(zip(chosen, wins), 1):
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            draw(ax, w, legend=True)
            ip = indiv / f"H{H}_best_{rank:02d}.png"
            fig.tight_layout(); fig.savefig(ip, dpi=140); plt.close(fig)
            notes = []
            if not np.isnan(xr_shape[i]) and not np.isnan(xc_shape[i]) and xc_shape[i] > xr_shape[i] + 0.1:
                notes.append("XC adds curvature vs XR-straight")
            if 0.85 <= xc_ratio[i] <= 1.15:
                notes.append("realistic length")
            if xc_cos[i] > 0.8:
                notes.append("well-aligned")
            inv_rows.append({"horizon": f"H{H}", "rank": rank, "recording": rec[i],
                             "track_id": str(tid[i]).split("__")[-1], "window_id": int(i),
                             "gt_net_displacement": round(float(gt_net[i]), 3),
                             "model_xc_net_displacement": round(float(xc_net[i]), 3),
                             "model_xc_net_ratio": round(float(xc_ratio[i]), 3),
                             "model_xc_cosine": round(float(xc_cos[i]), 3),
                             "model_xc_shape_ratio": round(float(xc_shape[i]), 3) if not np.isnan(xc_shape[i]) else None,
                             "model_xr_net_ratio": round(float(dR.nd_ratio.to_numpy()[i]), 3),
                             "visual_score": round(float(scoremap[i]), 4),
                             "individual_plot_path": str(ip.relative_to(HERE)),
                             "collage_path": str(collage_path.relative_to(HERE)),
                             "selection_notes": "; ".join(notes) if notes else "readable architectural path"})
        # collage 2x3
        fig, axes = plt.subplots(2, 3, figsize=(19, 11))
        for k, ax in enumerate(axes.flat):
            if k < len(wins):
                draw(ax, wins[k], legend=(k == 0), small=True)
            else:
                ax.axis("off")
        fig.suptitle(f"6 best architectural prediction paths — H{H} (~{ {20:1,60:3,100:5,200:10,400:20}[H] } m)", fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(collage_path, dpi=140); plt.close(fig)
        print(f"[vis] H{H}: {len(idx)} candidates -> {len(chosen)} selected ({dict(per_rec)})")

    inv = pd.DataFrame(inv_rows); inv.to_csv(HERE / "selected_visuals_inventory.csv", index=False)
    (HERE / "_missing.json").write_text(json.dumps(missing, indent=2))
    print(f"[vis] inventory rows: {len(inv)}")
    if missing:
        print("[vis] notes:", missing)
    # top-3 overall by score
    if len(inv):
        top = inv.sort_values("visual_score", ascending=False).head(3)
        print("\nTop-3 visuals overall:")
        print(top[["horizon", "recording", "track_id", "model_xc_net_ratio", "model_xc_shape_ratio", "visual_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
