"""
aggregate_xc.py — combine MODEL_XC variant results into the required CSVs, figures, and
MODEL_XC_REPORT data. Reads experiments/<variant>/{eval_base.json,horizon.json} + checkpoints
train_summary. Re-rolls control + best variant for trajectory-example figures (equal box scaling).
Inference only; MODEL_X / MODEL_XR untouched.
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
sys.path.insert(0, str(HERE))
import xc_common as XCC  # noqa
L = XCC.L
REP = HERE / "reports"; FIG = HERE / "figures"; REP.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
HORIZONS = [20, 60, 100, 200, 400]
CONTROL = "MODEL_XC_A_BMAG_CONTROL"
# MODEL_XR_B_MAG reference (Phase 4/5A)
BMAG_REF = {"ADE": 0.362, "FDE": 0.603, "step1_sm": 1.036, "nd": 0.935, "cosine": 0.956, "shape": 0.043}
# guards (success criteria)
STEP1_OK = (0.85, 1.10); ND_OK = (0.85, 1.15); COS_MIN = 0.90; OVERSHOOT_MAX = 25.0


def variants():
    vs = []
    for c in ["A_BMAG_CONTROL", "B_CURV_LIGHT", "C_CURV_MED", "D_CURV_STRONG", "E_CURV_DIR_LIGHT"]:
        v = f"MODEL_XC_{c}"
        if (HERE / "experiments" / v / "eval_base.json").exists():
            vs.append(v)
    return vs


def load(v):
    ev = json.loads((HERE / "experiments" / v / "eval_base.json").read_text())
    hz = json.loads((HERE / "experiments" / v / "horizon.json").read_text())
    try:
        tr = json.loads((HERE / "checkpoints" / v / "train_summary.json").read_text())
    except Exception:
        tr = {}
    return ev, hz, tr


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    VS = variants(); data = {v: load(v) for v in VS}
    short = {v: v.replace("MODEL_XC_", "") for v in VS}
    cols = plt.cm.tab10(np.linspace(0, 1, len(VS))); vc = {v: cols[i] for i, v in enumerate(VS)}

    # ---- summary (base H10) ----
    srows = []
    for v in VS:
        ev, _, tr = data[v]; a = ev["ALL"]
        srows.append({"variant": v, "lambda_mag": ev["lambda_mag"], "lambda_curv": ev["lambda_curv"],
                      "lambda_dir": ev.get("lambda_dir", 0.0), "best_val_ADE": tr.get("best_val_ADE"),
                      "ADE": a["ADE"], "FDE": a["FDE"], "step1_sm_ratio_med": a["step1_sm_ratio_med"],
                      "nd_ratio_med": a["nd_ratio_med"], "sm_ratio_med": a["sm_ratio_med"],
                      "overshoot_rate": a["overshoot_rate_nd_gt125"], "cosine_med": a["cosine_med"],
                      "heading_err_med": a["heading_err_med"], "pct_cos_gt075": a["pct_cos_gt075"],
                      "pct_cos_gt050": a["pct_cos_gt050"], "shape_ratio_med": a["shape_ratio_med"],
                      "tort_ratio_med": a["tort_ratio_med"], "turn_count_ratio_med": a["turn_count_ratio_med"],
                      "curv_density_ratio_med": a["curv_density_ratio_med"],
                      "pred_macro_turns_med": a["pred_macro_turns_med"]})
    summ = pd.DataFrame(srows); summ.to_csv(REP / "model_xc_summary_metrics.csv", index=False)

    # ---- curvature-only csv ----
    summ[["variant", "shape_ratio_med", "tort_ratio_med", "turn_count_ratio_med",
          "curv_density_ratio_med", "pred_macro_turns_med"]].to_csv(REP / "model_xc_curvature_metrics.csv", index=False)

    # ---- horizon ----
    hrows = []
    for v in VS:
        _, hz, _ = data[v]
        for H in HORIZONS:
            a = hz["horizons"][str(H)]["ALL"]
            hrows.append({"variant": v, "horizon": H, "ADE": a["ADE"], "FDE": a["FDE"],
                          "nd_ratio_med": a["nd_ratio_med"], "sm_ratio_med": a["sm_ratio_med"],
                          "cosine_med": a["cosine_med"], "shape_ratio_med": a["shape_ratio_med"],
                          "overshoot_rate": a["overshoot_rate_nd_gt125"]})
    hz_df = pd.DataFrame(hrows); hz_df.to_csv(REP / "model_xc_horizon_metrics.csv", index=False)

    # ---- per-recording (base) ----
    prows = []
    for v in VS:
        ev, _, _ = data[v]
        for r, b in ev["per_recording"].items():
            prows.append({"variant": v, "recording": r, "ADE": b["ADE"], "nd_ratio_med": b["nd_ratio_med"],
                          "cosine_med": b["cosine_med"], "shape_ratio_med": b["shape_ratio_med"],
                          "turn_count_ratio_med": b["turn_count_ratio_med"]})
    per = pd.DataFrame(prows); per.to_csv(REP / "model_xc_per_recording_metrics.csv", index=False)

    # ---- topology ----
    trows = []
    for v in VS:
        ev, _, _ = data[v]
        for grp in ("OPEN", "CONSTRAINED"):
            b = ev["topology"][grp]
            trows.append({"variant": v, "topology": grp, "shape_ratio_med": b["shape_ratio_med"],
                          "nd_ratio_med": b["nd_ratio_med"], "cosine_med": b["cosine_med"]})
    topo = pd.DataFrame(trows); topo.to_csv(REP / "model_xc_topology_metrics.csv", index=False)

    # ---- variant comparison + flags ----
    ctrl = summ[summ.variant == CONTROL].iloc[0]
    crows, frows = [], []
    for _, r in summ.iterrows():
        v = r.variant
        crows.append({"variant": v, "shape_ratio_med": r.shape_ratio_med,
                      "d_shape_vs_ctrl": round(r.shape_ratio_med - ctrl.shape_ratio_med, 4),
                      "step1_sm_ratio_med": r.step1_sm_ratio_med, "nd_ratio_med": r.nd_ratio_med,
                      "cosine_med": r.cosine_med, "ADE": r.ADE, "d_ade_vs_ctrl": round(r.ADE - ctrl.ADE, 4),
                      "overshoot_rate": r.overshoot_rate})
        if v != CONTROL:
            fl = []
            if not (STEP1_OK[0] <= r.step1_sm_ratio_med <= STEP1_OK[1]): fl.append("magnitude_step1_out_of_band")
            if not (ND_OK[0] <= r.nd_ratio_med <= ND_OK[1]): fl.append("nd_ratio_out_of_band")
            if r.cosine_med < COS_MIN: fl.append("direction_degraded")
            if r.overshoot_rate > OVERSHOOT_MAX: fl.append("overshoot")
            if r.shape_ratio_med <= ctrl.shape_ratio_med + 0.01: fl.append("no_shape_gain")
            frows.append({"variant": v, "failure_flags": ";".join(fl) if fl else "none",
                          "shape_ratio_med": r.shape_ratio_med, "step1_sm_ratio_med": r.step1_sm_ratio_med,
                          "nd_ratio_med": r.nd_ratio_med, "cosine_med": r.cosine_med,
                          "ADE": r.ADE, "overshoot_rate": r.overshoot_rate})
    pd.DataFrame(crows).to_csv(REP / "model_xc_variant_comparison.csv", index=False)
    fail = pd.DataFrame(frows); fail.to_csv(REP / "model_xc_failure_cases.csv", index=False)

    # ---- choose best ----
    cand = summ[summ.variant != CONTROL].copy()
    def ok(r):
        return (STEP1_OK[0] <= r.step1_sm_ratio_med <= STEP1_OK[1] and ND_OK[0] <= r.nd_ratio_med <= ND_OK[1]
                and r.cosine_med >= COS_MIN and r.overshoot_rate <= OVERSHOOT_MAX)
    passing = cand[cand.apply(ok, axis=1)]
    pool = passing if len(passing) else cand
    best = pool.sort_values("shape_ratio_med", ascending=False).iloc[0].variant
    print(f"[agg] best variant = {best} (passing guards: {list(passing.variant)})")
    (REP / "_best_variant.json").write_text(json.dumps({"best": best, "passing": list(passing.variant)}, indent=2))

    # ===== figures =====
    def bars(metric, title, ylab, fname, ref=None, reflab="B_MAG ref"):
        fig, ax = plt.subplots(figsize=(9, 5))
        xs = [short[v] for v in VS]; ys = [summ[summ.variant == v][metric].iloc[0] for v in VS]
        ax.bar(xs, ys, color=[vc[v] for v in VS])
        if ref is not None: ax.axhline(ref, ls="--", c="k", alpha=.5, label=reflab); ax.legend()
        for i, y in enumerate(ys): ax.text(i, y, f"{y:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel(ylab); ax.set_title(title); ax.grid(alpha=.3, axis="y"); plt.xticks(rotation=12)
        fig.tight_layout(); fig.savefig(FIG / fname, dpi=130); plt.close(fig)

    # ADE/FDE
    fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(VS)); w = 0.38
    ax.bar(x - w/2, [summ[summ.variant == v].ADE.iloc[0] for v in VS], w, label="ADE", color="#1f77b4")
    ax.bar(x + w/2, [summ[summ.variant == v].FDE.iloc[0] for v in VS], w, label="FDE", color="#ff7f0e")
    ax.axhline(BMAG_REF["ADE"], ls="--", c="#1f77b4", alpha=.6); ax.axhline(BMAG_REF["FDE"], ls="--", c="#ff7f0e", alpha=.6)
    ax.set_xticks(x); ax.set_xticklabels([short[v] for v in VS], rotation=12); ax.set_ylabel("m")
    ax.set_title("ADE/FDE by variant (dashed = MODEL_XR_B_MAG ref)"); ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "xc_variant_ADE_FDE_comparison.png", dpi=130); plt.close(fig)

    # magnitude preservation (step1_sm + nd)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, [summ[summ.variant == v].step1_sm_ratio_med.iloc[0] for v in VS], w, label="step1 smoothed", color="#2ca02c")
    ax.bar(x + w/2, [summ[summ.variant == v].nd_ratio_med.iloc[0] for v in VS], w, label="net-disp ratio", color="#9467bd")
    ax.axhline(1.0, ls="--", c="k", alpha=.5); ax.axhspan(0.85, 1.10, color="green", alpha=.07)
    ax.set_xticks(x); ax.set_xticklabels([short[v] for v in VS], rotation=12); ax.set_ylabel("ratio")
    ax.set_title("Magnitude preservation (shaded = success band)"); ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "xc_magnitude_preservation.png", dpi=130); plt.close(fig)

    bars("cosine_med", "Direction cosine by variant", "cosine", "xc_cosine_similarity_comparison.png", ref=BMAG_REF["cosine"])
    bars("shape_ratio_med", "Shape ratio (pred/GT cum heading) by variant", "shape ratio", "xc_shape_ratio_comparison.png", ref=BMAG_REF["shape"])

    # shape ratio vs horizon
    fig, ax = plt.subplots(figsize=(9, 5))
    for v in VS:
        s = hz_df[hz_df.variant == v].sort_values("horizon")
        ax.plot(s.horizon, s.shape_ratio_med, "-o", label=short[v], color=vc[v])
    ax.set_xticks(HORIZONS); ax.set_xlabel("horizon"); ax.set_ylabel("shape ratio")
    ax.set_title("Shape ratio vs horizon"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "xc_curvature_vs_horizon.png", dpi=130); plt.close(fig)

    # open vs constrained shape
    fig, ax = plt.subplots(figsize=(9, 5))
    op = [topo[(topo.topology == "OPEN") & (topo.variant == v)].shape_ratio_med.iloc[0] for v in VS]
    cn = [topo[(topo.topology == "CONSTRAINED") & (topo.variant == v)].shape_ratio_med.iloc[0] for v in VS]
    ax.bar(x - w/2, op, w, label="OPEN", color="#1f77b4"); ax.bar(x + w/2, cn, w, label="CONSTRAINED", color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels([short[v] for v in VS], rotation=12); ax.set_ylabel("shape ratio")
    ax.set_title("Shape preservation: open vs constrained"); ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "xc_open_vs_constrained_shape.png", dpi=130); plt.close(fig)

    # per-site comparisons
    def site(rec, fname):
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        for k, (mc, lab) in enumerate([("shape_ratio_med", "shape"), ("nd_ratio_med", "net-disp"), ("cosine_med", "cosine")]):
            ys = [float(per[(per.variant == v) & (per.recording == rec)][mc].iloc[0]) if len(per[(per.variant == v) & (per.recording == rec)]) else np.nan for v in VS]
            ax[k].bar([short[v] for v in VS], ys, color=[vc[v] for v in VS]); ax[k].set_title(lab)
            ax[k].grid(alpha=.3, axis="y"); ax[k].tick_params(axis="x", rotation=20, labelsize=6)
            if mc == "nd_ratio_med": ax[k].axhline(1.0, ls="--", c="k", alpha=.4)
        fig.suptitle(f"{rec} — XC variants (base H10)"); fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(FIG / fname, dpi=130); plt.close(fig)
    site("placa_espanya_01", "xc_placa_espanya_comparison.png")
    site("stairs_montjuic_01", "xc_stairs_comparison.png")

    # tradeoff bar (normalized: shape, nd-closeness, cosine, step1-closeness, ADE-inv)
    fig, ax = plt.subplots(figsize=(11, 5))
    metrics = ["shape", "nd_close", "cosine", "step1_close", "ADE_inv"]
    xt = np.arange(len(metrics)); bw = 0.8 / len(VS)
    for i, v in enumerate(VS):
        r = summ[summ.variant == v].iloc[0]
        vals = [min(r.shape_ratio_med, 1.0), max(0, 1 - abs(r.nd_ratio_med - 1)), r.cosine_med,
                max(0, 1 - abs(r.step1_sm_ratio_med - 1)), BMAG_REF["ADE"] / max(r.ADE, 1e-9)]
        ax.bar(xt + (i - len(VS)/2) * bw + bw/2, vals, bw, label=short[v], color=vc[v])
    ax.set_xticks(xt); ax.set_xticklabels(metrics); ax.set_ylabel("normalized (higher=better)")
    ax.set_title("MODEL_XC tradeoff across objectives (shape vs magnitude/direction/accuracy)")
    ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y"); fig.tight_layout()
    fig.savefig(FIG / "xc_tradeoff_radar_or_bar.png", dpi=130); plt.close(fig)

    # ===== example figures (control vs best vs GT) @ H100 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = XCC.load_df(); split = XCC.load_split(); wb = L.recording_world_bounds(df)
    Hx = 100; need = XCC.OBS + Hx; lens = df.groupby("trajectory_id").size()
    te_ids = [t for t in split[split.split == "test"].trajectory_id if lens.get(t, 0) >= need]

    def lm(v):
        sc = json.loads((HERE / "checkpoints" / v / "scalers.json").read_text())
        f = L.ColumnScaler.from_dict(sc["feature_scaler"]); t = L.ColumnScaler.from_dict(sc["target_scaler"])
        m = L.TrajectoryLSTM(L.N_FEAT).to(device)
        m.load_state_dict(torch.load(HERE / "checkpoints" / v / "model_best.pth", map_location=device)); m.eval()
        return m, f, t
    mc, fc, tc = lm(CONTROL); mb, fb, tb = lm(best)
    pc, gt, rec, tid = XCC.XC.rollout_all(mc, df, te_ids, wb, fc, tc, Hx, device)
    pb, _, _, _ = XCC.XC.rollout_all(mb, df, te_ids, wb, fb, tb, Hx, device)
    dC = XCC.enrich_full(pc, gt, rec, tid, Hx); dB = XCC.enrich_full(pb, gt, rec, tid, Hx)
    _, _, _, _, meta = L.build_eval_windows(df, te_ids, wb, n_steps=Hx, stride=1)
    starts = np.array([m["start_idx"] for m in meta])
    df_by = {t: g.sort_values("timestep") for t, g in df[df.trajectory_id.isin(set(tid))].groupby("trajectory_id")}

    def obs_path(i):
        g = df_by[tid[i]]; s = g.iloc[starts[i]:starts[i] + XCC.OBS]
        return np.column_stack([s.world_x.to_numpy(float), s.world_y.to_numpy(float)])

    def panel(ax, i):
        ob = obs_path(i)
        ax.plot(ob[:, 0], ob[:, 1], "-", color="black", lw=1.3, label="observed")
        ax.plot(gt[i, :, 0], gt[i, :, 1], "-", color="#7f7f7f", lw=2.4, label="GT")
        ax.plot(pc[i, :, 0], pc[i, :, 1], "--", color="#1f77b4", lw=1.4, label="control(B_MAG)")
        ax.plot(pb[i, :, 0], pb[i, :, 1], "-", color="#e0218a", lw=1.8, label=short[best])
        ax.plot(ob[-1, 0], ob[-1, 1], "o", color="#333", ms=4)
        ax.set_aspect("equal", adjustable="box"); ax.tick_params(labelsize=6)
        ax.set_title(f"{rec[i].replace('_01','')} shapeR {dC.shape_ratio.iloc[i]:.2f}->{dB.shape_ratio.iloc[i]:.2f}", fontsize=7)

    # best shape examples: control straight (shape<0.3), best improved (shape>control+0.15), moving, decent cosine
    curvy = dC.curvy.to_numpy() & dB.curvy.to_numpy()
    impr = (curvy & (dC.shape_ratio.to_numpy() < 0.35)
            & (dB.shape_ratio.to_numpy() > dC.shape_ratio.to_numpy() + 0.12)
            & (dB.cosine.to_numpy() > 0.6))
    idx = np.where(impr)[0]
    rng = np.random.default_rng(42); rng.shuffle(idx)
    chosen, seen = [], {}
    for i in idx:
        seen[rec[i]] = seen.get(rec[i], 0)
        if seen[rec[i]] < 2: chosen.append(i); seen[rec[i]] += 1
        if len(chosen) == 9: break
    for i in idx:
        if len(chosen) == 9: break
        if i not in chosen: chosen.append(i)
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for k, axx in enumerate(axes.flat):
        if k < len(chosen):
            panel(axx, chosen[k])
            if k == 0: axx.legend(fontsize=6)
        else: axx.axis("off")
    fig.suptitle(f"Best shape recovery — control(B_MAG, blue) vs {short[best]} (magenta) vs GT (H{Hx})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(FIG / "xc_best_shape_examples.png", dpi=140); plt.close(fig)

    # worst failures of best variant: highest ADE among curvy
    fsel = dB[dB.curvy].sort_values("ade", ascending=False).head(9).index.tolist()
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for k, axx in enumerate(axes.flat):
        if k < len(fsel):
            i = fsel[k]; ob = obs_path(i)
            axx.plot(ob[:, 0], ob[:, 1], "-", color="black", lw=1.3)
            axx.plot(gt[i, :, 0], gt[i, :, 1], "-", color="#7f7f7f", lw=2.4, label="GT")
            axx.plot(pb[i, :, 0], pb[i, :, 1], "-", color="#e0218a", lw=1.8, label=short[best])
            axx.plot(ob[-1, 0], ob[-1, 1], "o", color="#333", ms=4)
            axx.set_aspect("equal", adjustable="box"); axx.tick_params(labelsize=6)
            axx.set_title(f"{rec[i].replace('_01','')} ADE={dB.ade.iloc[i]:.1f} shapeR={dB.shape_ratio.iloc[i]:.2f}", fontsize=7)
            if k == 0: axx.legend(fontsize=6)
        else: axx.axis("off")
    fig.suptitle(f"Worst cases — {short[best]} (H{Hx})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(FIG / "xc_worst_failure_examples.png", dpi=140); plt.close(fig)

    print("[agg] done. summary:")
    print(summ[["variant", "ADE", "step1_sm_ratio_med", "nd_ratio_med", "cosine_med", "shape_ratio_med", "overshoot_rate"]].to_string(index=False))
    print("\nfailure flags:\n", fail.to_string(index=False))


if __name__ == "__main__":
    main()
