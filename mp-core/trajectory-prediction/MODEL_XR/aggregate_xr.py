"""
aggregate_xr.py — combine all MODEL_XR variant results into the required CSVs, figures,
and MODEL_XR_REPORT.md. Reads experiments/<variant>/{eval_base.json,horizon.json,
train_summary.json from checkpoints}. Re-rolls control + best variant for trajectory-example
figures (equal metric scaling, adjustable=box). Inference only; MODEL_X untouched.
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
sys.path.insert(0, str(HERE))
import xr_common as XC  # noqa: E402
L = XC.L

VARIANTS = ["MODEL_XR_A_CONTROL", "MODEL_XR_B_MAG", "MODEL_XR_C_MAG_LIGHT_DIR", "MODEL_XR_D_MAG_STRONGER"]
CONTROL = "MODEL_XR_A_CONTROL"
REP = HERE / "reports"; FIG = HERE / "figures"; REP.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
HORIZONS = [20, 60, 100, 200, 400]
MODELX_REF = {"ADE": 0.305, "FDE": 0.480}   # MODEL_X published test numbers
# success thresholds (documented in report)
STEP1_TARGET = (0.85, 1.05)
OVERSHOOT = 1.25            # step1_sm or nd_ratio above this = overshoot
ADE_WORSE_TOL = 1.15        # ADE > control*this = "substantially worse"
COS_DROP_TOL = 0.90         # cosine < control*this = "direction degraded"


def load(variant):
    ev = json.loads((HERE / "experiments" / variant / "eval_base.json").read_text())
    hz = json.loads((HERE / "experiments" / variant / "horizon.json").read_text())
    try:
        tr = json.loads((HERE / "checkpoints" / variant / "train_summary.json").read_text())
    except Exception:
        tr = {}
    return ev, hz, tr


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data = {v: load(v) for v in VARIANTS}

    # ── summary metrics (base H10, ALL) ──
    srows = []
    for v in VARIANTS:
        ev, _, tr = data[v]; a = ev["ALL"]
        srows.append({"variant": v, "lambda_mag": ev["lambda_mag"], "lambda_dir": ev["lambda_dir"],
                      "best_val_ADE": tr.get("best_val_ADE"), "best_epoch": tr.get("best_epoch"),
                      "ADE": a["ADE"], "FDE": a["FDE"], "ADE_med": a["ADE_med"], "FDE_med": a["FDE_med"],
                      "step1_ratio_med": a["step1_ratio_med"], "step1_sm_ratio_med": a["step1_sm_ratio_med"],
                      "nd_ratio_med": a["nd_ratio_med"], "sm_ratio_med": a["sm_ratio_med"],
                      "cum_ratio_med": a["cum_ratio_med"], "pct_nd_lt05": a["pct_nd_lt05"],
                      "cosine_med": a["cosine_med"], "cosine_t1_med": a["cosine_t1_med"],
                      "pct_cos_gt075": a["pct_cos_gt075"], "pct_cos_gt050": a["pct_cos_gt050"]})
    summ = pd.DataFrame(srows); summ.to_csv(REP / "model_xr_summary_metrics.csv", index=False)

    # ── step1 metrics ──
    summ[["variant", "step1_ratio_med", "step1_sm_ratio_med", "cosine_t1_med"]].to_csv(
        REP / "model_xr_step1_metrics.csv", index=False)

    # ── horizon metrics ──
    hrows = []
    for v in VARIANTS:
        _, hz, _ = data[v]
        for H in HORIZONS:
            a = hz["horizons"][str(H)]["ALL"]
            hrows.append({"variant": v, "horizon": H, "approx_dist_m": hz["horizons"][str(H)]["approx_dist_m"],
                          "ADE": a["ADE"], "FDE": a["FDE"], "cum_ratio_med": a["cum_ratio_med"],
                          "nd_ratio_med": a["nd_ratio_med"], "sm_ratio_med": a["sm_ratio_med"],
                          "cosine_med": a["cosine_med"], "step1_sm_ratio_med": a["step1_sm_ratio_med"]})
    hz_df = pd.DataFrame(hrows); hz_df.to_csv(REP / "model_xr_horizon_metrics.csv", index=False)

    # ── per-recording (base H10) ──
    prows = []
    for v in VARIANTS:
        ev, _, _ = data[v]
        for r, b in ev["per_recording"].items():
            prows.append({"variant": v, "recording": r, "n_windows": b["n_windows"],
                          "ADE": b["ADE"], "FDE": b["FDE"], "step1_sm_ratio_med": b["step1_sm_ratio_med"],
                          "nd_ratio_med": b["nd_ratio_med"], "sm_ratio_med": b["sm_ratio_med"],
                          "cosine_med": b["cosine_med"], "pct_nd_lt05": b["pct_nd_lt05"]})
    per = pd.DataFrame(prows); per.to_csv(REP / "model_xr_per_recording_metrics.csv", index=False)

    # ── variant comparison + failure flags ──
    ctrl = summ[summ.variant == CONTROL].iloc[0]
    crows, frows = [], []
    for _, row in summ.iterrows():
        v = row.variant
        d_step1 = row.step1_sm_ratio_med - ctrl.step1_sm_ratio_med
        d_nd = row.nd_ratio_med - ctrl.nd_ratio_med
        d_ade = row.ADE - ctrl.ADE
        d_cos = row.cosine_med - ctrl.cosine_med
        # per-recording: did plazas improve (espanya) not just red_bridge?
        esp = per[(per.variant == v) & (per.recording == "placa_espanya_01")]
        rb = per[(per.variant == v) & (per.recording == "red_bridge_combined_01")]
        esp_ctrl = per[(per.variant == CONTROL) & (per.recording == "placa_espanya_01")]
        esp_nd_gain = (float(esp.nd_ratio_med.iloc[0]) - float(esp_ctrl.nd_ratio_med.iloc[0])
                       if len(esp) and len(esp_ctrl) else float("nan"))
        crows.append({"variant": v, "ADE": row.ADE, "FDE": row.FDE,
                      "step1_sm_ratio_med": row.step1_sm_ratio_med, "d_step1_vs_ctrl": round(d_step1, 4),
                      "nd_ratio_med": row.nd_ratio_med, "d_nd_vs_ctrl": round(d_nd, 4),
                      "cosine_med": row.cosine_med, "d_cos_vs_ctrl": round(d_cos, 4),
                      "d_ade_vs_ctrl": round(d_ade, 4),
                      "espanya_nd_gain": round(esp_nd_gain, 4)})
        # failure flags (skip control)
        if v != CONTROL:
            flags = []
            if row.step1_sm_ratio_med > OVERSHOOT or row.nd_ratio_med > OVERSHOOT:
                flags.append("overshoot")
            if row.ADE > ctrl.ADE * ADE_WORSE_TOL:
                flags.append("ade_substantially_worse")
            if row.cosine_med < ctrl.cosine_med * COS_DROP_TOL:
                flags.append("direction_degraded")
            if (not np.isnan(esp_nd_gain)) and esp_nd_gain <= 0.0 and len(rb):
                rb_ctrl = per[(per.variant == CONTROL) & (per.recording == "red_bridge_combined_01")]
                if len(rb_ctrl) and float(rb.nd_ratio_med.iloc[0]) - float(rb_ctrl.nd_ratio_med.iloc[0]) > 0.02:
                    flags.append("improves_red_bridge_not_plazas")
            frows.append({"variant": v, "failure_flags": ";".join(flags) if flags else "none",
                          "step1_sm_ratio_med": row.step1_sm_ratio_med, "nd_ratio_med": row.nd_ratio_med,
                          "ADE": row.ADE, "cosine_med": row.cosine_med, "espanya_nd_gain": round(esp_nd_gain, 4)})
    pd.DataFrame(crows).to_csv(REP / "model_xr_variant_comparison.csv", index=False)
    fail_df = pd.DataFrame(frows); fail_df.to_csv(REP / "model_xr_failure_cases.csv", index=False)

    # ── choose best variant ──
    cand = summ[summ.variant != CONTROL].copy()
    def score(r):
        s = -abs(r.step1_sm_ratio_med - 1.0)                       # closeness to 1.0 (primary)
        if r.cosine_med < ctrl.cosine_med * COS_DROP_TOL: s -= 1.0  # punish direction loss
        if r.ADE > ctrl.ADE * ADE_WORSE_TOL: s -= 1.0              # punish ADE blowup
        if r.step1_sm_ratio_med > OVERSHOOT or r.nd_ratio_med > OVERSHOOT: s -= 0.5
        return s
    cand["score"] = cand.apply(score, axis=1)
    best = cand.sort_values("score", ascending=False).iloc[0].variant
    print(f"[agg] best variant = {best}")

    # ───────── figures ─────────
    vshort = {v: v.replace("MODEL_XR_", "") for v in VARIANTS}
    vc = {v: c for v, c in zip(VARIANTS, ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"])}

    def bar(metric, title, ylab, fname, ref=None):
        fig, ax = plt.subplots(figsize=(8, 5))
        xs = [vshort[v] for v in VARIANTS]; ys = [summ[summ.variant == v][metric].iloc[0] for v in VARIANTS]
        ax.bar(xs, ys, color=[vc[v] for v in VARIANTS])
        if ref is not None:
            ax.axhline(ref, ls="--", c="k", alpha=.5, label="MODEL_X ref" if metric in ("ADE", "FDE") else "ref")
            ax.legend()
        for i, y in enumerate(ys):
            ax.text(i, y, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(ylab); ax.set_title(title); ax.grid(alpha=.3, axis="y")
        plt.xticks(rotation=12); fig.tight_layout(); fig.savefig(FIG / fname, dpi=130); plt.close(fig)

    # ADE/FDE comparison (grouped)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(VARIANTS)); w = 0.38
    ax.bar(x - w/2, [summ[summ.variant == v].ADE.iloc[0] for v in VARIANTS], w, label="ADE", color="#1f77b4")
    ax.bar(x + w/2, [summ[summ.variant == v].FDE.iloc[0] for v in VARIANTS], w, label="FDE", color="#ff7f0e")
    ax.axhline(MODELX_REF["ADE"], ls="--", c="#1f77b4", alpha=.6); ax.axhline(MODELX_REF["FDE"], ls="--", c="#ff7f0e", alpha=.6)
    ax.set_xticks(x); ax.set_xticklabels([vshort[v] for v in VARIANTS], rotation=12)
    ax.set_ylabel("m"); ax.set_title("ADE/FDE by variant (dashed = MODEL_X ref)"); ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "variant_ADE_FDE_comparison.png", dpi=130); plt.close(fig)

    bar("step1_sm_ratio_med", "Step-1 smoothed magnitude ratio (target 0.85-1.05)",
        "pred/GT smoothed step1", "variant_step1_ratio_comparison.png", ref=1.0)
    bar("cosine_med", "Direction cosine (base H10) by variant", "cosine", "variant_cosine_similarity_comparison.png", ref=ctrl.cosine_med)

    # nd / sm ratio by horizon
    def hz_line(metric, title, ylab, fname):
        fig, ax = plt.subplots(figsize=(9, 5))
        for v in VARIANTS:
            s = hz_df[hz_df.variant == v].sort_values("horizon")
            ax.plot(s.horizon, s[metric], "-o", color=vc[v], label=vshort[v])
        ax.axhline(1.0, ls="--", c="k", alpha=.4)
        ax.set_xlabel("horizon"); ax.set_ylabel(ylab); ax.set_title(title); ax.set_xticks(HORIZONS)
        ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(FIG / fname, dpi=130); plt.close(fig)
    hz_line("nd_ratio_med", "Net-displacement ratio by horizon", "net-disp pred/GT",
            "variant_net_displacement_ratio_by_horizon.png")
    hz_line("sm_ratio_med", "Smoothed path-length ratio by horizon", "smoothed pred/GT",
            "variant_smoothed_path_ratio_by_horizon.png")

    # per-recording magnitude (nd_ratio) grouped by recording
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(XC.RECS)); w = 0.2
    for j, v in enumerate(VARIANTS):
        ys = [float(per[(per.variant == v) & (per.recording == r)].nd_ratio_med.iloc[0])
              if len(per[(per.variant == v) & (per.recording == r)]) else np.nan for r in XC.RECS]
        ax.bar(x + (j - 1.5) * w, ys, w, label=vshort[v], color=vc[v])
    ax.axhline(1.0, ls="--", c="k", alpha=.4)
    ax.set_xticks(x); ax.set_xticklabels([r.replace("_01", "") for r in XC.RECS], rotation=15, fontsize=8)
    ax.set_ylabel("net-disp pred/GT (H10)"); ax.set_title("Per-recording magnitude (net-disp ratio) by variant")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y"); fig.tight_layout()
    fig.savefig(FIG / "per_recording_magnitude_comparison.png", dpi=130); plt.close(fig)

    # focus recordings (espanya / stairs / red_bridge): ADE, nd_ratio, cosine bars
    def focus_fig(rec, fname):
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        for k, (m, lab) in enumerate([("ADE", "ADE (m)"), ("nd_ratio_med", "net-disp ratio"), ("cosine_med", "cosine")]):
            ys = [float(per[(per.variant == v) & (per.recording == rec)][m].iloc[0])
                  if len(per[(per.variant == v) & (per.recording == rec)]) else np.nan for v in VARIANTS]
            ax[k].bar([vshort[v] for v in VARIANTS], ys, color=[vc[v] for v in VARIANTS])
            ax[k].set_title(lab); ax[k].grid(alpha=.3, axis="y"); ax[k].tick_params(axis="x", rotation=20, labelsize=7)
            if m == "nd_ratio_med": ax[k].axhline(1.0, ls="--", c="k", alpha=.4)
        fig.suptitle(f"{rec} — variant comparison (base H10)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(FIG / fname, dpi=130); plt.close(fig)
    focus_fig("placa_espanya_01", "placa_espanya_variant_comparison.png")
    focus_fig("stairs_montjuic_01", "stairs_variant_comparison.png")
    focus_fig("red_bridge_combined_01", "red_bridge_variant_comparison.png")

    # ── trajectory example figures (control vs best, equal box) ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = XC.load_df(); split = XC.load_split(); wb = L.recording_world_bounds(df)
    Hx = 100; need = XC.OBS + Hx
    lens = df.groupby("trajectory_id").size()
    te_ids = [t for t in split[split.split == "test"].trajectory_id if lens.get(t, 0) >= need]

    def load_model(v):
        sc = json.loads((HERE / "checkpoints" / v / "scalers.json").read_text())
        f = L.ColumnScaler.from_dict(sc["feature_scaler"]); t = L.ColumnScaler.from_dict(sc["target_scaler"])
        m = L.TrajectoryLSTM(L.N_FEAT).to(device)
        m.load_state_dict(torch.load(HERE / "checkpoints" / v / "model_best.pth", map_location=device)); m.eval()
        return m, f, t

    mc, fc, tc = load_model(CONTROL); mb, fb, tb = load_model(best)
    pc, gt, rec, tid = XC.rollout_all(mc, df, te_ids, wb, fc, tc, Hx, device)
    pb, _, _, _ = XC.rollout_all(mb, df, te_ids, wb, fb, tb, Hx, device)
    dC = XC.enrich(pc, gt, rec, tid, Hx)
    # observed paths
    df_by = {t: g.sort_values("timestep") for t, g in df[df.trajectory_id.isin(set(tid))].groupby("trajectory_id")}
    meta_idx = {}
    # rebuild start_idx per window via build_eval_windows order
    _, _, _, _, meta = L.build_eval_windows(df, te_ids, wb, n_steps=Hx, stride=1)
    starts = np.array([m["start_idx"] for m in meta])

    def obs_path(i):
        g = df_by[tid[i]]; s = g.iloc[starts[i]:starts[i] + XC.OBS]
        return np.column_stack([s.world_x.to_numpy(float), s.world_y.to_numpy(float)])

    def panel(ax, i, show_ctrl=True):
        ob = obs_path(i)
        ax.plot(ob[:, 0], ob[:, 1], "-", color="black", lw=1.6, label="observed")
        ax.plot(gt[i, :, 0], gt[i, :, 1], "-", color="#7f7f7f", lw=2.4, label="GT")
        if show_ctrl:
            ax.plot(pc[i, :, 0], pc[i, :, 1], "--", color="#1f77b4", lw=1.5, label="control (MSE)")
        ax.plot(pb[i, :, 0], pb[i, :, 1], "-", color="#e0218a", lw=1.8, label=f"{vshort[best]}")
        ax.plot(ob[-1, 0], ob[-1, 1], "o", color="#333", ms=4)
        ax.set_aspect("equal", adjustable="box"); ax.tick_params(labelsize=6)
        ax.set_title(f"{rec[i].replace('_01','')} ndR c={dC.nd_ratio.iloc[i]:.2f}", fontsize=7)

    # best examples: long-GT moving windows
    mv = dC[dC.moving].copy()
    sel = mv.sort_values("nd_gt", ascending=False).head(60).sample(min(9, len(mv)), random_state=42).index.tolist() \
        if len(mv) >= 9 else mv.index.tolist()[:9]
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for k, ax in enumerate(axes.flat):
        if k < len(sel):
            panel(ax, sel[k])
            if k == 0: ax.legend(fontsize=6)
        else: ax.axis("off")
    fig.suptitle(f"Best variant ({vshort[best]}) vs control — rollout examples (H{Hx})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(FIG / "best_variant_rollout_examples.png", dpi=140); plt.close(fig)

    # failure examples: best-variant worst ADE or overshoot
    db = XC.enrich(pb, gt, rec, tid, Hx)
    fsel = db.sort_values("ade", ascending=False).head(9).index.tolist()
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for k, ax in enumerate(axes.flat):
        if k < len(fsel):
            i = fsel[k]; ob = obs_path(i)
            ax.plot(ob[:, 0], ob[:, 1], "-", color="black", lw=1.5, label="observed")
            ax.plot(gt[i, :, 0], gt[i, :, 1], "-", color="#7f7f7f", lw=2.4, label="GT")
            ax.plot(pb[i, :, 0], pb[i, :, 1], "-", color="#e0218a", lw=1.8, label=vshort[best])
            ax.plot(ob[-1, 0], ob[-1, 1], "o", color="#333", ms=4)
            ax.set_aspect("equal", adjustable="box"); ax.tick_params(labelsize=6)
            ax.set_title(f"{rec[i].replace('_01','')} ADE={db.ade.iloc[i]:.1f} ndR={db.nd_ratio.iloc[i]:.2f}", fontsize=7)
            if k == 0: ax.legend(fontsize=6)
        else: ax.axis("off")
    fig.suptitle(f"Failure / worst cases — {vshort[best]} (H{Hx})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(FIG / "failure_case_examples.png", dpi=140); plt.close(fig)

    (REP / "_best_variant.json").write_text(json.dumps({"best": best}, indent=2))
    print("[agg] CSVs + figures written. Summary:")
    print(summ[["variant", "ADE", "FDE", "step1_sm_ratio_med", "nd_ratio_med", "cosine_med"]].to_string(index=False))
    print("\nfailure flags:\n", fail_df.to_string(index=False))


if __name__ == "__main__":
    main()
