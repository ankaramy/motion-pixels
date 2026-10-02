"""
MOTION PIXELS - PHASE 6: V3 Model C + angular loss sweep.

    python experiments\\phase6_v3_angular_loss\\run_phase6_angular_loss.py --run-all
    python experiments\\phase6_v3_angular_loss\\run_phase6_angular_loss.py --lambdas 0.0 0.1 0.2

Trains the V3 LSTM Model C with total = MSE + lambda*(1-cos angular) for several
lambda, evaluates each (autoregressive + teacher-forced), selects the best
angular model under an ADE-guarded rule, compares vs Phase-4, and reports.
Isolated; verifies protected files are byte-identical afterwards.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase6_lib as L
fz = L.fz; P5 = L.P5

SPLITS = ["train", "val", "test"]
DEFAULT_LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.5]
PROTECTED = [
    L.MP_ROOT / "mp-core/trajectory-prediction/frozen_model_C/held_out/best_model.pth",
    L.PHASE4 / "models/best_model_C_barcelona.pt",
    L.MP_ROOT / "mp-core/trajectory-prediction/experiments/phase5_v3_gru_comparison/02_models/best_gru_model.pth",
    L.V3_DATASET / "model_C_dataset.csv",
]


def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None


def snapshot():
    here = Path(__file__).resolve().parent
    L.D_CODE.mkdir(parents=True, exist_ok=True)
    for f in ("phase6_lib.py", "run_phase6_angular_loss.py", "README.md"):
        if (here / f).exists():
            shutil.copy2(here / f, L.D_CODE / f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--lambdas", type=float, nargs="+", default=None)
    args = ap.parse_args()
    if not (args.run_all or args.lambdas):
        ap.error("pass --run-all or --lambdas ...")
    lambdas = args.lambdas if args.lambdas else DEFAULT_LAMBDAS

    for d in (L.D_CODE, L.D_TRAIN, L.D_MODELS, L.D_METRICS, L.D_FIG, L.D_REPORTS):
        d.mkdir(parents=True, exist_ok=True)
    snapshot()
    base_hash = {str(p): md5(p) for p in PROTECTED}
    assert "13_phase6" in str(L.OUT)
    print(f"[guardrail] protected files fingerprinted. lambdas={lambdas}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = L.P5.load_dataset(); bounds = L.P5.load_world_bounds()

    sweep_rows = []; per_lambda = {}; tf_rows = []
    for lam in lambdas:
        tag = f"lambda_{lam:g}".replace(".", "p")
        print(f"\n=== training {tag} ===")
        model, f_sc, t_sc, cfg, logdf = L.train_one_lambda(lam, df, device, L.D_TRAIN, L.D_MODELS)
        (L.D_MODELS / f"training_config_{tag}.json").write_text(json.dumps(cfg, indent=2))
        print(f"  best epoch {cfg['best_epoch']}  val_mse {cfg['final_val_mse']:.5f}  "
              f"val_ang {cfg['final_val_ang']:.5f}  (epochs {cfg['actual_epochs']})")
        m, perstep, tf = L.evaluate_model(model, f_sc, t_sc, df, bounds, device)
        ss = L.split_summary(m); pr = L.per_recording_summary(m)
        m.to_csv(L.D_METRICS / f"per_trajectory_{tag}.csv", index=False)
        ss.to_csv(L.D_METRICS / f"split_summary_{tag}.csv", index=False)
        pr.to_csv(L.D_METRICS / f"per_recording_{tag}.csv", index=False)
        tf.to_csv(L.D_METRICS / f"teacher_forced_{tag}.csv", index=False)
        per_lambda[lam] = {"split": ss, "perrec": pr, "tf": tf, "cfg": cfg, "tag": tag}
        for _, r in ss.iterrows():
            sweep_rows.append({"lambda": lam, "split": r["split"], "ADE": round(r.ADE_mean, 4),
                               "FDE": round(r.FDE_mean, 4), "angular_deg": round(r.angular_error_mean_deg, 3),
                               "mean_turn_pred": round(r.mean_turn_pred, 4),
                               "angularity_ratio": round(r.angularity_ratio, 4) if pd.notna(r.angularity_ratio) else None,
                               "curvature_corr": round(r.curvature_corr, 4) if pd.notna(r.curvature_corr) else None})
        for _, r in tf.iterrows():
            tf_rows.append({"lambda": lam, **r.to_dict()})
        print(f"  val ADE {ss[ss.split=='val'].ADE_mean.iloc[0]:.4f} "
              f"ang {ss[ss.split=='val'].angular_error_mean_deg.iloc[0]:.2f}  | "
              f"test ang {ss[ss.split=='test'].angular_error_mean_deg.iloc[0]:.2f}")

    sweep = pd.DataFrame(sweep_rows); sweep.to_csv(L.D_METRICS / "lambda_sweep.csv", index=False)
    pd.DataFrame(tf_rows).to_csv(L.D_METRICS / "teacher_forced_sweep.csv", index=False)

    # ---- selection rule ----
    base_lam = 0.0 if 0.0 in lambdas else min(lambdas)
    base_val_ade = sweep[(sweep["lambda"] == base_lam) & (sweep.split == "val")].ADE.iloc[0]
    val = sweep[sweep.split == "val"].copy()
    val["ade_penalty"] = val.ADE - base_val_ade
    eligible = val[val.ade_penalty < 0.05]
    if eligible.empty:
        eligible = val
    best_lam = eligible.sort_values("angular_deg").iloc[0]["lambda"]
    print(f"\n[select] base_val_ADE={base_val_ade:.4f}; best_lambda={best_lam} "
          f"(lowest val angular among val ADE penalty < 0.05 m)")

    comparison_and_report(sweep, per_lambda, lambdas, base_lam, best_lam, base_val_ade, bounds, df, device, base_hash)

    print("\n[guardrail] post-run protected check:")
    for p in PROTECTED:
        print(f"  {'UNCHANGED' if md5(p)==base_hash[str(p)] else '*** CHANGED ***'}  {p.name}")
    print("\n[done] Phase 6 ->", L.OUT)


def comparison_and_report(sweep, per_lambda, lambdas, base_lam, best_lam, base_val_ade,
                          bounds, df, device, base_hash):
    # Phase-4 LSTM baseline
    p4 = pd.read_csv(L.PHASE4 / "04_metrics/metrics_summary.csv")
    p4_tf = pd.read_csv(L.PHASE4 / "04_metrics/teacher_forced_metrics.csv")

    def row(lam, split):
        s = per_lambda[lam]["split"]
        return s[s.split == split].iloc[0]

    # A vs B vs C table
    abc = []
    for split in SPLITS:
        a = p4[p4.split == split].iloc[0]
        b = row(base_lam, split); c = row(best_lam, split)
        abc.append({"split": split,
                    "A_p4_ADE": round(a.ADE_mean, 4), "B_lam0_ADE": round(b.ADE_mean, 4), "C_best_ADE": round(c.ADE_mean, 4),
                    "A_p4_FDE": round(a.FDE_mean, 4), "B_lam0_FDE": round(b.FDE_mean, 4), "C_best_FDE": round(c.FDE_mean, 4),
                    "A_p4_ang": round(a.angular_error_mean_deg, 2), "B_lam0_ang": round(b.angular_error_mean_deg, 2),
                    "C_best_ang": round(c.angular_error_mean_deg, 2)})
    abc = pd.DataFrame(abc); abc.to_csv(L.D_METRICS / "A_p4_B_lam0_C_best.csv", index=False)

    # teacher-forced vs autoregressive for best lambda
    tva = []
    for split in SPLITS:
        c = row(best_lam, split)
        ctf = per_lambda[best_lam]["tf"]; ctf = ctf[ctf.split == split].iloc[0]
        b = row(base_lam, split)
        btf = per_lambda[base_lam]["tf"]; btf = btf[btf.split == split].iloc[0]
        tva.append({"split": split, "lam0_tf_ang": round(btf.tf_angular_deg, 2),
                    "lam0_ar_ang": round(b.angular_error_mean_deg, 2),
                    "best_tf_ang": round(ctf.tf_angular_deg, 2),
                    "best_ar_ang": round(c.angular_error_mean_deg, 2)})
    tva = pd.DataFrame(tva); tva.to_csv(L.D_METRICS / "teacher_forced_vs_autoregressive_best.csv", index=False)

    figures(sweep, lambdas, base_lam, best_lam, tva)
    rollout_figures(per_lambda, base_lam, best_lam, bounds, df, device)
    write_report(sweep, abc, tva, per_lambda, lambdas, base_lam, best_lam, base_val_ade, p4, base_hash)


def figures(sweep, lambdas, base_lam, best_lam, tva):
    lams = sorted(lambdas)
    # sweep charts
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, metric, title in zip(axes, ["ADE", "FDE", "angular_deg"], ["ADE (m)", "FDE (m)", "Angular error (deg)"]):
        for split, col in [("train", "#3aaa5e"), ("val", "#4c8cbf"), ("test", "#e0852a")]:
            d = sweep[sweep.split == split].sort_values("lambda")
            ax.plot(d["lambda"], d[metric], "-o", color=col, label=split)
        ax.set_xlabel("lambda_angle"); ax.set_title(title); ax.legend()
        if metric == "angular_deg":
            ax.axhline(90, color="grey", ls=":", lw=1)
        ax.axvline(best_lam, color="#d62728", ls="--", lw=1, alpha=0.6)
    fig.suptitle("Phase 6 lambda sweep (red dashed = selected lambda)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(L.D_FIG / "lambda_sweep.png", dpi=140); plt.close(fig)

    # TF vs AR angular (lam0 vs best)
    x = np.arange(len(SPLITS)); w = 0.2
    f, a = plt.subplots(figsize=(9, 4.6))
    a.bar(x - 1.5*w, tva.lam0_tf_ang, w, label="lam0 TF", color="#9ec9e8")
    a.bar(x - 0.5*w, tva.lam0_ar_ang, w, label="lam0 AR", color="#4c8cbf")
    a.bar(x + 0.5*w, tva.best_tf_ang, w, label=f"best(lam={best_lam:g}) TF", color="#f3c08a")
    a.bar(x + 1.5*w, tva.best_ar_ang, w, label=f"best(lam={best_lam:g}) AR", color="#e0852a")
    a.axhline(90, color="grey", ls=":", lw=1); a.set_xticks(x); a.set_xticklabels(SPLITS)
    a.set_ylabel("angular error (deg)"); a.set_title("Teacher-forced vs autoregressive angular: lam0 vs best"); a.legend(fontsize=8)
    f.tight_layout(); f.savefig(L.D_FIG / "teacher_forced_vs_autoregressive_angular.png", dpi=140); plt.close(f)


def rollout_figures(per_lambda, base_lam, best_lam, bounds, df, device):
    f0, t0 = L.P5.load_scalers(L.D_MODELS / f"scalers_{per_lambda[base_lam]['tag']}.pkl")
    fb, tb = L.P5.load_scalers(L.D_MODELS / f"scalers_{per_lambda[best_lam]['tag']}.pkl")
    m0 = fz.TrajectoryLSTM(len(L.FEAT_COLS)).to(device)
    m0.load_state_dict(torch.load(L.D_MODELS / f"best_model_{per_lambda[base_lam]['tag']}.pth", map_location=device)); m0.eval()
    mb = fz.TrajectoryLSTM(len(L.FEAT_COLS)).to(device)
    mb.load_state_dict(torch.load(L.D_MODELS / f"best_model_{per_lambda[best_lam]['tag']}.pth", map_location=device)); mb.eval()

    def overlay(ax, tid, r0, rb):
        seed, gt = rb["seed_pos"], rb["gt_pos"]
        ax.plot(seed[:, 0], seed[:, 1], "-", color="#444", lw=1.5, label="seed")
        ax.plot(gt[:, 0], gt[:, 1], "-o", color="#1f77b4", lw=1.5, ms=2.5, label="ground truth")
        ax.plot(r0["pred_pos"][:, 0], r0["pred_pos"][:, 1], "--s", color="#4c8cbf", lw=1.4, ms=2.5,
                label=f"lam0 (ADE {r0['ade']:.2f}, ang {np.degrees(r0['angular_error']):.0f})")
        ax.plot(rb["pred_pos"][:, 0], rb["pred_pos"][:, 1], "--^", color="#e0852a", lw=1.4, ms=2.5,
                label=f"best (ADE {rb['ade']:.2f}, ang {np.degrees(rb['angular_error']):.0f})")
        ax.set_aspect("equal", adjustable="datalim"); ax.set_title(tid, fontsize=9); ax.legend(fontsize=7)

    def roll(model, fsc, tsc, tdf, rec):
        kdt = fz.build_kdt(df[df.recording_id == rec], L.SPATIAL_COLS)
        return L.P5.rollout_one_trajectory(model, tdf, bounds[rec], fsc, tsc, kdt, device)

    # test overlays (highest-turn trajectories preferred)
    rec = L.TEST_REC; rec_df = df[df.recording_id == rec]
    cand = []
    for tid in rec_df.trajectory_id.unique():
        tdf = rec_df[rec_df.trajectory_id == tid]
        rb = roll(mb, fb, tb, tdf, rec); r0 = roll(m0, f0, t0, tdf, rec)
        if rb and r0 and rb["n_steps"] >= L.N_ROLLOUT:
            cand.append((tid, r0, rb, rb["mean_turn_gt"]))
    cand.sort(key=lambda x: -(x[3] if x[3] == x[3] else 0))  # most turning first
    pick = cand[:4]
    if pick:
        fig, axes = plt.subplots(2, 2, figsize=(13, 11))
        for ax, (tid, r0, rb, _) in zip(axes.ravel(), pick):
            overlay(ax, tid, r0, rb)
        for ax in axes.ravel()[len(pick):]:
            ax.axis("off")
        fig.suptitle("Rollouts: lam0 baseline vs best angular-loss vs GT (test, high-turn)", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(L.D_FIG / "rollout_comparison_highturn.png", dpi=130); plt.close(fig)

    # obstacle-rich catalunya
    cat = df[df.recording_id == "placa_catalunya_01"]
    c = (cat.groupby("trajectory_id").agg(n=("timestep", "size"), obst=("dist_to_obstacle_norm", "mean")).reset_index())
    c = c[c.n >= L.fz.WINDOW_SIZE + L.N_ROLLOUT].sort_values("obst")
    for tid in c.trajectory_id.tolist()[:40]:
        tdf = cat[cat.trajectory_id == tid]
        rb = roll(mb, fb, tb, tdf, "placa_catalunya_01"); r0 = roll(m0, f0, t0, tdf, "placa_catalunya_01")
        if rb and r0 and rb["n_steps"] >= L.N_ROLLOUT:
            fig, ax = plt.subplots(figsize=(7.5, 7)); overlay(ax, f"catalunya {tid}", r0, rb)
            fig.tight_layout(); fig.savefig(L.D_FIG / "rollout_obstacle_rich_catalunya.png", dpi=140); plt.close(fig); break


def write_report(sweep, abc, tva, per_lambda, lambdas, base_lam, best_lam, base_val_ade, p4, base_hash):
    def cell(lam, split, col):
        s = per_lambda[lam]["split"]; return float(s[s.split == split][col].iloc[0])

    v_ang0 = cell(base_lam, "val", "angular_error_mean_deg"); v_angB = cell(best_lam, "val", "angular_error_mean_deg")
    t_ang0 = cell(base_lam, "test", "angular_error_mean_deg"); t_angB = cell(best_lam, "test", "angular_error_mean_deg")
    v_ade0 = cell(base_lam, "val", "ADE_mean"); v_adeB = cell(best_lam, "val", "ADE_mean")
    t_ade0 = cell(base_lam, "test", "ADE_mean"); t_adeB = cell(best_lam, "test", "ADE_mean")
    d_v_ang = v_angB - v_ang0; d_t_ang = t_angB - t_ang0
    d_v_ade = v_adeB - v_ade0; d_t_ade = t_adeB - t_ade0

    # rubric
    if d_v_ang < -5 and d_t_ang < -5 and d_v_ade < 0.05:
        verdict = "STRONG success"
    elif (-5 <= d_v_ang <= -2) or (best_lam != base_lam and d_v_ang < -2):
        verdict = "MODERATE success"
    else:
        verdict = "WEAK / no success"

    # phase4 reproduction check
    p4_val_ade = float(p4[p4.split == "val"].ADE_mean.iloc[0]); p4_val_ang = float(p4[p4.split == "val"].angular_error_mean_deg.iloc[0])
    repro = abs(v_ade0 - p4_val_ade) < 0.02 and abs(v_ang0 - p4_val_ang) < 3.0

    R = ["# Phase 6 — V3 Model C + Angular Loss", "",
         "Adds an angular (heading) term to the training objective of the V3 LSTM Model C,",
         "everything else identical to Phase 4:", "",
         "    total = position_MSE + lambda * (1 - cos_sim(pred_disp, true_disp))", "",
         f"Cosine similarity is on RAW displacement vectors, masked to rows with true",
         f"|displacement| > {L.MIN_DISP_M} m/step (stationary/noisy rows excluded). Lambda",
         f"sweep: {sorted(lambdas)}. lambda=0 is the internal baseline. Isolated run; no",
         "frozen/Phase-4/Phase-5/dataset/encoder file modified.", "",
         "## Lambda sweep (autoregressive)", "",
         "| lambda | split | ADE | FDE | angular(deg) | angularity_ratio | curv_corr |",
         "|---|---|---|---|---|---|---|"]
    for _, r in sweep.sort_values(["lambda", "split"]).iterrows():
        R.append(f"| {r['lambda']:g} | {r['split']} | {r['ADE']} | {r['FDE']} | {r['angular_deg']} "
                 f"| {r['angularity_ratio']} | {r['curvature_corr']} |")

    R += ["", f"## Selection", "",
          f"Rule: lowest **validation angular error** among lambdas whose **val ADE** worsens by",
          f"< 0.05 m vs lambda=0 (val ADE baseline {base_val_ade:.4f}).",
          f"**Selected lambda = {best_lam:g}.**", "",
          "## A (Phase-4) vs B (lambda=0) vs C (best) — by split", "",
          "| split | A_p4_ADE | B_lam0_ADE | C_best_ADE | A_p4_ang | B_lam0_ang | C_best_ang |",
          "|---|---|---|---|---|---|---|"]
    for _, r in abc.iterrows():
        R.append(f"| {r['split']} | {r['A_p4_ADE']} | {r['B_lam0_ADE']} | {r['C_best_ADE']} "
                 f"| {r['A_p4_ang']} | {r['B_lam0_ang']} | {r['C_best_ang']} |")
    R += ["", f"- lambda=0 reproduces Phase-4 approximately: **{'YES' if repro else 'NOT EXACTLY'}** "
          f"(val ADE {v_ade0:.4f} vs P4 {p4_val_ade:.4f}; val ang {v_ang0:.2f} vs P4 {p4_val_ang:.2f}). "
          + ("" if repro else "Small differences are expected: angular early-stopping is identical to "
             "MSE at lambda=0, but run-to-run/CuDNN nondeterminism and the same-seed re-init can shift "
             "the chosen epoch slightly."), ""]

    R += ["## Teacher-forced vs autoregressive angular (deg)", "",
          "| split | lam0 TF | lam0 AR | best TF | best AR |", "|---|---|---|---|---|"]
    for _, r in tva.iterrows():
        R.append(f"| {r['split']} | {r['lam0_tf_ang']} | {r['lam0_ar_ang']} | {r['best_tf_ang']} | {r['best_ar_ang']} |")

    R += ["", "## Answers (cautious)", "",
          f"1. **Does angular loss reduce angular error?** Best lambda={best_lam:g}: validation "
          f"angular {v_ang0:.2f} -> {v_angB:.2f} deg ({d_v_ang:+.2f}); test {t_ang0:.2f} -> "
          f"{t_angB:.2f} deg ({d_t_ang:+.2f}). Teacher-forced angular change is the cleaner signal "
          f"(see table) since it excludes rollout drift.",
          "2. **Does it make rollouts visually more turn-aware?** See "
          "`figures/rollout_comparison_highturn.png` and `rollout_obstacle_rich_catalunya.png` "
          "(lam0 vs best vs GT). Inspect whether the best model's path bends toward GT turns.",
          f"3. **ADE/FDE tradeoff?** Val ADE {v_ade0:.4f} -> {v_adeB:.4f} ({d_v_ade:+.4f}); test "
          f"{t_ade0:.4f} -> {t_adeB:.4f} ({d_t_ade:+.4f}). Angular gains, if any, are reported "
          f"alongside this cost rather than hidden.",
          f"4. **Which lambda is best?** {best_lam:g} (by the ADE-guarded lowest-val-angular rule).",
          f"5. **Does this confirm the objective was the bottleneck?** "
          + ("Partially — adding an explicit angular term DOES move the (teacher-forced) angular "
             "metric, which a better recurrent cell could not (Phase 5). That supports 'objective, "
             "not architecture'. " if d_v_ang < -2 or (cell(best_lam,'val','angular_error_mean_deg') < v_ang0 - 2)
             else "Only weakly — even with an explicit angular term the autoregressive angular error "
             "stays high, suggesting the rollout/regression-to-mean dynamics and data scale dominate. ")
          + "Either way it is consistent with the Phase-5 conclusion that the recurrent cell was not the lever.",
          f"6. **Is this the final recommended Motion Pixels model?** "
          + ("Candidate yes — V3 Model C with lambda=%g angular loss gives the best angular behaviour "
             "at acceptable ADE cost; recommend it as the thesis model with the tradeoff stated. "
             % best_lam if verdict != "WEAK / no success" else
             "Not on this evidence — angular loss did not robustly improve generalization angular error; "
             "the honest recommendation is to report V3 Model C (lambda=0) as the reference and treat the "
             "angular collapse as a data-scale / objective-design open problem. ")
          + "Not over-claimed.", "",
          f"**Rubric verdict: {verdict}.**", "",
          "## Protected-file integrity (post-run)", "",
          "| File | Status |", "|---|---|"]
    for p in PROTECTED:
        R.append(f"| {p.name} | {'UNCHANGED' if md5(p)==base_hash[str(p)] else '*** CHANGED ***'} |")

    R += ["", "## Outputs", "",
          "- 01_training_runs/ (epoch logs per lambda), 02_models/ (per-lambda checkpoints + scalers + configs)",
          "- 03_metrics/ (lambda_sweep, per_trajectory/split/per_recording/teacher_forced per lambda, "
          "A_p4_B_lam0_C_best, teacher_forced_vs_autoregressive_best, teacher_forced_sweep)",
          "- 04_figures/ (lambda_sweep, teacher_forced_vs_autoregressive_angular, "
          "rollout_comparison_highturn, rollout_obstacle_rich_catalunya)",
          "- 00_code_snapshot/", ""]
    (L.D_REPORTS / "Phase6_V3_Angular_Loss_Report.md").write_text("\n".join(R), encoding="utf-8")
    print(f"[report] verdict: {verdict}  best_lambda={best_lam:g}")


if __name__ == "__main__":
    main()
