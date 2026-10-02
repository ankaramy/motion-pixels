"""Phase 5 — global metrics CSVs + plots."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import common as C

SPLITS = ["train", "val", "test"]
SPLIT_COLOR = {"train": "#3aaa5e", "val": "#4c8cbf", "test": "#e05c2a"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    C.D_METRICS.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(C.D_METRICS / "per_trajectory_metrics.csv")
    perstep = json.loads((C.D_METRICS / "per_step_errors.json").read_text())
    m["ang_deg"] = np.degrees(m.angular_error)

    # 1 metrics_summary.csv
    rows = []
    for s in SPLITS:
        d = m[m.split == s]
        rows.append({"split": s, "ADE_mean": d.ade.mean(), "ADE_median": d.ade.median(),
                     "FDE_mean": d.fde.mean(), "FDE_median": d.fde.median(),
                     "angular_error_mean_deg": d.ang_deg.mean(), "angular_error_median_deg": d.ang_deg.median(),
                     "mean_turn_pred": d.mean_turn_pred.mean(), "mean_turn_gt": d.mean_turn_gt.mean(),
                     "n_rollouts": len(d), "n_tracks": d.trajectory_id.nunique()})
    pd.DataFrame(rows).round(4).to_csv(C.D_METRICS / "metrics_summary.csv", index=False)

    # 2 per_recording_metrics.csv
    rr = []
    for rec in C.SPLIT_MAP:
        d = m[m.recording_id == rec]
        rr.append({"recording_id": rec, "split": C.SPLIT_MAP[rec],
                   "ADE_mean": d.ade.mean(), "ADE_median": d.ade.median(),
                   "FDE_mean": d.fde.mean(), "FDE_median": d.fde.median(),
                   "angular_error_mean_deg": d.ang_deg.mean(),
                   "mean_turn_pred": d.mean_turn_pred.mean(), "mean_turn_gt": d.mean_turn_gt.mean(),
                   "n_rollouts": len(d)})
    pd.DataFrame(rr).round(4).to_csv(C.D_METRICS / "per_recording_metrics.csv", index=False)

    # 3 error_by_horizon.csv (mean error at each step, per split + overall)
    split_of = dict(zip(m.trajectory_id.astype(str), m.split))
    horizon = {s: [[] for _ in range(C.N_ROLLOUT)] for s in SPLITS}
    allh = [[] for _ in range(C.N_ROLLOUT)]
    for tid, errs in perstep.items():
        s = split_of.get(str(tid))
        for i, e in enumerate(errs[:C.N_ROLLOUT]):
            allh[i].append(e)
            if s: horizon[s][i].append(e)
    hrows = []
    for i in range(C.N_ROLLOUT):
        row = {"step": i + 1, "overall_mean_err": np.mean(allh[i]) if allh[i] else np.nan,
               "n_overall": len(allh[i])}
        for s in SPLITS:
            row[f"{s}_mean_err"] = np.mean(horizon[s][i]) if horizon[s][i] else np.nan
            row[f"{s}_n"] = len(horizon[s][i])
        hrows.append(row)
    pd.DataFrame(hrows).round(5).to_csv(C.D_METRICS / "error_by_horizon.csv", index=False)

    # --- recompute per-step displacement (pred vs true) + residuals for val+test ---
    df = C.load_dataset(); bounds = C.load_world_bounds()
    f_sc, t_sc = C.load_scalers(C.D_TRAIN / "scalers.pkl")
    model = C.load_model(C.D_MODELS / "best_model_C_barcelona.pt", device)
    disp = {"val": {"pdx": [], "pdy": [], "gdx": [], "gdy": [], "res": []},
            "test": {"pdx": [], "pdy": [], "gdx": [], "gdy": [], "res": []}}
    for rec, key in [(C.VAL_REC, "val"), (C.TEST_REC, "test")]:
        wb = bounds[rec]; kdt = C.build_recording_kdt(df, rec)
        rec_df = df[df.recording_id == rec]
        for tid in rec_df.trajectory_id.unique():
            r = C.rollout_one_trajectory(model, rec_df[rec_df.trajectory_id == tid], wb, f_sc, t_sc, kdt, device)
            if r is None:
                continue
            pp, gp = r["pred_pos"], r["gt_pos"]
            n = min(len(pp), len(gp))
            pd_ = np.diff(pp[:n], axis=0); gd_ = np.diff(gp[:n], axis=0)
            disp[key]["pdx"] += pd_[:, 0].tolist(); disp[key]["pdy"] += pd_[:, 1].tolist()
            disp[key]["gdx"] += gd_[:, 0].tolist(); disp[key]["gdy"] += gd_[:, 1].tolist()
            disp[key]["res"] += r["per_step_err"].tolist()

    # ---- plots ----
    # ade_fde_by_split
    f, a = plt.subplots(figsize=(7, 4.6))
    x = np.arange(len(SPLITS)); w = 0.35
    a.bar(x - w/2, [m[m.split == s].ade.mean() for s in SPLITS], w, label="ADE mean", color="#4c8cbf")
    a.bar(x + w/2, [m[m.split == s].fde.mean() for s in SPLITS], w, label="FDE mean", color="#e05c2a")
    a.set_xticks(x); a.set_xticklabels(SPLITS); a.set_ylabel("metres"); a.set_title("ADE / FDE by split"); a.legend()
    f.tight_layout(); f.savefig(C.D_METRICS / "ade_fde_by_split.png"); plt.close(f)

    # angular_error_by_split
    f, a = plt.subplots(figsize=(6, 4.6))
    a.bar(SPLITS, [m[m.split == s].ang_deg.mean() for s in SPLITS], color=[SPLIT_COLOR[s] for s in SPLITS])
    a.set_ylabel("degrees"); a.set_title("Mean angular error by split"); a.axhline(90, color="grey", ls=":", lw=1)
    f.tight_layout(); f.savefig(C.D_METRICS / "angular_error_by_split.png"); plt.close(f)

    # error_by_prediction_horizon
    hd = pd.DataFrame(hrows)
    f, a = plt.subplots(figsize=(7.5, 4.6))
    for s in SPLITS:
        a.plot(hd.step, hd[f"{s}_mean_err"], "-o", ms=3, color=SPLIT_COLOR[s], label=s)
    a.set_xlabel("rollout step"); a.set_ylabel("mean error (m)"); a.set_title("Error by prediction horizon"); a.legend()
    f.tight_layout(); f.savefig(C.D_METRICS / "error_by_prediction_horizon.png"); plt.close(f)

    # predicted vs true displacement scatter (val+test)
    f, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, comp, lbl in zip(axes, ["dx", "dy"], ["du (world_x step)", "dv (world_y step)"]):
        for key, col in [("val", "#4c8cbf"), ("test", "#e05c2a")]:
            p = np.array(disp[key]["p" + comp]); g = np.array(disp[key]["g" + comp])
            ax.scatter(g, p, s=3, alpha=0.2, color=col, label=key)
        lim = 3
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel(f"true {lbl}"); ax.set_ylabel(f"predicted {lbl}"); ax.set_title(f"Predicted vs true {comp}"); ax.legend()
    f.suptitle("Predicted vs true per-step displacement (y=x is perfect)", fontweight="bold")
    f.tight_layout(rect=[0, 0, 1, 0.95]); f.savefig(C.D_METRICS / "predicted_vs_true_displacement_scatter.png"); plt.close(f)

    # residual error distribution
    f, a = plt.subplots(figsize=(7, 4.6))
    for key, col in [("val", "#4c8cbf"), ("test", "#e05c2a")]:
        a.hist(disp[key]["res"], bins=80, density=True, histtype="step", lw=1.6, color=col, label=f"{key} (median {np.median(disp[key]['res']):.2f} m)")
    a.set_xlim(0, 3); a.set_xlabel("per-step error (m)"); a.set_title("Residual (per-step) error distribution"); a.legend()
    f.tight_layout(); f.savefig(C.D_METRICS / "residual_error_distribution.png"); plt.close(f)

    # distributions val vs test
    for metric, fname, xlabel, xclip in [("ade", "ADE_distribution_validation_vs_test.png", "ADE (m)", 2.0),
                                          ("fde", "FDE_distribution_validation_vs_test.png", "FDE (m)", 3.0),
                                          ("ang_deg", "angular_error_distribution_validation_vs_test.png", "angular error (deg)", 180)]:
        f, a = plt.subplots(figsize=(7, 4.6))
        for s, col in [("val", "#4c8cbf"), ("test", "#e05c2a")]:
            d = m[m.split == s][metric]
            a.hist(d, bins=50, density=True, histtype="step", lw=1.6, color=col, label=f"{s} (median {d.median():.2f})")
        a.set_xlim(0, xclip); a.set_xlabel(xlabel); a.set_title(f"{xlabel} — validation vs test"); a.legend()
        f.tight_layout(); f.savefig(C.D_METRICS / fname); plt.close(f)

    print("Phase 5 done. metrics_summary:")
    print(pd.read_csv(C.D_METRICS / "metrics_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
