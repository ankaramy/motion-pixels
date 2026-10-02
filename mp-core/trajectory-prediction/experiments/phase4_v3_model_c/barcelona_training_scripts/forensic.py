"""
forensic.py — READ-ONLY root-cause investigation of Model C angular failure.
No retraining, no model/dataset/code modification. Uses the frozen checkpoint
and frozen helpers via common.py.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import common as C

fz = C.fz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTDIR = C.OUT / "root_cause_figures"
SPLITS = ["train", "val", "test"]
SCOL = {"train": "#3aaa5e", "val": "#4c8cbf", "test": "#e05c2a"}


def ang_between(pdu, pdv, tdu, tdv):
    ph = np.arctan2(pdv, pdu); th = np.arctan2(tdv, tdu)
    d = np.arctan2(np.sin(ph - th), np.cos(ph - th))
    return np.abs(d)


def teacher_forced(df, f_sc, t_sc, model, split):
    """Single-step teacher-forced prediction over ALL windows in a split."""
    ids = df[df.split == split].trajectory_id.unique().tolist()
    X, Y = fz.make_windows(df, C.FEAT_COLS, ids)             # X:(N,10,10) Y:(N,2) true next disp
    Xs = f_sc.transform(X.reshape(-1, 10)).reshape(X.shape).astype(np.float32)
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(Xs), 8192):
            xb = torch.tensor(Xs[i:i+8192], device=device)
            preds.append(model(xb).cpu().numpy())
    P = t_sc.inverse(np.concatenate(preds))                  # predicted disp (m), raw units
    pdu, pdv = P[:, 0], P[:, 1]
    tdu, tdv = Y[:, 0], Y[:, 1]
    step_err = np.hypot(pdu - tdu, pdv - tdv)                # teacher-forced position step error (m)
    moving = (np.hypot(tdu, tdv) > 1e-3) & (np.hypot(pdu, pdv) > 1e-3)
    ang = ang_between(pdu[moving], pdv[moving], tdu[moving], tdv[moving])
    return {"n_windows": len(P), "tf_ade_step": float(step_err.mean()),
            "tf_step_median": float(np.median(step_err)),
            "tf_angular_deg": float(np.degrees(ang.mean())),
            "tf_angular_median_deg": float(np.degrees(np.median(ang))),
            "pred_disp_norm_mean": float(np.hypot(pdu, pdv).mean()),
            "true_disp_norm_mean": float(np.hypot(tdu, tdv).mean()),
            "pdu": pdu, "pdv": pdv, "tdu": tdu, "tdv": tdv}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = C.load_dataset()
    f_sc, t_sc = C.load_scalers(C.D_TRAIN / "scalers.pkl")
    model = C.load_model(C.D_MODELS / "best_model_C_barcelona.pt", device)
    ar = pd.read_csv(C.D_METRICS / "per_trajectory_metrics.csv")
    ar["ang_deg"] = np.degrees(ar.angular_error)

    # ---------- TASK 1 + 4: teacher forced ----------
    tf = {s: teacher_forced(df, f_sc, t_sc, model, s) for s in SPLITS}

    cmp_rows = []
    for s in SPLITS:
        a = ar[ar.split == s]
        cmp_rows.append({
            "split": s,
            "AR_ADE_m": round(a.ade.mean(), 4), "AR_FDE_m": round(a.fde.mean(), 4),
            "AR_angular_deg": round(a.ang_deg.mean(), 2),
            "TF_step_err_m": round(tf[s]["tf_ade_step"], 4),
            "TF_angular_deg": round(tf[s]["tf_angular_deg"], 2),
            "pred_disp_mean_m": round(tf[s]["pred_disp_norm_mean"], 4),
            "true_disp_mean_m": round(tf[s]["true_disp_norm_mean"], 4),
            "n_windows": tf[s]["n_windows"],
        })
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(OUTDIR / "teacher_forced_vs_autoregressive.csv", index=False)

    # TF displacement scatter (val + test)
    for s in ["val", "test"]:
        f, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, comp, P, T in [(axes[0], "du", tf[s]["pdu"], tf[s]["tdu"]),
                               (axes[1], "dv", tf[s]["pdv"], tf[s]["tdv"])]:
            ax.scatter(T, P, s=3, alpha=0.15, color=SCOL[s])
            lim = 3; ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
            # least-squares slope through origin-ish
            b = np.polyfit(T, P, 1)
            xs = np.linspace(-lim, lim, 10); ax.plot(xs, b[0]*xs + b[1], "r-", lw=1.2, label=f"fit slope={b[0]:.2f}")
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
            ax.set_xlabel(f"true {comp} (m)"); ax.set_ylabel(f"pred {comp} (m)")
            ax.set_title(f"{s}: teacher-forced {comp}"); ax.legend()
        f.suptitle(f"Teacher-forced predicted vs true displacement — {s} (slope<1 = regression to mean)", fontweight="bold")
        f.tight_layout(rect=[0, 0, 1, 0.95])
        f.savefig(OUTDIR / f"predicted_vs_true_displacement_teacher_forced_{s}.png"); plt.close(f)
    # combined name the brief asked for (test, the held-out exam)
    import shutil
    shutil.copy(OUTDIR / "predicted_vs_true_displacement_teacher_forced_test.png",
                OUTDIR / "predicted_vs_true_displacement_teacher_forced.png")

    # ---------- TASK 5: turn_rate distribution ----------
    tr_rows = []
    for s in SPLITS:
        tr = df.loc[df.split == s, "turn_rate"].to_numpy()
        atr = np.abs(tr)
        tr_rows.append({"split": s, "mean_abs_turn_rate": round(atr.mean(), 4),
                        "median_turn_rate": round(float(np.median(tr)), 4),
                        "median_abs_turn_rate": round(float(np.median(atr)), 4),
                        "p90_abs_turn_rate": round(float(np.percentile(atr, 90)), 4)})
    trdf = pd.DataFrame(tr_rows); trdf.to_csv(OUTDIR / "turn_rate_distribution_stats.csv", index=False)

    f, a = plt.subplots(figsize=(7.5, 4.6))
    for s in SPLITS:
        a.hist(np.abs(df.loc[df.split == s, "turn_rate"]), bins=80, density=True, histtype="step",
               lw=1.6, color=SCOL[s], label=f"{s} (mean|tr|={trdf[trdf.split==s].mean_abs_turn_rate.iloc[0]:.2f})")
    a.set_xlabel("|turn_rate| (rad)"); a.set_title("|turn_rate| distribution by split (dataset targets-in-window)"); a.legend()
    f.tight_layout(); f.savefig(OUTDIR / "turn_rate_distribution_by_split.png"); plt.close(f)

    # ---------- TASK 6: cross-site distributions train vs red_bridge ----------
    train_mask = df.split == "train"; test_mask = df.recording_id == C.TEST_REC
    def heading(d): return np.arctan2(d.heading_sin, d.heading_cos)
    ks = {}
    f, axes = plt.subplots(1, 3, figsize=(18, 5))
    # speed
    axes[0].hist(df.loc[train_mask, "speed"], bins=100, range=(0, 3), density=True, histtype="step", lw=1.8, color="#3aaa5e", label="train")
    axes[0].hist(df.loc[test_mask, "speed"], bins=100, range=(0, 3), density=True, histtype="step", lw=1.8, color="#e05c2a", label="test (bridge)")
    axes[0].set_title("speed (m/step)"); axes[0].legend()
    ks["speed"] = ks_2samp(df.loc[train_mask, "speed"], df.loc[test_mask, "speed"]).statistic
    # heading
    axes[1].hist(heading(df[train_mask]), bins=100, density=True, histtype="step", lw=1.8, color="#3aaa5e", label="train")
    axes[1].hist(heading(df[test_mask]), bins=100, density=True, histtype="step", lw=1.8, color="#e05c2a", label="test (bridge)")
    axes[1].set_title("heading (rad)"); axes[1].legend()
    ks["heading"] = ks_2samp(heading(df[train_mask]), heading(df[test_mask])).statistic
    # turn_rate
    axes[2].hist(df.loc[train_mask, "turn_rate"], bins=100, density=True, histtype="step", lw=1.8, color="#3aaa5e", label="train")
    axes[2].hist(df.loc[test_mask, "turn_rate"], bins=100, density=True, histtype="step", lw=1.8, color="#e05c2a", label="test (bridge)")
    axes[2].set_title("turn_rate (rad)"); axes[2].legend()
    ks["turn_rate"] = ks_2samp(df.loc[train_mask, "turn_rate"], df.loc[test_mask, "turn_rate"]).statistic
    f.suptitle("Cross-site distribution: train (pooled) vs red_bridge_combined_01", fontweight="bold")
    f.tight_layout(rect=[0, 0, 1, 0.95]); f.savefig(OUTDIR / "cross_site_distributions.png"); plt.close(f)
    ks = {k: round(float(v), 4) for k, v in ks.items()}

    results = {"teacher_forced_vs_autoregressive": cmp_rows,
               "turn_rate_stats": tr_rows,
               "cross_site_KS_statistic": ks}
    (OUTDIR / "_forensic_numbers.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("=== TF vs AR ==="); print(cmp.to_string(index=False))
    print("\n=== turn_rate stats ==="); print(trdf.to_string(index=False))
    print("\n=== cross-site KS (train vs bridge) ==="); print(ks)


if __name__ == "__main__":
    main()
