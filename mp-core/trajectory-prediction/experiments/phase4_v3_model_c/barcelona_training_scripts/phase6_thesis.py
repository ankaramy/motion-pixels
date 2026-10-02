"""Phase 6 — clean, presentation-ready thesis figures."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import common as C

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams.update({"font.size": 12, "savefig.dpi": 200})


def pick_grid(model, df, bounds, f_sc, t_sc, rec, ids):
    wb = bounds[rec]; kdt = C.build_recording_kdt(df, rec)
    rec_df = df[df.recording_id == rec]
    out = []
    for tid in ids:
        r = C.rollout_one_trajectory(model, rec_df[rec_df.trajectory_id == tid], wb, f_sc, t_sc, kdt, device)
        if r: out.append(r)
    return out


def thesis_contact(results, title, path, ncol=4):
    nrow = int(np.ceil(len(results) / ncol))
    f, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow))
    axes = np.atleast_2d(axes)
    for ax, r in zip(axes.flat, results):
        C.plot_rollout(ax, r, title=f"ADE={r['ade']:.2f} m  FDE={r['fde']:.2f} m")
    for ax in axes.flat[len(results):]:
        ax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    f.legend(h, l, loc="lower center", ncol=6, fontsize=10, frameon=False)
    f.suptitle(title, fontweight="bold", fontsize=15)
    f.tight_layout(rect=[0, 0.04, 1, 0.97]); f.savefig(path); plt.close(f)


def main():
    C.D_THESIS.mkdir(parents=True, exist_ok=True)
    df = C.load_dataset(); bounds = C.load_world_bounds()
    f_sc, t_sc = C.load_scalers(C.D_TRAIN / "scalers.pkl")
    model = C.load_model(C.D_MODELS / "best_model_C_barcelona.pt", device)
    m = pd.read_csv(C.D_METRICS / "per_trajectory_metrics.csv")
    m["ang_deg"] = np.degrees(m.angular_error)
    el = pd.read_csv(C.D_TRAIN / "epoch_log.csv")
    summ = pd.read_csv(C.D_METRICS / "metrics_summary.csv")
    hd = pd.read_csv(C.D_METRICS / "error_by_horizon.csv")
    best_epoch = int(el.loc[el.is_best == 1, "epoch"].iloc[-1])

    # 1 training loss thesis
    f, a = plt.subplots(figsize=(8, 5))
    a.plot(el.epoch, el.train_loss, color="#3aaa5e", lw=2, label="train")
    a.plot(el.epoch, el.val_loss, color="#4c8cbf", lw=2, label="validation")
    a.axvline(best_epoch, color="#d62728", ls="--", lw=1.5, label=f"best (epoch {best_epoch})")
    a.set_xlabel("epoch"); a.set_ylabel("MSE loss (standardized)")
    a.set_title("Frozen Model C — training on Barcelona (recording-level split)")
    a.legend(); f.tight_layout(); f.savefig(C.D_THESIS / "training_loss_thesis.png"); plt.close(f)

    # 2 validation rollouts thesis contact (8 representative)
    vm = m[(m.recording_id == C.VAL_REC) & (m.n_steps >= 8)]
    vids = (vm.nlargest(4, "mean_turn_gt").trajectory_id.tolist() + vm.nsmallest(4, "ade").trajectory_id.tolist())
    thesis_contact(pick_grid(model, df, bounds, f_sc, t_sc, C.VAL_REC, vids),
                   "Validation rollouts — stairs_montjuic_01 (seed grey · GT blue · pred red)",
                   C.D_THESIS / "validation_rollouts_thesis_contact.png")

    # 3 test rollouts thesis contact (8 representative, bridge crossings = longest)
    tm = m[(m.recording_id == C.TEST_REC) & (m.n_steps >= 8)]
    tids = (tm.nlargest(4, "length").trajectory_id.tolist() + tm.nlargest(4, "mean_turn_gt").trajectory_id.tolist())
    thesis_contact(pick_grid(model, df, bounds, f_sc, t_sc, C.TEST_REC, tids),
                   "Test rollouts — red_bridge_combined_01 (held-out site)",
                   C.D_THESIS / "test_rollouts_thesis_contact.png")

    # 4 metrics summary thesis (table-style bar)
    f, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    sp = summ.split.tolist(); colors = ["#3aaa5e", "#4c8cbf", "#e05c2a"]
    axes[0].bar(sp, summ.ADE_mean, color=colors); axes[0].set_title("ADE mean (m)")
    axes[1].bar(sp, summ.FDE_mean, color=colors); axes[1].set_title("FDE mean (m)")
    axes[2].bar(sp, summ.angular_error_mean_deg, color=colors); axes[2].axhline(90, color="grey", ls=":"); axes[2].set_title("Angular error mean (deg)")
    for ax in axes:
        for i, v in enumerate(ax.containers[0].datavalues):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    f.suptitle("Frozen Model C — Barcelona metrics by split", fontweight="bold", fontsize=15)
    f.tight_layout(rect=[0, 0, 1, 0.93]); f.savefig(C.D_THESIS / "metrics_summary_thesis.png"); plt.close(f)

    # 5 error by horizon thesis
    f, a = plt.subplots(figsize=(8, 5))
    for s, col in [("train", "#3aaa5e"), ("val", "#4c8cbf"), ("test", "#e05c2a")]:
        a.plot(hd.step, hd[f"{s}_mean_err"], "-o", ms=4, lw=2, color=col, label=s)
    a.set_xlabel("rollout step (autoregressive)"); a.set_ylabel("mean position error (m)")
    a.set_title("Error growth over prediction horizon"); a.legend()
    f.tight_layout(); f.savefig(C.D_THESIS / "error_by_horizon_thesis.png"); plt.close(f)

    # 6 best/worst predictions thesis (test): 3 best + 3 worst
    te = m[(m.recording_id == C.TEST_REC) & (m.n_steps >= 8)]
    best3 = te.nsmallest(3, "ade").trajectory_id.tolist()
    worst3 = te.nlargest(3, "ade").trajectory_id.tolist()
    rb = pick_grid(model, df, bounds, f_sc, t_sc, C.TEST_REC, best3)
    rw = pick_grid(model, df, bounds, f_sc, t_sc, C.TEST_REC, worst3)
    f, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, r in zip(axes[0], rb): C.plot_rollout(ax, r, title=f"BEST · ADE={r['ade']:.2f} m")
    for ax, r in zip(axes[1], rw): C.plot_rollout(ax, r, title=f"WORST · ADE={r['ade']:.2f} m")
    h, l = axes[0, 0].get_legend_handles_labels()
    f.legend(h, l, loc="lower center", ncol=6, fontsize=10, frameon=False)
    f.suptitle("Test predictions — best (top) vs worst (bottom): straight-line collapse on turns",
               fontweight="bold", fontsize=15)
    f.tight_layout(rect=[0, 0.04, 1, 0.96]); f.savefig(C.D_THESIS / "best_worst_predictions_thesis.png"); plt.close(f)

    print("Phase 6 done — 6 thesis figures written.")


if __name__ == "__main__":
    main()
