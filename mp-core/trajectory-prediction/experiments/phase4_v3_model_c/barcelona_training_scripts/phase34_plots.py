"""
Phases 3 & 4 — validation (stairs_montjuic_01) and test (red_bridge_combined_01)
rollout visuals: individual plots, contact sheets, metrics CSV, reports.
Reuses the frozen rollout via common.rollout_one_trajectory.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import common as C

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_representative(mdf: pd.DataFrame, n_each: int):
    """Pick longest / medium / high-turn / low-error / high-error, deduped."""
    cand = mdf[mdf.n_steps >= 5].copy()
    if cand.empty:
        cand = mdf.copy()
    picks = {}
    def add(label, sub):
        for tid in sub.trajectory_id.tolist():
            picks.setdefault(tid, label)
    add("longest", cand.nlargest(n_each, "length"))
    med = cand.iloc[(cand.length - cand.length.median()).abs().argsort()[:n_each]]
    add("medium", med)
    add("high_turn", cand.nlargest(n_each, "mean_turn_gt"))
    add("low_error", cand.nsmallest(n_each, "ade"))
    add("high_error", cand.nlargest(n_each, "ade"))
    return picks


def main_for(rec: str, split_name: str, out_dir: str, prefix: str, n_each: int, min_plots: int):
    out = out_dir
    out.mkdir(parents=True, exist_ok=True)
    df = C.load_dataset()
    bounds = C.load_world_bounds()
    f_sc, t_sc = C.load_scalers(C.D_TRAIN / "scalers.pkl")
    model = C.load_model(C.D_MODELS / "best_model_C_barcelona.pt", device)
    wb = bounds[rec]
    kdt = C.build_recording_kdt(df, rec)
    rec_df = df[df.recording_id == rec]

    mdf = pd.read_csv(C.D_METRICS / "per_trajectory_metrics.csv")
    mdf = mdf[mdf.recording_id == rec].copy()
    # split metrics CSV for this phase
    mdf.to_csv(out / f"{prefix}_rollout_metrics.csv", index=False)

    picks = select_representative(mdf, n_each)
    pick_ids = list(picks.keys())
    if len(pick_ids) < min_plots:  # top up by length
        extra = mdf[~mdf.trajectory_id.isin(pick_ids)].nlargest(min_plots - len(pick_ids), "length")
        for tid in extra.trajectory_id:
            picks[tid] = "extra"; pick_ids.append(tid)

    # render individual plots
    results = {}
    for tid in pick_ids:
        tdf = rec_df[rec_df.trajectory_id == tid]
        r = C.rollout_one_trajectory(model, tdf, wb, f_sc, t_sc, kdt, device)
        if r is None:
            continue
        results[tid] = r
        f, a = plt.subplots(figsize=(6, 6))
        C.plot_rollout(a, r)
        a.legend(fontsize=7, loc="best")
        safe = str(tid).replace("/", "_")
        f.tight_layout(); f.savefig(out / f"{prefix}_{safe}_rollout.png"); plt.close(f)

    # contact sheet (up to min_plots)
    ids_cs = [t for t in pick_ids if t in results][:max(min_plots, 20)]
    ncol = 5; nrow = int(np.ceil(len(ids_cs) / ncol))
    f, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    axes = np.atleast_2d(axes)
    for ax, tid in zip(axes.flat, ids_cs):
        C.plot_rollout(ax, results[tid], title=f"{picks[tid]} | ADE={results[tid]['ade']:.2f} FDE={results[tid]['fde']:.2f}")
    for ax in axes.flat[len(ids_cs):]:
        ax.axis("off")
    f.suptitle(f"{split_name} rollouts — {rec} (seed=grey, GT=blue, pred=red)", fontweight="bold", fontsize=14)
    f.tight_layout(rect=[0, 0, 1, 0.98]); f.savefig(out / f"{prefix}_rollouts_contact_sheet.png", dpi=140); plt.close(f)

    # best/worst 5 by ADE (over trajectories with full-ish horizon)
    elig = mdf[mdf.n_steps >= 5]
    best5 = elig.nsmallest(5, "ade").trajectory_id.tolist()
    worst5 = elig.nlargest(5, "ade").trajectory_id.tolist()
    f, axes = plt.subplots(2, 5, figsize=(20, 8))
    for j, tid in enumerate(best5):
        r = results.get(tid) or C.rollout_one_trajectory(model, rec_df[rec_df.trajectory_id == tid], wb, f_sc, t_sc, kdt, device)
        if r: C.plot_rollout(axes[0, j], r, title=f"BEST ADE={r['ade']:.2f}")
    for j, tid in enumerate(worst5):
        r = results.get(tid) or C.rollout_one_trajectory(model, rec_df[rec_df.trajectory_id == tid], wb, f_sc, t_sc, kdt, device)
        if r: C.plot_rollout(axes[1, j], r, title=f"WORST ADE={r['ade']:.2f}")
    f.suptitle(f"{split_name} best (top) vs worst (bottom) 5 by ADE — {rec}", fontweight="bold", fontsize=14)
    f.tight_layout(rect=[0, 0, 1, 0.97]); f.savefig(out / f"{prefix}_best_worst_contact_sheet.png", dpi=140); plt.close(f)

    # report
    def stats(col):
        return mdf[col].mean(), mdf[col].median()
    ade_m, ade_md = stats("ade"); fde_m, fde_md = stats("fde")
    ang_m, ang_md = mdf.angular_error.mean(), mdf.angular_error.median()
    L = [f"# {split_name} Rollout Report — {rec}\n",
         f"- Trajectories evaluated: **{len(mdf)}** (rollout horizon up to {C.N_ROLLOUT} steps, autoregressive)",
         f"- Individual plots written: **{len(results)}**\n",
         "## Metrics\n",
         f"- ADE mean / median: **{ade_m:.3f} / {ade_md:.3f} m**",
         f"- FDE mean / median: **{fde_m:.3f} / {fde_md:.3f} m**",
         f"- Angular error mean / median: **{np.degrees(ang_m):.1f} / {np.degrees(ang_md):.1f} deg**",
         f"- Mean turn-rate predicted / ground-truth: **{mdf.mean_turn_pred.mean():.3f} / {mdf.mean_turn_gt.mean():.3f} rad/step**\n",
         "## Best 5 by ADE\n",
         "| trajectory_id | ADE | FDE | ang(deg) | len |\n|---|---|---|---|---|"]
    for _, row in elig.nsmallest(5, "ade").iterrows():
        L.append(f"| {row.trajectory_id} | {row.ade:.3f} | {row.fde:.3f} | {np.degrees(row.angular_error):.1f} | {int(row.length)} |")
    L += ["\n## Worst 5 by ADE\n", "| trajectory_id | ADE | FDE | ang(deg) | len |\n|---|---|---|---|---|"]
    for _, row in elig.nlargest(5, "ade").iterrows():
        L.append(f"| {row.trajectory_id} | {row.ade:.3f} | {row.fde:.3f} | {np.degrees(row.angular_error):.1f} | {int(row.length)} |")
    straight = mdf.mean_turn_pred.mean() < 0.5 * mdf.mean_turn_gt.mean()
    L += ["\n## Qualitative notes\n",
          f"- **Straight-line collapse:** {'YES' if straight else 'no'} — predicted mean |turn| "
          f"({mdf.mean_turn_pred.mean():.3f}) is {mdf.mean_turn_gt.mean()/max(mdf.mean_turn_pred.mean(),1e-6):.1f}x "
          f"smaller than ground-truth ({mdf.mean_turn_gt.mean():.3f} rad/step).",
          f"- **Angular movement captured:** {'weak' if np.degrees(ang_m) > 45 else 'partial'} "
          f"(mean angular error {np.degrees(ang_m):.0f} deg).",
          "- **Spatial context influence:** rollouts re-query dist_to_obstacle/boundary via per-recording "
          "KDTree each step; positional ADE stays low but heading is under-curved, so spatial features are "
          "not visibly steering turning behavior at this data scale.",
          f"- Positional accuracy is otherwise strong (median ADE {ade_md:.2f} m) because per-step "
          "displacements on this site are small."]
    (out / f"{prefix}_rollout_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[{split_name}] {rec}: {len(results)} plots, ADE {ade_m:.3f}/{ade_md:.3f}, FDE {fde_m:.3f}/{fde_md:.3f}, ang {np.degrees(ang_m):.0f}deg")


if __name__ == "__main__":
    main_for(C.VAL_REC, "Validation", C.D_VAL, "val", n_each=5, min_plots=20)
    main_for(C.TEST_REC, "Test", C.D_TEST, "test", n_each=7, min_plots=30)
