"""
fix_best_examples.py — regenerate ONLY best_variant_rollout_examples.png with a
representative (non-outlier) selection that actually demonstrates the magnitude fix:
windows where the CONTROL undershot (nd_ratio<0.6) and the best variant corrected it
(nd_ratio in [0.7,1.3]) with preserved direction (cosine>0.7), GT net-disp in a moderate
2-9 m band (avoid jitter outliers). Equal metric scaling. Overwrites the figure in place.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xr_common as XC
L = XC.L
best = json.loads((HERE / "reports" / "_best_variant.json").read_text())["best"]
CONTROL = "MODEL_XR_A_CONTROL"; Hx = 100


def load_model(v, device):
    sc = json.loads((HERE / "checkpoints" / v / "scalers.json").read_text())
    f = L.ColumnScaler.from_dict(sc["feature_scaler"]); t = L.ColumnScaler.from_dict(sc["target_scaler"])
    m = L.TrajectoryLSTM(L.N_FEAT).to(device)
    m.load_state_dict(torch.load(HERE / "checkpoints" / v / "model_best.pth", map_location=device)); m.eval()
    return m, f, t


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = XC.load_df(); split = XC.load_split(); wb = L.recording_world_bounds(df)
    need = XC.OBS + Hx; lens = df.groupby("trajectory_id").size()
    te_ids = [t for t in split[split.split == "test"].trajectory_id if lens.get(t, 0) >= need]
    mc, fc, tc = load_model(CONTROL, device); mb, fb, tb = load_model(best, device)
    pc, gt, rec, tid = XC.rollout_all(mc, df, te_ids, wb, fc, tc, Hx, device)
    pb, _, _, _ = XC.rollout_all(mb, df, te_ids, wb, fb, tb, Hx, device)
    dC = XC.enrich(pc, gt, rec, tid, Hx); dB = XC.enrich(pb, gt, rec, tid, Hx)
    _, _, _, _, meta = L.build_eval_windows(df, te_ids, wb, n_steps=Hx, stride=1)
    starts = np.array([m["start_idx"] for m in meta])
    df_by = {t: g.sort_values("timestep") for t, g in df[df.trajectory_id.isin(set(tid))].groupby("trajectory_id")}

    nd_gt = dC.nd_gt.to_numpy()
    mask = ((dC.nd_ratio.to_numpy() < 0.6) & (dB.nd_ratio.to_numpy() > 0.7) & (dB.nd_ratio.to_numpy() < 1.3)
            & (dB.cosine.to_numpy() > 0.7) & (nd_gt > 2.0) & (nd_gt < 9.0))
    idx = np.where(mask)[0]
    # diversify across recordings
    rng = np.random.default_rng(42); rng.shuffle(idx)
    chosen, seen = [], {}
    for i in idx:
        r = rec[i]; seen[r] = seen.get(r, 0)
        if seen[r] < 2:
            chosen.append(i); seen[r] += 1
        if len(chosen) == 9:
            break
    for i in idx:
        if len(chosen) == 9: break
        if i not in chosen: chosen.append(i)

    def obs_path(i):
        g = df_by[tid[i]]; s = g.iloc[starts[i]:starts[i] + XC.OBS]
        return np.column_stack([s.world_x.to_numpy(float), s.world_y.to_numpy(float)])

    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for k, ax in enumerate(axes.flat):
        if k < len(chosen):
            i = chosen[k]; ob = obs_path(i)
            ax.plot(ob[:, 0], ob[:, 1], "-", color="black", lw=1.6, label="observed")
            ax.plot(gt[i, :, 0], gt[i, :, 1], "-", color="#7f7f7f", lw=2.6, label="GT")
            ax.plot(pc[i, :, 0], pc[i, :, 1], "--", color="#1f77b4", lw=1.6, label="control (MSE)")
            ax.plot(pb[i, :, 0], pb[i, :, 1], "-", color="#e0218a", lw=1.9, label=best.replace("MODEL_XR_", ""))
            ax.plot(ob[-1, 0], ob[-1, 1], "o", color="#333", ms=4)
            ax.set_aspect("equal", adjustable="box"); ax.tick_params(labelsize=6)
            ax.set_title(f"{rec[i].replace('_01','')}  ndR ctrl={dC.nd_ratio.iloc[i]:.2f}->"
                         f"{best.replace('MODEL_XR_','')}={dB.nd_ratio.iloc[i]:.2f}  cos={dB.cosine.iloc[i]:.2f}",
                         fontsize=7)
            if k == 0: ax.legend(fontsize=6)
        else:
            ax.axis("off")
    fig.suptitle(f"Magnitude fix — control (short, blue) vs {best.replace('MODEL_XR_','')} (magenta) vs GT (grey), H{Hx}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(HERE / "figures" / "best_variant_rollout_examples.png", dpi=140); plt.close(fig)
    print(f"[fix] regenerated with {len(chosen)} representative windows (recordings: {seen})")


if __name__ == "__main__":
    main()
