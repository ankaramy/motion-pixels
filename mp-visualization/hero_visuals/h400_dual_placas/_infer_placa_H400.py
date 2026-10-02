"""Frozen-model H400 inference over placa_catalunya + placa_espanya test tracks.
Loads the frozen MODEL_X_HORIZON_SWEEP/H400 checkpoint (NO retraining), rolls out
predictions, and dumps obs/gt/pred world-coordinate windows + metrics so the hero
visual can select the best placa trajectories.
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

SWEEP = Path(r"C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-core\trajectory-prediction\experiments\MODEL_X_HORIZON_SWEEP")
MODELX = SWEEP.parent / "MODEL_X"
sys.path.insert(0, str(MODELX))
import model_x_lib as L

OUT = Path(r"C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-visualization\hero_visuals\h400_dual_placas")
OUT.mkdir(parents=True, exist_ok=True)
H = 400
PLACAS = ["placa_catalunya_01", "placa_espanya_01"]
CHUNK = 8192

def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    L.set_seed(L.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Hdir = SWEEP / "H400"
    print("device", device)

    cols = ["recording_id","trajectory_id","timestep","world_x","world_y"] + L.FEAT_COLS + L.TARGET_COLS
    CSV = L.CONFIG["dataset"]["csv_path"]
    print("reading", CSV)
    df = pd.read_csv(CSV, usecols=lambda c: c in set(cols))
    df = df[df.recording_id.isin(PLACAS)].copy()
    print("placa rows", len(df), "tracks", df.trajectory_id.nunique())

    split = pd.read_csv(MODELX / "splits" / "model_x_track_split.csv")
    Lens = df.groupby("trajectory_id").size()
    wb = L.recording_world_bounds(df)
    need = L.WINDOW_SIZE + H

    te = split[(split.split == "test") & (split.recording_id.isin(PLACAS))]
    te_ids = [t for t in te.trajectory_id if Lens.get(t, 0) >= need]
    print("placa test tracks qualifying H400:", len(te_ids))
    by_rec = {}
    for t in te_ids:
        r = df[df.trajectory_id == t].recording_id.iloc[0]
        by_rec.setdefault(r, []).append(t)
    for r in PLACAS:
        print("  ", r, "tracks:", len(by_rec.get(r, [])))

    # scalers from frozen H400
    sc = json.loads((Hdir / "scalers.json").read_text())
    f_sc = L.ColumnScaler.from_dict(sc["feature_scaler"])
    t_sc = L.ColumnScaler.from_dict(sc["target_scaler"])

    model = L.TrajectoryLSTM(L.N_FEAT).to(device)
    model.load_state_dict(torch.load(Hdir / "model_best.pth", map_location=device))
    model.eval()

    seeds, start, gt, wba, meta = L.build_eval_windows(df, te_ids, wb, n_steps=H, stride=1)
    M = len(seeds)
    print("eval windows:", M)
    preds = np.empty((M, H + 1, 2))
    for i in range(0, M, CHUNK):
        sl = slice(i, i + CHUNK)
        preds[sl] = L.rollout_world_batch(model, seeds[sl], start[sl],
                                          {k: v[sl] for k, v in wba.items()}, f_sc, t_sc, H, device)
        print(f"  rollout {min(i+CHUNK,M)}/{M}")

    grp = {tid: g.sort_values("timestep") for tid, g in df.groupby("trajectory_id", sort=False)}
    def obs_path(tid, si):
        g = grp[tid]
        sub = g.iloc[si:si + L.WINDOW_SIZE]
        return np.column_stack([sub.world_x.to_numpy(float), sub.world_y.to_numpy(float)])

    rows = []
    for k in range(M):
        m = L.window_metrics(preds[k], gt[k])
        m.update({"recording_id": meta[k]["recording_id"], "trajectory_id": meta[k]["trajectory_id"],
                  "start_idx": meta[k]["start_idx"], "window_idx": k})
        rows.append(m)
    metr = pd.DataFrame(rows)
    metr.to_csv(OUT / "placa_H400_all_window_metrics.csv", index=False)
    print("metrics saved; cols:", list(metr.columns))

    # one best window per trajectory (lowest ADE), then dump everything for selection
    bundle = {"H": H, "windows": {}}
    for rec in PLACAS:
        sub = metr[metr.recording_id == rec]
        if sub.empty:
            bundle["windows"][rec] = []
            continue
        # best-ADE window per trajectory
        best_per_traj = sub.loc[sub.groupby("trajectory_id").ade.idxmin()]
        packs = []
        for _, r in best_per_traj.iterrows():
            k = int(r.window_idx)
            packs.append({
                "obs": obs_path(r.trajectory_id, int(r.start_idx)),
                "gt": gt[k], "pred": preds[k],
                "ade": float(r.ade), "fde": float(r.fde),
                "angular_err_deg": float(r.angular_err_deg) if pd.notna(r.angular_err_deg) else None,
                "pred_len": float(r.pred_path_len), "gt_len": float(r.gt_path_len),
                "len_ratio": float(r.len_ratio),
                "recording_id": r.recording_id, "trajectory_id": r.trajectory_id,
                "start_idx": int(r.start_idx),
            })
        bundle["windows"][rec] = packs
        print(rec, "distinct trajectories:", len(packs))

    with open(OUT / "placa_H400_best_per_track.pkl", "wb") as fh:
        pickle.dump(bundle, fh)
    print("DONE -> placa_H400_best_per_track.pkl")

if __name__ == "__main__":
    main()
