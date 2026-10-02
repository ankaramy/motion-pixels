"""
compute_rollouts.py — autoregressive rollouts (frozen rollout()) for EVERY
trajectory in all 5 recordings, using the best checkpoint + per-recording world
bounds + per-recording KDTree. Writes one per-trajectory metrics table that
Phases 3/4/5/6 consume. Best model is never trained here (read-only eval).
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
import common as C


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = C.load_dataset()
    bounds = C.load_world_bounds()
    f_sc, t_sc = C.load_scalers(C.D_TRAIN / "scalers.pkl")
    model = C.load_model(C.D_MODELS / "best_model_C_barcelona.pt", device)

    rows = []
    perstep = {}   # trajectory_id -> per-step error list
    for rec in C.SPLIT_MAP:
        rec_df = df[df.recording_id == rec]
        wb = bounds[rec]
        kdt = C.build_recording_kdt(df, rec)
        ids = rec_df.trajectory_id.unique().tolist()
        n_ok = 0
        for tid in ids:
            tdf = rec_df[rec_df.trajectory_id == tid]
            r = C.rollout_one_trajectory(model, tdf, wb, f_sc, t_sc, kdt, device)
            if r is None:
                continue
            n_ok += 1
            rows.append({
                "recording_id": rec, "trajectory_id": tid, "split": C.SPLIT_MAP[rec],
                "length": r["length"], "n_steps": r["n_steps"],
                "ade": r["ade"], "fde": r["fde"], "angular_error": r["angular_error"],
                "mean_turn_pred": r["mean_turn_pred"], "mean_turn_gt": r["mean_turn_gt"],
                "full20": int(r["n_steps"] >= C.N_ROLLOUT),
            })
            perstep[tid] = [float(x) for x in r["per_step_err"]]
        print(f"  [{rec:24s}] {C.SPLIT_MAP[rec]:5s}  {n_ok}/{len(ids)} trajectories rolled out")

    C.D_METRICS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(C.D_METRICS / "per_trajectory_metrics.csv", index=False)
    (C.D_METRICS / "per_step_errors.json").write_text(json.dumps(perstep), encoding="utf-8")
    print(f"[done] {len(rows)} trajectories -> per_trajectory_metrics.csv")


if __name__ == "__main__":
    main()
