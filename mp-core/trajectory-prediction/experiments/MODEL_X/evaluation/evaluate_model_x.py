"""
evaluate_model_x.py — evaluate final MODEL_X on the test split.

10-step autoregressive rollout (world metres) over ALL test windows (stride 1).
Computes per-window metrics, then aggregates: overall, per-recording, genuine-turn
subset, straight subset, long-displacement subset. Saves metric tables + a pickle of
selected windows (with observed/GT/pred world paths) for plotting.

Primary metric: overall ADE/FDE. Turn Capture Rate is reported, not primary.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import model_x_lib as L  # noqa: E402

CFG = L.CONFIG
MODELS = ROOT / "models"
CKPT = MODELS / "best_by_val_ADE.pth"     # selected final checkpoint
CHUNK = 16384


def signed_net_heading_deg(path: np.ndarray) -> float:
    if len(path) < 3:
        return 0.0
    d = np.diff(path, axis=0)
    h0 = math.atan2(d[0, 1], d[0, 0]); h1 = math.atan2(d[-1, 1], d[-1, 0])
    return math.degrees(L.wrap_angle(h1 - h0))


def agg(metr: pd.DataFrame) -> dict:
    if len(metr) == 0:
        return {"n": 0}
    return {
        "n": int(len(metr)),
        "ADE_mean": float(metr.ade.mean()), "FDE_mean": float(metr.fde.mean()),
        "RMSE_mean": float(metr.rmse.mean()),
        "ADE_median": float(metr.ade.median()), "FDE_median": float(metr.fde.median()),
        "angular_err_mean_deg": float(metr.angular_err_deg.mean(skipna=True)),
        "angular_err_median_deg": float(metr.angular_err_deg.median(skipna=True)),
        "pred_path_len_median": float(metr.pred_path_len.median()),
        "gt_path_len_median": float(metr.gt_path_len.median()),
        "len_ratio_median": float(metr.len_ratio.median(skipna=True)),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}  checkpoint={CKPT.name}")

    sc = json.loads((MODELS / "scalers.json").read_text())
    f_sc = L.ColumnScaler.from_dict(sc["feature_scaler"])
    t_sc = L.ColumnScaler.from_dict(sc["target_scaler"])
    model = L.TrajectoryLSTM(L.N_FEAT).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device)); model.eval()

    csv = CFG["dataset"]["csv_path"]
    cols = ["recording_id", "trajectory_id", "timestep", "world_x", "world_y"] + L.FEAT_COLS
    df = pd.read_csv(csv, usecols=lambda c: c in set(cols))
    split = pd.read_csv(ROOT / "splits" / "model_x_track_split.csv")
    te_ids = split[split.split == "test"].trajectory_id.tolist()
    wb = L.recording_world_bounds(df)

    print(f"[eval] building all test windows (stride 1) ...")
    seeds, start, gt, wba, meta = L.build_eval_windows(df, te_ids, wb, n_steps=L.HORIZON, stride=1)
    M = len(seeds); print(f"[eval] {M} test windows")

    # rollout in chunks
    preds = np.empty((M, L.HORIZON + 1, 2), np.float64)
    for i in range(0, M, CHUNK):
        sl = slice(i, i + CHUNK)
        wb_c = {k: v[sl] for k, v in wba.items()}
        preds[sl] = L.rollout_world_batch(model, seeds[sl], start[sl], wb_c,
                                          f_sc, t_sc, L.HORIZON, device)
        print(f"[eval]   rolled {min(i+CHUNK, M)}/{M}")

    # per-window metrics
    rows = []
    for k in range(M):
        m = L.window_metrics(preds[k], gt[k])
        m["trajectory_id"] = meta[k]["trajectory_id"]
        m["recording_id"] = meta[k]["recording_id"]
        m["start_idx"] = meta[k]["start_idx"]
        m["window_idx"] = k
        m["is_turn"] = L.is_genuine_turn(m)
        m["pred_net_heading_deg"] = signed_net_heading_deg(preds[k])
        m["gt_net_heading_signed_deg"] = signed_net_heading_deg(gt[k])
        rows.append(m)
    metr = pd.DataFrame(rows)
    metr.to_csv(HERE / "per_window_metrics.csv", index=False)

    # subsets
    turns = metr[metr.is_turn]
    straight = metr[(metr.gt_net_heading_deg < 15) & (metr.gt_net_disp >= 0.5)]
    long_q = metr.gt_net_disp.quantile(0.75)
    longd = metr[metr.gt_net_disp >= long_q]

    # Turn Capture Rate (reported, not primary)
    tcr = float("nan")
    if len(turns) > 0:
        cap = ((turns.pred_net_heading_deg.abs() > L.TURN_HEADING_DEG)
               & (np.sign(turns.pred_net_heading_deg) == np.sign(turns.gt_net_heading_signed_deg)))
        tcr = float(cap.mean())

    subset_metrics = {
        "checkpoint": CKPT.name,
        "all_test": agg(metr),
        "genuine_turns": {**agg(turns), "turn_capture_rate": tcr},
        "straight": agg(straight),
        "long_displacement_top25pct": {**agg(longd), "gt_net_disp_threshold_m": float(long_q)},
    }
    (HERE / "subset_metrics.json").write_text(json.dumps(subset_metrics, indent=2))

    # per-recording
    per_rec = []
    for rec, g in metr.groupby("recording_id"):
        a = agg(g); a["recording_id"] = rec
        a["n_turns"] = int(g.is_turn.sum())
        per_rec.append(a)
    per_rec_df = pd.DataFrame(per_rec).set_index("recording_id")
    per_rec_df.to_csv(HERE / "per_recording_metrics.csv")

    # distribution arrays for plotting
    np.savez(HERE / "all_window_arrays.npz",
             ade=metr.ade.to_numpy(), fde=metr.fde.to_numpy(),
             pred_len=metr.pred_path_len.to_numpy(), gt_len=metr.gt_path_len.to_numpy(),
             recording=metr.recording_id.to_numpy(), is_turn=metr.is_turn.to_numpy())

    # selected windows for plots (store obs/gt/pred world paths)
    def obs_path(tid, si):
        g = df[df.trajectory_id == tid].sort_values("timestep")
        sub = g.iloc[si: si + L.WINDOW_SIZE]
        return np.column_stack([sub.world_x.to_numpy(float), sub.world_y.to_numpy(float)])

    def pack(idx_list):
        out = []
        for k in idx_list:
            r = metr.iloc[k] if False else metr[metr.window_idx == k].iloc[0]
            out.append({
                "obs": obs_path(r.trajectory_id, int(r.start_idx)),
                "gt": gt[k], "pred": preds[k],
                "ade": float(r.ade), "fde": float(r.fde),
                "angular_err_deg": float(r.angular_err_deg),
                "recording_id": r.recording_id, "trajectory_id": r.trajectory_id,
                "is_turn": bool(r.is_turn),
            })
        return out

    best6 = metr.nsmallest(6, "ade").window_idx.tolist()
    worst6 = metr.nlargest(6, "ade").window_idx.tolist()
    turn_best6 = turns.nsmallest(6, "ade").window_idx.tolist() if len(turns) >= 1 else []
    # representative 9: spread across ADE quantiles, mixed recordings
    qs = np.linspace(0.05, 0.95, 9)
    rep9 = [int(metr.iloc[(metr.ade - metr.ade.quantile(q)).abs().argmin()].window_idx) for q in qs]
    # presentation: best turns if available else best overall with real motion
    pres_pool = turns if len(turns) >= 6 else metr[metr.gt_net_disp >= 1.0]
    pres = pres_pool.nsmallest(9, "ade").window_idx.tolist()

    plot_data = {"best6": pack(best6), "worst6": pack(worst6),
                 "turn_best6": pack(turn_best6), "rep9": pack(rep9),
                 "presentation": pack(pres)}
    with open(HERE / "plot_windows.pkl", "wb") as fh:
        pickle.dump(plot_data, fh)

    print("[eval] DONE")
    print(json.dumps(subset_metrics, indent=2))
    print("\nper-recording:\n", per_rec_df[["n", "ADE_mean", "FDE_mean", "n_turns"]].to_string())


if __name__ == "__main__":
    main()
