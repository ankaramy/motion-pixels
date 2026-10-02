"""
MOTION PIXELS - PHASE 6: V3 Model C + angular (heading) loss.

Same V3 LSTM Model C as Phase 4 (architecture, features, split, window, optimizer,
batch, early stopping, normalization, rollout). The ONLY change is the training
loss:

    total = position_mse + lambda_angle * angular_loss

where angular_loss = 1 - cos_sim between the predicted and true next-step
displacement vectors, computed on RAW (unscaled) du/dv and masked to rows whose
true displacement magnitude exceeds a small threshold (so stationary / noisy
rows do not dominate).

Reuses frozen recipe symbols (run_bridge_ablation.py) and the model-agnostic
rollout / teacher-forced helpers from Phase 5 (phase5_gru_lib.py), all read-only.
Modifies no frozen / Phase-4 / Phase-5 / dataset / encoder file.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# frozen recipe (read-only)
MP_ROOT = Path(r"C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels")
FROZEN_DIR = MP_ROOT / "mp-core" / "trajectory-prediction" / "experiments" / "schema_ablation_bridge"
PHASE5_DIR = MP_ROOT / "mp-core" / "trajectory-prediction" / "experiments" / "phase5_v3_gru_comparison"
sys.path.insert(0, str(FROZEN_DIR))
sys.path.insert(0, str(PHASE5_DIR))
import run_bridge_ablation as fz       # noqa: E402  frozen recipe
import phase5_gru_lib as P5            # noqa: E402  model-agnostic helpers + V3 dataset access

# paths
V3_DATASET = P5.V3_DATASET
PHASE4 = P5.PHASE4
OUT = Path(r"C:\Users\OWNER\Desktop\MotionPixels_Thesis_Outputs\13_phase6_v3_angular_loss")
D_CODE = OUT / "00_code_snapshot"
D_TRAIN = OUT / "01_training_runs"
D_MODELS = OUT / "02_models"
D_METRICS = OUT / "03_metrics"
D_FIG = OUT / "04_figures"
D_REPORTS = OUT / "05_reports"

FEAT_COLS = P5.FEAT_COLS
TARGET_COLS = P5.TARGET_COLS
SPATIAL_COLS = P5.SPATIAL_COLS
SPLIT_MAP = P5.SPLIT_MAP
VAL_REC = P5.VAL_REC
TEST_REC = P5.TEST_REC
N_ROLLOUT = fz.N_ROLLOUT

MIN_DISP_M = 0.01   # rows with true |du,dv| below this (m/step) are masked from angular loss


def angular_loss(pred_scaled, y_scaled, tgt_mean, tgt_std, eps=1e-6, min_disp=MIN_DISP_M):
    """1 - cosine similarity between predicted and true RAW displacement vectors,
    masked to genuinely-moving rows. pred/y are SCALED (model space)."""
    pred_raw = pred_scaled * tgt_std + tgt_mean
    y_raw = y_scaled * tgt_std + tgt_mean
    true_mag = torch.linalg.norm(y_raw, dim=1)
    mask = true_mag > min_disp
    if mask.sum() == 0:
        return pred_scaled.sum() * 0.0
    pr = pred_raw[mask]; yr = y_raw[mask]
    cos = (pr * yr).sum(1) / (torch.linalg.norm(pr, dim=1) * torch.linalg.norm(yr, dim=1) + eps)
    return (1.0 - cos).mean()


def train_one_lambda(lam, df, device, out_train_dir, out_model_dir):
    fz.set_seed(fz.SEED)
    train_df = df[df.split == "train"]; val_df = df[df.split == "val"]
    train_ids = train_df.trajectory_id.unique().tolist()
    val_ids = val_df.trajectory_id.unique().tolist()

    f_sc = fz.ColumnScaler().fit(train_df, FEAT_COLS)
    t_sc = fz.ColumnScaler().fit(train_df, TARGET_COLS)
    X_tr, Y_tr = fz.make_windows(df, FEAT_COLS, train_ids)
    X_va, Y_va = fz.make_windows(df, FEAT_COLS, val_ids)
    X_tr = f_sc.transform(X_tr.reshape(-1, len(FEAT_COLS))).reshape(X_tr.shape).astype(np.float32)
    X_va = f_sc.transform(X_va.reshape(-1, len(FEAT_COLS))).reshape(X_va.shape).astype(np.float32)
    Y_tr = t_sc.transform(Y_tr).astype(np.float32)
    Y_va = t_sc.transform(Y_va).astype(np.float32)

    tr_loader = DataLoader(fz.WindowDataset(X_tr, Y_tr), batch_size=fz.BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(fz.WindowDataset(X_va, Y_va), batch_size=fz.BATCH_SIZE * 2, shuffle=False)

    tgt_mean = torch.tensor(t_sc.means, dtype=torch.float32, device=device)
    tgt_std = torch.tensor(t_sc.stds, dtype=torch.float32, device=device)

    model = fz.TrajectoryLSTM(len(FEAT_COLS)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=fz.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(optim, fz.LR_STEP, fz.LR_GAMMA)
    mse_fn = nn.MSELoss()

    best_val, no_imp, best_state, best_epoch = float("inf"), 0, None, 0
    log = []; t0 = time.time()
    for epoch in range(1, fz.MAX_EPOCHS + 1):
        model.train(); tl = tm = ta = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            mse = mse_fn(pred, yb)
            ang = angular_loss(pred, yb, tgt_mean, tgt_std)
            loss = mse + lam * ang
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tl += loss.item() * len(xb); tm += mse.item() * len(xb); ta += ang.item() * len(xb)
        n = max(len(X_tr), 1); tl /= n; tm /= n; ta /= n

        model.eval(); vl = vm = va = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                mse = mse_fn(pred, yb); ang = angular_loss(pred, yb, tgt_mean, tgt_std)
                vl += (mse + lam * ang).item() * len(xb); vm += mse.item() * len(xb); va += ang.item() * len(xb)
        nv = max(len(X_va), 1); vl /= nv; vm /= nv; va /= nv
        lr_now = optim.param_groups[0]["lr"]; sched.step()

        # early stopping on the TOTAL objective (== MSE when lambda=0, so reproduces Phase 4)
        is_best = vl < best_val
        if is_best:
            best_val, no_imp, best_epoch = vl, 0, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        log.append({"epoch": epoch, "train_total": tl, "train_mse": tm, "train_ang": ta,
                    "val_total": vl, "val_mse": vm, "val_ang": va, "lr": lr_now, "is_best": int(is_best)})
        if no_imp >= fz.PATIENCE:
            break

    model.load_state_dict(best_state)
    tag = f"lambda_{lam:g}".replace(".", "p")
    out_model_dir.mkdir(parents=True, exist_ok=True); out_train_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_model_dir / f"best_model_{tag}.pth")
    P5.save_scalers(out_model_dir / f"scalers_{tag}.pkl", f_sc, t_sc)
    pd.DataFrame(log).to_csv(out_train_dir / f"epoch_log_{tag}.csv", index=False)
    cfg = {"lambda_angle": lam, "model": "TrajectoryLSTM (V3 Model C + angular loss)",
           "loss": "MSE + lambda*(1-cos) on raw displacement, masked |disp|>%.3g m" % MIN_DISP_M,
           "actual_epochs": int(log[-1]["epoch"]), "best_epoch": int(best_epoch),
           "best_val_total": float(best_val), "final_train_mse": float(log[-1]["train_mse"]),
           "final_val_mse": float(log[-1]["val_mse"]), "final_val_ang": float(log[-1]["val_ang"]),
           "early_stopped": bool(no_imp >= fz.PATIENCE), "train_time_s": round(time.time() - t0, 1),
           "seed": fz.SEED, "feature_order": FEAT_COLS}
    return model, f_sc, t_sc, cfg, pd.DataFrame(log)


def evaluate_model(model, f_sc, t_sc, df, bounds, device):
    """Autoregressive rollout (all recordings) + teacher-forced. Returns dataframes."""
    rows, perstep = [], {}
    for rec in SPLIT_MAP:
        rec_df = df[df.recording_id == rec]; wb = bounds[rec]
        kdt = fz.build_kdt(rec_df, SPATIAL_COLS)
        for tid in rec_df.trajectory_id.unique():
            r = P5.rollout_one_trajectory(model, rec_df[rec_df.trajectory_id == tid], wb, f_sc, t_sc, kdt, device)
            if r is None:
                continue
            p_uv = np.column_stack([(r["pred_pos"][:, 0] - wb["xmin"]) / wb["xrng"],
                                    (r["pred_pos"][:, 1] - wb["ymin"]) / wb["yrng"]])
            g_uv = np.column_stack([(r["gt_pos"][:, 0] - wb["xmin"]) / wb["xrng"],
                                    (r["gt_pos"][:, 1] - wb["ymin"]) / wb["yrng"]])
            ex = fz._per_track_extras(p_uv, g_uv)
            rows.append({"recording_id": rec, "trajectory_id": tid, "split": SPLIT_MAP[rec],
                         "n_steps": r["n_steps"], "ade": r["ade"], "fde": r["fde"],
                         "angular_error": r["angular_error"], "mean_turn_pred": r["mean_turn_pred"],
                         "mean_turn_gt": r["mean_turn_gt"], "path_length_ratio": ex["path_length_ratio"],
                         "angularity_ratio": ex["angularity_ratio"], "curvature_corr": ex["curvature_corr"]})
            perstep[tid] = [float(x) for x in r["per_step_err"]]
    m = pd.DataFrame(rows); m["ang_deg"] = np.degrees(m.angular_error)

    tf = pd.DataFrame([P5.teacher_forced(df, f_sc, t_sc, model, s, device) for s in ("train", "val", "test")])
    return m, perstep, tf


def split_summary(m):
    rows = []
    for s in ("train", "val", "test"):
        d = m[m.split == s]
        rows.append({"split": s, "ADE_mean": d.ade.mean(), "FDE_mean": d.fde.mean(),
                     "angular_error_mean_deg": d.ang_deg.mean(),
                     "angular_error_median_deg": d.ang_deg.median(),
                     "mean_turn_pred": d.mean_turn_pred.mean(), "mean_turn_gt": d.mean_turn_gt.mean(),
                     "path_length_ratio": d.path_length_ratio.mean(),
                     "angularity_ratio": d.angularity_ratio.mean(),
                     "curvature_corr": d.curvature_corr.mean(), "n_rollouts": len(d)})
    return pd.DataFrame(rows)


def per_recording_summary(m):
    rows = []
    for rec in SPLIT_MAP:
        d = m[m.recording_id == rec]
        rows.append({"recording_id": rec, "split": SPLIT_MAP[rec], "ADE_mean": d.ade.mean(),
                     "FDE_mean": d.fde.mean(), "angular_error_mean_deg": d.ang_deg.mean(),
                     "mean_turn_pred": d.mean_turn_pred.mean(), "n_rollouts": len(d)})
    return pd.DataFrame(rows)
