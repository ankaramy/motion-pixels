"""
MP_X shared trainer — turn-balanced Model C.

Same frozen Model C architecture / optimiser / schedule / early-stopping as
Phase-4 (imported read-only from run_bridge_ablation.py). The ONLY change vs the
baseline is the training-window SAMPLING: a WeightedRandomSampler oversamples
turn windows by bin (straight 1 / mild 8 / sharp 12 / U-turn 16). Reused by exp01
(full data) and exp03 (horizon sweep). An optional `train_window_filter` allows
exp02 (turn-only) to drop straight windows entirely.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import mpx_common as C
fz = C.fz


def train_balanced_model_c(df, out_dir: Path, device, *, horizon=C.N_ROLLOUT,
                           use_sampler=True, weights=None, seed=42,
                           tag="exp01", train_window_filter=None,
                           val_window_filter=None, max_epochs=None, verbose=True):
    """Train Model C and save checkpoint, scalers, config, and loss CSVs.

    Parameters
    ----------
    train_window_filter : callable(meta_df) -> bool mask, optional
        Keep only training windows where the mask is True (e.g. genuine turns).
    val_window_filter : callable(meta_df) -> bool mask, optional
        Keep only validation windows where the mask is True. Used by the
        turn-only diagnostic so EARLY STOPPING tracks a turn-relevant signal
        instead of the ~99.96%-straight validation MSE (which would otherwise
        re-select a straight-collapse checkpoint). Ignored if it would empty val.
    Returns a dict summary (paths, bin counts, best epoch/loss, etc.).
    """
    weights = dict(C.BIN_WEIGHTS) if weights is None else dict(weights)
    max_epochs = fz.MAX_EPOCHS if max_epochs is None else max_epochs
    out_dir = Path(out_dir)
    models_dir = out_dir / "models"
    train_dir = out_dir / "training"
    for d in (models_dir, train_dir):
        d.mkdir(parents=True, exist_ok=True)

    fz.set_seed(seed)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    train_ids = train_df["trajectory_id"].unique().tolist()
    val_ids = val_df["trajectory_id"].unique().tolist()

    # Scalers fit on TRAIN rows only (frozen recipe), features + targets.
    f_sc = fz.ColumnScaler().fit(train_df, C.FEAT_COLS)
    t_sc = fz.ColumnScaler().fit(train_df, C.TARGET_COLS)

    if verbose:
        print(f"[windows] building labeled windows (horizon={horizon}) ...")
    X_tr, Y_tr, meta_tr = C.make_labeled_windows(df, C.FEAT_COLS, train_ids, horizon)
    X_va, Y_va, meta_va = C.make_labeled_windows(df, C.FEAT_COLS, val_ids, horizon)

    if train_window_filter is not None:
        mask = np.asarray(train_window_filter(meta_tr), dtype=bool)
        X_tr, Y_tr = X_tr[mask], Y_tr[mask]
        meta_tr = meta_tr[mask].reset_index(drop=True)

    if val_window_filter is not None:
        vmask = np.asarray(val_window_filter(meta_va), dtype=bool)
        if vmask.sum() == 0:
            if verbose:
                print("[warn] val_window_filter emptied val -> keeping full val for early stopping")
        else:
            X_va, Y_va = X_va[vmask], Y_va[vmask]
            meta_va = meta_va[vmask].reset_index(drop=True)

    counts_tr = C.bin_counts(meta_tr)
    counts_va = C.bin_counts(meta_va)
    if verbose:
        print(f"[windows] train={len(X_tr):,}  val={len(X_va):,}")
        print(f"[windows] train turn-bins: {counts_tr}")
        print(f"[windows] val   turn-bins: {counts_va}")

    nf = len(C.FEAT_COLS)
    X_tr_s = f_sc.transform(X_tr.reshape(-1, nf)).reshape(X_tr.shape).astype(np.float32)
    X_va_s = f_sc.transform(X_va.reshape(-1, nf)).reshape(X_va.shape).astype(np.float32)
    Y_tr_s = t_sc.transform(Y_tr).astype(np.float32)
    Y_va_s = t_sc.transform(Y_va).astype(np.float32)

    if use_sampler:
        w = meta_tr["bin"].map(weights).to_numpy(np.float64)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(w), replacement=True)
        tr_loader = DataLoader(fz.WindowDataset(X_tr_s, Y_tr_s),
                               batch_size=fz.BATCH_SIZE, sampler=sampler)
    else:
        tr_loader = DataLoader(fz.WindowDataset(X_tr_s, Y_tr_s),
                               batch_size=fz.BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(fz.WindowDataset(X_va_s, Y_va_s),
                           batch_size=fz.BATCH_SIZE * 2, shuffle=False)

    model = fz.TrajectoryLSTM(nf).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=fz.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(optim, fz.LR_STEP, fz.LR_GAMMA)
    mse = nn.MSELoss()

    best_val, no_imp, best_state, best_ep = float("inf"), 0, None, 0
    log = []
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        tl = 0.0
        nseen = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = mse(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tl += loss.item() * len(xb)
            nseen += len(xb)
        tl /= max(nseen, 1)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl += mse(model(xb), yb).item() * len(xb)
        vl /= max(len(X_va_s), 1)
        lr_now = optim.param_groups[0]["lr"]
        sched.step()

        is_best = vl < best_val
        if is_best:
            best_val, no_imp, best_ep = vl, 0, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        log.append({"epoch": epoch, "train_loss": tl, "val_loss": vl, "lr": lr_now,
                    "is_best": int(is_best)})
        if verbose and (epoch == 1 or epoch % 5 == 0 or is_best):
            print(f"  ep {epoch:3d}  train={tl:.5f}  val={vl:.5f}  best={best_val:.5f}")
        if no_imp >= fz.PATIENCE:
            if verbose:
                print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    ckpt = models_dir / f"best_model_{tag}.pth"
    scal = models_dir / f"scalers_{tag}.pkl"
    torch.save(best_state, ckpt)
    C.P5.save_scalers(scal, f_sc, t_sc)

    log_df = pd.DataFrame(log)
    log_df.to_csv(train_dir / "epoch_log.csv", index=False)
    log_df[["epoch", "train_loss"]].to_csv(train_dir / "training_loss.csv", index=False)
    log_df[["epoch", "val_loss"]].to_csv(train_dir / "validation_loss.csv", index=False)

    cfg = {
        "experiment": tag, "model": "TrajectoryLSTM (frozen Model C)",
        "feature_order": C.FEAT_COLS, "targets": C.TARGET_COLS,
        "window": C.WINDOW_SIZE, "label_horizon": horizon,
        "use_weighted_sampler": use_sampler, "sampler_weights": weights,
        "train_window_filter": (train_window_filter is not None),
        "val_window_filter": (val_window_filter is not None),
        "batch_size": fz.BATCH_SIZE, "lr": fz.LR, "weight_decay": 1e-4,
        "scheduler": f"StepLR(step={fz.LR_STEP}, gamma={fz.LR_GAMMA})",
        "grad_clip": 1.0, "max_epochs": max_epochs, "patience": fz.PATIENCE,
        "seed": seed, "device": str(device),
        "n_train_windows": int(len(X_tr_s)), "n_val_windows": int(len(X_va_s)),
        "train_bin_counts": counts_tr, "val_bin_counts": counts_va,
        "actual_epochs": int(log_df["epoch"].iloc[-1]), "best_epoch": best_ep,
        "best_val_loss": float(best_val), "train_time_s": round(time.time() - t0, 1),
        "dataset": str(C.P5.MODEL_C_CSV),
    }
    (train_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if verbose:
        print(f"[saved] {ckpt}")
        print(f"[saved] {scal}")
        print(f"[done] best epoch {best_ep}  val {best_val:.5f}  "
              f"({cfg['train_time_s']}s)")
    return {"ckpt": ckpt, "scalers": scal, "config": cfg,
            "f_sc": f_sc, "t_sc": t_sc, "model": model,
            "train_bin_counts": counts_tr, "val_bin_counts": counts_va}
