"""
train_model_xr.py --config configs/<variant>.json

Trains one MODEL_XR variant. IDENTICAL recipe to MODEL_X (same split, seed, architecture,
scalers, batch, optimizer, single-step next-displacement objective, val ADE checkpointing) —
the ONLY change is the composite loss (losses.composite_loss). Control (lambda_mag=lambda_dir=0)
is therefore byte-for-byte the MODEL_X objective.

Writes checkpoints/<variant>/{model_best.pth, scalers.json, sigma.json, train_log.csv}.
Does NOT touch MODEL_X.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xr_common as XC  # noqa: E402
import losses as LO     # noqa: E402
L = XC.L

VAL_ROLLOUT_WINDOWS = 3000
EPOCHS_MAX = 80
PATIENCE = 12
BATCH = 256


def ade_fde(pred, gt):
    n = min(pred.shape[1], gt.shape[1])
    d = np.linalg.norm(pred[:, :n] - gt[:, :n], axis=2)
    return float(d[:, 1:].mean()), float(d[:, -1].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = XC.load_config(args.config)
    variant = cfg["variant"]; lmag = cfg["lambda_mag"]; ldir = cfg["lambda_dir"]
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time(); L.set_seed(XC.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = HERE / "checkpoints" / variant; out.mkdir(parents=True, exist_ok=True)
    print(f"[{variant}] device={device} lambda_mag={lmag} lambda_dir={ldir}")

    df = XC.load_df(); split = XC.load_split(); wb = L.recording_world_bounds(df)
    tr_ids = split[split.split == "train"].trajectory_id.tolist()
    va_ids = split[split.split == "val"].trajectory_id.tolist()

    Xtr, Ytr = L.make_train_windows(df, tr_ids)
    print(f"[{variant}] train windows {Xtr.shape}")
    f_sc = L.ColumnScaler().fit(Xtr.reshape(-1, L.N_FEAT), L.FEAT_COLS)
    t_sc = L.ColumnScaler().fit(Ytr, L.TARGET_COLS)
    sigma_iso = float(np.sqrt((Ytr ** 2).sum(axis=1).mean()))   # RMS metric step size
    Xtr_s = torch.tensor(f_sc.transform(Xtr).astype(np.float32))
    Ytr_s = torch.tensor(t_sc.transform(Ytr).astype(np.float32))

    Xva, Yva = L.make_train_windows(df, va_ids)
    Xva_d = torch.tensor(f_sc.transform(Xva).astype(np.float32)).to(device)
    Yva_d = torch.tensor(t_sc.transform(Yva).astype(np.float32)).to(device)
    vseeds, vstart, vgt, vwb, _ = L.build_eval_windows(df, va_ids, wb, n_steps=XC.BASE_H,
                                                       stride=2, max_windows=VAL_ROLLOUT_WINDOWS, seed=XC.SEED)

    tgt_std = torch.tensor(t_sc.stds, dtype=torch.float32, device=device)
    tgt_mean = torch.tensor(t_sc.means, dtype=torch.float32, device=device)

    (out / "scalers.json").write_text(json.dumps(
        {"feature_scaler": f_sc.to_dict(), "target_scaler": t_sc.to_dict()}))
    (out / "sigma.json").write_text(json.dumps({"sigma_iso": sigma_iso}))

    dl = DataLoader(TensorDataset(Xtr_s, Ytr_s), batch_size=BATCH, shuffle=True,
                    pin_memory=(device.type == "cuda"))
    model = L.TrajectoryLSTM(L.N_FEAT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    hist, best, best_ep, since = [], float("inf"), -1, 0
    for ep in range(1, EPOCHS_MAX + 1):
        model.train(); te = time.time(); agg = {}
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss, parts = LO.composite_loss(model(xb), yb, tgt_std, tgt_mean, sigma_iso, lmag, ldir)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
        nb = len(dl); agg = {k: v / nb for k, v in agg.items()}
        model.eval()
        pred = L.rollout_world_batch(model, vseeds, vstart, vwb, f_sc, t_sc, XC.BASE_H, device)
        v_ade, v_fde = ade_fde(pred, vgt)
        flag = ""
        if v_ade < best:
            best, best_ep, since = v_ade, ep, 0
            torch.save(model.state_dict(), out / "model_best.pth"); flag = "*"
        else:
            since += 1
        rec = {"epoch": ep, "val_ade": v_ade, "val_fde": v_fde, **agg}
        hist.append(rec)
        print(f"[{variant} ep{ep:3d}] loss={agg.get('total',0):.4f} mse={agg.get('mse',0):.4f} "
              f"mag={agg.get('mag',0):.4f} dir={agg.get('dir',0):.4f} "
              f"ADE={v_ade:.4f} FDE={v_fde:.4f} {flag} ({time.time()-te:.1f}s) best={best:.4f}@{best_ep}")
        if since >= PATIENCE:
            print(f"[{variant}] early stop @ {ep}"); break

    import pandas as pd
    pd.DataFrame(hist).to_csv(out / "train_log.csv", index=False)
    (out / "train_summary.json").write_text(json.dumps(
        {"variant": variant, "lambda_mag": lmag, "lambda_dir": ldir,
         "best_val_ADE": best, "best_epoch": best_ep, "epochs": len(hist),
         "sigma_iso": sigma_iso, "minutes": round((time.time()-t0)/60, 2)}, indent=2))
    print(f"[{variant}] DONE best_val_ADE={best:.4f}@{best_ep} ({round((time.time()-t0)/60,2)} min)")


if __name__ == "__main__":
    main()
