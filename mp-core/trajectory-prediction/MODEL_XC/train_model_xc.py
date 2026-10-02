"""
train_model_xc.py --config configs/<variant>.json

Trains one MODEL_XC variant. Identical recipe to MODEL_X / MODEL_XR (same split, seed, arch,
scalers, batch, AdamW, single-step objective, val-ADE checkpoint at H10) — only the loss adds a
curvature term (losses_xc.composite_loss). Control (lambda_curv=0, lambda_mag=1) == MODEL_XR_B_MAG.
Writes checkpoints/<variant>/{model_best.pth, scalers.json, sigma.json, train_log.csv, train_summary.json}.
Does NOT touch MODEL_X or MODEL_XR.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xc_common as XCC      # noqa
import losses_xc as LO       # noqa
L = XCC.L; XC = XCC.XC

VAL_ROLLOUT_WINDOWS = 3000
EPOCHS_MAX = 80; PATIENCE = 12; BATCH = 256


def ade_fde(pred, gt):
    n = min(pred.shape[1], gt.shape[1]); d = np.linalg.norm(pred[:, :n] - gt[:, :n], axis=2)
    return float(d[:, 1:].mean()), float(d[:, -1].mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = XCC.load_config(ap.parse_args().config)
    variant = cfg["variant"]; lmag = cfg["lambda_mag"]; lcurv = cfg["lambda_curv"]; ldir = cfg.get("lambda_dir", 0.0)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time(); L.set_seed(XCC.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = HERE / "checkpoints" / variant; out.mkdir(parents=True, exist_ok=True)
    print(f"[{variant}] device={device} lmag={lmag} lcurv={lcurv} ldir={ldir}")

    df = XCC.load_df(); split = XCC.load_split(); wb = L.recording_world_bounds(df)
    tr_ids = split[split.split == "train"].trajectory_id.tolist()
    va_ids = split[split.split == "val"].trajectory_id.tolist()

    Xtr, Ytr, Vtr = XCC.make_train_windows_xc(df, tr_ids)
    print(f"[{variant}] train windows {Xtr.shape}")
    f_sc = L.ColumnScaler().fit(Xtr.reshape(-1, L.N_FEAT), L.FEAT_COLS)
    t_sc = L.ColumnScaler().fit(Ytr, L.TARGET_COLS)
    sigma_iso = float(np.sqrt((Ytr ** 2).sum(axis=1).mean()))
    Xtr_s = torch.tensor(f_sc.transform(Xtr).astype(np.float32))
    Ytr_s = torch.tensor(t_sc.transform(Ytr).astype(np.float32))
    Vtr_t = torch.tensor(Vtr)

    Xva, Yva, _ = XCC.make_train_windows_xc(df, va_ids)
    Xva_d = torch.tensor(f_sc.transform(Xva).astype(np.float32)).to(device)
    Yva_d = torch.tensor(t_sc.transform(Yva).astype(np.float32)).to(device)
    vseeds, vstart, vgt, vwb, _ = L.build_eval_windows(df, va_ids, wb, n_steps=XCC.BASE_H,
                                                       stride=2, max_windows=VAL_ROLLOUT_WINDOWS, seed=XCC.SEED)
    tgt_std = torch.tensor(t_sc.stds, dtype=torch.float32, device=device)
    tgt_mean = torch.tensor(t_sc.means, dtype=torch.float32, device=device)
    (out / "scalers.json").write_text(json.dumps({"feature_scaler": f_sc.to_dict(), "target_scaler": t_sc.to_dict()}))
    (out / "sigma.json").write_text(json.dumps({"sigma_iso": sigma_iso}))

    dl = DataLoader(TensorDataset(Xtr_s, Ytr_s, Vtr_t), batch_size=BATCH, shuffle=True,
                    pin_memory=(device.type == "cuda"))
    model = L.TrajectoryLSTM(L.N_FEAT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    hist, best, best_ep, since = [], float("inf"), -1, 0
    for ep in range(1, EPOCHS_MAX + 1):
        model.train(); te = time.time(); agg = {}
        for xb, yb, vb in dl:
            xb, yb, vb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True), vb.to(device, non_blocking=True)
            opt.zero_grad()
            loss, parts = LO.composite_loss(model(xb), yb, vb, tgt_std, tgt_mean, sigma_iso, lmag, lcurv, ldir)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
        nb = len(dl); agg = {k: v / nb for k, v in agg.items()}
        model.eval()
        pred = L.rollout_world_batch(model, vseeds, vstart, vwb, f_sc, t_sc, XCC.BASE_H, device)
        v_ade, v_fde = ade_fde(pred, vgt)
        flag = ""
        if v_ade < best:
            best, best_ep, since = v_ade, ep, 0
            torch.save(model.state_dict(), out / "model_best.pth"); flag = "*"
        else:
            since += 1
        hist.append({"epoch": ep, "val_ade": v_ade, "val_fde": v_fde, **agg})
        print(f"[{variant} ep{ep:3d}] tot={agg.get('total',0):.4f} mse={agg.get('mse',0):.4f} "
              f"mag={agg.get('mag',0):.4f} curv={agg.get('curv',0):.4f} ADE={v_ade:.4f} FDE={v_fde:.4f} "
              f"{flag} ({time.time()-te:.1f}s) best={best:.4f}@{best_ep}")
        if since >= PATIENCE:
            print(f"[{variant}] early stop @ {ep}"); break

    import pandas as pd
    pd.DataFrame(hist).to_csv(out / "train_log.csv", index=False)
    (out / "train_summary.json").write_text(json.dumps(
        {"variant": variant, "lambda_mag": lmag, "lambda_curv": lcurv, "lambda_dir": ldir,
         "best_val_ADE": best, "best_epoch": best_ep, "epochs": len(hist),
         "sigma_iso": sigma_iso, "minutes": round((time.time()-t0)/60, 2)}, indent=2))
    print(f"[{variant}] DONE best_val_ADE={best:.4f}@{best_ep} ({round((time.time()-t0)/60,2)} min)")


if __name__ == "__main__":
    main()
