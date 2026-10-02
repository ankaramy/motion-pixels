"""
train_model_x.py — train MODEL_X (Horizon-10 Model C retraining).

Training target: single-step next-displacement (target_du, target_dv), MSE on
scaled targets — exactly the frozen Model C objective. Horizon=10 is applied as a
10-step autoregressive rollout at validation/eval time. Checkpoints are selected by
validation ADE (primary) and FDE over the 10-step rollout.

Terminal-only progress; nothing is printed to chat. Outputs -> training/ and models/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import model_x_lib as L  # noqa: E402

CFG = L.CONFIG
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
VAL_ROLLOUT_WINDOWS = 3000   # subsample size for per-epoch val ADE/FDE


def ade_fde(pred: np.ndarray, gt: np.ndarray):
    n = min(pred.shape[1], gt.shape[1])
    disp = np.linalg.norm(pred[:, :n] - gt[:, :n], axis=2)   # (M, n)
    ade = float(disp[:, 1:].mean())
    fde = float(disp[:, -1].mean())
    return ade, fde


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    L.set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  torch={torch.__version__}")

    tr_cfg = CFG["training"]
    csv = CFG["dataset"]["csv_path"]
    cols = ["recording_id", "trajectory_id", "timestep", "world_x", "world_y"] \
        + L.FEAT_COLS + L.TARGET_COLS
    print(f"[train] loading {csv}")
    df = pd.read_csv(csv, usecols=lambda c: c in set(cols))

    split = pd.read_csv(ROOT / "splits" / "model_x_track_split.csv")
    tr_ids = split[split.split == "train"].trajectory_id.tolist()
    va_ids = split[split.split == "val"].trajectory_id.tolist()
    wb = L.recording_world_bounds(df)

    # ── training windows (single-step) ──
    print(f"[train] building train windows from {len(tr_ids)} tracks ...")
    Xtr, Ytr = L.make_train_windows(df, tr_ids)
    print(f"[train] train windows: X={Xtr.shape}  Y={Ytr.shape}")

    # ── scalers (fit on train) ──
    f_sc = L.ColumnScaler().fit(Xtr.reshape(-1, L.N_FEAT), L.FEAT_COLS)
    t_sc = L.ColumnScaler().fit(Ytr, L.TARGET_COLS)
    Xtr_s = f_sc.transform(Xtr).astype(np.float32)
    Ytr_s = t_sc.transform(Ytr).astype(np.float32)

    # ── val single-step windows (for val MSE) ──
    Xva, Yva = L.make_train_windows(df, va_ids)
    Xva_s = torch.tensor(f_sc.transform(Xva).astype(np.float32))
    Yva_s = torch.tensor(t_sc.transform(Yva).astype(np.float32))

    # ── val rollout subsample (for ADE/FDE) ──
    vseeds, vstart, vgt, vwb, _ = L.build_eval_windows(
        df, va_ids, wb, n_steps=L.HORIZON, stride=2,
        max_windows=VAL_ROLLOUT_WINDOWS, seed=L.SEED)
    print(f"[train] val rollout subsample: {vseeds.shape[0]} windows")

    train_dl = DataLoader(TensorDataset(torch.tensor(Xtr_s), torch.tensor(Ytr_s)),
                          batch_size=tr_cfg["batch_size"], shuffle=True,
                          num_workers=0, pin_memory=(device.type == "cuda"))

    model = L.TrajectoryLSTM(L.N_FEAT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr_cfg["lr"],
                            weight_decay=tr_cfg["weight_decay"])
    crit = nn.MSELoss()

    # save scalers immediately
    (MODELS / "scalers.json").write_text(json.dumps(
        {"feature_scaler": f_sc.to_dict(), "target_scaler": t_sc.to_dict()}, indent=2))

    hist = []
    best_ade, best_fde, best_ade_ep, best_fde_ep = float("inf"), float("inf"), -1, -1
    patience, since_improve = tr_cfg["early_stopping_patience"], 0
    Xva_dev = Xva_s.to(device); Yva_dev = Yva_s.to(device)

    for ep in range(1, tr_cfg["epochs_max"] + 1):
        model.train(); te = time.time(); run = 0.0; nb = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tr_cfg["grad_clip"])
            opt.step()
            run += loss.item(); nb += 1
        train_loss = run / max(nb, 1)

        # val MSE
        model.eval()
        with torch.no_grad():
            vloss_parts = []
            for i in range(0, len(Xva_dev), 8192):
                vloss_parts.append(crit(model(Xva_dev[i:i+8192]), Yva_dev[i:i+8192]).item()
                                   * len(Xva_dev[i:i+8192]))
            val_loss = sum(vloss_parts) / max(len(Xva_dev), 1)

        # val rollout ADE/FDE
        pred = L.rollout_world_batch(model, vseeds, vstart, vwb, f_sc, t_sc,
                                     L.HORIZON, device)
        val_ade, val_fde = ade_fde(pred, vgt)

        improved = ""
        if val_ade < best_ade:
            best_ade, best_ade_ep = val_ade, ep; since_improve = 0
            torch.save(model.state_dict(), MODELS / "best_by_val_ADE.pth"); improved += "A"
        else:
            since_improve += 1
        if val_fde < best_fde:
            best_fde, best_fde_ep = val_fde, ep
            torch.save(model.state_dict(), MODELS / "best_by_val_FDE.pth"); improved += "F"
        torch.save(model.state_dict(), MODELS / "last_epoch.pth")

        hist.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss,
                     "val_ade": val_ade, "val_fde": val_fde,
                     "lr": opt.param_groups[0]["lr"]})
        print(f"[ep {ep:3d}/{tr_cfg['epochs_max']}] "
              f"train={train_loss:.5f} val={val_loss:.5f} "
              f"ADE={val_ade:.4f} FDE={val_fde:.4f} {improved:2s} "
              f"({time.time()-te:.1f}s)  best_ADE={best_ade:.4f}@{best_ade_ep}")

        if since_improve >= patience:
            print(f"[train] early stop at epoch {ep} (no val ADE improvement for {patience})")
            break

    # ── logs ──
    H = pd.DataFrame(hist)
    H[["epoch", "train_loss"]].to_csv(HERE / "train_loss.csv", index=False)
    H[["epoch", "val_loss"]].to_csv(HERE / "val_loss.csv", index=False)
    H.to_csv(HERE / "val_metrics_by_epoch.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(H.epoch, H.train_loss, label="train loss"); ax.plot(H.epoch, H.val_loss, label="val loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (scaled)"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("MODEL_X loss"); fig.tight_layout(); fig.savefig(HERE / "loss_curve.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(H.epoch, H.val_ade, label="val ADE (m)"); ax.plot(H.epoch, H.val_fde, label="val FDE (m)")
    ax.axvline(best_ade_ep, ls="--", c="green", alpha=.6, label=f"best ADE @ {best_ade_ep}")
    ax.set_xlabel("epoch"); ax.set_ylabel("metres"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("MODEL_X val ADE/FDE (10-step rollout)"); fig.tight_layout()
    fig.savefig(HERE / "ade_fde_curve.png", dpi=130); plt.close(fig)

    summary = {
        "epochs_completed": int(H.epoch.max()),
        "best_val_ADE": best_ade, "best_val_ADE_epoch": best_ade_ep,
        "best_val_FDE": best_fde, "best_val_FDE_epoch": best_fde_ep,
        "final_train_loss": float(H.train_loss.iloc[-1]),
        "final_val_loss": float(H.val_loss.iloc[-1]),
        "train_windows": int(Xtr.shape[0]),
        "val_rollout_windows": int(vseeds.shape[0]),
        "device": str(device),
        "selected_checkpoint": "best_by_val_ADE.pth",
        "minutes": round((time.time() - t0) / 60, 2),
    }
    (HERE / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print("[train] DONE", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
