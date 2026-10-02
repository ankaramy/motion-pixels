"""
Phase 2 — train Frozen Model C on the Barcelona recording-level split.

Mirrors the frozen `train_one` loop EXACTLY (same architecture, scaler, optimizer,
scheduler, grad-clip, batch size, epochs, patience, seed — all imported from
run_bridge_ablation.py) and ADDS the required logging artifacts (final
checkpoint, epoch_log.csv, training_config.json, lr curve). Splits come from the
`split` column, NOT the frozen random split_ids().
"""
from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import common as C
fz = C.fz


def main():
    fz.set_seed(fz.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    df = C.load_dataset()
    train_df = df[df.split == "train"]
    val_df = df[df.split == "val"]
    train_ids = train_df.trajectory_id.unique().tolist()
    val_ids = val_df.trajectory_id.unique().tolist()
    print(f"[split] train {len(train_ids)} tracks / {len(train_df):,} rows | "
          f"val {len(val_ids)} tracks / {len(val_df):,} rows")

    # --- scalers fit on TRAIN ONLY ---
    f_sc = fz.ColumnScaler().fit(train_df, C.FEAT_COLS)
    t_sc = fz.ColumnScaler().fit(train_df, C.TARGET_COLS)

    # --- windows (frozen make_windows; never cross trajectory/recording since
    #     trajectory_id is recording-namespaced) ---
    X_tr, Y_tr = fz.make_windows(df, C.FEAT_COLS, train_ids)
    X_va, Y_va = fz.make_windows(df, C.FEAT_COLS, val_ids)
    print(f"[windows] train {X_tr.shape} val {X_va.shape}")

    X_tr = f_sc.transform(X_tr.reshape(-1, len(C.FEAT_COLS))).reshape(X_tr.shape).astype(np.float32)
    X_va = f_sc.transform(X_va.reshape(-1, len(C.FEAT_COLS))).reshape(X_va.shape).astype(np.float32)
    Y_tr = t_sc.transform(Y_tr).astype(np.float32)
    Y_va = t_sc.transform(Y_va).astype(np.float32)

    tr_loader = DataLoader(fz.WindowDataset(X_tr, Y_tr), batch_size=fz.BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(fz.WindowDataset(X_va, Y_va), batch_size=fz.BATCH_SIZE * 2, shuffle=False)

    model = fz.TrajectoryLSTM(len(C.FEAT_COLS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=fz.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, fz.LR_STEP, fz.LR_GAMMA)
    criterion = nn.MSELoss()

    best_val, no_improve, best_state, best_epoch = float("inf"), 0, None, 0
    epoch_log = []
    t0 = time.time()
    for epoch in range(1, fz.MAX_EPOCHS + 1):
        ep_t = time.time()
        model.train(); tr_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= max(len(X_tr), 1)

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_loss += criterion(model(xb), yb).item() * len(xb)
        va_loss /= max(len(X_va), 1)

        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if not np.isfinite(tr_loss) or not np.isfinite(va_loss):
            raise SystemExit(f"[FATAL] non-finite loss at epoch {epoch}: train={tr_loss} val={va_loss}")

        is_best = va_loss < best_val
        if is_best:
            best_val, no_improve, best_epoch = va_loss, 0, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        epoch_log.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                          "lr": lr_now, "epoch_time_s": time.time() - ep_t, "is_best": int(is_best)})
        if epoch % 5 == 0 or epoch == 1:
            print(f"  ep {epoch:3d}  train={tr_loss:.6f}  val={va_loss:.6f}  best={best_val:.6f}  lr={lr_now:.2e}")
        if no_improve >= fz.PATIENCE:
            print(f"  early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    final_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    early_stopped = no_improve >= fz.PATIENCE
    actual_epochs = epoch_log[-1]["epoch"]

    # --- save artifacts ---
    C.D_TRAIN.mkdir(parents=True, exist_ok=True); C.D_MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, C.D_MODELS / "best_model_C_barcelona.pt")
    torch.save(final_state, C.D_MODELS / "final_model_C_barcelona.pt")
    C.save_scalers(C.D_TRAIN / "scalers.pkl", f_sc, t_sc, C.FEAT_COLS, C.TARGET_COLS)
    pd.DataFrame(epoch_log).to_csv(C.D_TRAIN / "epoch_log.csv", index=False)

    cfg = {
        "model": "TrajectoryLSTM (frozen Model C, C_motion_position_spatial)",
        "input_dim": len(C.FEAT_COLS), "hidden_dim": fz.HIDDEN_SIZE,
        "num_layers": fz.NUM_LAYERS, "dropout": fz.DROPOUT, "output_dim": 2,
        "feature_order": C.FEAT_COLS, "targets": C.TARGET_COLS,
        "window": fz.WINDOW_SIZE, "stride": 1,
        "optimizer": "AdamW", "lr": fz.LR, "weight_decay": 1e-4,
        "scheduler": f"StepLR(step={fz.LR_STEP}, gamma={fz.LR_GAMMA})",
        "grad_clip": 1.0, "batch_size": fz.BATCH_SIZE, "max_epochs": fz.MAX_EPOCHS,
        "patience": fz.PATIENCE, "loss": "MSE", "seed": fz.SEED,
        "scaler": "column-wise StandardScaler, fit on TRAIN only, applied to features+targets",
        "split_source": "recording-level `split` column (not frozen random split_ids)",
        "train_recordings": [r for r in C.SPLIT_MAP if C.SPLIT_MAP[r] == "train"],
        "val_recordings": [C.VAL_REC], "test_recordings": [C.TEST_REC],
        "device": str(device),
        "actual_epochs": int(actual_epochs), "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "final_train_loss": float(epoch_log[-1]["train_loss"]),
        "final_val_loss": float(epoch_log[-1]["val_loss"]),
        "early_stopped": bool(early_stopped),
        "total_train_time_s": round(time.time() - t0, 1),
    }
    (C.D_TRAIN / "training_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # --- plots ---
    el = pd.DataFrame(epoch_log)
    f, a = plt.subplots(figsize=(8, 4.6))
    a.plot(el.epoch, el.train_loss, color="#3aaa5e", lw=1.6, label="train")
    a.plot(el.epoch, el.val_loss, color="#4c8cbf", lw=1.6, label="val")
    a.axvline(best_epoch, color="#d62728", ls="--", lw=1.2, label=f"best epoch {best_epoch}")
    a.set_xlabel("epoch"); a.set_ylabel("MSE (scaled)"); a.set_title("Model C training loss (Barcelona)"); a.legend()
    f.tight_layout(); f.savefig(C.D_TRAIN / "loss_curve_train_val.png"); plt.close(f)

    f, a = plt.subplots(figsize=(8, 4.6))
    a.semilogy(el.epoch, el.train_loss, color="#3aaa5e", lw=1.6, label="train")
    a.semilogy(el.epoch, el.val_loss, color="#4c8cbf", lw=1.6, label="val")
    a.axvline(best_epoch, color="#d62728", ls="--", lw=1.2, label=f"best epoch {best_epoch}")
    a.set_xlabel("epoch"); a.set_ylabel("MSE (log)"); a.set_title("Model C training loss (log scale)"); a.legend()
    f.tight_layout(); f.savefig(C.D_TRAIN / "loss_curve_log_scale.png"); plt.close(f)

    f, a = plt.subplots(figsize=(8, 4.2))
    a.plot(el.epoch, el.lr, color="#8e44ad", lw=1.6)
    a.set_xlabel("epoch"); a.set_ylabel("learning rate"); a.set_title("Learning-rate schedule (StepLR 25, 0.35)")
    f.tight_layout(); f.savefig(C.D_TRAIN / "lr_curve.png"); plt.close(f)

    # --- report ---
    L = ["# Phase 2 — Model C Training Report (Barcelona)\n",
         f"- Architecture: TrajectoryLSTM, input={len(C.FEAT_COLS)}, hidden={fz.HIDDEN_SIZE}, "
         f"layers={fz.NUM_LAYERS}, dropout={fz.DROPOUT}, head Linear(128->2)",
         f"- Feature order: `{C.FEAT_COLS}`",
         f"- Targets: `{C.TARGET_COLS}`",
         f"- Train recordings: {cfg['train_recordings']}",
         f"- Val recording: {cfg['val_recordings']}",
         f"- Optimizer/recipe: AdamW lr {fz.LR} wd 1e-4, StepLR({fz.LR_STEP},{fz.LR_GAMMA}), "
         f"grad-clip 1.0, batch {fz.BATCH_SIZE}, max {fz.MAX_EPOCHS} ep, patience {fz.PATIENCE}, MSE, seed {fz.SEED}",
         f"- Scaler: StandardScaler fit on TRAIN only (features+targets) -> `scalers.pkl`\n",
         "## Results\n",
         f"- Actual epochs run: **{actual_epochs}**",
         f"- Best epoch: **{best_epoch}**",
         f"- Best validation loss (scaled MSE): **{best_val:.6f}**",
         f"- Final train loss: **{epoch_log[-1]['train_loss']:.6f}**",
         f"- Final val loss: **{epoch_log[-1]['val_loss']:.6f}**",
         f"- Early stopping triggered: **{'yes' if early_stopped else 'no'}**",
         f"- Total train time: {cfg['total_train_time_s']} s ({device})\n",
         "## Artifacts\n",
         "- `models/best_model_C_barcelona.pt`",
         "- `models/final_model_C_barcelona.pt`",
         "- `01_training_run/scalers.pkl`",
         "- `01_training_run/training_config.json`",
         "- `01_training_run/epoch_log.csv`",
         "- `01_training_run/loss_curve_train_val.png`, `loss_curve_log_scale.png`, `lr_curve.png`"]
    (C.D_TRAIN / "training_report.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\n[done] best epoch {best_epoch}  best_val {best_val:.6f}  "
          f"final_train {epoch_log[-1]['train_loss']:.6f}  early_stopped {early_stopped}")


if __name__ == "__main__":
    main()
