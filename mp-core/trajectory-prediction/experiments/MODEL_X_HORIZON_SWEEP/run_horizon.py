"""
run_horizon.py <H> — train + evaluate one MODEL_X horizon variant.

Same Model X recipe (Model C TrajectoryLSTM, 10 features, v3 manual masks, mixed
track-level split, single-step next-displacement MSE objective). Horizon H governs:
  (1) which tracks qualify (need >= obs + H frames -> reported data availability),
  (2) the autoregressive rollout depth at val/eval,
  (3) checkpoint selection by H-step rollout validation ADE.

The training OBJECTIVE stays single-step (Model X / Model C) — we do NOT use a
multi-step rollout loss (infeasible at H=400) and we do NOT change the architecture.
Each horizon model is trained only on trajectories that actually persist >= obs+H
frames, so the long-horizon models see the relevant (persistent-walker) distribution.

Usage:  python run_horizon.py 20
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
MODELX = HERE.parent / "MODEL_X"
sys.path.insert(0, str(MODELX))
import model_x_lib as L  # noqa: E402

APPROX_DIST_M = {20: 1.0, 60: 3.0, 100: 5.0, 200: 10.0, 400: 20.0}
CSV = L.CONFIG["dataset"]["csv_path"]
OBS = L.WINDOW_SIZE
SEED = L.SEED
VAL_ROLLOUT_WINDOWS = 2500
CHUNK = 16384


def ade_fde(pred, gt):
    n = min(pred.shape[1], gt.shape[1])
    disp = np.linalg.norm(pred[:, :n] - gt[:, :n], axis=2)
    return float(disp[:, 1:].mean()), float(disp[:, -1].mean())


def make_train_windows_qual(df, traj_ids):
    """Single-step windows from qualifying tracks (already length-filtered)."""
    return L.make_train_windows(df, traj_ids)


def diag_panel(ax, w):
    obs, gt, pred = w["obs"], w["gt"], w["pred"]
    ax.plot(obs[:, 0], obs[:, 1], "-o", color="#1f77b4", ms=2, lw=1.2, label="observed")
    ax.plot(gt[:, 0], gt[:, 1], "-", color="#2ca02c", lw=1.6, label="GT future")
    ax.plot(pred[:, 0], pred[:, 1], "--", color="#d62728", lw=1.6, label="prediction")
    ax.plot(obs[-1, 0], obs[-1, 1], "o", color="black", ms=5)
    ax.set_aspect("equal", "datalim"); ax.tick_params(labelsize=6)
    ax.set_title(f"{w['recording_id']}\nADE={w['ade']:.2f} FDE={w['fde']:.2f} "
                 f"len={w['pred_len']:.1f}/{w['gt_len']:.1f}m", fontsize=7)


def pres_panel(ax, w):
    obs, gt, pred = w["obs"], w["gt"], w["pred"]
    ax.plot(obs[:, 0], obs[:, 1], "-", color="black", lw=2.0, solid_capstyle="round")
    ax.plot(gt[:, 0], gt[:, 1], "-", color="#b0b0b0", lw=3.0, solid_capstyle="round")
    ax.plot(pred[:, 0], pred[:, 1], "-", color="#e0218a", lw=2.2, solid_capstyle="round")
    if len(pred) >= 2:
        ax.annotate("", xy=pred[-1], xytext=pred[-2],
                    arrowprops=dict(arrowstyle="-|>", color="#e0218a", lw=2.0))
    ax.plot(obs[-1, 0], obs[-1, 1], "o", color="#333", ms=4)
    ax.set_aspect("equal", "datalim"); ax.axis("off")


def grid(panel_fn, windows, title, out, n=6, shape=(2, 3)):
    import matplotlib.pyplot as plt
    if not windows:
        print(f"  skip {out.name} (no windows)"); return
    fig, axes = plt.subplots(*shape, figsize=(13, 8 if shape[0] == 2 else 12))
    for i, ax in enumerate(np.atleast_1d(axes).flat):
        if i < min(n, len(windows)):
            panel_fn(ax, windows[i])
        else:
            ax.axis("off")
    fig.suptitle(title, fontsize=12); fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=140); plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H = int(sys.argv[1])
    OUT = HERE / f"H{H}"
    OUT.mkdir(exist_ok=True)
    need = OBS + H
    t0 = time.time()
    L.set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n===== HORIZON H{H} (need>={need} frames, approx {APPROX_DIST_M.get(H,'?')} m) "
          f"device={device} =====")

    cols = ["recording_id", "trajectory_id", "timestep", "world_x", "world_y"] + L.FEAT_COLS + L.TARGET_COLS
    df = pd.read_csv(CSV, usecols=lambda c: c in set(cols))
    split = pd.read_csv(MODELX / "splits" / "model_x_track_split.csv")
    Lens = df.groupby("trajectory_id").size()
    wb = L.recording_world_bounds(df)

    def qual(spl):
        ids = split[split.split == spl].trajectory_id
        return [t for t in ids if Lens.get(t, 0) >= need]
    tr_ids, va_ids, te_ids = qual("train"), qual("val"), qual("test")

    # ── data-availability report ──
    all_tr = split[split.split == "train"]
    avail = {"horizon": H, "approx_distance_m": APPROX_DIST_M.get(H),
             "need_frames": need,
             "tracks_total": int(split.trajectory_id.nunique()),
             "tracks_qualifying": len(tr_ids) + len(va_ids) + len(te_ids),
             "train_tracks": len(tr_ids), "val_tracks": len(va_ids), "test_tracks": len(te_ids),
             "tracks_dropped_short": int(split.trajectory_id.nunique()
                                         - (len(tr_ids) + len(va_ids) + len(te_ids)))}
    per_rec_avail = {}
    for rec in sorted(split.recording_id.unique()):
        ids = split[(split.recording_id == rec)]
        q = {s: sum(1 for t in ids[ids.split == s].trajectory_id if Lens.get(t, 0) >= need)
             for s in ("train", "val", "test")}
        per_rec_avail[rec] = q
    avail["per_recording_qualifying_tracks"] = per_rec_avail

    print(f"  qualifying tracks: train={len(tr_ids)} val={len(va_ids)} test={len(te_ids)} "
          f"(dropped {avail['tracks_dropped_short']} short)")

    # ── train windows (single-step) ──
    Xtr, Ytr = make_train_windows_qual(df, tr_ids)
    print(f"  train windows: {Xtr.shape}")
    f_sc = L.ColumnScaler().fit(Xtr.reshape(-1, L.N_FEAT), L.FEAT_COLS)
    t_sc = L.ColumnScaler().fit(Ytr, L.TARGET_COLS)
    Xtr_s = torch.tensor(f_sc.transform(Xtr).astype(np.float32))
    Ytr_s = torch.tensor(t_sc.transform(Ytr).astype(np.float32))

    Xva, Yva = make_train_windows_qual(df, va_ids)
    Xva_dev = torch.tensor(f_sc.transform(Xva).astype(np.float32)).to(device)
    Yva_dev = torch.tensor(t_sc.transform(Yva).astype(np.float32)).to(device)

    vstride = max(1, H // 10)
    vseeds, vstart, vgt, vwb, _ = L.build_eval_windows(
        df, va_ids, wb, n_steps=H, stride=vstride, max_windows=VAL_ROLLOUT_WINDOWS, seed=SEED)
    print(f"  val rollout subsample: {vseeds.shape[0]}")

    bs = 256
    dl = DataLoader(TensorDataset(Xtr_s, Ytr_s), batch_size=bs, shuffle=True,
                    pin_memory=(device.type == "cuda"))
    model = L.TrajectoryLSTM(L.N_FEAT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.MSELoss()

    hist, best_ade, best_ep, since = [], float("inf"), -1, 0
    for ep in range(1, 81):
        model.train(); run = nb = 0; te = time.time()
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            run += loss.item(); nb += 1
        train_loss = run / max(nb, 1)
        model.eval()
        with torch.no_grad():
            vl = sum(crit(model(Xva_dev[i:i+8192]), Yva_dev[i:i+8192]).item()
                     * len(Xva_dev[i:i+8192]) for i in range(0, len(Xva_dev), 8192)) / max(len(Xva_dev), 1)
        pred = L.rollout_world_batch(model, vseeds, vstart, vwb, f_sc, t_sc, H, device)
        v_ade, v_fde = ade_fde(pred, vgt)
        flag = ""
        if v_ade < best_ade:
            best_ade, best_ep, since = v_ade, ep, 0
            torch.save(model.state_dict(), OUT / "model_best.pth"); flag = "*"
        else:
            since += 1
        hist.append({"epoch": ep, "train_loss": train_loss, "val_loss": vl,
                     "val_ade": v_ade, "val_fde": v_fde})
        print(f"  [ep {ep:3d}] train={train_loss:.4f} val={vl:.4f} "
              f"ADE={v_ade:.3f} FDE={v_fde:.3f} {flag} ({time.time()-te:.1f}s) best={best_ade:.3f}@{best_ep}")
        if since >= 12:
            print(f"  early stop @ {ep}"); break

    Hh = pd.DataFrame(hist)
    Hh[["epoch", "train_loss"]].to_csv(OUT / "train_loss.csv", index=False)
    Hh.to_csv(OUT / "val_metrics_by_epoch.csv", index=False)
    (OUT / "scalers.json").write_text(json.dumps(
        {"feature_scaler": f_sc.to_dict(), "target_scaler": t_sc.to_dict()}))

    # curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(Hh.epoch, Hh.train_loss, label="train"); ax[0].plot(Hh.epoch, Hh.val_loss, label="val")
    ax[0].set_title("loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(Hh.epoch, Hh.val_ade, label="val ADE"); ax[1].plot(Hh.epoch, Hh.val_fde, label="val FDE")
    ax[1].axvline(best_ep, ls="--", c="g", alpha=.5); ax[1].set_title(f"H{H} val ADE/FDE (m)")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "loss_curve.png", dpi=120); plt.close(fig)

    # ── final eval on full test ──
    model.load_state_dict(torch.load(OUT / "model_best.pth", map_location=device)); model.eval()
    seeds, start, gt, wba, meta = L.build_eval_windows(df, te_ids, wb, n_steps=H, stride=1)
    M = len(seeds); print(f"  test windows: {M}")
    preds = np.empty((M, H + 1, 2))
    for i in range(0, M, CHUNK):
        sl = slice(i, i + CHUNK)
        preds[sl] = L.rollout_world_batch(model, seeds[sl], start[sl],
                                          {k: v[sl] for k, v in wba.items()}, f_sc, t_sc, H, device)

    rows = []
    for k in range(M):
        m = L.window_metrics(preds[k], gt[k])
        m.update({"recording_id": meta[k]["recording_id"], "trajectory_id": meta[k]["trajectory_id"],
                  "start_idx": meta[k]["start_idx"], "window_idx": k})
        rows.append(m)
    metr = pd.DataFrame(rows)
    metr.to_csv(OUT / "per_window_metrics.csv", index=False)

    gt_len = metr.gt_path_len.to_numpy(); pred_len = metr.pred_path_len.to_numpy()
    collapse_rate = float(np.mean(pred_len < 0.25 * np.maximum(gt_len, 1e-9)))
    summary = {
        "horizon": H, "approx_distance_m": APPROX_DIST_M.get(H),
        "data_availability": avail,
        "best_epoch": best_ep, "val_ADE": best_ade,
        "test_windows": int(M),
        "train_windows": int(Xtr.shape[0]),
        "val_rollout_windows": int(vseeds.shape[0]),
        "ADE": float(metr.ade.mean()), "FDE": float(metr.fde.mean()),
        "ADE_median": float(metr.ade.median()), "FDE_median": float(metr.fde.median()),
        "angular_err_mean_deg": float(metr.angular_err_deg.mean(skipna=True)),
        "angular_err_median_deg": float(metr.angular_err_deg.median(skipna=True)),
        "gt_len_median": float(np.median(gt_len)), "pred_len_median": float(np.median(pred_len)),
        "pred_len_p75": float(np.percentile(pred_len, 75)),
        "gt_len_p75": float(np.percentile(gt_len, 75)),
        "pred_gt_ratio_median": float(np.median(pred_len / np.maximum(gt_len, 1e-9))),
        "collapse_rate_pred_lt_25pct_gt": collapse_rate,
        "minutes": round((time.time() - t0) / 60, 2),
    }
    per_rec = metr.groupby("recording_id").agg(
        n=("ade", "size"), ADE=("ade", "mean"), FDE=("fde", "mean")).round(4)
    per_rec.to_csv(OUT / "per_recording_metrics.csv")
    summary["per_recording"] = {r: {"n": int(v.n), "ADE": float(v.ADE), "FDE": float(v.FDE)}
                                for r, v in per_rec.iterrows()}
    (OUT / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

    # config.json
    (OUT / "config.json").write_text(json.dumps({
        "horizon": H, "approx_distance_m": APPROX_DIST_M.get(H),
        "obs_len": OBS, "features": L.FEAT_COLS, "targets": L.TARGET_COLS,
        "arch": {"hidden": 128, "layers": 2, "dropout": 0.2},
        "training": {"objective": "single-step next-displacement MSE (Model X)",
                     "rollout_depth_eval": H, "optimizer": "AdamW", "lr": 1e-3,
                     "weight_decay": 1e-5, "batch_size": bs, "grad_clip": 1.0,
                     "max_epochs": 80, "patience": 12, "seed": SEED,
                     "checkpoint": "best val H-step rollout ADE"},
        "track_filter": f"len >= obs+H = {need}",
        "dataset": CSV,
    }, indent=2))

    # ── plots ──
    def obs_path(tid, si):
        g = df[df.trajectory_id == tid].sort_values("timestep")
        sub = g.iloc[si:si + OBS]
        return np.column_stack([sub.world_x.to_numpy(float), sub.world_y.to_numpy(float)])

    def pack(idxs):
        out = []
        for k in idxs:
            r = metr.iloc[k]
            out.append({"obs": obs_path(r.trajectory_id, int(r.start_idx)),
                        "gt": gt[k], "pred": preds[k], "ade": float(r.ade), "fde": float(r.fde),
                        "pred_len": float(r.pred_path_len), "gt_len": float(r.gt_path_len),
                        "recording_id": r.recording_id})
        return out

    best6 = metr.nsmallest(6, "ade").index.tolist()
    worst6 = metr.nlargest(6, "ade").index.tolist()
    long_pool = metr[metr.gt_path_len >= metr.gt_path_len.quantile(0.75)]
    long6 = long_pool.nsmallest(6, "ade").index.tolist()
    grid(diag_panel, pack(best6), f"H{H} — best 6 overall (lowest ADE)",
         OUT / "best_6_overall" / "best_6_overall.png")
    grid(diag_panel, pack(worst6), f"H{H} — worst 6 overall (highest ADE)",
         OUT / "worst_6_overall" / "worst_6_overall.png")
    grid(diag_panel, pack(long6), f"H{H} — best 6 long-GT predictions",
         OUT / "best_6_long_predictions" / "best_6_long_predictions.png")

    # presentation collage (3x3 of long, good-ADE windows)
    pres = long_pool.nsmallest(9, "ade").index.tolist()
    grid(pres_panel, pack(pres), f"MODEL_X H{H} (~{APPROX_DIST_M.get(H)} m) — predictions",
         OUT / "presentation_collage.png", n=9, shape=(3, 3))

    # distributions
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(pred_len, bins=60, color="#d62728", alpha=.6, label="predicted")
    ax.hist(gt_len, bins=60, color="#2ca02c", alpha=.5, label="GT")
    ax.set_xlabel(f"path length over {H} steps (m)"); ax.set_ylabel("count")
    ax.legend(); ax.grid(alpha=.3); ax.set_title(f"H{H} path-length distribution")
    fig.tight_layout(); fig.savefig(OUT / "prediction_length_distribution.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    mx = max(gt_len.max(), pred_len.max())
    ax.scatter(gt_len, pred_len, s=4, alpha=.15, color="#1f77b4")
    ax.plot([0, mx], [0, mx], "k--", lw=1, label="y=x")
    ax.plot([0, mx], [0, 0.25 * mx], "r--", lw=1, label="collapse (25%)")
    ax.set_xlabel("GT path length (m)"); ax.set_ylabel("pred path length (m)")
    ax.legend(); ax.grid(alpha=.3); ax.set_title(f"H{H} GT vs pred length")
    fig.tight_layout(); fig.savefig(OUT / "gt_vs_pred_length_scatter.png", dpi=130); plt.close(fig)

    # metrics_summary.md
    md = [f"# MODEL_X H{H} — metrics summary", "",
          f"Approx distance ~**{APPROX_DIST_M.get(H)} m**. Best epoch {best_ep} (val ADE {best_ade:.3f}).",
          f"Single-step training objective; {H}-step autoregressive rollout at eval; spatial features frozen.",
          "",
          "## Data availability",
          f"- need >= {need} frames; qualifying tracks: train {len(tr_ids)} / val {len(va_ids)} / test {len(te_ids)}",
          f"- dropped {avail['tracks_dropped_short']} short tracks; train windows {Xtr.shape[0]}; test windows {M}",
          "",
          "## Test metrics (all windows)",
          f"- ADE **{summary['ADE']:.3f} m** (median {summary['ADE_median']:.3f})",
          f"- FDE **{summary['FDE']:.3f} m** (median {summary['FDE_median']:.3f})",
          f"- angular err {summary['angular_err_mean_deg']:.1f}° mean / {summary['angular_err_median_deg']:.1f}° median",
          f"- GT len median {summary['gt_len_median']:.2f} m | pred len median {summary['pred_len_median']:.2f} m "
          f"| ratio **{summary['pred_gt_ratio_median']:.2f}**",
          f"- pred len p75 {summary['pred_len_p75']:.2f} m (GT p75 {summary['gt_len_p75']:.2f} m)",
          f"- **collapse rate (pred<25% GT): {collapse_rate*100:.1f}%**",
          "", "## Per-recording", "", "| recording | n | ADE | FDE |", "|---|---:|---:|---:|"]
    for r, v in per_rec.iterrows():
        md.append(f"| {r} | {int(v.n)} | {v.ADE:.3f} | {v.FDE:.3f} |")
    (OUT / "metrics_summary.md").write_text("\n".join(md), encoding="utf-8")

    with open(OUT / "plot_windows.pkl", "wb") as fh:
        pickle.dump({"best6": pack(best6), "long6": pack(long6)}, fh)

    print(f"  DONE H{H}: ADE={summary['ADE']:.3f} FDE={summary['FDE']:.3f} "
          f"ratio={summary['pred_gt_ratio_median']:.2f} collapse={collapse_rate*100:.1f}% "
          f"({summary['minutes']} min)")


if __name__ == "__main__":
    main()
