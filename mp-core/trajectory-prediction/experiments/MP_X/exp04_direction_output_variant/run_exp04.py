"""
MP_X / exp04 — Direction-output variant of Model C.

Tests whether the turn-DIRECTION bottleneck (exp01/exp02: the model produces
turn magnitude but lands ~90° angular error) is the displacement-only output
head. Same Model C trunk + 10 inputs, but the head predicts 4 values:

    target_du, target_dv, future_heading_sin, future_heading_cos

Loss = position_mse + 0.2 * heading_mse (standardized target space). Trained
turn-only (same genuine-turn filter as exp02) with genuine-turn validation for
early stopping. At rollout the explicit predicted heading drives the heading
input channels. All turn metrics are measured from realized positions, so the
numbers are directly comparable to exp01 and exp02.

Run:
    python run_exp04.py                 # train + evaluate + plot
    python run_exp04.py --skip-train    # reuse models/best_model_exp04.pth

Nothing here modifies frozen_model_C, the dataset, or any other experiment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "shared"
EXP01 = HERE.parent / "exp01_turn_balanced_model_c"
EXP02 = HERE.parent / "exp02_turn_only_diagnostic"
sys.path.insert(0, str(SHARED))

import mpx_common as C       # noqa: E402
import mpx_dir as D          # noqa: E402
import mpx_plots as V        # noqa: E402

TAG = "exp04"
HORIZON = C.N_ROLLOUT
HEAD_LOSS_W = 0.2
EVAL_SPLITS = ["val", "test"]
CAPACITY_SPLIT = ["train"]
PLOTS = HERE / "plots"


def genuine_mask(meta):
    return meta["is_genuine_turn"].to_numpy(dtype=bool)


# ─────────────────────────────────────────────────────────────────────────────
# Training (4-output, position_mse + 0.2*heading_mse)
# ─────────────────────────────────────────────────────────────────────────────
def train(df, device, seed=42):
    models_dir = HERE / "models"
    train_dir = HERE / "training"
    for d in (models_dir, train_dir):
        d.mkdir(parents=True, exist_ok=True)

    C.fz.set_seed(seed)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    train_ids = train_df["trajectory_id"].unique().tolist()
    val_ids = val_df["trajectory_id"].unique().tolist()

    f_sc = C.fz.ColumnScaler().fit(train_df, C.FEAT_COLS)
    t_sc = C.fz.ColumnScaler().fit(train_df, D.DIR_TARGET_COLS)

    print(f"[windows] building labeled windows (horizon={HORIZON}) ...")
    X_tr, Y_tr, meta_tr = C.make_labeled_windows(df, C.FEAT_COLS, train_ids, HORIZON,
                                                 target_cols=D.DIR_TARGET_COLS)
    X_va, Y_va, meta_va = C.make_labeled_windows(df, C.FEAT_COLS, val_ids, HORIZON,
                                                 target_cols=D.DIR_TARGET_COLS)
    # turn-only filter on BOTH train and val (val = early-stop signal)
    mtr, mva = genuine_mask(meta_tr), genuine_mask(meta_va)
    X_tr, Y_tr, meta_tr = X_tr[mtr], Y_tr[mtr], meta_tr[mtr].reset_index(drop=True)
    X_va, Y_va, meta_va = X_va[mva], Y_va[mva], meta_va[mva].reset_index(drop=True)
    counts_tr, counts_va = C.bin_counts(meta_tr), C.bin_counts(meta_va)
    print(f"[windows] train(turn-only)={len(X_tr):,}  val(turn-only)={len(X_va):,}")
    print(f"[windows] train bins: {counts_tr}")

    nf = len(C.FEAT_COLS)
    X_tr_s = f_sc.transform(X_tr.reshape(-1, nf)).reshape(X_tr.shape).astype(np.float32)
    X_va_s = f_sc.transform(X_va.reshape(-1, nf)).reshape(X_va.shape).astype(np.float32)
    Y_tr_s = t_sc.transform(Y_tr).astype(np.float32)
    Y_va_s = t_sc.transform(Y_va).astype(np.float32)

    tr_loader = DataLoader(C.fz.WindowDataset(X_tr_s, Y_tr_s), batch_size=C.fz.BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(C.fz.WindowDataset(X_va_s, Y_va_s), batch_size=C.fz.BATCH_SIZE * 2, shuffle=False)

    model = D.TrajectoryLSTMDir(nf).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=C.fz.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(optim, C.fz.LR_STEP, C.fz.LR_GAMMA)

    best_val, no_imp, best_state, best_ep = float("inf"), 0, None, 0
    log, t0 = [], time.time()
    for epoch in range(1, C.fz.MAX_EPOCHS + 1):
        model.train()
        tl = tp = th = 0.0
        nseen = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            pos, head = D.heading_loss_components(pred, yb)
            loss = pos + HEAD_LOSS_W * head
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            bs = len(xb)
            tl += loss.item() * bs; tp += pos.item() * bs; th += head.item() * bs; nseen += bs
        tl /= max(nseen, 1); tp /= max(nseen, 1); th /= max(nseen, 1)

        model.eval()
        vl = vp = vh = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                pos, head = D.heading_loss_components(pred, yb)
                bs = len(xb)
                vl += (pos + HEAD_LOSS_W * head).item() * bs; vp += pos.item() * bs; vh += head.item() * bs
        nv = max(len(X_va_s), 1); vl /= nv; vp /= nv; vh /= nv
        lr_now = optim.param_groups[0]["lr"]; sched.step()

        is_best = vl < best_val
        if is_best:
            best_val, no_imp, best_ep = vl, 0, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        log.append({"epoch": epoch, "train_total": tl, "train_pos_mse": tp, "train_head_mse": th,
                    "val_total": vl, "val_pos_mse": vp, "val_head_mse": vh, "lr": lr_now,
                    "is_best": int(is_best)})
        if epoch == 1 or epoch % 5 == 0 or is_best:
            print(f"  ep {epoch:3d}  train={tl:.5f} (pos {tp:.4f} head {th:.4f})  val={vl:.5f}  best={best_val:.5f}")
        if no_imp >= C.fz.PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    ckpt = models_dir / f"best_model_{TAG}.pth"
    scal = models_dir / f"scalers_{TAG}.pkl"
    torch.save(best_state, ckpt)
    D.save_dir_scalers(scal, f_sc, t_sc)
    log_df = pd.DataFrame(log)
    log_df.to_csv(train_dir / "epoch_log.csv", index=False)
    log_df[["epoch", "train_total"]].rename(columns={"train_total": "train_loss"}).to_csv(train_dir / "training_loss.csv", index=False)
    log_df[["epoch", "val_total"]].rename(columns={"val_total": "val_loss"}).to_csv(train_dir / "validation_loss.csv", index=False)
    cfg = {"experiment": TAG, "model": "TrajectoryLSTMDir (Model C trunk + 4-output head)",
           "feature_order": C.FEAT_COLS, "targets": D.DIR_TARGET_COLS,
           "loss": f"position_mse + {HEAD_LOSS_W}*heading_mse (standardized target space)",
           "training": "turn-only (genuine-turn filter on train + val)",
           "window": C.WINDOW_SIZE, "label_horizon": HORIZON,
           "batch_size": C.fz.BATCH_SIZE, "lr": C.fz.LR, "weight_decay": 1e-4,
           "scheduler": f"StepLR(step={C.fz.LR_STEP}, gamma={C.fz.LR_GAMMA})",
           "max_epochs": C.fz.MAX_EPOCHS, "patience": C.fz.PATIENCE, "seed": seed,
           "device": str(device), "n_train_windows": int(len(X_tr_s)),
           "n_val_windows": int(len(X_va_s)), "train_bin_counts": counts_tr,
           "val_bin_counts": counts_va, "actual_epochs": int(log_df["epoch"].iloc[-1]),
           "best_epoch": best_ep, "best_val_loss": float(best_val),
           "train_time_s": round(time.time() - t0, 1), "dataset": str(C.P5.MODEL_C_CSV)}
    (train_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[saved] {ckpt}\n[done] best epoch {best_ep} val {best_val:.5f} ({cfg['train_time_s']}s)")
    return model, f_sc, t_sc, cfg, counts_tr, counts_va


# ─────────────────────────────────────────────────────────────────────────────
def per_bin_table(rolled):
    if rolled is None or len(rolled) == 0:
        return pd.DataFrame([C.aggregate(None, f"bin:{b}") for b in C.BIN_ORDER])
    rows = []
    for b in C.BIN_ORDER:
        sub = (rolled[(rolled["bin"] == b) & (rolled["in_all_sample"])] if b == "straight"
               else rolled[(rolled["bin"] == b) & (rolled["is_genuine_turn"])])
        rows.append(C.aggregate(sub, f"bin:{b}"))
    return pd.DataFrame(rows)


def eval_group(model, f_sc, t_sc, df, bounds, device, splits, cap_all, seed):
    rolled, cand, gen = C.evaluate_windows(
        model, f_sc, t_sc, df, bounds, device, splits, HORIZON,
        cap_all=cap_all, seed=seed, rollout_fn=D.rollout_one_trajectory_dir)
    all_rows = rolled[rolled["in_all_sample"]] if len(rolled) else rolled
    gen_rows = rolled[rolled["is_genuine_turn"]] if len(rolled) else rolled
    summ = pd.DataFrame([C.aggregate(all_rows, "all_windows"),
                         C.aggregate(gen_rows, "genuine_turns")])
    return rolled, cand, gen, summ


def load_prev_genuine(folder):
    p = folder / "metrics_all_vs_genuine_heldout.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    r = d[d.subset == "genuine_turns"]
    return r.iloc[0] if len(r) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--cap-all", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[load] {C.P5.MODEL_C_CSV}")
    df = C.P5.load_dataset()
    bounds = C.P5.load_world_bounds()
    df = D.add_heading_targets(df)   # adds future_heading_sin/cos
    n_rows, n_tracks = len(df), df["trajectory_id"].nunique()
    print(f"       {n_rows:,} rows · {n_tracks:,} tracks (+ heading targets)")

    models_dir = HERE / "models"
    ckpt = models_dir / f"best_model_{TAG}.pth"
    scal = models_dir / f"scalers_{TAG}.pkl"
    if args.skip_train and ckpt.exists() and scal.exists():
        print(f"[skip-train] loading {ckpt.name}")
        model = D.TrajectoryLSTMDir(len(C.FEAT_COLS)).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()
        f_sc, t_sc = D.load_dir_scalers(scal)
        cfg = json.loads((HERE / "training" / "config.json").read_text(encoding="utf-8"))
        counts_tr, counts_va = cfg.get("train_bin_counts", {}), cfg.get("val_bin_counts", {})
    else:
        model, f_sc, t_sc, cfg, counts_tr, counts_va = train(df, device, seed=args.seed)
    model.eval()

    print(f"\n[eval] held-out {EVAL_SPLITS} (4-output rollout, horizon={HORIZON}) ...")
    rolled, cand, gen, hsum = eval_group(model, f_sc, t_sc, df, bounds, device, EVAL_SPLITS, args.cap_all, args.seed)
    print(f"[eval] capacity {CAPACITY_SPLIT} ...")
    rolled_tr, cand_tr, gen_tr, tsum = eval_group(model, f_sc, t_sc, df, bounds, device, CAPACITY_SPLIT, args.cap_all, args.seed)

    eval_bins = {b: int((cand["bin"] == b).sum()) for b in C.BIN_ORDER} if len(cand) else {}
    if len(rolled):
        rolled.assign(group="held_out").to_csv(HERE / "metrics.csv", index=False)
    pb, pb_tr = per_bin_table(rolled), per_bin_table(rolled_tr)
    hsum.to_csv(HERE / "metrics_all_vs_genuine_heldout.csv", index=False)
    pb.to_csv(HERE / "metrics_per_bin_heldout.csv", index=False)
    tsum.to_csv(HERE / "metrics_all_vs_genuine_train.csv", index=False)
    pb_tr.to_csv(HERE / "metrics_per_bin_train.csv", index=False)
    if len(rolled_tr):
        rolled_tr.assign(group="train_capacity").to_csv(HERE / "metrics_train_capacity.csv", index=False)

    plot_rolls = gen if len(gen) >= 6 else gen + gen_tr
    plot_src = "held-out (val+test)" if len(gen) >= 6 else "val+test+train (held-out had <6 genuine turns)"
    nb, nw = V.save_best_worst(plot_rolls, PLOTS, k=6)
    V.save_collage(plot_rolls, PLOTS / "collage_9_genuine_turns.png", k=9)
    V.heading_change_comparison(rolled if len(rolled) else rolled_tr, PLOTS / "heading_change_comparison.png")
    V.final_heading_scatter(plot_rolls, PLOTS / "final_heading_scatter.png")
    print(f"[plots] best={nb} worst={nw} from {plot_src}; collage + heading comparison + final-heading scatter saved")

    e1 = load_prev_genuine(EXP01)
    e2 = load_prev_genuine(EXP02)
    write_outputs(cfg, df, n_rows, n_tracks, counts_tr, counts_va, eval_bins,
                  hsum, pb, tsum, len(gen), len(gen_tr), e1, e2, plot_src)

    h_all = hsum[hsum.subset == "all_windows"].iloc[0]
    h_gen = hsum[hsum.subset == "genuine_turns"].iloc[0]
    print("\n" + "=" * 70)
    print("EXP04 — DIRECTION-OUTPUT VARIANT — SUMMARY")
    print("=" * 70)
    print(f"held-out ALL windows   (n={int(h_all.n):5d}): ADE {h_all.ade_mean:.3f}m  FDE {h_all.fde_mean:.3f}m  ang {h_all.angular_err_deg_mean:.1f}°")
    print(f"held-out GENUINE turns (n={int(h_gen.n):5d}): ADE {h_gen.ade_mean:.3f}m  FDE {h_gen.fde_mean:.3f}m  "
          f"ang {h_gen.angular_err_deg_mean:.1f}°  predΔhead {h_gen.pred_head_change_deg_mean:.0f}°  "
          f"angularity {h_gen.angularity_ratio_mean:.2f}  TCR {h_gen.turn_capture_rate:.2f}")
    if e1 is not None and e2 is not None:
        print(f"\n[held-out genuine, angular err]  exp01 {e1.angular_err_deg_mean:.1f}°  "
              f"exp02 {e2.angular_err_deg_mean:.1f}°  exp04 {h_gen.angular_err_deg_mean:.1f}°")
        print(f"[held-out genuine, TCR]          exp01 {e1.turn_capture_rate:.2f}  "
              f"exp02 {e2.turn_capture_rate:.2f}  exp04 {h_gen.turn_capture_rate:.2f}")
    print(f"\nplots: {PLOTS}")


def _fmt_bins(d):
    return ", ".join(f"{b}={d.get(b, 0)}" for b in C.BIN_ORDER)


def _row(s, subset):
    r = s[s.subset == subset]
    return r.iloc[0] if len(r) else None


def write_outputs(cfg, df, n_rows, n_tracks, counts_tr, counts_va, eval_bins,
                  hsum, pb, tsum, n_gen, n_gen_tr, e1, e2, plot_src):
    h_all, h_gen = _row(hsum, "all_windows"), _row(hsum, "genuine_turns")
    t_gen = _row(tsum, "genuine_turns")

    # verdict logic vs exp02 (the closest comparator: turn-only, no heading head)
    base = e2 if e2 is not None else e1
    if base is not None and h_gen is not None:
        d_ang = h_gen.angular_err_deg_mean - base.angular_err_deg_mean
        d_angy = h_gen.angularity_ratio_mean - base.angularity_ratio_mean
        d_tcr = h_gen.turn_capture_rate - base.turn_capture_rate
        d_ade = h_gen.ade_mean - base.ade_mean
        q1 = (f"Angular error {base.angular_err_deg_mean:.1f}° (exp02) → {h_gen.angular_err_deg_mean:.1f}° "
              f"(exp04), {d_ang:+.1f}°. {'YES — reduced.' if d_ang < -3 else ('No — essentially unchanged (still ≈chance ~90°).' if abs(d_ang) <= 3 else 'No — it got worse.')}")
        q2 = (f"Angularity {base.angularity_ratio_mean:.2f} → {h_gen.angularity_ratio_mean:.2f} ({d_angy:+.2f}); "
              f"predicted Δheading {base.pred_head_change_deg_mean:.0f}° → {h_gen.pred_head_change_deg_mean:.0f}°. "
              f"{'Preserved/increased.' if d_angy >= -0.05 else 'Decreased.'}")
        q3 = (f"TCR {base.turn_capture_rate:.2f} → {h_gen.turn_capture_rate:.2f} ({d_tcr:+.2f}). "
              f"{'Improved.' if d_tcr > 0.03 else ('~unchanged.' if abs(d_tcr) <= 0.03 else 'Decreased.')}")
        q4 = (f"Genuine-turn ADE {base.ade_mean:.3f} → {h_gen.ade_mean:.3f} m ({d_ade:+.3f}); "
              f"FDE {base.fde_mean:.3f} → {h_gen.fde_mean:.3f} m. "
              f"{'No meaningful damage.' if d_ade <= 0.05 else 'Some ADE cost.'}")
        improves_dir = d_ang < -5
    else:
        q1 = q2 = q3 = q4 = "(prior-experiment comparison unavailable)"
        improves_dir = h_gen is not None and h_gen.angular_err_deg_mean < 75

    if improves_dir:
        q5 = ("Explicit heading output DID improve turn direction — the displacement-only head "
              "was a real part of the bottleneck; pursue the 4-output head + scale turn data.")
    else:
        q5 = ("Explicit heading output improves turn MAGNITUDE/expression but NOT direction "
              "(angular error stays ≈chance ~90°). The remaining issue is therefore **not** the "
              "output representation but **insufficient directional context / ambiguity in the "
              "input window**: a 10-frame seed of mostly-straight approach motion does not "
              "determine which way the pedestrian will turn at a decision point. Likely fixes are "
              "longer/again-richer context, decision-point-aware spatial features (which way is "
              "open), or multi-modal (mixture) outputs rather than a single mean heading.")

    R = [
        "# exp04 — Direction-output variant of Model C\n",
        "## Purpose\n",
        "Test whether the turn-direction bottleneck (exp01/exp02 produced turn magnitude but "
        "~90° angular error) is the **displacement-only output head**. The head is widened "
        "2→4 to also predict `future_heading_sin/cos`; loss = position_mse + 0.2·heading_mse; "
        "trained turn-only. At rollout the explicit predicted heading drives the heading input "
        "channels.\n",
        "## Dataset, targets & filter\n",
        f"`{C.P5.MODEL_C_CSV.name}` — {n_rows:,} rows · {n_tracks:,} tracks. Targets = "
        "`target_du, target_dv, future_heading_sin, future_heading_cos` "
        "(heading target = unit next-step displacement). Trained only on genuine-turn windows "
        "(disp>2m, head>30°, max-step<0.6m), genuine-turn validation for early stopping.\n",
        "## Windows & turn-bin counts\n",
        f"- Train (turn-only): {sum(counts_tr.values()):,} — {_fmt_bins(counts_tr)}",
        f"- Val (turn-only, early-stop): {sum(counts_va.values()):,} — {_fmt_bins(counts_va)}",
        f"- Held-out full-horizon eval windows: {_fmt_bins(eval_bins)} (genuine rolled: {n_gen}; train genuine rolled: {n_gen_tr})\n",
        "## Training command\n", "```\npython run_exp04.py\n```\n",
        f"Loss `position_mse + {HEAD_LOSS_W}*heading_mse`. Best epoch {cfg['best_epoch']}, "
        f"val total {cfg['best_val_loss']:.5f}, {cfg['actual_epochs']} epochs, {cfg['train_time_s']}s, seed {cfg['seed']}.\n",
        "## Main results (held-out val+test)\n",
    ]
    if h_all is not None and h_gen is not None:
        R += [
            f"- All windows (n={int(h_all.n)}): ADE {h_all.ade_mean:.3f} m, FDE {h_all.fde_mean:.3f} m, angular {h_all.angular_err_deg_mean:.1f}°.",
            f"- Genuine turns (n={int(h_gen.n)}): ADE {h_gen.ade_mean:.3f} m, FDE {h_gen.fde_mean:.3f} m, "
            f"angular {h_gen.angular_err_deg_mean:.1f}°, predicted Δheading {h_gen.pred_head_change_deg_mean:.0f}° "
            f"(GT {h_gen.gt_head_change_deg_mean:.0f}°), angularity {h_gen.angularity_ratio_mean:.2f}, **TCR {h_gen.turn_capture_rate:.2f}**.",
        ]
    if t_gen is not None:
        R.append(f"- Train capacity (n={int(t_gen.n)} genuine): ADE {t_gen.ade_mean:.3f} m, angular {t_gen.angular_err_deg_mean:.1f}°, TCR {t_gen.turn_capture_rate:.2f}.")

    # three-way comparison table
    if h_gen is not None and e1 is not None and e2 is not None:
        R += ["\n## Three-way comparison (held-out genuine turns)\n",
              "| metric | exp01 balanced | exp02 turn-only | exp04 dir-output |",
              "|---|---|---|---|",
              f"| ADE (m) | {e1.ade_mean:.3f} | {e2.ade_mean:.3f} | {h_gen.ade_mean:.3f} |",
              f"| FDE (m) | {e1.fde_mean:.3f} | {e2.fde_mean:.3f} | {h_gen.fde_mean:.3f} |",
              f"| angular err (°) | {e1.angular_err_deg_mean:.1f} | {e2.angular_err_deg_mean:.1f} | {h_gen.angular_err_deg_mean:.1f} |",
              f"| pred Δhead (°) | {e1.pred_head_change_deg_mean:.0f} | {e2.pred_head_change_deg_mean:.0f} | {h_gen.pred_head_change_deg_mean:.0f} |",
              f"| angularity | {e1.angularity_ratio_mean:.2f} | {e2.angularity_ratio_mean:.2f} | {h_gen.angularity_ratio_mean:.2f} |",
              f"| TCR | {e1.turn_capture_rate:.2f} | {e2.turn_capture_rate:.2f} | {h_gen.turn_capture_rate:.2f} |"]

    R += [
        "\n## Verdict\n",
        "**1. Does explicit heading output reduce angular error?**", q1, "",
        "**2. Does it preserve/increase angularity?**", q2, "",
        "**3. Does it improve Turn Capture Rate?**", q3, "",
        "**4. Does it damage ADE/FDE?**", q4, "",
        "**5. Magnitude vs direction — where does this leave us?**", q5, "",
        "## Plots\n",
        f"`plots/best_6_genuine_turns/` + `.png`, `plots/worst_6_genuine_turns/` + `.png`, "
        f"`plots/collage_9_genuine_turns.png`, `plots/heading_change_comparison.png`, "
        f"`plots/final_heading_scatter.png` (GT vs predicted final direction; from {plot_src}).\n",
        "## Files\n",
        "`metrics.csv`, `metrics_all_vs_genuine_heldout.csv`, `metrics_per_bin_heldout.csv`, "
        "`metrics_train_capacity.csv` (+ train variants), `metrics_summary.md`, "
        "`training/` (config.json, epoch_log.csv, training_loss.csv, validation_loss.csv), "
        "`models/best_model_exp04.pth`, `models/scalers_exp04.pkl`.\n",
    ]
    (HERE / "README.md").write_text("\n".join(R), encoding="utf-8")
    print(f"[saved] {HERE / 'README.md'}")

    S = ["# exp04 — Direction-output variant — metrics summary\n",
         f"4-output head, loss position_mse + {HEAD_LOSS_W}*heading_mse, turn-only training. "
         f"Best epoch {cfg['best_epoch']} (val {cfg['best_val_loss']:.5f}), {cfg['train_time_s']}s.\n",
         "## Held-out: all windows vs genuine turns\n",
         "| subset | n | ADE (m) | FDE (m) | ang err (°) | GT Δhead (°) | pred Δhead (°) | angularity | TCR |",
         "|---|---|---|---|---|---|---|---|---|"]
    for subset in ("all_windows", "genuine_turns"):
        r = _row(hsum, subset)
        if r is None:
            continue
        S.append(f"| {subset} | {int(r.n)} | {r.ade_mean:.3f} | {r.fde_mean:.3f} | {r.angular_err_deg_mean:.1f} | "
                 f"{r.gt_head_change_deg_mean:.1f} | {r.pred_head_change_deg_mean:.1f} | {r.angularity_ratio_mean:.2f} | {r.turn_capture_rate:.2f} |")
    S += ["\n## Held-out: per turn bin\n",
          "| bin | n | ADE (m) | FDE (m) | ang err (°) | GT Δhead (°) | pred Δhead (°) | angularity | TCR |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in pb.iterrows():
        S.append(f"| {r.subset.replace('bin:', '')} | {int(r.n)} | {r.ade_mean:.3f} | {r.fde_mean:.3f} | {r.angular_err_deg_mean:.1f} | "
                 f"{r.gt_head_change_deg_mean:.1f} | {r.pred_head_change_deg_mean:.1f} | {r.angularity_ratio_mean:.2f} | {r.turn_capture_rate:.2f} |")
    (HERE / "metrics_summary.md").write_text("\n".join(S), encoding="utf-8")
    print(f"[saved] {HERE / 'metrics_summary.md'}")


if __name__ == "__main__":
    main()
