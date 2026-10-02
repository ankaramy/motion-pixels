"""
MP_X / exp01 — Turn-balanced Model C.

Tests the hypothesis that Model C's straight-line collapse is caused by turn
scarcity / class imbalance, NOT by a fundamental incapacity. We retrain the frozen
Model C architecture with a WeightedRandomSampler that oversamples turn windows
(straight 1 / mild 8 / sharp 12 / U-turn 16), then evaluate "all windows" and
"genuine-turn windows" separately with turn-specific metrics and trajectory plots.

Run:
    python run_exp01.py                 # train + evaluate + plot
    python run_exp01.py --skip-train    # reuse models/best_model_exp01.pth
    python run_exp01.py --cap-all 6000  # bigger "all windows" eval sample

Nothing here modifies frozen_model_C, the dataset, or any other experiment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "shared"
sys.path.insert(0, str(SHARED))

import mpx_common as C       # noqa: E402
import mpx_train as T        # noqa: E402
import mpx_plots as V        # noqa: E402

TAG = "exp01"
HORIZON = C.N_ROLLOUT        # 20, the frozen rollout horizon
EVAL_SPLITS = ["val", "test"]          # held-out (honest)
CAPACITY_SPLIT = ["train"]             # capacity reference (model saw these)

PLOTS = HERE / "plots"
METRICS_CSV = HERE / "metrics.csv"
SUMMARY_MD = HERE / "metrics_summary.md"


def per_bin_table(rolled: pd.DataFrame) -> pd.DataFrame:
    """One aggregate row per turn bin. 'straight' is drawn from the uniform
    all-windows sample; genuine bins use every rolled genuine-turn window."""
    rows = []
    if rolled is None or len(rolled) == 0:
        return pd.DataFrame([C.aggregate(None, f"bin:{b}") for b in C.BIN_ORDER])
    for b in C.BIN_ORDER:
        if b == "straight":
            sub = rolled[(rolled["bin"] == b) & (rolled["in_all_sample"])]
        else:
            sub = rolled[(rolled["bin"] == b) & (rolled["is_genuine_turn"])]
        agg = C.aggregate(sub, f"bin:{b}")
        rows.append(agg)
    return pd.DataFrame(rows)


def evaluate_split_group(model, f_sc, t_sc, df, bounds, device, splits, cap_all, seed):
    rolled, cand, genuine_rolls = C.evaluate_windows(
        model, f_sc, t_sc, df, bounds, device, splits, HORIZON,
        cap_all=cap_all, seed=seed)
    all_rows = rolled[rolled["in_all_sample"]] if len(rolled) else rolled
    gen_rows = rolled[rolled["is_genuine_turn"]] if len(rolled) else rolled
    summary = [
        C.aggregate(all_rows, "all_windows"),
        C.aggregate(gen_rows, "genuine_turns"),
    ]
    return rolled, cand, genuine_rolls, pd.DataFrame(summary)


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
    n_rows = len(df)
    n_tracks = df["trajectory_id"].nunique()
    print(f"       {n_rows:,} rows · {n_tracks:,} tracks")
    for s in ("train", "val", "test"):
        d = df[df["split"] == s]
        print(f"       {s:5s}: {len(d):,} rows · {d['trajectory_id'].nunique():,} tracks "
              f"· recs {sorted(d['recording_id'].unique())}")

    models_dir = HERE / "models"
    ckpt = models_dir / f"best_model_{TAG}.pth"
    scal = models_dir / f"scalers_{TAG}.pkl"

    # ── train (or reuse) ─────────────────────────────────────────────────────
    if args.skip_train and ckpt.exists() and scal.exists():
        print(f"[skip-train] loading {ckpt.name}")
        model = C.fz.TrajectoryLSTM(len(C.FEAT_COLS)).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        f_sc, t_sc = C.P5.load_scalers(scal)
        cfg = json.loads((HERE / "training" / "config.json").read_text(encoding="utf-8"))
        train_bins = cfg.get("train_bin_counts", {})
        val_bins = cfg.get("val_bin_counts", {})
    else:
        out = T.train_balanced_model_c(
            df, HERE, device, horizon=HORIZON, use_sampler=True,
            weights=C.BIN_WEIGHTS, seed=args.seed, tag=TAG)
        model, f_sc, t_sc = out["model"], out["f_sc"], out["t_sc"]
        cfg = out["config"]
        train_bins, val_bins = out["train_bin_counts"], out["val_bin_counts"]
    model.eval()

    # ── evaluation ───────────────────────────────────────────────────────────
    print(f"\n[eval] held-out splits {EVAL_SPLITS} (sliding windows, horizon={HORIZON}) ...")
    rolled, cand, genuine_rolls, hsum = evaluate_split_group(
        model, f_sc, t_sc, df, bounds, device, EVAL_SPLITS, args.cap_all, args.seed)

    print(f"[eval] capacity split {CAPACITY_SPLIT} (reference; model trained on these) ...")
    rolled_tr, cand_tr, genuine_rolls_tr, tsum = evaluate_split_group(
        model, f_sc, t_sc, df, bounds, device, CAPACITY_SPLIT, args.cap_all, args.seed)

    # full-horizon window bin counts (every candidate, rolled or not)
    eval_bin_counts = {b: int((cand["bin"] == b).sum()) for b in C.BIN_ORDER} if len(cand) else {}
    eval_gen_counts = {b: int((cand_tr["bin"] == b).sum()) for b in C.BIN_ORDER} if len(cand_tr) else {}

    # ── persist metrics ──────────────────────────────────────────────────────
    if len(rolled):
        rolled.assign(group="held_out").to_csv(METRICS_CSV, index=False)
    pb = per_bin_table(rolled)
    pb_tr = per_bin_table(rolled_tr)
    hsum.to_csv(HERE / "metrics_all_vs_genuine_heldout.csv", index=False)
    pb.to_csv(HERE / "metrics_per_bin_heldout.csv", index=False)
    tsum.to_csv(HERE / "metrics_all_vs_genuine_train.csv", index=False)
    pb_tr.to_csv(HERE / "metrics_per_bin_train.csv", index=False)
    if len(rolled_tr):
        rolled_tr.assign(group="train_capacity").to_csv(HERE / "metrics_train_capacity.csv", index=False)

    # ── plots (pool held-out genuine; fall back to train if too few) ─────────
    plot_rolls = genuine_rolls
    plot_src = "held-out (val+test)"
    if len(plot_rolls) < 6:
        print(f"[plots] only {len(plot_rolls)} held-out genuine turns -> adding train capacity turns")
        plot_rolls = genuine_rolls + genuine_rolls_tr
        plot_src = "val+test+train (held-out had <6 genuine turns)"
    nb, nw = V.save_best_worst(plot_rolls, PLOTS, k=6)
    V.save_collage(plot_rolls, PLOTS / "collage_9_genuine_turns.png", k=9)
    all_rolled = pd.concat([rolled, rolled_tr], ignore_index=True) if len(rolled_tr) else rolled
    V.heading_change_comparison(rolled if len(rolled) else all_rolled,
                                PLOTS / "heading_change_comparison.png")
    print(f"[plots] best={nb} worst={nw} genuine-turn plots from {plot_src}; collage + heading comparison saved")

    # ── markdown summary ─────────────────────────────────────────────────────
    write_summary(cfg, df, n_rows, n_tracks, train_bins, val_bins,
                  eval_bin_counts, eval_gen_counts, hsum, pb, tsum, pb_tr,
                  len(genuine_rolls), len(genuine_rolls_tr), plot_src)
    write_readme(cfg, df, train_bins, val_bins, eval_bin_counts, hsum, pb,
                 len(genuine_rolls), len(genuine_rolls_tr))

    # ── console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXP01 — TURN-BALANCED MODEL C — SUMMARY")
    print("=" * 70)
    h_all = hsum[hsum.subset == "all_windows"].iloc[0]
    h_gen = hsum[hsum.subset == "genuine_turns"].iloc[0]
    print(f"held-out ALL windows   (n={int(h_all.n):5d}): "
          f"ADE {h_all.ade_mean:.3f}m  FDE {h_all.fde_mean:.3f}m  ang {h_all.angular_err_deg_mean:.1f}°")
    print(f"held-out GENUINE turns (n={int(h_gen.n):5d}): "
          f"ADE {h_gen.ade_mean:.3f}m  FDE {h_gen.fde_mean:.3f}m  ang {h_gen.angular_err_deg_mean:.1f}°  "
          f"TCR {h_gen.turn_capture_rate:.2f}")
    t_gen = tsum[tsum.subset == "genuine_turns"].iloc[0]
    print(f"train    GENUINE turns (n={int(t_gen.n):5d}): "
          f"ADE {t_gen.ade_mean:.3f}m  ang {t_gen.angular_err_deg_mean:.1f}°  TCR {t_gen.turn_capture_rate:.2f}  (capacity ref)")
    print(f"\nplots: {PLOTS}")
    print("next: exp02 (turn-only diagnostic) if genuine-turn TCR is still low; "
          "exp03 (horizon sweep) if turns appear at short horizons; "
          "exp04 (direction output) if heading commitment is the bottleneck.")


def _fmt_bins(d):
    return ", ".join(f"{b}={d.get(b, 0)}" for b in C.BIN_ORDER)


def _sumrow(df_sum, subset):
    r = df_sum[df_sum.subset == subset]
    return r.iloc[0] if len(r) else None


def write_summary(cfg, df, n_rows, n_tracks, train_bins, val_bins,
                  eval_bins, eval_gen_bins, hsum, pb, tsum, pb_tr,
                  n_gen_heldout, n_gen_train, plot_src):
    L = []
    A = L.append
    A("# exp01 — Turn-balanced Model C — metrics summary\n")
    A(f"Dataset: `{C.P5.MODEL_C_CSV.name}` — {n_rows:,} rows · {n_tracks:,} tracks.\n")
    A("## Sampler\n")
    A(f"WeightedRandomSampler weights: {cfg['sampler_weights']}.  "
      f"Label horizon = {cfg['label_horizon']} steps.  "
      f"Best epoch {cfg['best_epoch']} (val MSE {cfg['best_val_loss']:.5f}), "
      f"{cfg['actual_epochs']} epochs in {cfg['train_time_s']}s.\n")
    A("## Training-window turn bins\n")
    A(f"- train windows ({sum(train_bins.values()):,}): {_fmt_bins(train_bins)}")
    A(f"- val windows ({sum(val_bins.values()):,}): {_fmt_bins(val_bins)}\n")
    A("## Held-out evaluation full-horizon window bins (val+test)\n")
    A(f"- {_fmt_bins(eval_bins)}  (total {sum(eval_bins.values()):,})")
    A(f"- genuine-turn windows rolled out: {n_gen_heldout}\n")
    A("## Held-out: all windows vs genuine turns\n")
    A("| subset | n | ADE (m) | FDE (m) | ang err (°) | GT Δhead (°) | pred Δhead (°) | "
      "GT path (m) | pred path (m) | angularity | TCR |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for subset in ("all_windows", "genuine_turns"):
        r = _sumrow(hsum, subset)
        if r is None:
            continue
        A(f"| {subset} | {int(r.n)} | {r.ade_mean:.3f} | {r.fde_mean:.3f} | "
          f"{r.angular_err_deg_mean:.1f} | {r.gt_head_change_deg_mean:.1f} | "
          f"{r.pred_head_change_deg_mean:.1f} | {r.gt_path_len_m_mean:.2f} | "
          f"{r.pred_path_len_m_mean:.2f} | {r.angularity_ratio_mean:.2f} | {r.turn_capture_rate:.2f} |")
    A("\n## Held-out: per turn bin\n")
    A("| bin | n | ADE (m) | FDE (m) | ang err (°) | GT Δhead (°) | pred Δhead (°) | angularity | TCR |")
    A("|---|---|---|---|---|---|---|---|---|")
    for _, r in pb.iterrows():
        A(f"| {r.subset.replace('bin:', '')} | {int(r.n)} | {r.ade_mean:.3f} | {r.fde_mean:.3f} | "
          f"{r.angular_err_deg_mean:.1f} | {r.gt_head_change_deg_mean:.1f} | "
          f"{r.pred_head_change_deg_mean:.1f} | {r.angularity_ratio_mean:.2f} | {r.turn_capture_rate:.2f} |")
    A("\n## Train (capacity reference — model trained on these)\n")
    A(f"genuine-turn windows: {n_gen_train}.  Full-horizon bins: {_fmt_bins(eval_gen_bins)}.\n")
    A("| subset | n | ADE (m) | ang err (°) | pred Δhead (°) | angularity | TCR |")
    A("|---|---|---|---|---|---|---|")
    for subset in ("all_windows", "genuine_turns"):
        r = _sumrow(tsum, subset)
        if r is None:
            continue
        A(f"| {subset} | {int(r.n)} | {r.ade_mean:.3f} | {r.angular_err_deg_mean:.1f} | "
          f"{r.pred_head_change_deg_mean:.1f} | {r.angularity_ratio_mean:.2f} | {r.turn_capture_rate:.2f} |")
    A(f"\nPlots in `plots/` (best/worst/collage genuine turns from {plot_src}; "
      "heading_change_comparison.png).\n")
    SUMMARY_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[saved] {SUMMARY_MD}")


def write_readme(cfg, df, train_bins, val_bins, eval_bins, hsum, pb,
                 n_gen_heldout, n_gen_train):
    h_all = _sumrow(hsum, "all_windows")
    h_gen = _sumrow(hsum, "genuine_turns")
    gen_pb = pb[pb.subset.isin([f"bin:{b}" for b in C.GENUINE_BINS])]
    improved = (h_gen is not None and not np.isnan(h_gen.turn_capture_rate)
                and h_gen.turn_capture_rate > 0.10
                and h_gen.angular_err_deg_mean < 80.0)
    verdict = ("**Visible angularity IMPROVED** on genuine turns vs the Phase-4 baseline "
               "(see metrics)." if improved else
               "**Visible angularity did NOT clearly improve** on held-out genuine turns "
               "(see metrics + plots); turn-balancing alone is insufficient at this data scale.")
    L = [
        "# exp01 — Turn-balanced Model C\n",
        "## Purpose\n",
        "Test whether Model C's straight-line collapse is caused by turn scarcity / "
        "class imbalance rather than a fundamental incapacity. Same frozen Model C "
        "architecture and recipe; the only change is a `WeightedRandomSampler` that "
        "oversamples turn windows.\n",
        "## Dataset\n",
        f"`{C.P5.MODEL_C_CSV}` — {len(df):,} rows · {df['trajectory_id'].nunique():,} tracks "
        "(Barcelona_v3_manual_master, recording-level split).\n",
        "## Windows & turn-bin counts\n",
        f"- Train windows ({sum(train_bins.values()):,}): {_fmt_bins(train_bins)}",
        f"- Val windows ({sum(val_bins.values()):,}): {_fmt_bins(val_bins)}",
        f"- Held-out full-horizon eval windows: {_fmt_bins(eval_bins)} "
        f"(genuine rolled: {n_gen_heldout}; train genuine rolled: {n_gen_train})\n",
        "## Sampler weights\n",
        f"`{cfg['sampler_weights']}` (straight / mild_30_90 / sharp_90_150 / uturn_150_180).\n",
        "## Training command\n",
        "```\npython run_exp01.py\n```\n",
        f"Best epoch {cfg['best_epoch']}, val MSE {cfg['best_val_loss']:.5f}, "
        f"{cfg['actual_epochs']} epochs, {cfg['train_time_s']}s, seed {cfg['seed']}.\n",
        "## Main results (held-out val+test)\n",
    ]
    if h_all is not None and h_gen is not None:
        L += [
            f"- All windows (n={int(h_all.n)}): ADE {h_all.ade_mean:.3f} m, FDE {h_all.fde_mean:.3f} m, "
            f"angular {h_all.angular_err_deg_mean:.1f}°.",
            f"- Genuine turns (n={int(h_gen.n)}): ADE {h_gen.ade_mean:.3f} m, FDE {h_gen.fde_mean:.3f} m, "
            f"angular {h_gen.angular_err_deg_mean:.1f}°, angularity ratio {h_gen.angularity_ratio_mean:.2f}, "
            f"**Turn Capture Rate {h_gen.turn_capture_rate:.2f}**.",
        ]
    if len(gen_pb):
        L.append("- Per genuine bin (held-out): " + "; ".join(
            f"{r.subset.replace('bin:', '')} n={int(r.n)} ang {r.angular_err_deg_mean:.0f}° TCR {r.turn_capture_rate:.2f}"
            for _, r in gen_pb.iterrows()) + ".")
    L += [
        "\n## Visible angularity improved?\n",
        verdict + "\n",
        "## Plots\n",
        "- `plots/best_6_genuine_turns/` and `best_6_genuine_turns.png`",
        "- `plots/worst_6_genuine_turns/` and `worst_6_genuine_turns.png`",
        "- `plots/collage_9_genuine_turns.png`",
        "- `plots/heading_change_comparison.png`\n",
        "## Files\n",
        "`metrics.csv` (per held-out window), `metrics_all_vs_genuine_heldout.csv`, "
        "`metrics_per_bin_heldout.csv`, `metrics_train_capacity.csv` (+ train variants), "
        "`metrics_summary.md`, `training/` (config.json, epoch_log.csv, training_loss.csv, "
        "validation_loss.csv), `models/best_model_exp01.pth`, `models/scalers_exp01.pkl`.\n",
    ]
    (HERE / "README.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[saved] {HERE / 'README.md'}")


if __name__ == "__main__":
    main()
