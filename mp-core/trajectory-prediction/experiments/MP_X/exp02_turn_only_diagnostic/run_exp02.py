"""
MP_X / exp02 — Turn-only diagnostic Model C.

Trains the frozen Model C architecture on GENUINE-TURN windows ONLY (GT net
displacement > 2 m, GT heading change > 30°, max single-step < 0.6 m), with NO
WeightedRandomSampler (every training window is already a turn). Early stopping
tracks genuine-turn VALIDATION windows (not the ~99.96%-straight full val set) so
the checkpoint is not pulled back toward straight-line collapse.

Question: when straight-motion dilution is removed entirely, can the
displacement-only Model C learn turn DIRECTION — or does it still mis-direct,
implicating the displacement-only output head / turn diversity as the next
bottleneck?

Run:
    python run_exp02.py                 # train + evaluate + plot
    python run_exp02.py --skip-train    # reuse models/best_model_exp02.pth

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
EXP01 = HERE.parent / "exp01_turn_balanced_model_c"
sys.path.insert(0, str(SHARED))

import mpx_common as C       # noqa: E402
import mpx_train as T        # noqa: E402
import mpx_plots as V        # noqa: E402

TAG = "exp02"
HORIZON = C.N_ROLLOUT
EVAL_SPLITS = ["val", "test"]
CAPACITY_SPLIT = ["train"]

PLOTS = HERE / "plots"
SUMMARY_MD = HERE / "metrics_summary.md"


def genuine_mask(meta: pd.DataFrame):
    """Boolean mask: keep only genuine-turn windows."""
    return meta["is_genuine_turn"].to_numpy(dtype=bool)


def per_bin_table(rolled: pd.DataFrame) -> pd.DataFrame:
    if rolled is None or len(rolled) == 0:
        return pd.DataFrame([C.aggregate(None, f"bin:{b}") for b in C.BIN_ORDER])
    rows = []
    for b in C.BIN_ORDER:
        if b == "straight":
            sub = rolled[(rolled["bin"] == b) & (rolled["in_all_sample"])]
        else:
            sub = rolled[(rolled["bin"] == b) & (rolled["is_genuine_turn"])]
        rows.append(C.aggregate(sub, f"bin:{b}"))
    return pd.DataFrame(rows)


def evaluate_split_group(model, f_sc, t_sc, df, bounds, device, splits, cap_all, seed):
    rolled, cand, genuine_rolls = C.evaluate_windows(
        model, f_sc, t_sc, df, bounds, device, splits, HORIZON, cap_all=cap_all, seed=seed)
    all_rows = rolled[rolled["in_all_sample"]] if len(rolled) else rolled
    gen_rows = rolled[rolled["is_genuine_turn"]] if len(rolled) else rolled
    summary = pd.DataFrame([C.aggregate(all_rows, "all_windows"),
                            C.aggregate(gen_rows, "genuine_turns")])
    return rolled, cand, genuine_rolls, summary


def load_exp01_genuine():
    """Read exp01's held-out genuine-turn row for a direct comparison (best-effort)."""
    p = EXP01 / "metrics_all_vs_genuine_heldout.csv"
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
    n_rows, n_tracks = len(df), df["trajectory_id"].nunique()
    print(f"       {n_rows:,} rows · {n_tracks:,} tracks")

    models_dir = HERE / "models"
    ckpt = models_dir / f"best_model_{TAG}.pth"
    scal = models_dir / f"scalers_{TAG}.pkl"

    # ── train turn-only (or reuse) ───────────────────────────────────────────
    if args.skip_train and ckpt.exists() and scal.exists():
        print(f"[skip-train] loading {ckpt.name}")
        model = C.fz.TrajectoryLSTM(len(C.FEAT_COLS)).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        f_sc, t_sc = C.P5.load_scalers(scal)
        cfg = json.loads((HERE / "training" / "config.json").read_text(encoding="utf-8"))
        train_bins, val_bins = cfg.get("train_bin_counts", {}), cfg.get("val_bin_counts", {})
    else:
        out = T.train_balanced_model_c(
            df, HERE, device, horizon=HORIZON, use_sampler=False, seed=args.seed,
            tag=TAG, train_window_filter=genuine_mask, val_window_filter=genuine_mask)
        model, f_sc, t_sc = out["model"], out["f_sc"], out["t_sc"]
        cfg = out["config"]
        train_bins, val_bins = out["train_bin_counts"], out["val_bin_counts"]
    model.eval()

    # ── evaluation ───────────────────────────────────────────────────────────
    print(f"\n[eval] held-out {EVAL_SPLITS} (sliding windows, horizon={HORIZON}) ...")
    rolled, cand, genuine_rolls, hsum = evaluate_split_group(
        model, f_sc, t_sc, df, bounds, device, EVAL_SPLITS, args.cap_all, args.seed)
    print(f"[eval] capacity {CAPACITY_SPLIT} (reference) ...")
    rolled_tr, cand_tr, genuine_rolls_tr, tsum = evaluate_split_group(
        model, f_sc, t_sc, df, bounds, device, CAPACITY_SPLIT, args.cap_all, args.seed)

    eval_bins = {b: int((cand["bin"] == b).sum()) for b in C.BIN_ORDER} if len(cand) else {}

    # ── persist metrics ──────────────────────────────────────────────────────
    if len(rolled):
        rolled.assign(group="held_out").to_csv(HERE / "metrics.csv", index=False)
    pb, pb_tr = per_bin_table(rolled), per_bin_table(rolled_tr)
    hsum.to_csv(HERE / "metrics_all_vs_genuine_heldout.csv", index=False)
    pb.to_csv(HERE / "metrics_per_bin_heldout.csv", index=False)
    tsum.to_csv(HERE / "metrics_all_vs_genuine_train.csv", index=False)
    pb_tr.to_csv(HERE / "metrics_per_bin_train.csv", index=False)
    if len(rolled_tr):
        rolled_tr.assign(group="train_capacity").to_csv(HERE / "metrics_train_capacity.csv", index=False)

    # ── plots ────────────────────────────────────────────────────────────────
    plot_rolls = genuine_rolls
    plot_src = "held-out (val+test)"
    if len(plot_rolls) < 6:
        plot_rolls = genuine_rolls + genuine_rolls_tr
        plot_src = "val+test+train (held-out had <6 genuine turns)"
    nb, nw = V.save_best_worst(plot_rolls, PLOTS, k=6)
    V.save_collage(plot_rolls, PLOTS / "collage_9_genuine_turns.png", k=9)
    V.heading_change_comparison(rolled if len(rolled) else rolled_tr,
                                PLOTS / "heading_change_comparison.png")
    print(f"[plots] best={nb} worst={nw} from {plot_src}; collage + heading comparison saved")

    # ── verdict + reports ────────────────────────────────────────────────────
    exp01_gen = load_exp01_genuine()
    write_outputs(cfg, df, n_rows, n_tracks, train_bins, val_bins, eval_bins,
                  hsum, pb, tsum, len(genuine_rolls), len(genuine_rolls_tr),
                  exp01_gen, plot_src)

    # ── console summary ──────────────────────────────────────────────────────
    h_all = hsum[hsum.subset == "all_windows"].iloc[0]
    h_gen = hsum[hsum.subset == "genuine_turns"].iloc[0]
    t_gen = tsum[tsum.subset == "genuine_turns"].iloc[0]
    print("\n" + "=" * 70)
    print("EXP02 — TURN-ONLY DIAGNOSTIC — SUMMARY")
    print("=" * 70)
    print(f"trained on {cfg['n_train_windows']:,} genuine-turn windows (no sampler); "
          f"best epoch {cfg['best_epoch']}, {cfg['train_time_s']}s")
    print(f"held-out ALL windows   (n={int(h_all.n):5d}): ADE {h_all.ade_mean:.3f}m  "
          f"FDE {h_all.fde_mean:.3f}m  ang {h_all.angular_err_deg_mean:.1f}°")
    print(f"held-out GENUINE turns (n={int(h_gen.n):5d}): ADE {h_gen.ade_mean:.3f}m  "
          f"FDE {h_gen.fde_mean:.3f}m  ang {h_gen.angular_err_deg_mean:.1f}°  "
          f"pred Δhead {h_gen.pred_head_change_deg_mean:.0f}°  angularity {h_gen.angularity_ratio_mean:.2f}  "
          f"TCR {h_gen.turn_capture_rate:.2f}")
    print(f"train    GENUINE turns (n={int(t_gen.n):5d}): ADE {t_gen.ade_mean:.3f}m  "
          f"ang {t_gen.angular_err_deg_mean:.1f}°  TCR {t_gen.turn_capture_rate:.2f}  (capacity ref)")
    if exp01_gen is not None:
        print(f"\n[vs exp01 held-out genuine] ADE {exp01_gen.ade_mean:.3f}->{h_gen.ade_mean:.3f}  "
              f"ang {exp01_gen.angular_err_deg_mean:.1f}->{h_gen.angular_err_deg_mean:.1f}  "
              f"predΔhead {exp01_gen.pred_head_change_deg_mean:.0f}->{h_gen.pred_head_change_deg_mean:.0f}  "
              f"TCR {exp01_gen.turn_capture_rate:.2f}->{h_gen.turn_capture_rate:.2f}")
    print(f"\nplots: {PLOTS}")


# ─────────────────────────────────────────────────────────────────────────────
def _fmt_bins(d):
    return ", ".join(f"{b}={d.get(b, 0)}" for b in C.BIN_ORDER)


def _row(df_sum, subset):
    r = df_sum[df_sum.subset == subset]
    return r.iloc[0] if len(r) else None


def _delta(new, old):
    if old is None or np.isnan(old):
        return "n/a"
    d = new - old
    return f"{d:+.3f}" if abs(d) < 10 else f"{d:+.1f}"


def write_outputs(cfg, df, n_rows, n_tracks, train_bins, val_bins, eval_bins,
                  hsum, pb, tsum, n_gen_heldout, n_gen_train, exp01_gen, plot_src):
    h_all, h_gen = _row(hsum, "all_windows"), _row(hsum, "genuine_turns")
    t_gen = _row(tsum, "genuine_turns")

    # answers to the four required verdict questions (held-out genuine vs exp01)
    if exp01_gen is not None and h_gen is not None:
        q1_dir = h_gen.pred_head_change_deg_mean - exp01_gen.pred_head_change_deg_mean
        q1 = (f"Predicted heading change {exp01_gen.pred_head_change_deg_mean:.0f}° (exp01) → "
              f"{h_gen.pred_head_change_deg_mean:.0f}° (exp02), {'INCREASED' if q1_dir > 1 else ('~unchanged' if abs(q1_dir) <= 1 else 'DECREASED')} "
              f"by {q1_dir:+.0f}°; TCR {exp01_gen.turn_capture_rate:.2f}→{h_gen.turn_capture_rate:.2f}.")
        q2_dir = h_gen.angular_err_deg_mean - exp01_gen.angular_err_deg_mean
        q2 = (f"Angular error {exp01_gen.angular_err_deg_mean:.1f}° → {h_gen.angular_err_deg_mean:.1f}° "
              f"({q2_dir:+.1f}°); {'IMPROVED' if q2_dir < -3 else ('~unchanged (still ~chance ~90°)' if abs(q2_dir) <= 3 else 'WORSE')}.")
        q3_ade = h_gen.ade_mean - exp01_gen.ade_mean
        q3 = (f"Genuine-turn ADE {exp01_gen.ade_mean:.3f} → {h_gen.ade_mean:.3f} m ({q3_ade:+.3f}); "
              f"FDE {exp01_gen.fde_mean:.3f} → {h_gen.fde_mean:.3f} m. "
              f"{'REDUCED' if q3_ade < -0.05 else ('~unchanged' if abs(q3_ade) <= 0.05 else 'INCREASED')}.")
        angle_appeared = (h_gen.pred_head_change_deg_mean > 20) or (h_gen.turn_capture_rate > 0.3)
        dir_wrong = h_gen.angular_err_deg_mean > 75
    else:
        q1 = q2 = q3 = "(exp01 comparison unavailable)"
        angle_appeared = h_gen is not None and h_gen.pred_head_change_deg_mean > 20
        dir_wrong = h_gen is not None and h_gen.angular_err_deg_mean > 75

    if angle_appeared and dir_wrong:
        q4 = ("Angularity appears but turn DIRECTION stays wrong (≈chance ~90°). Removing "
              "straight dilution did not teach direction, so the next bottleneck is most likely "
              "the **displacement-only output head** (no explicit heading target to commit to) "
              "and/or **insufficient turn diversity** (few, repetitive turn geometries). → motivates "
              "exp04 (explicit heading_sin/cos output + heading loss).")
    elif angle_appeared and not dir_wrong:
        q4 = ("Turn-only training BOTH produces angularity AND improves direction — straight "
              "dilution was the dominant cause; scaling balanced turn data is the path forward.")
    else:
        q4 = ("Even with zero straight dilution the model under-turns — the limit is turn data "
              "volume/diversity, not just balance; prioritise turn-rich data collection.")

    # ---- README ----
    R = [
        "# exp02 — Turn-only diagnostic Model C\n",
        "## Purpose\n",
        "Train Model C on **genuine-turn windows only** (no straight dilution, no sampler) to "
        "test whether the displacement-only architecture can learn turn DIRECTION when it sees "
        "nothing but turns. Early stopping tracks genuine-turn validation windows so the "
        "checkpoint is not pulled back to straight collapse.\n",
        "## Dataset & filter\n",
        f"`{C.P5.MODEL_C_CSV.name}` — {n_rows:,} rows · {n_tracks:,} tracks. Training windows kept "
        "iff GT net displacement > 2 m AND GT heading change > 30° AND max single-step < 0.6 m.\n",
        "## Windows & turn-bin counts\n",
        f"- Train (genuine-turn only): {sum(train_bins.values()):,} windows — {_fmt_bins(train_bins)}",
        f"- Val (genuine-turn only, early-stop signal): {sum(val_bins.values()):,} — {_fmt_bins(val_bins)}",
        f"- Held-out full-horizon eval windows: {_fmt_bins(eval_bins)} "
        f"(genuine rolled: {n_gen_heldout}; train genuine rolled: {n_gen_train})\n",
        "## Training command\n",
        "```\npython run_exp02.py\n```\n",
        f"No WeightedRandomSampler. Best epoch {cfg['best_epoch']}, val MSE {cfg['best_val_loss']:.5f}, "
        f"{cfg['actual_epochs']} epochs, {cfg['train_time_s']}s, seed {cfg['seed']}.\n",
        "## Main results (held-out val+test)\n",
    ]
    if h_all is not None and h_gen is not None:
        R += [
            f"- All windows (n={int(h_all.n)}): ADE {h_all.ade_mean:.3f} m, FDE {h_all.fde_mean:.3f} m, "
            f"angular {h_all.angular_err_deg_mean:.1f}°.",
            f"- Genuine turns (n={int(h_gen.n)}): ADE {h_gen.ade_mean:.3f} m, FDE {h_gen.fde_mean:.3f} m, "
            f"angular {h_gen.angular_err_deg_mean:.1f}°, predicted Δheading {h_gen.pred_head_change_deg_mean:.0f}° "
            f"(GT {h_gen.gt_head_change_deg_mean:.0f}°), angularity {h_gen.angularity_ratio_mean:.2f}, "
            f"**TCR {h_gen.turn_capture_rate:.2f}**.",
        ]
    if t_gen is not None:
        R.append(f"- Train capacity (n={int(t_gen.n)} genuine): ADE {t_gen.ade_mean:.3f} m, "
                 f"angular {t_gen.angular_err_deg_mean:.1f}°, TCR {t_gen.turn_capture_rate:.2f}.")
    if exp01_gen is not None and h_gen is not None:
        R += ["\n## exp02 vs exp01 (held-out genuine turns)\n",
              "| metric | exp01 balanced | exp02 turn-only | Δ |", "|---|---|---|---|",
              f"| ADE (m) | {exp01_gen.ade_mean:.3f} | {h_gen.ade_mean:.3f} | {_delta(h_gen.ade_mean, exp01_gen.ade_mean)} |",
              f"| FDE (m) | {exp01_gen.fde_mean:.3f} | {h_gen.fde_mean:.3f} | {_delta(h_gen.fde_mean, exp01_gen.fde_mean)} |",
              f"| angular err (°) | {exp01_gen.angular_err_deg_mean:.1f} | {h_gen.angular_err_deg_mean:.1f} | {_delta(h_gen.angular_err_deg_mean, exp01_gen.angular_err_deg_mean)} |",
              f"| pred Δhead (°) | {exp01_gen.pred_head_change_deg_mean:.0f} | {h_gen.pred_head_change_deg_mean:.0f} | {_delta(h_gen.pred_head_change_deg_mean, exp01_gen.pred_head_change_deg_mean)} |",
              f"| angularity | {exp01_gen.angularity_ratio_mean:.2f} | {h_gen.angularity_ratio_mean:.2f} | {_delta(h_gen.angularity_ratio_mean, exp01_gen.angularity_ratio_mean)} |",
              f"| TCR | {exp01_gen.turn_capture_rate:.2f} | {h_gen.turn_capture_rate:.2f} | {_delta(h_gen.turn_capture_rate, exp01_gen.turn_capture_rate)} |"]
    R += [
        "\n## Verdict\n",
        "**1. Does turn-only training increase predicted heading change?**", q1, "",
        "**2. Does it improve angular direction accuracy?**", q2, "",
        "**3. Does it reduce ADE/FDE on genuine turns?**", q3, "",
        "**4. Where is the next bottleneck?**", q4, "",
        "## Plots\n",
        f"`plots/best_6_genuine_turns/` + `.png`, `plots/worst_6_genuine_turns/` + `.png`, "
        f"`plots/collage_9_genuine_turns.png`, `plots/heading_change_comparison.png` (from {plot_src}).\n",
        "## Files\n",
        "`metrics.csv`, `metrics_all_vs_genuine_heldout.csv`, `metrics_per_bin_heldout.csv`, "
        "`metrics_train_capacity.csv` (+ train variants), `metrics_summary.md`, "
        "`training/` (config.json, epoch_log.csv, training_loss.csv, validation_loss.csv), "
        "`models/best_model_exp02.pth`, `models/scalers_exp02.pkl`.\n",
    ]
    (HERE / "README.md").write_text("\n".join(R), encoding="utf-8")
    print(f"[saved] {HERE / 'README.md'}")

    # ---- metrics_summary.md ----
    S = ["# exp02 — Turn-only diagnostic — metrics summary\n",
         f"Trained on {cfg['n_train_windows']:,} genuine-turn windows (no sampler); "
         f"best epoch {cfg['best_epoch']} (val MSE {cfg['best_val_loss']:.5f}), {cfg['train_time_s']}s.\n",
         "## Held-out: all windows vs genuine turns\n",
         "| subset | n | ADE (m) | FDE (m) | ang err (°) | GT Δhead (°) | pred Δhead (°) | angularity | TCR |",
         "|---|---|---|---|---|---|---|---|---|"]
    for subset in ("all_windows", "genuine_turns"):
        r = _row(hsum, subset)
        if r is None:
            continue
        S.append(f"| {subset} | {int(r.n)} | {r.ade_mean:.3f} | {r.fde_mean:.3f} | "
                 f"{r.angular_err_deg_mean:.1f} | {r.gt_head_change_deg_mean:.1f} | "
                 f"{r.pred_head_change_deg_mean:.1f} | {r.angularity_ratio_mean:.2f} | {r.turn_capture_rate:.2f} |")
    S += ["\n## Held-out: per turn bin\n",
          "| bin | n | ADE (m) | FDE (m) | ang err (°) | GT Δhead (°) | pred Δhead (°) | angularity | TCR |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in pb.iterrows():
        S.append(f"| {r.subset.replace('bin:', '')} | {int(r.n)} | {r.ade_mean:.3f} | {r.fde_mean:.3f} | "
                 f"{r.angular_err_deg_mean:.1f} | {r.gt_head_change_deg_mean:.1f} | "
                 f"{r.pred_head_change_deg_mean:.1f} | {r.angularity_ratio_mean:.2f} | {r.turn_capture_rate:.2f} |")
    SUMMARY_MD.write_text("\n".join(S), encoding="utf-8")
    print(f"[saved] {SUMMARY_MD}")


if __name__ == "__main__":
    main()
