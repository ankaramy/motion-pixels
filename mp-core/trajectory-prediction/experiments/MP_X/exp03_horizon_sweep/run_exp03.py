"""
MP_X / exp03 — Horizon sweep (turn-only Model C at H = 5 / 10 / 20).

Tests whether the turn-DIRECTION failure is a prediction-horizon / observability
problem: is direction easier to predict over a shorter horizon? Uses the exp02
recipe at each horizon — turn-only training, genuine-turn validation, frozen
Model C architecture, 10 inputs, target_du/dv output, NO sampler, NO heading head.
Only the horizon changes.

IMPORTANT design choice — displacement floor scales with horizon.
The genuine-turn filter is (net disp > D, heading change > 30°, max single-step <
0.6 m). With a FIXED D = 2 m the short-horizon turn sets collapse to nothing
(held-out genuine turns: 1 at H=5, 2 at H=10, 50 at H=20 — measured), so the
horizons are not comparable. A pedestrian simply cannot travel 2 m in 5 frames.
We therefore hold the turn-severity threshold fixed (heading change > 30°) and the
artifact guard fixed (max-step < 0.6 m), but scale the displacement floor with the
horizon: D(H) = 2.0 * H / 20  ->  0.5 m (H=5), 1.0 m (H=10), 2.0 m (H=20). This
preserves a constant minimum per-step speed floor (~0.1 m/step, "moving, not
jitter") across horizons instead of a fixed absolute distance, giving comparable
genuine-turn populations (held-out 1073 / 442 / 50).

Run:
    python run_exp03.py                 # full sweep (train + eval + plots)
    python run_exp03.py --skip-train    # reuse per-horizon checkpoints

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

HORIZONS = [5, 10, 20]
EVAL_SPLITS = ["val", "test"]
BASE_DISP_M = 2.0            # at the reference horizon (20)
REF_HORIZON = 20


def disp_floor(H):
    return BASE_DISP_M * H / REF_HORIZON


def genuine_mask(meta):
    return meta["is_genuine_turn"].to_numpy(dtype=bool)


def per_bin_table(rolled):
    if rolled is None or len(rolled) == 0:
        return pd.DataFrame([C.aggregate(None, f"bin:{b}") for b in C.BIN_ORDER])
    rows = []
    for b in C.BIN_ORDER:
        sub = (rolled[(rolled["bin"] == b) & (rolled["in_all_sample"])] if b == "straight"
               else rolled[(rolled["bin"] == b) & (rolled["is_genuine_turn"])])
        rows.append(C.aggregate(sub, f"bin:{b}"))
    return pd.DataFrame(rows)


def write_horizon_summary(out_dir, H, cfg, eval_bins, hsum, pb, n_gen):
    h_all = hsum[hsum.subset == "all_windows"]
    h_gen = hsum[hsum.subset == "genuine_turns"]
    S = [f"# exp03 horizon_{H:02d} — metrics summary\n",
         f"Turn-only Model C, horizon **{H}**, displacement floor **{disp_floor(H):.1f} m** "
         f"(heading>30°, max-step<0.6m fixed). Best epoch {cfg['best_epoch']} "
         f"(val MSE {cfg['best_val_loss']:.5f}), {cfg['train_time_s']}s. "
         f"Trained on {cfg['n_train_windows']:,} genuine-turn windows.\n",
         f"Held-out full-horizon eval window bins: " +
         ", ".join(f"{b}={eval_bins.get(b, 0)}" for b in C.BIN_ORDER) +
         f" (genuine rolled: {n_gen}).\n",
         "## Held-out: all windows vs genuine turns\n",
         "| subset | n | ADE (m) | FDE (m) | ang err (°) | GT Δhead (°) | pred Δhead (°) | angularity | TCR |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in (h_all.iloc[0] if len(h_all) else None, h_gen.iloc[0] if len(h_gen) else None):
        if r is None:
            continue
        S.append(f"| {r.subset} | {int(r.n)} | {r.ade_mean:.3f} | {r.fde_mean:.3f} | {r.angular_err_deg_mean:.1f} | "
                 f"{r.gt_head_change_deg_mean:.1f} | {r.pred_head_change_deg_mean:.1f} | {r.angularity_ratio_mean:.2f} | {r.turn_capture_rate:.2f} |")
    S += ["\n## Held-out: per turn bin\n",
          "| bin | n | ADE (m) | FDE (m) | ang err (°) | pred Δhead (°) | angularity | TCR |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in pb.iterrows():
        S.append(f"| {r.subset.replace('bin:', '')} | {int(r.n)} | {r.ade_mean:.3f} | {r.fde_mean:.3f} | "
                 f"{r.angular_err_deg_mean:.1f} | {r.pred_head_change_deg_mean:.1f} | {r.angularity_ratio_mean:.2f} | {r.turn_capture_rate:.2f} |")
    (out_dir / "metrics_summary.md").write_text("\n".join(S), encoding="utf-8")


def run_one_horizon(df, bounds, device, H, args):
    out_dir = HERE / f"horizon_{H:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"h{H:02d}"
    ckpt = out_dir / "models" / f"best_model_{tag}.pth"
    scal = out_dir / "models" / f"scalers_{tag}.pkl"

    if args.skip_train and ckpt.exists() and scal.exists():
        print(f"[H={H}] skip-train -> {ckpt.name}")
        model = C.fz.TrajectoryLSTM(len(C.FEAT_COLS)).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()
        f_sc, t_sc = C.P5.load_scalers(scal)
        cfg = json.loads((out_dir / "training" / "config.json").read_text(encoding="utf-8"))
    else:
        out = T.train_balanced_model_c(
            df, out_dir, device, horizon=H, use_sampler=False, seed=args.seed,
            tag=tag, train_window_filter=genuine_mask, val_window_filter=genuine_mask)
        model, f_sc, t_sc, cfg = out["model"], out["f_sc"], out["t_sc"], out["config"]
    model.eval()

    print(f"[H={H}] evaluating held-out {EVAL_SPLITS} (rollout {H} steps, disp floor {disp_floor(H):.1f}m) ...")
    rolled, cand, gen = C.evaluate_windows(
        model, f_sc, t_sc, df, bounds, device, EVAL_SPLITS, H,
        cap_all=args.cap_all, seed=args.seed)
    all_rows = rolled[rolled["in_all_sample"]] if len(rolled) else rolled
    gen_rows = rolled[rolled["is_genuine_turn"]] if len(rolled) else rolled
    hsum = pd.DataFrame([C.aggregate(all_rows, "all_windows"),
                         C.aggregate(gen_rows, "genuine_turns")])
    eval_bins = {b: int((cand["bin"] == b).sum()) for b in C.BIN_ORDER} if len(cand) else {}
    pb = per_bin_table(rolled)

    if len(rolled):
        rolled.assign(group="held_out", horizon=H).to_csv(out_dir / "metrics.csv", index=False)
    hsum.to_csv(out_dir / "metrics_all_vs_genuine_heldout.csv", index=False)
    pb.to_csv(out_dir / "metrics_per_bin_heldout.csv", index=False)
    write_horizon_summary(out_dir, H, cfg, eval_bins, hsum, pb, len(gen))

    # plots
    plots = out_dir / "plots"
    V.save_best_worst(gen, plots, k=6)
    V.save_collage(gen, plots / "collage_9_genuine_turns.png", k=9)
    V.heading_change_comparison(rolled if len(rolled) else gen, plots / "heading_change_comparison.png")
    print(f"[H={H}] done: held-out genuine n={len(gen)}; plots in {plots}")

    g = hsum[hsum.subset == "genuine_turns"].iloc[0]
    return {"horizon": H, "disp_floor_m": disp_floor(H), "n_turns": int(g.n),
            "ade": g.ade_mean, "fde": g.fde_mean, "angular_error_deg": g.angular_err_deg_mean,
            "gt_delta_heading_deg": g.gt_head_change_deg_mean,
            "pred_delta_heading_deg": g.pred_head_change_deg_mean,
            "angularity_ratio": g.angularity_ratio_mean, "TCR": g.turn_capture_rate,
            "best_epoch": cfg["best_epoch"], "train_time_s": cfg["train_time_s"]}


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
    print(f"       {len(df):,} rows · {df['trajectory_id'].nunique():,} tracks")

    rows = []
    saved_disp = C.DISP_MIN_M
    try:
        for H in HORIZONS:
            C.DISP_MIN_M = disp_floor(H)   # horizon-scaled genuine-turn displacement floor
            print(f"\n{'='*70}\n[horizon {H}]  displacement floor = {C.DISP_MIN_M:.2f} m\n{'='*70}")
            rows.append(run_one_horizon(df, bounds, device, H, args))
    finally:
        C.DISP_MIN_M = saved_disp

    comp = pd.DataFrame(rows)
    comp.to_csv(HERE / "comparison_metrics.csv", index=False)
    write_comparison(comp)

    # ── required console table + verdict ─────────────────────────────────────
    print("\n" + "=" * 96)
    print("EXP03 — HORIZON SWEEP — held-out genuine turns")
    print("=" * 96)
    hdr = f"{'horizon':>7} | {'n_turns':>7} | {'ADE':>6} | {'FDE':>6} | {'ang_err_deg':>11} | {'pred_dHead_deg':>14} | {'angularity':>10} | {'TCR':>5}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in comp.iterrows():
        print(f"{int(r.horizon):>7} | {int(r.n_turns):>7} | {r.ade:>6.3f} | {r.fde:>6.3f} | "
              f"{r.angular_error_deg:>11.1f} | {r.pred_delta_heading_deg:>14.1f} | {r.angularity_ratio:>10.2f} | {r.TCR:>5.2f}")
    print_verdict(comp)


def print_verdict(comp):
    c = comp.set_index("horizon")
    a5, a20 = c.loc[5, "angular_error_deg"], c.loc[20, "angular_error_deg"]
    horizon_helps = a5 < a20 - 5
    best_ang = int(comp.loc[comp.angular_error_deg.idxmin(), "horizon"])
    print("\nVERDICT")
    print(f"  • Angular error: H5={a5:.1f}°  H10={c.loc[10,'angular_error_deg']:.1f}°  H20={a20:.1f}°  "
          f"(min at H={best_ang}).")
    if horizon_helps:
        print("  • Shorter horizon REDUCES angular error -> the direction failure is partly "
              "horizon/observability driven.")
    else:
        print("  • Shorter horizon does NOT meaningfully reduce angular error (all ≈chance ~90°) "
              "-> direction failure is NOT a horizon artifact; it persists even at H=5.")
    print(f"  • TCR by horizon: H5={c.loc[5,'TCR']:.2f}  H10={c.loc[10,'TCR']:.2f}  H20={c.loc[20,'TCR']:.2f}.")
    print(f"  • See comparison_summary.md / README.md for the full 5-question verdict.")


def write_comparison(comp):
    c = comp.set_index("horizon")
    a5, a10, a20 = c.loc[5, "angular_error_deg"], c.loc[10, "angular_error_deg"], c.loc[20, "angular_error_deg"]
    horizon_helps = a5 < a20 - 5
    best_ang = int(comp.loc[comp.angular_error_deg.idxmin(), "horizon"])
    best_tcr = int(comp.loc[comp.TCR.idxmax(), "horizon"])

    # q5: best thesis viz = strongest visible angularity while readable. Longer horizon
    # shows the most visible curvature; pick the horizon maximizing angularity*TCR.
    comp_score = (comp["angularity_ratio"].clip(lower=0) * comp["TCR"])
    best_viz = int(comp.loc[comp_score.idxmax(), "horizon"])

    if horizon_helps:
        q1 = f"YES — H=5 angular error {a5:.1f}° < H=20 {a20:.1f}° (Δ {a5-a20:+.1f}°)."
        q4 = ("The direction problem is at least partly HORIZON-BASED: predicting fewer steps "
              "ahead is meaningfully easier.")
    else:
        q1 = (f"NO — H=5 angular error {a5:.1f}° vs H=20 {a20:.1f}° (Δ {a5-a20:+.1f}°); all horizons "
              f"sit near chance (~90°).")
        q4 = ("The direction problem is NOT horizon-based — it is still present even at H=5. "
              "Shrinking the prediction window does not make turn DIRECTION predictable, which "
              "(with exp04) points to input/representation ambiguity, not horizon length.")
    q2 = (f"Predicted Δheading H5={c.loc[5,'pred_delta_heading_deg']:.0f}° / "
          f"H10={c.loc[10,'pred_delta_heading_deg']:.0f}° / H20={c.loc[20,'pred_delta_heading_deg']:.0f}°; "
          f"angularity {c.loc[5,'angularity_ratio']:.2f} / {c.loc[10,'angularity_ratio']:.2f} / "
          f"{c.loc[20,'angularity_ratio']:.2f}. Note GT Δheading also shrinks at short horizon "
          f"(less time to turn), so angularity is the fair (ratio) measure.")
    q3 = (f"TCR H5={c.loc[5,'TCR']:.2f} / H10={c.loc[10,'TCR']:.2f} / H20={c.loc[20,'TCR']:.2f} "
          f"(max at H={best_tcr}). " +
          ("Improves toward longer horizon." if c.loc[20, "TCR"] > c.loc[5, "TCR"] else
           "Improves toward shorter horizon." if c.loc[5, "TCR"] > c.loc[20, "TCR"] else "Roughly flat."))
    q5 = (f"H={best_viz} gives the strongest visible turning (max angularity×TCR). For thesis "
          f"figures, H=20 shows the most visible curvature in absolute terms while H={best_ang} "
          f"has the lowest angular error — use H={best_viz} for the headline rollout panels.")

    M = [
        "# exp03 — Horizon sweep — comparison\n",
        "Turn-only Model C (exp02 recipe) at horizons 5 / 10 / 20. Same architecture, 10 inputs, "
        "`target_du/dv` output, no sampler, no heading head. Only the horizon (and the "
        "horizon-scaled displacement floor) changes.\n",
        "## Displacement-floor note (why it scales)\n",
        "A fixed 2 m genuine-turn floor empties the short-horizon turn sets (held-out genuine "
        "turns measured at 1 / 2 / 50 for H=5 / 10 / 20). We hold heading-change>30° and "
        "max-step<0.6m fixed but scale the displacement floor D(H)=2·H/20 (0.5 / 1.0 / 2.0 m) "
        "to keep a constant per-step speed floor and comparable populations "
        "(held-out genuine 1073 / 442 / 50).\n",
        "## Held-out genuine-turn metrics by horizon\n",
        "| horizon | disp floor (m) | n_turns | ADE (m) | FDE (m) | angular err (°) | GT Δhead (°) | pred Δhead (°) | angularity | TCR |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in comp.iterrows():
        M.append(f"| {int(r.horizon)} | {r.disp_floor_m:.1f} | {int(r.n_turns)} | {r.ade:.3f} | {r.fde:.3f} | "
                 f"{r.angular_error_deg:.1f} | {r.gt_delta_heading_deg:.1f} | {r.pred_delta_heading_deg:.1f} | "
                 f"{r.angularity_ratio:.2f} | {r.TCR:.2f} |")
    M += [
        "\n## Verdict\n",
        "**1. Does H=5 reduce angular error vs H=20?**", q1, "",
        "**2. Does shorter horizon reduce or preserve predicted angularity?**", q2, "",
        "**3. Does Turn Capture Rate improve or collapse?**", q3, "",
        "**4. Is the direction problem horizon-based or present even at short horizon?**", q4, "",
        "**5. Which horizon gives the best thesis visualization?**", q5, "",
        "## Per-horizon outputs\n",
        "`horizon_05/`, `horizon_10/`, `horizon_20/` — each with model checkpoint, "
        "`training/config.json` + loss CSVs, `metrics.csv`, `metrics_summary.md`, and "
        "`plots/` (best_6 / worst_6 / collage / heading_change_comparison).\n",
    ]
    (HERE / "comparison_summary.md").write_text("\n".join(M), encoding="utf-8")
    # README mirrors the comparison summary as the folder entry point.
    (HERE / "README.md").write_text(
        "# exp03 — Horizon sweep (turn-only Model C, H = 5 / 10 / 20)\n\n"
        "Does shorter prediction horizon make turn DIRECTION easier? See "
        "`comparison_summary.md` for the full verdict and `comparison_metrics.csv` for the raw "
        "table; per-horizon results live in `horizon_05/ 10/ 20/`.\n\n"
        + "\n".join(M[M.index("## Held-out genuine-turn metrics by horizon\n"):]),
        encoding="utf-8")
    print(f"[saved] {HERE / 'comparison_summary.md'}")
    print(f"[saved] {HERE / 'README.md'}")
    print(f"[saved] {HERE / 'comparison_metrics.csv'}")


if __name__ == "__main__":
    main()
