"""
03_rollout_stabilization_abc.py
-------------------------------
Inference-only rollout stabilisation experiments.  No retraining.

A  Delta amplification         : (du, dv) *= alpha       (alpha in [1.25, 1.5, 2.0, adaptive])
B  Velocity persistence blend  : delta = beta * model + (1-beta) * prev_velocity
C  Minimum-motion floor        : if |delta| < tau, reuse previous direction at magnitude tau

All variants reuse the trained LSTM at:
    mp-data/outputs/prediction/experiments/phase-2b-final/lstm/

All outputs go to:
    mp-data/outputs/prediction_rollout_fix/
        plots/abc_per_trajectory_<pid>.png
        plots/abc_summary.png
        debug/abc_metrics.csv
        reports/abc_stabilization.md

Existing artefacts in prediction/ are NEVER touched.
"""

import json
import pickle
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE     = Path(__file__).resolve().parent
MP_ROOT  = HERE.parent.parent.parent.parent
MP_DATA  = MP_ROOT / "mp-data"
OVERFIT  = MP_ROOT / "mp-core" / "trajectory-prediction" / "experiments" / "overfit-10x"
sys.path.insert(0, str(OVERFIT))

from train_phase2b_final import (   # noqa: E402
    TrajectoryLSTM, SpatialInterpolator,
    add_deltas, FEATURE_COLS, WINDOW_SIZE, HIDDEN_SIZE, IDW_K, N_ROLLOUT,
)

SRC_CSV  = MP_DATA / "processed" / "encoded" / "trajectories_encoded.csv"
LSTM_DIR = MP_DATA / "outputs" / "prediction" / "experiments" / "phase-2b-final" / "lstm"
OUT      = MP_DATA / "outputs" / "prediction_rollout_fix"
PLOTS    = OUT / "plots"
DEBUG    = OUT / "debug"
REPORTS  = OUT / "reports"

PROBE_PIDS = [310, 50, 100, 200, 400, 500]
MIN_LEN    = WINDOW_SIZE + N_ROLLOUT

ALPHA_VALUES   = [1.0, 1.25, 1.5, 2.0]
BETA_VALUES    = [0.6, 0.7, 0.8]
FLOOR_VALUES   = [0.01, 0.02, 0.03]   # m/step
SEED_VEL_FRAMES = 3                    # average last K seed deltas for velocity init


# ── Model load ─────────────────────────────────────────────────────────────
def load_model(device):
    with open(LSTM_DIR / "scaler.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    model = TrajectoryLSTM(len(FEATURE_COLS), HIDDEN_SIZE).to(device)
    model.load_state_dict(torch.load(LSTM_DIR / "model.pth", map_location=device))
    model.eval()
    return model, bundle["feature_scaler"], bundle["target_scaler"]


# ── Generic rollout with a pluggable delta-corrector ───────────────────────
def rollout_with_corrector(model, fs, ts, track, interp, device,
                           corrector: Callable):
    """corrector(raw_du, raw_dv, step_idx, state) -> (du, dv)
       state is a mutable dict the corrector can read/write between steps.
    """
    feats = track[FEATURE_COLS].to_numpy(dtype=np.float32)
    pos   = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    feats_s = fs.transform(feats)
    window = feats_s[:WINDOW_SIZE].copy()
    prev_x = float(pos[WINDOW_SIZE - 1, 0])
    prev_y = float(pos[WINDOW_SIZE - 1, 1])

    # seed velocity = average of last K seed-window deltas (in raw units)
    seed_du = np.diff(pos[:WINDOW_SIZE, 0])
    seed_dv = np.diff(pos[:WINDOW_SIZE, 1])
    if len(seed_du) >= SEED_VEL_FRAMES:
        v0_u = float(np.mean(seed_du[-SEED_VEL_FRAMES:]))
        v0_v = float(np.mean(seed_dv[-SEED_VEL_FRAMES:]))
    else:
        v0_u = float(np.mean(seed_du)) if len(seed_du) else 0.0
        v0_v = float(np.mean(seed_dv)) if len(seed_dv) else 0.0
    state = {"prev_du": v0_u, "prev_dv": v0_v}

    pred_du, pred_dv, pred_x, pred_y = [], [], [], []
    for step in range(N_ROLLOUT):
        x_in = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            p_s = model(x_in).cpu().numpy()[0]
        raw_du, raw_dv = ts.inverse_transform(p_s[np.newaxis])[0]
        raw_du, raw_dv = float(raw_du), float(raw_dv)

        du, dv = corrector(raw_du, raw_dv, step, state)

        wx, wy = prev_x + du, prev_y + dv
        pred_du.append(du); pred_dv.append(dv)
        pred_x.append(wx);  pred_y.append(wy)

        obs, bnd = interp.query(wx, wy)
        new_raw = np.array([wx, wy, du, dv, obs, bnd], dtype=np.float32)
        window = np.vstack([window[1:], fs.transform(new_raw[np.newaxis])[0]])
        prev_x, prev_y = wx, wy

        # update prev velocity for blending/floor correctors
        state["prev_du"], state["prev_dv"] = du, dv

    return (np.array(pred_du), np.array(pred_dv),
            np.array(pred_x),  np.array(pred_y))


# ── Correctors ──────────────────────────────────────────────────────────────
def corrector_baseline():
    def f(du, dv, step, state):
        return du, dv
    return f, "baseline"


def corrector_alpha(alpha):
    def f(du, dv, step, state):
        return alpha * du, alpha * dv
    return f, f"alpha={alpha}"


def corrector_alpha_adaptive(target_mag=0.04):
    """Re-scales each step so |delta| == target_mag when raw output is non-zero.
    target_mag was chosen to match GT median (~0.026 m) but a touch larger to
    counteract trailing decay; can be tuned."""
    def f(du, dv, step, state):
        m = np.hypot(du, dv)
        if m < 1e-9:
            return du, dv
        s = target_mag / m
        return s * du, s * dv
    return f, f"alpha_adaptive(target={target_mag:g})"


def corrector_velocity_blend(beta):
    def f(du, dv, step, state):
        bdu = beta * du + (1.0 - beta) * state["prev_du"]
        bdv = beta * dv + (1.0 - beta) * state["prev_dv"]
        return bdu, bdv
    return f, f"velocity_blend(beta={beta})"


def corrector_motion_floor(threshold):
    def f(du, dv, step, state):
        m = np.hypot(du, dv)
        if m >= threshold:
            return du, dv
        pdu, pdv = state["prev_du"], state["prev_dv"]
        pm = np.hypot(pdu, pdv)
        if pm < 1e-9:
            return du, dv  # nothing useful to fall back to; let it be
        # Re-use previous direction at magnitude = threshold
        return threshold * pdu / pm, threshold * pdv / pm
    return f, f"motion_floor(tau={threshold:g})"


# ── Eval helpers ───────────────────────────────────────────────────────────
def gt_future(track):
    pos = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    seed_end = pos[WINDOW_SIZE - 1]
    fx = pos[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 0]
    fy = pos[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT, 1]
    gdu = np.r_[fx[0] - seed_end[0], np.diff(fx)]
    gdv = np.r_[fy[0] - seed_end[1], np.diff(fy)]
    return fx, fy, gdu, gdv


def metrics(pred_du, pred_dv, pred_x, pred_y, gt_x, gt_y, gt_du, gt_dv):
    ade  = float(np.mean(np.hypot(pred_x - gt_x, pred_y - gt_y)))
    fde  = float(np.hypot(pred_x[-1] - gt_x[-1], pred_y[-1] - gt_y[-1]))
    cum_pred = float(np.sum(np.hypot(pred_du, pred_dv)))
    cum_gt   = float(np.sum(np.hypot(gt_du,  gt_dv)))
    return {
        "ade": ade, "fde": fde,
        "cum_pred": cum_pred, "cum_gt": cum_gt,
        "cum_ratio": cum_pred / cum_gt if cum_gt > 1e-9 else float("nan"),
        "mean_mag_pred": float(np.mean(np.hypot(pred_du, pred_dv))),
        "mean_mag_gt":   float(np.mean(np.hypot(gt_du,  gt_dv))),
    }


# ── Figures ────────────────────────────────────────────────────────────────
def per_traj_plot(pid, gt_x, gt_y, gt_du, gt_dv, runs, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Stabilisation experiments A/B/C — pid {pid}",
                 fontsize=12, fontweight="bold")

    # Plan view
    ax = axes[0, 0]
    ax.plot(gt_x, gt_y, "k-o", ms=3, lw=1.5, label="GT future", alpha=0.7)
    for r in runs:
        ax.plot(r["x"], r["y"], "-o", ms=3, lw=1.1, alpha=0.75,
                label=f"{r['name']}  cum={r['m']['cum_ratio']:.2f}")
    ax.set_xlabel("world_x (m)"); ax.set_ylabel("world_y (m)")
    ax.set_title("Plan view")
    ax.legend(fontsize=7, loc="best"); ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_aspect("equal", adjustable="datalim")

    # Cumulative displacement
    ax = axes[0, 1]
    steps = np.arange(1, N_ROLLOUT + 1)
    ax.plot(steps, np.cumsum(np.hypot(gt_du, gt_dv)), "k-",
            lw=2, label="GT")
    for r in runs:
        ax.plot(steps, np.cumsum(np.hypot(r["du"], r["dv"])),
                lw=1.0, alpha=0.85, label=r["name"])
    ax.set_xlabel("Rollout step"); ax.set_ylabel("Cumul path length (m)")
    ax.set_title("Cumulative displacement")
    ax.legend(fontsize=7); ax.grid(True, lw=0.3, alpha=0.5)

    # Per-step delta magnitude
    ax = axes[1, 0]
    ax.plot(steps, np.hypot(gt_du, gt_dv), "k-", lw=2, label="GT")
    for r in runs:
        ax.plot(steps, np.hypot(r["du"], r["dv"]),
                lw=1.0, alpha=0.85, label=r["name"])
    ax.set_xlabel("Rollout step"); ax.set_ylabel("|delta| (m)")
    ax.set_title("Per-step delta magnitude")
    ax.legend(fontsize=7); ax.grid(True, lw=0.3, alpha=0.5)

    # Positional error
    ax = axes[1, 1]
    for r in runs:
        err = np.hypot(r["x"] - gt_x, r["y"] - gt_y)
        ax.plot(steps, err, lw=1.0, alpha=0.85, label=r["name"])
    ax.set_xlabel("Rollout step"); ax.set_ylabel("L2 position error (m)")
    ax.set_title("Drift accumulation")
    ax.legend(fontsize=7); ax.grid(True, lw=0.3, alpha=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summary_plot(records, names, out_path):
    """Mean cumulative ratio + mean ADE across PIDs, per corrector."""
    means_cum = []
    means_ade = []
    means_fde = []
    for name in names:
        rs = [rec[name]["m"] for rec in records.values()]
        means_cum.append(np.mean([r["cum_ratio"] for r in rs]))
        means_ade.append(np.mean([r["ade"] for r in rs]))
        means_fde.append(np.mean([r["fde"] for r in rs]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Stabilisation summary across probe trajectories",
                 fontsize=12, fontweight="bold")

    x = np.arange(len(names))
    axes[0].bar(x, means_cum, color="#2980b9", alpha=0.8)
    axes[0].axhline(1.0, color="k", lw=0.6, ls="--")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=35, ha="right",
                                                   fontsize=7)
    axes[0].set_ylabel("mean cum-ratio  (1.0 ideal)")
    axes[0].set_title("Cumulative-displacement ratio")
    axes[0].grid(True, lw=0.3, alpha=0.5, axis="y")

    axes[1].bar(x, means_ade, color="#c0392b", alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=35, ha="right",
                                                   fontsize=7)
    axes[1].set_ylabel("mean ADE (m)")
    axes[1].set_title("ADE (lower better)")
    axes[1].grid(True, lw=0.3, alpha=0.5, axis="y")

    axes[2].bar(x, means_fde, color="#8e44ad", alpha=0.8)
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=35, ha="right",
                                                   fontsize=7)
    axes[2].set_ylabel("mean FDE (m)")
    axes[2].set_title("FDE (lower better)")
    axes[2].grid(True, lw=0.3, alpha=0.5, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    DEBUG.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device : {device}")

    df_full = pd.read_csv(SRC_CSV)
    if "track_id" in df_full.columns and "person_id" not in df_full.columns:
        df_full = df_full.rename(columns={"track_id": "person_id"})
    if "frame_idx" in df_full.columns and "frame_number" not in df_full.columns:
        df_full = df_full.rename(columns={"frame_idx": "frame_number"})

    interp = SpatialInterpolator(df_full, k=IDW_K)
    df = add_deltas(df_full)

    track_lengths = df.groupby("person_id").size()
    eligible = set(track_lengths[track_lengths >= MIN_LEN].index.tolist())
    probe_in = [p for p in PROBE_PIDS if p in eligible]

    # Deduplicate against tracks with identical first-MIN_LEN positions
    # (the overfit-10x dataset duplicates 54 base tracks across 540 pids).
    seen_hashes = set()
    probe = []
    for pid in probe_in:
        rows = (df[df["person_id"] == pid]
                .sort_values("frame_number")
                .head(MIN_LEN)[["world_x", "world_y"]]
                .round(6).to_numpy().tobytes())
        if rows in seen_hashes:
            continue
        seen_hashes.add(rows)
        probe.append(pid)

    # If we lost too many to duplication, top up by scanning eligible pids
    EXTRA_TARGET = 6
    if len(probe) < EXTRA_TARGET:
        for pid in sorted(eligible):
            if pid in probe:
                continue
            rows = (df[df["person_id"] == pid]
                    .sort_values("frame_number")
                    .head(MIN_LEN)[["world_x", "world_y"]]
                    .round(6).to_numpy().tobytes())
            if rows in seen_hashes:
                continue
            seen_hashes.add(rows)
            probe.append(pid)
            if len(probe) >= EXTRA_TARGET:
                break

    print(f"[INFO]  Probing PIDs (deduplicated): {probe}")

    model, fs, ts = load_model(device)

    correctors = []
    correctors.append(corrector_baseline())
    for a in ALPHA_VALUES:
        correctors.append(corrector_alpha(a))
    correctors.append(corrector_alpha_adaptive(target_mag=0.04))
    for b in BETA_VALUES:
        correctors.append(corrector_velocity_blend(b))
    for t in FLOOR_VALUES:
        correctors.append(corrector_motion_floor(t))

    names = [n for (_, n) in correctors]

    records = {}
    all_rows = []
    for pid in probe:
        track = (df[df["person_id"] == pid]
                 .reset_index(drop=True)
                 .iloc[:MIN_LEN].copy())
        gt_x, gt_y, gt_du, gt_dv = gt_future(track)

        runs = []
        per_traj_data = {}
        for fn, name in correctors:
            du, dv, x, y = rollout_with_corrector(
                model, fs, ts, track, interp, device, fn)
            m = metrics(du, dv, x, y, gt_x, gt_y, gt_du, gt_dv)
            runs.append({"name": name, "du": du, "dv": dv, "x": x, "y": y, "m": m})
            per_traj_data[name] = {"du": du, "dv": dv, "x": x, "y": y, "m": m}
            all_rows.append({"pid": pid, "corrector": name, **m})

        records[pid] = per_traj_data
        per_traj_plot(pid, gt_x, gt_y, gt_du, gt_dv, runs,
                      PLOTS / f"abc_per_trajectory_{pid}.png")

        # Compact print
        print(f"  pid={pid}")
        for r in runs:
            print(f"    {r['name']:<28}  cum_ratio={r['m']['cum_ratio']:.3f}  "
                  f"ADE={r['m']['ade']:.3f}  FDE={r['m']['fde']:.3f}")

    summary_plot(records, names, PLOTS / "abc_summary.png")
    print(f"[OK]    Summary plot: abc_summary.png")

    df_metrics = pd.DataFrame(all_rows)
    df_metrics.to_csv(DEBUG / "abc_metrics.csv", index=False)
    print(f"[OK]    Metrics CSV : abc_metrics.csv")

    # Aggregate stats per corrector
    agg = (df_metrics.groupby("corrector")
                     .agg(mean_ade=("ade", "mean"),
                          mean_fde=("fde", "mean"),
                          mean_cum_ratio=("cum_ratio", "mean"),
                          median_cum_ratio=("cum_ratio", "median"))
                     .reindex(names))

    # Score: prefer cum_ratio close to 1 AND low ADE/FDE.
    # Define a composite where deviation from 1 is penalised.
    agg["cum_ratio_dev"] = (agg["mean_cum_ratio"] - 1.0).abs()
    agg["composite"] = agg["cum_ratio_dev"] + 0.1 * agg["mean_ade"]
    agg_sorted = agg.sort_values("composite")

    print("\n---- AGG RESULTS (sorted by composite score) ----")
    print(agg_sorted.to_string())

    # ── Markdown ────────────────────────────────────────────────────────────
    md = [
        "# Rollout stabilisation — experiments A, B, C",
        "",
        "Inference-only corrections applied on top of the trained LSTM.  No",
        "retraining.  All rollouts produced from the same checkpoint at",
        f"`{LSTM_DIR.relative_to(MP_ROOT).as_posix()}`.",
        "",
        "## Correctors evaluated",
        "",
        "- **baseline**            — production rollout (no correction).",
        f"- **alpha=A**             — `(du, dv) *= A` after `inverse_transform`. Tested A ∈ {ALPHA_VALUES}.",
        f"- **alpha_adaptive**      — rescales each step so `|delta| = 0.04 m` whenever the raw output is non-zero.",
        f"- **velocity_blend(β)**   — `delta = β·model + (1−β)·prev_velocity`. Tested β ∈ {BETA_VALUES}.",
        f"- **motion_floor(τ)**     — if `|model_delta| < τ`, reuse previous direction at magnitude τ. Tested τ ∈ {FLOOR_VALUES} m.",
        "",
        "Initial `prev_velocity` for blending / floor = mean of the last "
        f"{SEED_VEL_FRAMES} seed-window deltas.",
        "",
        f"Probed PIDs: {probe}",
        "",
        "## Aggregate metrics",
        "",
        "| corrector | mean ADE | mean FDE | mean cum-ratio | median cum-ratio | composite (lower better) |",
        "|---|---|---|---|---|---|",
    ]
    for name in agg_sorted.index:
        r = agg_sorted.loc[name]
        md.append(
            f"| `{name}` | {r['mean_ade']:.3f} | {r['mean_fde']:.3f} "
            f"| {r['mean_cum_ratio']:.3f} | {r['median_cum_ratio']:.3f} "
            f"| {r['composite']:.3f} |"
        )

    best_name = agg_sorted.index[0]
    md += [
        "",
        f"**Top corrector by composite score:** `{best_name}`",
        "",
        "*Composite = |mean cum-ratio − 1| + 0.1 · mean ADE.*  Penalises both",
        "magnitude collapse / explosion AND positional drift.",
        "",
        "## Summary figure",
        "",
        f"![summary]({(PLOTS / 'abc_summary.png').as_posix()})",
        "",
        "## Per-trajectory figures",
        "",
    ]
    for pid in probe:
        md.append(f"- pid {pid} — ![pid {pid}]({(PLOTS / f'abc_per_trajectory_{pid}.png').as_posix()})")

    md += [
        "",
        "---",
        "*Generated by `experiments/rollout-fix/03_rollout_stabilization_abc.py`*",
    ]

    out = REPORTS / "abc_stabilization.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK]    Report : {out}")

    # Persist the per-corrector aggregate for the final report
    agg_sorted.to_csv(DEBUG / "abc_aggregate.csv")


if __name__ == "__main__":
    main()
