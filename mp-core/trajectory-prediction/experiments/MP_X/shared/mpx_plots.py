"""
MP_X shared plotting — genuine-turn rollout plots + heading-change comparison.

A single rollout plot shows: seed/history (grey), GT future (blue), predicted
future (orange), start marker, GT final-direction arrow, predicted final-direction
arrow, and a rich title (ids, ADE, FDE, GT/pred turn angle, bin). Reused by every
experiment.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_SEED = "#444444"
C_GT = "#1f77b4"
C_PRED = "#e0852a"

_BIN_LABEL = {"straight": "straight", "mild_30_90": "mild 30-90",
              "sharp_90_150": "sharp 90-150", "uturn_150_180": "U-turn 150-180"}


def _short_id(tid) -> str:
    s = str(tid)
    return s.split("__")[-1] if "__" in s else s


def _final_dir_arrow(ax, pts, color):
    """Draw a short arrow along the final segment of a path."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return
    p0, p1 = pts[-2], pts[-1]
    d = p1 - p0
    nrm = np.linalg.norm(d)
    if nrm < 1e-9:
        return
    # scale arrow to ~25% of the path's bounding box for visibility
    span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1e-6)
    d = d / nrm * 0.25 * span
    ax.annotate("", xy=(p1[0] + d[0], p1[1] + d[1]), xytext=(p1[0], p1[1]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0))


def plot_single_rollout(ax, row: dict, r: dict, title_extra: str = ""):
    """Draw one rollout on `ax`. `row` is a metric row; `r` is a P5 rollout dict."""
    seed = np.asarray(r["seed_pos"], dtype=np.float64)
    gt = np.asarray(r["gt_pos"], dtype=np.float64)
    pr = np.asarray(r["pred_pos"], dtype=np.float64)

    ax.plot(seed[:, 0], seed[:, 1], "-", color=C_SEED, lw=1.6, label="seed (history)")
    ax.plot(gt[:, 0], gt[:, 1], "-o", color=C_GT, lw=2.0, ms=3.2, label="GT future")
    ax.plot(pr[:, 0], pr[:, 1], "--^", color=C_PRED, lw=1.8, ms=3.0, label="predicted")
    ax.plot(gt[0, 0], gt[0, 1], "*", color="k", ms=14, zorder=6, label="start")
    _final_dir_arrow(ax, gt, C_GT)
    _final_dir_arrow(ax, pr, C_PRED)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    tid = _short_id(row["trajectory_id"])
    ttl = (f"{row['recording_id'][:14]} {tid} @w{row['win_start']}  [{_BIN_LABEL.get(row['bin'], row['bin'])}]\n"
           f"ADE {row['ade']:.2f}m  FDE {row['fde']:.2f}m  "
           f"GT {row['gt_head_change_deg']:.0f}° / pred {row['pred_head_change_deg']:.0f}°")
    if title_extra:
        ttl += f"\n{title_extra}"
    ax.set_title(ttl, fontsize=8)
    ax.legend(fontsize=6, loc="best")


def _dedup_by_trajectory(rolls, k):
    """Pick up to k rolls, at most one per trajectory (avoids near-duplicate
    overlapping windows of the same turn)."""
    picked, seen = [], set()
    for row, r in rolls:
        if row["trajectory_id"] in seen:
            continue
        seen.add(row["trajectory_id"])
        picked.append((row, r))
        if len(picked) >= k:
            break
    if len(picked) < k:  # backfill if not enough unique trajectories
        for rr in rolls:
            if rr not in picked:
                picked.append(rr)
            if len(picked) >= k:
                break
    return picked[:k]


def grid_plot(rolls, out_path: Path, suptitle: str, ncol=3):
    n = len(rolls)
    if n == 0:
        return False
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (row, r) in zip(axes, rolls):
        plot_single_rollout(ax, row, r)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(suptitle, fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def save_best_worst(genuine_rolls, out_dir: Path, k=6, rank_key="angular_err_deg"):
    """Save k best and k worst genuine-turn rollouts (one PNG each + a grid).

    "best"  = smallest angular error (model matched the turn direction best),
    "worst" = largest angular error. Deduplicated by trajectory.
    """
    out_dir = Path(out_dir)
    best_dir = out_dir / "best_6_genuine_turns"
    worst_dir = out_dir / "worst_6_genuine_turns"
    best_dir.mkdir(parents=True, exist_ok=True)
    worst_dir.mkdir(parents=True, exist_ok=True)
    if not genuine_rolls:
        return 0, 0
    s = sorted(genuine_rolls, key=lambda rr: rr[0][rank_key])
    best = _dedup_by_trajectory(s, k)
    worst = _dedup_by_trajectory(list(reversed(s)), k)

    for tag, rolls, d in (("best", best, best_dir), ("worst", worst, worst_dir)):
        for j, (row, r) in enumerate(rolls, 1):
            fig, ax = plt.subplots(figsize=(5.6, 5.2))
            plot_single_rollout(ax, row, r)
            fig.tight_layout()
            fig.savefig(d / f"{tag}_{j:02d}_{_short_id(row['trajectory_id'])}_w{row['win_start']}.png", dpi=120)
            plt.close(fig)
    grid_plot(best, out_dir / "best_6_genuine_turns.png",
              "6 BEST genuine-turn predictions (lowest angular error)")
    grid_plot(worst, out_dir / "worst_6_genuine_turns.png",
              "6 WORST genuine-turn predictions (highest angular error)")
    return len(best), len(worst)


def save_collage(genuine_rolls, out_path: Path, k=9):
    """9-panel collage of representative genuine turns, spread across the GT
    heading-change range and deduplicated by trajectory."""
    if not genuine_rolls:
        return False
    s = sorted(genuine_rolls, key=lambda rr: rr[0]["gt_head_change_deg"])
    # spread evenly across the sorted range
    idx = np.linspace(0, len(s) - 1, num=min(k * 3, len(s))).astype(int)
    spread = [s[i] for i in dict.fromkeys(idx)]
    picks = _dedup_by_trajectory(spread, k)
    return grid_plot(picks, Path(out_path),
                     "Representative genuine turns (GT vs predicted)", ncol=3)


def final_heading_scatter(genuine_rolls, out_path: Path):
    """GT vs predicted FINAL net-displacement direction on genuine turns.

    Left: scatter of GT net-direction (deg) vs predicted net-direction (deg) with
    a y=x reference (points on the diagonal = correct turn direction).
    Right: polar view — GT (blue) and predicted (orange) net-direction unit points.
    "final heading" = atan2 of (end - start) over the future path."""
    out_path = Path(out_path)
    if not genuine_rolls:
        return False
    gt_dir, pr_dir = [], []
    for _row, r in genuine_rolls:
        gt = np.asarray(r["gt_pos"], dtype=np.float64)
        pr = np.asarray(r["pred_pos"], dtype=np.float64)
        if len(gt) < 2 or len(pr) < 2:
            continue
        gv = gt[-1] - gt[0]
        pv = pr[-1] - pr[0]
        if np.linalg.norm(gv) < 1e-9 or np.linalg.norm(pv) < 1e-9:
            continue
        gt_dir.append(math.degrees(math.atan2(gv[1], gv[0])))
        pr_dir.append(math.degrees(math.atan2(pv[1], pv[0])))
    if not gt_dir:
        return False
    gt_dir = np.array(gt_dir)
    pr_dir = np.array(pr_dir)

    fig = plt.figure(figsize=(13, 5.6))
    ax = fig.add_subplot(1, 2, 1)
    err = np.abs((pr_dir - gt_dir + 180) % 360 - 180)
    sc = ax.scatter(gt_dir, pr_dir, c=err, cmap="viridis_r", vmin=0, vmax=180,
                    s=40, edgecolor="k", lw=0.3)
    ax.plot([-180, 180], [-180, 180], "--", color="k", lw=1.0, label="y = x (correct direction)")
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xlabel("GT final direction (deg)")
    ax.set_ylabel("predicted final direction (deg)")
    ax.set_title(f"GT vs predicted final direction (n={len(gt_dir)})")
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="lower right")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("direction error (deg)")

    axp = fig.add_subplot(1, 2, 2, projection="polar")
    axp.scatter(np.radians(gt_dir), np.ones_like(gt_dir), s=36, color=C_GT, alpha=0.7, label="GT")
    axp.scatter(np.radians(pr_dir), np.full_like(pr_dir, 0.7), s=36, color=C_PRED, alpha=0.7, label="predicted")
    axp.set_rticks([]); axp.set_rmax(1.15)
    axp.set_title("Final direction (polar): GT vs predicted")
    axp.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.1, -0.05))

    med = float(np.median(err))
    fig.suptitle(f"Final-direction agreement on genuine turns — median direction error {med:.0f}°",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def heading_change_comparison(rolled, out_path: Path):
    """Scatter + histogram of GT vs predicted heading change.

    Left: scatter (all eval windows + genuine highlighted), y=x reference, the
    30°/20° turn thresholds. Right: overlaid histograms (genuine turns)."""
    import pandas as pd  # local
    out_path = Path(out_path)
    if rolled is None or len(rolled) == 0:
        return False
    df = rolled
    gen = df[df["is_genuine_turn"]]
    allw = df[df["in_all_sample"]] if "in_all_sample" in df.columns else df

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    ax = axes[0]
    ax.scatter(allw["gt_head_change_deg"], allw["pred_head_change_deg"],
               s=14, alpha=0.35, color="#999999", label=f"all windows (n={len(allw)})")
    if len(gen):
        ax.scatter(gen["gt_head_change_deg"], gen["pred_head_change_deg"],
                   s=34, alpha=0.85, color=C_PRED, edgecolor="k", lw=0.3,
                   label=f"genuine turns (n={len(gen)})")
    lim = max(10.0, float(np.nanmax(df["gt_head_change_deg"])) * 1.05)
    ax.plot([0, lim], [0, lim], "--", color="k", lw=1.0, label="y = x (perfect)")
    ax.axvline(30, color="#1f77b4", ls=":", lw=1.0, label="GT turn = 30°")
    ax.axhline(20, color=C_PRED, ls=":", lw=1.0, label="pred turn = 20°")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("GT heading change (deg)")
    ax.set_ylabel("predicted heading change (deg)")
    ax.set_title("Predicted vs GT heading change")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    if len(gen):
        bins = np.linspace(0, 180, 19)
        ax.hist(gen["gt_head_change_deg"], bins=bins, alpha=0.55, color=C_GT, label="GT (genuine turns)")
        ax.hist(gen["pred_head_change_deg"], bins=bins, alpha=0.55, color=C_PRED, label="predicted")
        ax.axvline(20, color=C_PRED, ls=":", lw=1.0)
        ax.axvline(30, color=C_GT, ls=":", lw=1.0)
    ax.set_xlabel("heading change (deg)")
    ax.set_ylabel("count")
    ax.set_title("Heading-change distribution on genuine turns")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("Heading-change comparison — does the model commit to turns?",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True
