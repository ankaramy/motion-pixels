"""
generate_curated_long_horizon_collages.py
==========================================
Curated PRESENTATION collages from the existing long-horizon stress-test rollouts.

>>> EXPLORATORY EXTRAPOLATION BEYOND VALIDATED RANGE <<<

No retraining, no new experiments, no model/dataset/encoder changes. This script does NOT create
new predictions: it reuses the *exact same* deterministic rollout helpers from
`generate_long_horizon_stress_test.py` (frozen MODEL_XC_B_CURV_LIGHT, same seeds, same code path),
which reproduce byte-identical trajectories to the figures already in the stress-test folder. The
only differences here are (a) a manual track subset and (b) the requested visual change:

    The ENTIRE prediction trajectory is drawn in the dashed/dotted extrapolation style —
    there is NO solid "validated" segment. History and GT styling are unchanged.

Outputs (PNG + PDF) -> $MP_PLOTS_OUT/Long_Horizon_Stress_Test/ (default: ./outputs)
Run:  python generate_curated_long_horizon_collages.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_long_horizon_stress_test as G   # reuse frozen, deterministic rollout + style

ROOT = G.ROOT
C_HIST, C_GT, C_PRED, C_GRID = G.C_HIST, G.C_GT, G.C_PRED, G.C_GRID
SITE, EXTRAP_LABEL, MODEL_NAME = G.SITE, G.EXTRAP_LABEL, G.MODEL_NAME

# manual curated subset per horizon (track ids)
CURATED = {
    800:  ["2039", "540", "1178"],
    1000: ["2039", "540", "1178"],
    1200: ["5350", "2039"],
}


def draw_dashed(ax, hist, gt, pred, H, title, legend=False, small=False):
    """Same visual language as the stress-test plots, but the WHOLE prediction is dashed
    (extrapolation style) — no solid validated segment."""
    # history (unchanged)
    ax.plot(hist[:, 0], hist[:, 1], "-", color=C_HIST, lw=2.2, zorder=4)
    # GT where real (unchanged)
    if len(gt) > 1:
        ax.plot(gt[:, 0], gt[:, 1], "-", color=C_GT, lw=1.4, alpha=0.9, zorder=3)
        me = max(1, len(gt) // 18)
        ax.plot(gt[::me, 0], gt[::me, 1], "o", color=C_GT, ms=2.6, zorder=3)
    # prediction — ENTIRE path dashed
    ax.plot(pred[:, 0], pred[:, 1], "--", color=C_PRED, lw=1.9, dashes=(4, 3), zorder=5)
    # seed point + final point (no validated-edge ring)
    ax.plot(pred[0, 0], pred[0, 1], "o", color="black", ms=6, zorder=7)
    ax.plot(pred[-1, 0], pred[-1, 1], "s", mfc="white", mec=C_PRED, mew=1.4, ms=6, zorder=7)

    pts = np.vstack([hist, pred] + ([gt] if len(gt) > 1 else []))
    xl, yl = G.pad_limits(pts); ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=C_GRID, lw=0.5, alpha=0.5); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#bbbbbb"); s.set_linewidth(0.8)
    ax.tick_params(labelsize=7 if small else 8, color="#bbbbbb", length=3)
    ax.set_title(title, fontsize=10 if small else 11, loc="left", color="#222222")
    if legend:
        handles = [Line2D([0], [0], color=C_HIST, lw=2.2, label="history (seed)"),
                   Line2D([0], [0], color=C_GT, lw=1.4, marker="o", ms=3, label="GT (real, where available)"),
                   Line2D([0], [0], color=C_PRED, lw=1.9, ls="--", label="prediction (exploratory extrapolation)")]
        ax.legend(handles=handles, fontsize=7.5, loc="best", framealpha=0.9, edgecolor="#dddddd")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df = G.XCC.load_df()
    wb_all = G.L.recording_world_bounds(df)
    model, f_sc, t_sc = G.load_xc_model()
    # deterministic replay of the SAME inputs/rollout used in the stress test
    seeds, starts, wbk, hist, gt_full, tracks = G.build_inputs(df, wb_all)
    by_track = {trk: b for b, (_, trk, _) in enumerate(tracks)}

    created = []
    for H, trk_ids in CURATED.items():
        pred = G.rollout(model, f_sc, t_sc, seeds, starts, wbk, H)       # identical to stress test
        n = len(trk_ids)
        fig, axes = plt.subplots(1, n, figsize=(7.0 * n, 6.6), squeeze=False)
        axes = axes[0]
        fig.patch.set_facecolor("white")
        for ax, trk in zip(axes, trk_ids):
            b = by_track[trk]
            rec = tracks[b][0]
            gt = gt_full[b][:H + 1]
            title = f"{SITE.get(rec, rec)}   ·   track {trk}   ·   H{H}"
            draw_dashed(ax, hist[b], gt, pred[b], H, title, legend=(ax is axes[0]))
        fig.suptitle(f"{MODEL_NAME} — curated long-horizon selection · H{H}\n{EXTRAP_LABEL}",
                     fontsize=15, color="#222222")
        # generous breathing room
        fig.subplots_adjust(left=0.05, right=0.97, top=0.86, bottom=0.08, wspace=0.18)
        png = ROOT / f"H{H}_curated_collage.png"
        pdf = ROOT / f"H{H}_curated_collage.pdf"
        fig.savefig(png, dpi=200, facecolor="white")
        fig.savefig(pdf, facecolor="white")           # trivial vector export
        plt.close(fig)
        created += [png, pdf]
        print(f"[H{H}] curated collage: {trk_ids} -> {png.name} (+ pdf)")

    print("\nCreated:")
    for p in created:
        print(" ", p)


if __name__ == "__main__":
    main()
