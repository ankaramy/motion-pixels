"""
generate_long_horizon_stress_test.py
====================================
LONG-HORIZON EXTRAPOLATION STUDY for the frozen MODEL_XC_B_CURV_LIGHT.

>>> EXPLORATORY EXTRAPOLATION BEYOND VALIDATED RANGE <<<

This pushes the final frozen thesis model far past its validated/presented horizon (<= H400)
to H800 / H1000 / H1200 purely to *visualize behaviour under stress*. It is NOT a benchmark,
NOT a performance claim, NOT a new model. No training, retraining, loss/encoder/dataset/
hyper-parameter changes. Only MODEL_XC_B_CURV_LIGHT is used; no model comparison.

The model is replayed deterministically (autoregressive rollout, spatial obstacle/boundary
features frozen at the last observation — the documented MODEL_X rollout convention).

Prediction line: SOLID within the validated range (<= H400), DASHED for the extrapolated
continuation (> H400). GT is drawn only where the real track actually provides it; nothing is
invented beyond real data.

Outputs -> $MP_PLOTS_OUT/Long_Horizon_Stress_Test/ (default: ./outputs)
Run:  python generate_long_horizon_stress_test.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as manim

# --------------------------------------------------------------------------------------
# Project wiring (read-only reuse)
# --------------------------------------------------------------------------------------
import os
TP = Path(__file__).resolve().parents[2] / "mp-core" / "trajectory-prediction"
sys.path.insert(0, str(TP / "MODEL_XC"))
import xc_common as XCC          # noqa: E402  (pulls xr_common + model_x_lib)
L = XCC.L

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = TP / "MODEL_XC" / "checkpoints" / "MODEL_XC_B_CURV_LIGHT" / "model_best.pth"
SCALERS = TP / "MODEL_XC" / "checkpoints" / "MODEL_XC_B_CURV_LIGHT" / "scalers.json"

ROOT = Path(os.environ.get("MP_PLOTS_OUT", Path(__file__).resolve().parent / "outputs")) / "Long_Horizon_Stress_Test"
DIRS = {h: ROOT / d for h, d in {800: "H800", 1000: "H1000", 1200: "H1200_optional"}.items()}
FIG_DIR = ROOT / "figures"
ANIM_DIR = ROOT / "animations"
REPORT_DIR = ROOT / "report"
for d in [ROOT, *DIRS.values(), FIG_DIR, ANIM_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Study configuration
# --------------------------------------------------------------------------------------
HORIZONS = [800, 1000, 1200]          # H1200 included (rollout is computationally trivial)
VALIDATED_H = 400                     # max presented/validated horizon -> solid below, dashed above
OBS = XCC.OBS                         # 10 observed seed steps
MODEL_NAME = "MODEL_XC_B_CURV_LIGHT"
EXTRAP_LABEL = "EXPLORATORY EXTRAPOLATION BEYOND VALIDATED RANGE"

# Selection: priority tracks first, then diversity across all five recordings (~19 total).
PRIORITY = [("esplanade_espanya_01", "750"), ("placa_espanya_01", "4394"),
            ("placa_espanya_01", "6268"), ("red_bridge_combined_01", "1178")]
EXTRA = [("esplanade_espanya_01", "4152"), ("esplanade_espanya_01", "585"),
         ("esplanade_espanya_01", "112"), ("esplanade_espanya_01", "1882"),
         ("placa_espanya_01", "1647"), ("placa_espanya_01", "3790"), ("placa_espanya_01", "3"),
         ("placa_catalunya_01", "1846"), ("placa_catalunya_01", "4703"),
         ("placa_catalunya_01", "5350"), ("placa_catalunya_01", "3084"),
         ("red_bridge_combined_01", "2039"), ("red_bridge_combined_01", "1969"),
         ("stairs_montjuic_01", "1261"), ("stairs_montjuic_01", "540")]
SELECTION = PRIORITY + EXTRA

# tracks shown in the per-horizon collage (representative, all sites)
COLLAGE_TRACKS = [("esplanade_espanya_01", "750"), ("placa_espanya_01", "4394"),
                  ("placa_espanya_01", "6268"), ("red_bridge_combined_01", "1178"),
                  ("esplanade_espanya_01", "585"), ("placa_espanya_01", "3"),
                  ("placa_catalunya_01", "5350"), ("red_bridge_combined_01", "2039"),
                  ("stairs_montjuic_01", "540")]
# hero animations (kept few, for clarity)
ANIM_TRACKS = [("esplanade_espanya_01", "750"), ("placa_espanya_01", "4394"),
               ("placa_espanya_01", "6268"), ("red_bridge_combined_01", "1178")]
ANIM_HORIZON = 1000

SITE = {"placa_espanya_01": "Plaça Espanya", "placa_catalunya_01": "Plaça Catalunya",
        "esplanade_espanya_01": "Esplanade Espanya", "stairs_montjuic_01": "Stairs Montjuïc",
        "red_bridge_combined_01": "Red Bridge"}

C_HIST = "#333333"        # history dark grey
C_GT = "#1f77b4"          # GT blue
C_PRED = "#ff7f0e"        # prediction orange (no glow)
C_GRID = "#d5d5d5"


# --------------------------------------------------------------------------------------
# Model + rollout
# --------------------------------------------------------------------------------------
def load_xc_model():
    s = json.loads(Path(SCALERS).read_text())
    f_sc = L.ColumnScaler.from_dict(s["feature_scaler"])
    t_sc = L.ColumnScaler.from_dict(s["target_scaler"])
    m = L.TrajectoryLSTM(L.N_FEAT).to(DEVICE)
    m.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    m.eval()
    return m, f_sc, t_sc


def build_inputs(df, wb_all):
    """Seed each selected track from its first OBS frames (start_idx=0). Returns arrays."""
    grp = {t: g.sort_values("timestep") for t, g in df.groupby("trajectory_id")}
    seeds, starts, wbk, tracks = [], [], {"xmin": [], "xrng": [], "ymin": [], "yrng": []}, []
    hist, gt_full = [], []
    for rec, trk in SELECTION:
        tid = f"{rec}__{trk}"
        g = grp.get(tid)
        if g is None or len(g) < OBS + 5:
            print(f"[skip] {tid} missing/too short")
            continue
        feat = g[L.FEAT_COLS].to_numpy(np.float32)
        wxy = np.column_stack([g.world_x.to_numpy(float), g.world_y.to_numpy(float)])
        seeds.append(feat[:OBS])
        starts.append(wxy[OBS - 1])
        wb = wb_all[rec]
        for k in wbk:
            wbk[k].append(wb[k])
        hist.append(wxy[:OBS])
        gt_full.append(wxy[OBS - 1:])          # from anchor to end (real future, any length)
        tracks.append((rec, trk, tid))
    return (np.asarray(seeds, np.float32), np.asarray(starts, float),
            {k: np.asarray(v, float) for k, v in wbk.items()}, hist, gt_full, tracks)


def rollout(model, f_sc, t_sc, seeds, starts, wbk, H):
    return L.rollout_world_batch(model, seeds, starts, wbk, f_sc, t_sc, H, DEVICE)  # (B,H+1,2)


# --------------------------------------------------------------------------------------
# Stress diagnostics (descriptive only — NOT validated metrics)
# --------------------------------------------------------------------------------------
def wrap_deg(a):
    return (a + 180) % 360 - 180


def diagnostics(pred):
    """pred (H+1,2). Returns dict of descriptive long-horizon behaviour stats + labels."""
    steps = np.diff(pred, axis=0)
    d = np.linalg.norm(steps, axis=1)
    H = len(d)
    path = float(d.sum())
    net = float(np.linalg.norm(pred[-1] - pred[0]))
    p2n = path / net if net > 1e-6 else np.inf
    mean_step = float(d.mean())
    final_step = float(d[-1])
    k = max(10, H // 20)
    early = float(d[:k].mean()); late = float(d[-k:].mean())
    decay = late / early if early > 1e-9 else 0.0
    head = np.degrees(np.arctan2(steps[:, 1], steps[:, 0]))
    turn = wrap_deg(np.diff(head))
    head_total = float(np.abs(turn).sum())              # total absolute turning
    head_net = float(abs(turn.sum()))                   # net rotation (spiral indicator)
    sign_changes = int((np.diff(np.sign(turn[np.abs(turn) > 1e-3])) != 0).sum())
    straightness = net / path if path > 1e-6 else 0.0

    # dominant failure / behaviour mode (order matters: collapse & magnitude-runaway first)
    if decay < 0.20 and late < 0.3 * mean_step:
        mode = "collapse_to_stop"                       # steps vanish / path knots to a point
    elif net > 150 or (decay > 3.5 and net > 80):
        mode = "runaway_acceleration"                   # magnitude explodes (non-physical reach)
    elif head_net > 540 and straightness < 0.5:
        mode = "spiral"                                  # monotonic winding, low net progress
    elif straightness > 0.9 and head_total < 400:
        mode = "straight_line_drift"                     # collapses onto a steady straight heading
    elif sign_changes > 0.20 * H and head_total > 720:
        mode = "oscillation"                             # repeated back-and-forth heading flips
    else:
        mode = "meander_drift"                           # wanders but stays readable

    steady = (0.3 < decay < 2.0) and (net < 30)
    if mode == "collapse_to_stop":
        plaus = "collapsed"
    elif mode in ("runaway_acceleration", "spiral"):
        plaus = "artificial"
    elif mode == "oscillation":
        plaus = "unstable"
    elif mode == "straight_line_drift":
        plaus = "plausible" if steady else "stretched_but_readable"
    else:  # meander_drift
        plaus = "plausible" if (steady and straightness > 0.92) else "stretched_but_readable"

    return dict(total_pred_path_length=round(path, 3), net_displacement=round(net, 3),
                path_to_net_ratio=round(p2n, 3) if np.isfinite(p2n) else 9999.0,
                mean_step_length=round(mean_step, 4), final_step_length=round(final_step, 4),
                step_decay_ratio=round(decay, 3), heading_change_total=round(head_total, 1),
                heading_net_rotation=round(head_net, 1), straightness=round(straightness, 3),
                dominant_failure_mode=mode, visual_plausibility_label=plaus)


# --------------------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------------------
def pad_limits(pts, pad_frac=0.12, pad_min=1.5):
    xmin, ymin = pts.min(0); xmax, ymax = pts.max(0)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    xspan, yspan = xmax - xmin, ymax - ymin
    pad = max(pad_frac * max(xspan, yspan), pad_min)
    long = max(xspan, yspan) + 2 * pad
    w = max(xspan + 2 * pad, 0.5 * long); h = max(yspan + 2 * pad, 0.5 * long)
    return (cx - w / 2, cx + w / 2), (cy - h / 2, cy + h / 2)


def split_validated(pred, H):
    v = min(VALIDATED_H, H)
    return pred[:v + 1], pred[v:]          # solid (<=H400), dashed (>H400), share the joint


def draw_case(ax, hist, gt, pred, H, title, legend=False, small=False):
    solid, dashed = split_validated(pred, H)
    # history
    ax.plot(hist[:, 0], hist[:, 1], "-", color=C_HIST, lw=2.2, zorder=4)
    # GT where it exists (real only)
    if len(gt) > 1:
        ax.plot(gt[:, 0], gt[:, 1], "-", color=C_GT, lw=1.4, alpha=0.9, zorder=3)
        me = max(1, len(gt) // 18)
        ax.plot(gt[::me, 0], gt[::me, 1], "o", color=C_GT, ms=2.6, zorder=3)
    # prediction: solid validated, dashed extrapolation
    ax.plot(solid[:, 0], solid[:, 1], "-", color=C_PRED, lw=2.4, zorder=5)
    ax.plot(dashed[:, 0], dashed[:, 1], "--", color=C_PRED, lw=1.8, dashes=(4, 3), zorder=5)
    # markers: seed point (black), validated edge (orange ring), final (open)
    ax.plot(pred[0, 0], pred[0, 1], "o", color="black", ms=6, zorder=7)
    if H > VALIDATED_H:
        ax.plot(pred[VALIDATED_H, 0], pred[VALIDATED_H, 1], "o", mfc="white",
                mec=C_PRED, mew=1.6, ms=7, zorder=7)
    ax.plot(pred[-1, 0], pred[-1, 1], "s", mfc="white", mec=C_PRED, mew=1.4, ms=6, zorder=7)

    pts = np.vstack([hist, pred] + ([gt] if len(gt) > 1 else []))
    xl, yl = pad_limits(pts); ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=C_GRID, lw=0.5, alpha=0.5); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#bbbbbb"); s.set_linewidth(0.8)
    ax.tick_params(labelsize=6 if small else 7, color="#bbbbbb", length=3)
    ax.set_title(title, fontsize=8 if small else 9.5, loc="left", color="#222222")
    if legend:
        handles = [Line2D([0], [0], color=C_HIST, lw=2.2, label="history (seed)"),
                   Line2D([0], [0], color=C_GT, lw=1.4, marker="o", ms=3, label="GT (real, where available)"),
                   Line2D([0], [0], color=C_PRED, lw=2.4, label="prediction ≤ H400 (validated)"),
                   Line2D([0], [0], color=C_PRED, lw=1.8, ls="--", label="prediction > H400 (extrapolation)")]
        ax.legend(handles=handles, fontsize=6.5, loc="best", framealpha=0.9, edgecolor="#dddddd")


def fig_title(rec, trk, H):
    return (f"{SITE.get(rec, rec)}   ·   track {trk}   ·   H{H}\n"
            f"{EXTRAP_LABEL}  (solid ≤ H400 · dashed > H400)")


# --------------------------------------------------------------------------------------
# Animation (GIF — no ffmpeg available)
# --------------------------------------------------------------------------------------
def make_animation(hist, gt, pred, rec, trk, H, out_path):
    solid, dashed = split_validated(pred, H)
    # decimate prediction for a smooth, light GIF
    nseg = 110
    s_idx = np.linspace(0, len(solid) - 1, min(nseg, len(solid))).astype(int)
    d_idx = np.linspace(0, len(dashed) - 1, min(nseg, len(dashed))).astype(int)
    solid_s = solid[s_idx]; dashed_s = dashed[d_idx]

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    fig.patch.set_facecolor("white")
    pts = np.vstack([hist, pred] + ([gt] if len(gt) > 1 else []))
    xl, yl = pad_limits(pts); ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=C_GRID, lw=0.5, alpha=0.5); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#bbbbbb"); s.set_linewidth(0.8)
    ax.tick_params(labelsize=7, color="#bbbbbb")
    ax.set_title(fig_title(rec, trk, H), fontsize=10, loc="left")
    # GT static context (real)
    if len(gt) > 1:
        ax.plot(gt[:, 0], gt[:, 1], "-", color=C_GT, lw=1.2, alpha=0.55, zorder=2)

    l_hist, = ax.plot([], [], "-", color=C_HIST, lw=2.4, zorder=4)
    l_val, = ax.plot([], [], "-", color=C_PRED, lw=2.6, zorder=5)
    l_ext, = ax.plot([], [], "--", color=C_PRED, lw=1.9, dashes=(4, 3), zorder=5)
    dot, = ax.plot([], [], "o", color="black", ms=6, zorder=7)
    banner = ax.text(0.5, 0.045, "", transform=ax.transAxes, ha="center", va="bottom",
                     fontsize=11, color="#b00050", weight="bold",
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#e0a0bd", alpha=0.9))

    F_HIST, F_VAL, F_EXT, F_HOLD = 12, 46, 46, 16
    total = F_HIST + F_VAL + F_EXT + F_HOLD

    def frame(i):
        if i < F_HIST:
            n = max(2, int(len(hist) * (i + 1) / F_HIST))
            l_hist.set_data(hist[:n, 0], hist[:n, 1])
            dot.set_data([hist[0, 0]], [hist[0, 1]])
        elif i < F_HIST + F_VAL:
            l_hist.set_data(hist[:, 0], hist[:, 1])
            t = (i - F_HIST + 1) / F_VAL
            n = max(2, int(len(solid_s) * t))
            l_val.set_data(solid_s[:n, 0], solid_s[:n, 1])
            banner.set_text("validated prediction (≤ H400)")
        elif i < F_HIST + F_VAL + F_EXT:
            l_val.set_data(solid_s[:, 0], solid_s[:, 1])
            t = (i - F_HIST - F_VAL + 1) / F_EXT
            n = max(1, int(len(dashed_s) * t))
            l_ext.set_data(dashed_s[:n, 0], dashed_s[:n, 1])
            banner.set_text(EXTRAP_LABEL.title())
        else:
            l_ext.set_data(dashed_s[:, 0], dashed_s[:, 1])
            banner.set_text("Exploratory extrapolation beyond validated range")
        return l_hist, l_val, l_ext, dot, banner

    anim = manim.FuncAnimation(fig, frame, frames=total, interval=70, blit=False)
    anim.save(out_path, writer="pillow", fps=15)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df = XCC.load_df()
    wb_all = L.recording_world_bounds(df)
    model, f_sc, t_sc = load_xc_model()
    seeds, starts, wbk, hist, gt_full, tracks = build_inputs(df, wb_all)
    print(f"[sel] {len(tracks)} trajectories: {[t[1] for t in tracks]}")

    rows = []
    preds_by_h = {}
    for H in HORIZONS:
        pred = rollout(model, f_sc, t_sc, seeds, starts, wbk, H)     # (B,H+1,2)
        preds_by_h[H] = pred
        Hdir = DIRS[H]
        for b, (rec, trk, tid) in enumerate(tracks):
            gt = gt_full[b][:H + 1]                                  # real GT clipped to H
            # static figure
            fig, ax = plt.subplots(figsize=(7.6, 6.6))
            fig.patch.set_facecolor("white")
            draw_case(ax, hist[b], gt, pred[b], H, fig_title(rec, trk, H), legend=True)
            fig.tight_layout()
            fig.savefig(Hdir / f"H{H}_{rec}_track{trk}.png", dpi=170, facecolor="white")
            plt.close(fig)
            # diagnostics
            dg = diagnostics(pred[b])
            gt_steps = max(0, len(gt) - 1)
            note = f"GT covers {gt_steps} of {H} future steps"
            if gt_steps >= H:
                note += " (full real overlap)"
            rows.append(dict(horizon=f"H{H}", recording=rec, track_id=trk, **dg, notes=note))
        print(f"[H{H}] {len(tracks)} figures -> {Hdir}")

    # ---- collages per horizon (representative subset, 3x3) ----
    tindex = {(rec, trk): b for b, (rec, trk, _) in enumerate(tracks)}
    for H in HORIZONS:
        pred = preds_by_h[H]
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        fig.patch.set_facecolor("white")
        for ax, key in zip(axes.flat, COLLAGE_TRACKS):
            b = tindex.get(key)
            if b is None:
                ax.axis("off"); continue
            rec, trk = key
            gt = gt_full[b][:H + 1]
            short = f"{SITE.get(rec, rec)}   ·   track {trk}   ·   H{H}"
            draw_case(ax, hist[b], gt, pred[b], H, short,
                      legend=(ax is axes.flat[0]), small=True)
        fig.suptitle(f"{MODEL_NAME} — Long-horizon stress test H{H}   ·   {EXTRAP_LABEL}",
                     fontsize=15, color="#222222")
        fig.tight_layout(rect=[0, 0, 1, 0.975])
        cpath = DIRS[H] / f"H{H}_long_horizon_collage.png"
        fig.savefig(cpath, dpi=150, facecolor="white"); plt.close(fig)
        # mirror collage into figures/
        import shutil; shutil.copy(cpath, FIG_DIR / cpath.name)
    print("[collage] done")

    # ---- hero animations (few) ----
    predA = preds_by_h[ANIM_HORIZON]
    for rec, trk in ANIM_TRACKS:
        b = tindex.get((rec, trk))
        if b is None:
            continue
        gt = gt_full[b][:ANIM_HORIZON + 1]
        out = ANIM_DIR / f"anim_{rec}_track{trk}_H{ANIM_HORIZON}.gif"
        make_animation(hist[b], gt, predA[b], rec, trk, ANIM_HORIZON, out)
        print(f"[anim] {out.name}")

    # ---- diagnostics CSV ----
    cols = ["horizon", "recording", "track_id", "total_pred_path_length", "net_displacement",
            "path_to_net_ratio", "mean_step_length", "final_step_length", "step_decay_ratio",
            "heading_change_total", "heading_net_rotation", "straightness",
            "dominant_failure_mode", "visual_plausibility_label", "notes"]
    sdf = pd.DataFrame(rows)[cols]
    sdf.to_csv(ROOT / "summary_long_horizon_scores.csv", index=False)
    print(f"[csv] {len(sdf)} rows -> summary_long_horizon_scores.csv")

    # quick aggregates to console for the report
    for H in HORIZONS:
        sub = sdf[sdf.horizon == f"H{H}"]
        print(f"\n== H{H} mode counts ==")
        print(sub.dominant_failure_mode.value_counts().to_string())
        print(sub.visual_plausibility_label.value_counts().to_string())
    print(f"\n[ok] outputs -> {ROOT}")


if __name__ == "__main__":
    main()
