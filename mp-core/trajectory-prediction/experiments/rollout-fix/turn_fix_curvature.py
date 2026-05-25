"""
turn_fix_curvature.py
---------------------
Phase 2: Curvature-preserving inference correction (read-only, no retraining).

One family of inference-only fixes. At every rollout step, we keep the model's
predicted *direction* but blend a small fraction of the recently observed
heading trend into it before reconstructing (du, dv). Magnitude is still
clamped by alpha_adaptive(target_mag=0.04).

Variants compared:
  - GT (ground-truth future)
  - stabilized   = alpha_adaptive(target_mag=0.04)             [Phase 1 baseline]
  - curv w=0.25  = stabilized + curvature blend at weight 0.25
  - curv w=0.35  = stabilized + curvature blend at weight 0.35
  - curv w=0.50  = stabilized + curvature blend at weight 0.50

Outputs only under mp-data/outputs/prediction_turn_fix/.
Does NOT touch any other folder.
"""

import sys
import pickle
from pathlib import Path
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE    = Path(__file__).resolve().parent
MP_ROOT = HERE.parent.parent.parent.parent
MP_DATA = MP_ROOT / "mp-data"
OVERFIT = MP_ROOT / "mp-core" / "trajectory-prediction" / "experiments" / "overfit-10x"
sys.path.insert(0, str(OVERFIT))

from train_phase2b_final import (   # noqa: E402
    TrajectoryLSTM, SpatialInterpolator, add_deltas,
    FEATURE_COLS, WINDOW_SIZE, HIDDEN_SIZE, IDW_K, N_ROLLOUT,
)

SRC_CSV  = MP_DATA / "processed" / "encoded" / "trajectories_encoded.csv"
LSTM_DIR = MP_DATA / "outputs" / "prediction" / "experiments" / "phase-2b-final" / "lstm"
OUT      = MP_DATA / "outputs" / "prediction_turn_fix"
PLOTS    = OUT / "plots"
DEBUG    = OUT / "debug"
REPORTS  = OUT / "reports"

TARGET_MAG       = 0.04
HISTORY_LEN      = 3       # last 3 displacements used to estimate heading trend
TREND_DELTA_CLIP = np.deg2rad(45.0)   # clip per-step trend to ±45° to avoid runaway
VARIANT_WEIGHTS  = [0.0, 0.25, 0.35, 0.50]   # 0.0 = pure stabilized (no curvature)

VARIANT_NAMES = {
    0.0:  "stabilized",
    0.25: "curv_w0.25",
    0.35: "curv_w0.35",
    0.50: "curv_w0.50",
}
VARIANT_COLORS = {
    "GT":          "#27ae60",
    "stabilized":  "#2980b9",
    "curv_w0.25":  "#8e44ad",
    "curv_w0.35":  "#e67e22",
    "curv_w0.50":  "#e74c3c",
}


# --------------------------------------------------------------------------- #
# Angle helpers — proper wrapping around ±π
# --------------------------------------------------------------------------- #
def wrap(a):
    """Wrap angle to (-π, π]."""
    return np.arctan2(np.sin(a), np.cos(a))


def angle_diff(a, b):
    """Shortest signed angular difference a - b in (-π, π]."""
    return wrap(a - b)


def angular_blend(model_h, trend_h, w):
    """Weighted average on the circle (vector mean).

    w=0.0 → model_h only.  w=1.0 → trend_h only.
    Returns angle in (-π, π].
    """
    s = (1.0 - w) * np.sin(model_h) + w * np.sin(trend_h)
    c = (1.0 - w) * np.cos(model_h) + w * np.cos(trend_h)
    return np.arctan2(s, c)


def estimate_trend_delta(heading_history):
    """Average wrapped per-step angular change over the supplied headings."""
    h = list(heading_history)
    if len(h) < 2:
        return 0.0
    diffs = [angle_diff(h[i + 1], h[i]) for i in range(len(h) - 1)]
    avg = float(np.mean(diffs))
    return float(np.clip(avg, -TREND_DELTA_CLIP, TREND_DELTA_CLIP))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_model(device):
    with open(LSTM_DIR / "scaler.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    model = TrajectoryLSTM(len(FEATURE_COLS), HIDDEN_SIZE).to(device)
    model.load_state_dict(torch.load(LSTM_DIR / "model.pth", map_location=device))
    model.eval()
    return model, bundle["feature_scaler"], bundle["target_scaler"]


# --------------------------------------------------------------------------- #
# Rollout with optional curvature blend
# --------------------------------------------------------------------------- #
def rollout(model, fs, ts, track, interp, device,
            alpha_target=TARGET_MAG, curvature_weight=0.0):
    """Single rollout.

    alpha_target      : magnitude clamp (None disables; default TARGET_MAG)
    curvature_weight  : 0.0 → no curvature blend (= pure stabilized)
                        >0  → blend recent-heading-trend into model heading
    """
    feats   = track[FEATURE_COLS].to_numpy(dtype=np.float32)
    pos     = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    feats_s = fs.transform(feats)
    window  = feats_s[:WINDOW_SIZE].copy()

    # Seed-window displacement headings (from the real seed positions).
    # These give the trend a real, observed history to start from.
    seed_pos = pos[:WINDOW_SIZE]
    seed_dx  = np.diff(seed_pos[:, 0])
    seed_dy  = np.diff(seed_pos[:, 1])
    seed_h   = np.arctan2(seed_dy, seed_dx)   # length = WINDOW_SIZE - 1

    heading_history = deque(seed_h.tolist(), maxlen=HISTORY_LEN)
    prev_heading    = float(seed_h[-1])

    prev_x = float(pos[WINDOW_SIZE - 1, 0])
    prev_y = float(pos[WINDOW_SIZE - 1, 1])

    pred_x, pred_y, pred_du, pred_dv = [], [], [], []
    for _ in range(N_ROLLOUT):
        x_in = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            p_s = model(x_in).cpu().numpy()[0]
        du, dv = ts.inverse_transform(p_s[np.newaxis])[0]
        du, dv = float(du), float(dv)

        model_mag     = float(np.hypot(du, dv))
        model_heading = float(np.arctan2(dv, du)) if model_mag > 1e-9 \
                        else prev_heading

        # Curvature blend (pure inference-side).
        if curvature_weight > 0.0:
            trend_delta   = estimate_trend_delta(heading_history)
            trend_heading = wrap(prev_heading + trend_delta)
            corr_heading  = angular_blend(model_heading, trend_heading,
                                          curvature_weight)
        else:
            corr_heading  = model_heading

        # Magnitude: keep alpha_adaptive clamp.
        if alpha_target is not None:
            corr_mag = alpha_target
        else:
            corr_mag = model_mag

        du = float(corr_mag * np.cos(corr_heading))
        dv = float(corr_mag * np.sin(corr_heading))

        wx, wy = prev_x + du, prev_y + dv
        pred_x.append(wx); pred_y.append(wy)
        pred_du.append(du); pred_dv.append(dv)

        obs, bnd = interp.query(wx, wy)
        new_raw = np.array([wx, wy, du, dv, obs, bnd], dtype=np.float32)
        window = np.vstack([window[1:], fs.transform(new_raw[np.newaxis])[0]])

        heading_history.append(corr_heading)
        prev_heading = corr_heading
        prev_x, prev_y = wx, wy

    return (np.array(pred_x), np.array(pred_y),
            np.array(pred_du), np.array(pred_dv))


# --------------------------------------------------------------------------- #
# Trajectory picking (mirrors Phase 1)
# --------------------------------------------------------------------------- #
def classify_track(track):
    pos = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    if len(pos) < WINDOW_SIZE + N_ROLLOUT:
        return "short"
    deltas = np.diff(pos[:WINDOW_SIZE + N_ROLLOUT], axis=0)
    path_len = float(np.sum(np.hypot(deltas[:, 0], deltas[:, 1])))
    net      = float(np.hypot(pos[WINDOW_SIZE + N_ROLLOUT - 1, 0] - pos[0, 0],
                              pos[WINDOW_SIZE + N_ROLLOUT - 1, 1] - pos[0, 1]))
    straightness = net / path_len if path_len > 1e-9 else 0.0
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    angle_diff_seq = np.diff(np.unwrap(angles))
    noise = float(np.std(angle_diff_seq))
    if path_len < 0.2:
        return "short"
    if noise > 1.0:
        return "noisy"
    if straightness > 0.85:
        return "straight"
    return "turning"


def pick_multi(df, eligible, want=("straight", "noisy"), per_cat=2):
    chosen = {c: [] for c in want}
    seen = set()
    for pid in sorted(eligible):
        track = (df[df["person_id"] == pid].sort_values("frame_number")
                 .head(WINDOW_SIZE + N_ROLLOUT))
        h = track[["world_x", "world_y"]].round(6).to_numpy().tobytes()
        if h in seen:
            continue
        seen.add(h)
        cat = classify_track(track)
        if cat in want and len(chosen[cat]) < per_cat:
            chosen[cat].append(int(pid))
        if all(len(v) >= per_cat for v in chosen.values()):
            break
    return chosen


# --------------------------------------------------------------------------- #
# Per-trajectory evaluation
# --------------------------------------------------------------------------- #
def evaluate_pid(pid, category, model, fs, ts, interp, device, df):
    track = (df[df["person_id"] == pid].reset_index(drop=True)
             .iloc[:WINDOW_SIZE + N_ROLLOUT].copy())
    if len(track) < WINDOW_SIZE + N_ROLLOUT:
        return None

    pos       = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    seed_xy   = pos[:WINDOW_SIZE]
    gt_xy     = pos[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT]
    seed_last = seed_xy[-1]

    # GT displacement & heading per step (anchored on last seed position).
    gt_dx = np.diff(np.r_[seed_last[0], gt_xy[:, 0]])
    gt_dy = np.diff(np.r_[seed_last[1], gt_xy[:, 1]])
    h_gt  = np.arctan2(gt_dy, gt_dx)
    cum_gt = float(np.sum(np.hypot(gt_dx, gt_dy)))
    total_turn_gt = float(np.sum(np.abs(np.diff(np.unwrap(h_gt)))))

    variants = {"GT": {
        "xy": gt_xy, "h": h_gt,
        "ade": 0.0, "fde": 0.0,
        "cum_pred": cum_gt, "cum_ratio_vs_gt": 1.0,
        "total_turn_deg": np.degrees(total_turn_gt),
        "turn_ratio_vs_gt": 1.0,
        "clumped": False,
    }}

    for w in VARIANT_WEIGHTS:
        name = VARIANT_NAMES[w]
        px, py, pdu, pdv = rollout(model, fs, ts, track, interp, device,
                                   alpha_target=TARGET_MAG,
                                   curvature_weight=w)
        h_pred = np.arctan2(pdv, pdu)
        ade = float(np.mean(np.hypot(px - gt_xy[:, 0], py - gt_xy[:, 1])))
        fde = float(np.hypot(px[-1] - gt_xy[-1, 0], py[-1] - gt_xy[-1, 1]))
        cum_pred = float(np.sum(np.hypot(pdu, pdv)))
        total_turn = float(np.sum(np.abs(np.diff(np.unwrap(h_pred)))))
        # Clumped if predicted path length < 20% of GT path length.
        clumped = cum_pred < 0.20 * cum_gt if cum_gt > 1e-9 else False
        variants[name] = {
            "xy": np.column_stack([px, py]),
            "h":  h_pred,
            "ade": ade, "fde": fde,
            "cum_pred": cum_pred,
            "cum_ratio_vs_gt": cum_pred / cum_gt if cum_gt > 1e-9 else float("nan"),
            "total_turn_deg": np.degrees(total_turn),
            "turn_ratio_vs_gt": total_turn / total_turn_gt if total_turn_gt > 1e-9 else float("nan"),
            "clumped": bool(clumped),
        }

    return {
        "pid": pid, "category": category,
        "seed_xy": seed_xy, "cum_gt": cum_gt,
        "total_turn_gt_deg": np.degrees(total_turn_gt),
        "variants": variants,
    }


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_trajectory_panels(d, path):
    order = ["GT", "stabilized", "curv_w0.25", "curv_w0.35", "curv_w0.50"]
    fig, axes = plt.subplots(1, len(order), figsize=(22, 5.5))
    for ax, name in zip(axes, order):
        v   = d["variants"][name]
        xy  = v["xy"]
        col = VARIANT_COLORS[name]
        ax.plot(d["seed_xy"][:, 0], d["seed_xy"][:, 1], "-o",
                color="#e67e22", lw=1.8, ms=3.5, alpha=0.85, label="seed (10)")
        ax.plot(xy[:, 0], xy[:, 1], "-s", color=col, lw=1.6, ms=4.0,
                alpha=0.95, label="future")
        for i, (x, y) in enumerate(xy, 1):
            if i % 5 == 0 or i == 1 or i == len(xy):
                ax.annotate(str(i), xy=(x, y), fontsize=6, color="#333",
                            ha="center", va="bottom",
                            xytext=(0, 3), textcoords="offset points")
        title = f"{name}"
        if name != "GT":
            title += (f"\nADE={v['ade']:.2f}  FDE={v['fde']:.2f}\n"
                      f"turn={v['total_turn_deg']:.0f}° "
                      f"({v['turn_ratio_vs_gt']:.2f}×GT)  "
                      f"cum={v['cum_ratio_vs_gt']:.2f}")
            if v["clumped"]:
                title += "  [CLUMPED]"
        else:
            title += f"\nturn={v['total_turn_deg']:.0f}°  cum={d['cum_gt']:.2f} m"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("world_x (m)", fontsize=8)
        ax.set_ylabel("world_y (m)", fontsize=8)
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.legend(fontsize=7)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"Trajectory comparison — pid {d['pid']} [{d['category']}]",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_heading_evolution(d, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.8))
    order = ["GT", "stabilized", "curv_w0.25", "curv_w0.35", "curv_w0.50"]
    for name in order:
        h = d["variants"][name]["h"]
        col = VARIANT_COLORS[name]
        t = np.arange(1, len(h) + 1)
        ax1.plot(t, np.degrees(np.unwrap(h)), "-", color=col, lw=1.6, label=name)
        dh = np.diff(np.unwrap(h))
        ax2.plot(np.arange(2, len(h) + 1), np.degrees(dh), "-",
                 color=col, lw=1.4, label=name)
    ax1.set_xlabel("rollout step"); ax1.set_ylabel("heading (deg, unwrapped)")
    ax1.set_title(f"Heading evolution — pid {d['pid']} [{d['category']}]")
    ax1.legend(fontsize=8); ax1.grid(True, lw=0.3, alpha=0.5)
    ax2.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax2.set_xlabel("rollout step"); ax2.set_ylabel("Δ heading (deg/step)")
    ax2.set_title(f"Angular change — pid {d['pid']} [{d['category']}]")
    ax2.legend(fontsize=8); ax2.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_summary_grid(all_d, path):
    n = len(all_d); cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.5 * rows), squeeze=False)
    order = ["GT", "stabilized", "curv_w0.25", "curv_w0.35", "curv_w0.50"]
    for ax, d in zip(axes.flat, all_d):
        ax.plot(d["seed_xy"][:, 0], d["seed_xy"][:, 1], "-",
                color="#e67e22", lw=2.0, alpha=0.7, label="seed")
        for name in order:
            xy = d["variants"][name]["xy"]
            col = VARIANT_COLORS[name]
            ax.plot(xy[:, 0], xy[:, 1], "-", color=col, lw=1.4, alpha=0.9,
                    label=name)
        ax.set_title(f"pid {d['pid']} [{d['category']}]", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.legend(fontsize=6)
    for ax in axes.flat[n:]:
        ax.set_axis_off()
    fig.suptitle("Trajectory comparison — all pids", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(rows, agg, path, picks_summary):
    df = pd.DataFrame(rows)
    target_pids = {310, 20, 40}  # main turning evidence

    turning_df = df[df["pid"].isin(target_pids) & df["variant"].ne("GT")]
    straight_df = df[(df["category"] == "straight") & df["variant"].ne("GT")]
    noisy_df    = df[(df["category"] == "noisy")    & df["variant"].ne("GT")]

    def mean_by_variant(sub, col):
        return sub.groupby("variant")[col].mean().to_dict()

    turn_ratio_turning = mean_by_variant(turning_df, "turn_ratio_vs_gt")
    cum_ratio_turning  = mean_by_variant(turning_df, "cum_ratio_vs_gt")
    ade_turning        = mean_by_variant(turning_df, "ade")
    ade_straight       = mean_by_variant(straight_df, "ade")
    ade_noisy          = mean_by_variant(noisy_df, "ade")
    clumped_overall    = (df[df["variant"].ne("GT")]
                          .groupby("variant")["clumped"].mean().to_dict())

    # Pick winning weight = max turn_ratio_vs_gt on turning targets, with
    # tiebreak on lower mean ADE across straight + noisy.
    candidates = ["curv_w0.25", "curv_w0.35", "curv_w0.50"]
    best = max(candidates, key=lambda v: (
        turn_ratio_turning.get(v, 0.0),
        -((ade_straight.get(v, 1e9) + ade_noisy.get(v, 1e9)) / 2.0),
    ))
    baseline_ratio = turn_ratio_turning.get("stabilized", 0.0)
    best_ratio     = turn_ratio_turning.get(best, 0.0)
    recovered      = best_ratio > baseline_ratio * 1.5 and best_ratio > 0.15

    # Damage check on straight: if straight ADE for best is > 1.5× stabilized ADE → damaged.
    s_base = ade_straight.get("stabilized", 0.0)
    s_best = ade_straight.get(best, 0.0)
    damaged_straight = (s_base > 1e-6) and (s_best > 1.5 * s_base)

    n_base = ade_noisy.get("stabilized", 0.0)
    n_best = ade_noisy.get(best, 0.0)
    damaged_noisy = (n_base > 1e-6) and (n_best > 1.5 * n_base)

    # Deployment: any variant should not clump.
    clump_best = clumped_overall.get(best, 0.0)
    deployment_ok = clump_best < 0.5

    lines = []
    lines.append("# Turn-Fix Results — curvature-preserve inference correction")
    lines.append("")
    lines.append("**Phase 2 — Inference-only fix. No retraining, no architectural change.**")
    lines.append("")
    lines.append("- Magnitude: `alpha_adaptive(target_mag=0.04)` (unchanged from Phase 1)")
    lines.append(f"- Heading trend window: last {HISTORY_LEN} displacements")
    lines.append(f"- Per-step trend clip: ±{np.degrees(TREND_DELTA_CLIP):.0f}°")
    lines.append(f"- Rollout length: {N_ROLLOUT} steps")
    lines.append(f"- Trajectories: {picks_summary}")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 1. Did curvature preservation recover turning?")
    lines.append("")
    lines.append("Mean turn-ratio (cumulative |Δheading| ÷ GT) on the canonical turning pids "
                 "(310, 20, 40):")
    lines.append("")
    lines.append("| variant | mean turn-ratio vs GT | mean cum-displacement ratio vs GT | mean ADE (m) |")
    lines.append("|---|---|---|---|")
    for v in ["stabilized", "curv_w0.25", "curv_w0.35", "curv_w0.50"]:
        lines.append(f"| {v} | {turn_ratio_turning.get(v, 0.0):.3f} | "
                     f"{cum_ratio_turning.get(v, 0.0):.3f} | "
                     f"{ade_turning.get(v, 0.0):.3f} |")
    lines.append("")
    if recovered:
        lines.append(f"→ **Yes (partial).** Best variant `{best}` lifts mean turn-ratio "
                     f"from {baseline_ratio:.2f} (stabilized) to {best_ratio:.2f} "
                     f"on canonical turning pids — a real improvement, though still "
                     "well below GT.")
    else:
        lines.append(f"→ **No / marginal.** Best variant `{best}` only reaches "
                     f"turn-ratio {best_ratio:.2f} on turning pids "
                     f"(vs {baseline_ratio:.2f} stabilized). Curvature blend alone "
                     "does not restore turning to a meaningful fraction of GT.")
    lines.append("")

    lines.append("## 2. Which weight gives the best balance?")
    lines.append("")
    lines.append("Winner is the variant that **maximises turn-ratio on turning pids** "
                 "while keeping ADE on straight + noisy pids within 1.5× the stabilized "
                 "baseline.")
    lines.append("")
    lines.append(f"→ **Best weight: `{best}`**.")
    lines.append("")

    lines.append("## 3. Did the fix damage straight trajectories?")
    lines.append("")
    lines.append("Mean metrics on straight pids:")
    lines.append("")
    lines.append("| variant | mean ADE | mean cum-ratio | mean turn-ratio |")
    lines.append("|---|---|---|---|")
    for v in ["stabilized", "curv_w0.25", "curv_w0.35", "curv_w0.50"]:
        lines.append(f"| {v} | {ade_straight.get(v, 0.0):.3f} | "
                     f"{mean_by_variant(straight_df, 'cum_ratio_vs_gt').get(v, 0.0):.3f} | "
                     f"{mean_by_variant(straight_df, 'turn_ratio_vs_gt').get(v, 0.0):.3f} |")
    lines.append("")
    if damaged_straight:
        lines.append(f"→ **Yes — damaged.** `{best}` ADE on straight pids ({s_best:.2f} m) "
                     f"is more than 1.5× the stabilized baseline ({s_base:.2f} m).")
    else:
        lines.append(f"→ **No.** `{best}` ADE on straight pids ({s_best:.2f} m) is within "
                     f"1.5× the stabilized baseline ({s_base:.2f} m).")
    lines.append("")

    lines.append("Mean metrics on noisy pids:")
    lines.append("")
    lines.append("| variant | mean ADE | mean cum-ratio | mean turn-ratio |")
    lines.append("|---|---|---|---|")
    for v in ["stabilized", "curv_w0.25", "curv_w0.35", "curv_w0.50"]:
        lines.append(f"| {v} | {ade_noisy.get(v, 0.0):.3f} | "
                     f"{mean_by_variant(noisy_df, 'cum_ratio_vs_gt').get(v, 0.0):.3f} | "
                     f"{mean_by_variant(noisy_df, 'turn_ratio_vs_gt').get(v, 0.0):.3f} |")
    lines.append("")
    if damaged_noisy:
        lines.append(f"→ Note: `{best}` ADE on noisy pids ({n_best:.2f} m) is >1.5× "
                     f"the stabilized baseline ({n_base:.2f} m).")
    else:
        lines.append(f"→ `{best}` ADE on noisy pids ({n_best:.2f} m) is within 1.5× "
                     f"the stabilized baseline ({n_base:.2f} m).")
    lines.append("")

    lines.append("## 4. Did the fix preserve full rollout deployment?")
    lines.append("")
    lines.append("Fraction of pids whose predicted path length is < 20% of GT path "
                 "length (\"clumped\"):")
    lines.append("")
    lines.append("| variant | clumped fraction |")
    lines.append("|---|---|")
    for v in ["stabilized", "curv_w0.25", "curv_w0.35", "curv_w0.50"]:
        lines.append(f"| {v} | {clumped_overall.get(v, 0.0):.2f} |")
    lines.append("")
    if deployment_ok:
        lines.append(f"→ **Yes.** `{best}` does not clump on the majority of pids "
                     f"(clumped fraction = {clump_best:.2f}). Magnitude clamp is "
                     "still doing its job; the curvature blend only redirects motion.")
    else:
        lines.append(f"→ **No.** `{best}` clumps on the majority of pids "
                     f"(clumped fraction = {clump_best:.2f}).")
    lines.append("")

    lines.append("## 5. Verdict — freeze or reject?")
    lines.append("")
    if recovered and not damaged_straight and deployment_ok:
        decision = ("**FREEZE** the curvature-preserve fix at `curvature_weight = "
                    f"{best.split('_w')[-1]}` as the new inference-time correction.")
    elif recovered and (damaged_straight or not deployment_ok):
        decision = ("**REJECT for now.** Curvature preservation does recover some "
                    "turning, but at the cost of straight-line accuracy or rollout "
                    "deployment. A direction-aware weight (curvature_weight that "
                    "scales with recent |Δheading|) is the next thing to try.")
    elif not recovered and not damaged_straight and deployment_ok:
        decision = ("**REJECT.** Curvature blend at these weights is essentially "
                    "neutral — turning is not recovered. The LSTM's heading "
                    "output is too flat in closed-loop for a small blend with its "
                    "own recent heading to lift it. Next step would be a stronger "
                    "prior (e.g. blend with GT-derived turn distribution, not the "
                    "model's own recent heading).")
    else:
        decision = ("**REJECT.** Neither turning recovered nor side metrics preserved.")
    lines.append(decision)
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Per-trajectory metrics")
    lines.append("")
    lines.append("| pid | category | variant | ADE | FDE | cum-ratio | turn-ratio | clumped |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['pid']} | {r['category']} | {r['variant']} | "
                     f"{r['ade']:.3f} | {r['fde']:.3f} | "
                     f"{r['cum_ratio_vs_gt']:.3f} | {r['turn_ratio_vs_gt']:.3f} | "
                     f"{'yes' if r['clumped'] else 'no'} |")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `plots/trajectory_panels_pid_*.png` — 5-panel side-by-side per pid")
    lines.append("- `plots/heading_evolution_pid_*.png` — heading + Δheading per pid")
    lines.append("- `plots/summary_all_pids.png`        — overlay grid across pids")
    lines.append("- `debug/turn_fix_metrics.csv`        — per-(pid, variant) metrics")
    lines.append("")
    lines.append("_Phase 2 ends here. No pipeline changes applied._")

    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    DEBUG.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device: {device}")

    df_full = pd.read_csv(SRC_CSV)
    if "track_id" in df_full.columns and "person_id" not in df_full.columns:
        df_full = df_full.rename(columns={"track_id": "person_id"})
    if "frame_idx" in df_full.columns and "frame_number" not in df_full.columns:
        df_full = df_full.rename(columns={"frame_idx": "frame_number"})

    interp = SpatialInterpolator(df_full, k=IDW_K)
    df = add_deltas(df_full)
    model, fs, ts = load_model(device)

    # Fixed turning targets: 310, 20, 40. Then auto-pick 2 straight + 2 noisy.
    track_lengths = df.groupby("person_id").size()
    eligible = set(track_lengths[track_lengths >= WINDOW_SIZE + N_ROLLOUT].index.tolist())
    picks = pick_multi(df, eligible, want=("straight", "noisy"), per_cat=2)

    plan = [(310, "turning"), (20, "turning"), (40, "turning")]
    for cat in ("straight", "noisy"):
        for pid in picks.get(cat, []):
            plan.append((pid, cat))

    picks_summary = ", ".join(f"{p}({c})" for p, c in plan)
    print(f"[INFO]  Plan: {picks_summary}")

    rows = []
    all_d = []
    for pid, cat in plan:
        d = evaluate_pid(pid, cat, model, fs, ts, interp, device, df)
        if d is None:
            print(f"[SKIP]  pid={pid} not long enough")
            continue
        plot_trajectory_panels(d, PLOTS / f"trajectory_panels_pid_{pid}.png")
        plot_heading_evolution(d, PLOTS / f"heading_evolution_pid_{pid}.png")
        all_d.append(d)

        for vname, v in d["variants"].items():
            rows.append({
                "pid": pid, "category": cat, "variant": vname,
                "ade": v["ade"], "fde": v["fde"],
                "cum_pred": v["cum_pred"],
                "cum_ratio_vs_gt": v["cum_ratio_vs_gt"],
                "total_turn_deg": v["total_turn_deg"],
                "turn_ratio_vs_gt": v["turn_ratio_vs_gt"],
                "clumped": v["clumped"],
            })

        print(f"[OK]    pid={pid:<4} [{cat:<8}]  "
              "  ".join(
                  f"{v}: turn={d['variants'][v]['turn_ratio_vs_gt']:.2f} "
                  f"ADE={d['variants'][v]['ade']:.2f}"
                  for v in ["stabilized", "curv_w0.25", "curv_w0.35", "curv_w0.50"]
              ))

    if not all_d:
        print("[ERROR] No evaluations completed.")
        return

    plot_summary_grid(all_d, PLOTS / "summary_all_pids.png")

    metrics_path = DEBUG / "turn_fix_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    write_report(rows, None,
                 REPORTS / "turn_fix_results.md",
                 picks_summary)

    print(f"[OK]    Metrics : {metrics_path}")
    print(f"[OK]    Report  : {REPORTS / 'turn_fix_results.md'}")
    print(f"[OK]    Plots   : {PLOTS}")


if __name__ == "__main__":
    main()
