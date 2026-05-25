"""
diagnose_turning.py
-------------------
Phase 1: Turning Diagnosis (read-only analysis, no retraining, no fixes).

Compares heading evolution between:
  - GT future (ground truth)
  - baseline LSTM rollout
  - alpha_adaptive rollout (target_mag = 0.04)

Outputs only under mp-data/outputs/prediction_turn_diagnosis/.
Does NOT touch prediction/ or prediction_rollout_fix/.

Artifacts:
  plots/heading_evolution_pid_<pid>.png
  plots/angular_change_pid_<pid>.png
  plots/trajectory_diagnosis_pid_<pid>.png
  plots/heading_evolution_summary.png
  plots/angular_change_summary.png
  debug/turn_diagnosis_metrics.csv
  reports/turning_behavior_diagnosis.md
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
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
OUT      = MP_DATA / "outputs" / "prediction_turn_diagnosis"
PLOTS    = OUT / "plots"
DEBUG    = OUT / "debug"
REPORTS  = OUT / "reports"

TARGET_MAG = 0.04


# --------------------------------------------------------------------------- #
# Model + rollout
# --------------------------------------------------------------------------- #
def load_model(device):
    with open(LSTM_DIR / "scaler.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    model = TrajectoryLSTM(len(FEATURE_COLS), HIDDEN_SIZE).to(device)
    model.load_state_dict(torch.load(LSTM_DIR / "model.pth", map_location=device))
    model.eval()
    return model, bundle["feature_scaler"], bundle["target_scaler"]


def rollout(model, fs, ts, track, interp, device, alpha_target=None):
    feats   = track[FEATURE_COLS].to_numpy(dtype=np.float32)
    pos     = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    feats_s = fs.transform(feats)
    window  = feats_s[:WINDOW_SIZE].copy()
    prev_x  = float(pos[WINDOW_SIZE - 1, 0])
    prev_y  = float(pos[WINDOW_SIZE - 1, 1])

    pred_x, pred_y, pred_du, pred_dv = [], [], [], []
    for _ in range(N_ROLLOUT):
        x_in = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            p_s = model(x_in).cpu().numpy()[0]
        du, dv = ts.inverse_transform(p_s[np.newaxis])[0]
        du, dv = float(du), float(dv)

        if alpha_target is not None:
            m = float(np.hypot(du, dv))
            if m > 1e-9:
                s = alpha_target / m
                du, dv = s * du, s * dv

        wx, wy = prev_x + du, prev_y + dv
        pred_x.append(wx); pred_y.append(wy)
        pred_du.append(du); pred_dv.append(dv)

        obs, bnd = interp.query(wx, wy)
        new_raw = np.array([wx, wy, du, dv, obs, bnd], dtype=np.float32)
        window = np.vstack([window[1:], fs.transform(new_raw[np.newaxis])[0]])
        prev_x, prev_y = wx, wy

    return (np.array(pred_x), np.array(pred_y),
            np.array(pred_du), np.array(pred_dv))


# --------------------------------------------------------------------------- #
# Trajectory picking
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
    angle_diff = np.diff(np.unwrap(angles))
    noise = float(np.std(angle_diff))
    if path_len < 0.2:
        return "short"
    if noise > 1.0:
        return "noisy"
    if straightness > 0.85:
        return "straight"
    return "turning"


def pick_multi(df, eligible, want=("turning", "straight", "noisy"), per_cat=2):
    """Return {category: [pid, pid, ...]} with up to `per_cat` per category."""
    chosen = {c: [] for c in want}
    seen_hashes = set()
    for pid in sorted(eligible):
        track = (df[df["person_id"] == pid].sort_values("frame_number")
                 .head(WINDOW_SIZE + N_ROLLOUT))
        h = track[["world_x", "world_y"]].round(6).to_numpy().tobytes()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        cat = classify_track(track)
        if cat in want and len(chosen[cat]) < per_cat:
            chosen[cat].append(int(pid))
        if all(len(v) >= per_cat for v in chosen.values()):
            break
    return chosen


# --------------------------------------------------------------------------- #
# Heading helpers
# --------------------------------------------------------------------------- #
def headings_from_xy(xy_seed_last, xy_future):
    """Return heading_t = atan2(dy, dx) per step using deltas anchored on
    the last seed position. Length == len(xy_future)."""
    x = np.r_[xy_seed_last[0], xy_future[:, 0]]
    y = np.r_[xy_seed_last[1], xy_future[:, 1]]
    dx = np.diff(x)
    dy = np.diff(y)
    return np.arctan2(dy, dx), dx, dy


def headings_from_du_dv(du, dv):
    return np.arctan2(dv, du)


def angular_change(heading):
    """delta_heading_t = wrap(heading_t - heading_{t-1}). Length len(heading)-1."""
    d = np.diff(np.unwrap(heading))
    return d


# --------------------------------------------------------------------------- #
# Per-trajectory diagnosis
# --------------------------------------------------------------------------- #
def diagnose_pid(pid, category, model, fs, ts, interp, device, df):
    track = (df[df["person_id"] == pid].reset_index(drop=True)
             .iloc[:WINDOW_SIZE + N_ROLLOUT].copy())
    if len(track) < WINDOW_SIZE + N_ROLLOUT:
        return None

    pos      = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    seed_xy  = pos[:WINDOW_SIZE]
    gt_xy    = pos[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT]
    seed_last = seed_xy[-1]

    bx, by, bdu, bdv = rollout(model, fs, ts, track, interp, device, alpha_target=None)
    ax, ay, adu, adv = rollout(model, fs, ts, track, interp, device, alpha_target=TARGET_MAG)

    h_gt, _, _ = headings_from_xy(seed_last, gt_xy)
    h_bs       = headings_from_du_dv(bdu, bdv)
    h_al       = headings_from_du_dv(adu, adv)

    dh_gt = angular_change(h_gt)
    dh_bs = angular_change(h_bs)
    dh_al = angular_change(h_al)

    return {
        "pid": pid, "category": category,
        "seed_xy": seed_xy, "gt_xy": gt_xy,
        "baseline_xy": np.column_stack([bx, by]),
        "alpha_xy":    np.column_stack([ax, ay]),
        "h_gt": h_gt, "h_bs": h_bs, "h_al": h_al,
        "dh_gt": dh_gt, "dh_bs": dh_bs, "dh_al": dh_al,
    }


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_heading_evolution(d, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    t = np.arange(1, len(d["h_gt"]) + 1)
    ax.plot(t, np.degrees(np.unwrap(d["h_gt"])), "-o", color="#27ae60",
            lw=2.0, ms=4, label="GT")
    ax.plot(t, np.degrees(np.unwrap(d["h_bs"])), "-s", color="#2980b9",
            lw=1.6, ms=4, label="baseline")
    ax.plot(t, np.degrees(np.unwrap(d["h_al"])), "-^", color="#e74c3c",
            lw=1.6, ms=4, label=f"stabilized (alpha_adaptive, |Δ|={TARGET_MAG})")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("heading (deg, unwrapped)")
    ax.set_title(f"Heading evolution — pid {d['pid']} [{d['category']}]")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_angular_change(d, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    t = np.arange(2, len(d["h_gt"]) + 1)
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.plot(t, np.degrees(d["dh_gt"]), "-o", color="#27ae60",
            lw=1.8, ms=3.5, label="GT")
    ax.plot(t, np.degrees(d["dh_bs"]), "-s", color="#2980b9",
            lw=1.4, ms=3.5, label="baseline")
    ax.plot(t, np.degrees(d["dh_al"]), "-^", color="#e74c3c",
            lw=1.4, ms=3.5, label=f"stabilized (|Δ|={TARGET_MAG})")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("Δ heading (deg / step)")
    ax.set_title(f"Angular change — pid {d['pid']} [{d['category']}]")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_panel(d, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    titles = [
        f"GT future  (pid {d['pid']}, [{d['category']}])",
        "Baseline rollout",
        f"Stabilized rollout (alpha_adaptive, |Δ|={TARGET_MAG})",
    ]
    series = [d["gt_xy"], d["baseline_xy"], d["alpha_xy"]]
    colors = ["#27ae60", "#2980b9", "#e74c3c"]

    for ax, title, fut, c in zip(axes, titles, series, colors):
        ax.plot(d["seed_xy"][:, 0], d["seed_xy"][:, 1], "-o", color="#e67e22",
                lw=2.0, ms=4, alpha=0.85, label="seed (10)")
        ax.plot(fut[:, 0], fut[:, 1], "-s", color=c, lw=1.7, ms=4.5,
                alpha=0.95, label="future")
        for i, (x, y) in enumerate(fut, 1):
            if i % 3 == 0 or i == 1 or i == len(fut):
                ax.annotate(str(i), xy=(x, y), fontsize=6, color="#333",
                            ha="center", va="bottom",
                            xytext=(0, 4), textcoords="offset points")
        ax.set_xlabel("world_x (m)", fontsize=8)
        ax.set_ylabel("world_y (m)", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(f"Trajectory diagnosis — pid {d['pid']}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_summary_grid(diags, out_path, kind):
    n = len(diags)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.6 * rows),
                             squeeze=False)
    for ax, d in zip(axes.flat, diags):
        if kind == "heading":
            t = np.arange(1, len(d["h_gt"]) + 1)
            ax.plot(t, np.degrees(np.unwrap(d["h_gt"])), "-", color="#27ae60",
                    lw=1.8, label="GT")
            ax.plot(t, np.degrees(np.unwrap(d["h_bs"])), "-", color="#2980b9",
                    lw=1.4, label="baseline")
            ax.plot(t, np.degrees(np.unwrap(d["h_al"])), "-", color="#e74c3c",
                    lw=1.4, label="stabilized")
            ax.set_ylabel("heading (deg)")
        else:
            t = np.arange(2, len(d["h_gt"]) + 1)
            ax.axhline(0, color="black", lw=0.5, alpha=0.4)
            ax.plot(t, np.degrees(d["dh_gt"]), "-", color="#27ae60",
                    lw=1.6, label="GT")
            ax.plot(t, np.degrees(d["dh_bs"]), "-", color="#2980b9",
                    lw=1.3, label="baseline")
            ax.plot(t, np.degrees(d["dh_al"]), "-", color="#e74c3c",
                    lw=1.3, label="stabilized")
            ax.set_ylabel("Δ heading (deg/step)")
        ax.set_xlabel("rollout step")
        ax.set_title(f"pid {d['pid']} [{d['category']}]", fontsize=9)
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.legend(fontsize=7)
    # hide empties
    for ax in axes.flat[n:]:
        ax.set_axis_off()
    fig.suptitle("Heading evolution per trajectory" if kind == "heading"
                 else "Angular change per trajectory", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Metrics + report
# --------------------------------------------------------------------------- #
def summarise(d):
    """Per-trajectory summary statistics."""
    dh_gt = d["dh_gt"]; dh_bs = d["dh_bs"]; dh_al = d["dh_al"]
    # total turn = sum of |delta heading|
    total_gt = float(np.sum(np.abs(dh_gt)))
    total_bs = float(np.sum(np.abs(dh_bs)))
    total_al = float(np.sum(np.abs(dh_al)))
    # net signed turn
    net_gt = float(np.sum(dh_gt))
    net_bs = float(np.sum(dh_bs))
    net_al = float(np.sum(dh_al))
    # divergence step: first step where |h_pred - h_gt| > 30 deg
    def divergence_step(h_pred):
        diff = np.abs(np.unwrap(h_pred) - np.unwrap(d["h_gt"]))
        idx = np.where(diff > np.deg2rad(30))[0]
        return int(idx[0]) + 1 if len(idx) else -1
    return {
        "pid": d["pid"], "category": d["category"],
        "total_turn_gt_deg":      np.degrees(total_gt),
        "total_turn_baseline_deg":np.degrees(total_bs),
        "total_turn_alpha_deg":   np.degrees(total_al),
        "net_turn_gt_deg":        np.degrees(net_gt),
        "net_turn_baseline_deg":  np.degrees(net_bs),
        "net_turn_alpha_deg":     np.degrees(net_al),
        "turn_ratio_baseline":    total_bs / total_gt if total_gt > 1e-9 else float("nan"),
        "turn_ratio_alpha":       total_al / total_gt if total_gt > 1e-9 else float("nan"),
        "divergence_step_baseline": divergence_step(d["h_bs"]),
        "divergence_step_alpha":    divergence_step(d["h_al"]),
    }


def write_report(summaries, path):
    df = pd.DataFrame(summaries)
    turning = df[df["category"] == "turning"]

    lines = []
    lines.append("# Turning Behavior Diagnosis")
    lines.append("")
    lines.append("**Phase 1 — Read-only diagnosis. No retraining, no architectural change, no fixes.**")
    lines.append("")
    lines.append(f"- Stabilizer: `alpha_adaptive(target_mag={TARGET_MAG})`")
    lines.append(f"- Rollout length: {N_ROLLOUT} steps")
    lines.append(f"- Trajectories inspected: {len(df)} ({(df['category'].value_counts().to_dict())})")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## A vs B verdict")
    lines.append("")

    # Aggregate signal across turning trajectories.
    if len(turning) > 0:
        mean_ratio_bs = float(turning["turn_ratio_baseline"].mean())
        mean_ratio_al = float(turning["turn_ratio_alpha"].mean())
        mean_total_gt = float(turning["total_turn_gt_deg"].mean())
        mean_total_bs = float(turning["total_turn_baseline_deg"].mean())
        mean_total_al = float(turning["total_turn_alpha_deg"].mean())

        suppression_pct = (1.0 - mean_ratio_al) * 100.0
        very_severe = mean_ratio_al < 0.25
        moderate    = 0.25 <= mean_ratio_al < 0.60

        if very_severe:
            verdict = ("**A — turning is broken.** On turning trajectories the "
                       "stabilized rollout produces "
                       f"{mean_ratio_al:.0%} of the GT cumulative turn "
                       f"(baseline: {mean_ratio_bs:.0%}). The stabilizer "
                       f"suppresses {suppression_pct:.0f}% of angular change "
                       "relative to GT — the rollout no longer turns.")
        elif moderate:
            verdict = ("**A/B — turning is degraded.** On turning trajectories "
                       f"the stabilized rollout retains only {mean_ratio_al:.0%} "
                       f"of GT cumulative turn (baseline: {mean_ratio_bs:.0%}). "
                       "Direction-of-turn is mostly preserved but magnitude is "
                       "noticeably smoothed; this is closer to a smoothing "
                       "artifact than a complete failure.")
        else:
            verdict = ("**B — turning is acceptable but smoother.** Stabilized "
                       f"rollout retains {mean_ratio_al:.0%} of GT cumulative "
                       f"turn (baseline: {mean_ratio_bs:.0%}). The stabilizer "
                       "does smooth angular change but the shape of the turn "
                       "is preserved.")

        lines.append(verdict)
        lines.append("")
        lines.append("### Aggregate on turning trajectories")
        lines.append("")
        lines.append("| metric | GT | baseline | stabilized |")
        lines.append("|---|---|---|---|")
        lines.append(f"| mean total |Δheading| (deg) | {mean_total_gt:.1f} | "
                     f"{mean_total_bs:.1f} | {mean_total_al:.1f} |")
        lines.append(f"| mean turn-ratio vs GT       | 1.000 | "
                     f"{mean_ratio_bs:.3f} | {mean_ratio_al:.3f} |")
        lines.append("")
    else:
        lines.append("_No turning-category trajectories were found in the picked set._")
        lines.append("")

    lines.append("## Where does turning start to diverge?")
    lines.append("")
    lines.append("First rollout step where |heading_pred − heading_GT| > 30°. "
                 "`-1` = never diverges in 30 steps.")
    lines.append("")
    lines.append("| pid | category | divergence (baseline) | divergence (stabilized) |")
    lines.append("|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(f"| {r['pid']} | {r['category']} | "
                     f"{int(r['divergence_step_baseline'])} | "
                     f"{int(r['divergence_step_alpha'])} |")
    lines.append("")

    lines.append("## Does stabilization suppress angular change?")
    lines.append("")
    lines.append("Per-trajectory cumulative |Δheading| vs GT.")
    lines.append("")
    lines.append("| pid | category | GT (deg) | baseline (deg) | stabilized (deg) | "
                 "baseline ratio | stabilized ratio |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(f"| {r['pid']} | {r['category']} | "
                     f"{r['total_turn_gt_deg']:.1f} | "
                     f"{r['total_turn_baseline_deg']:.1f} | "
                     f"{r['total_turn_alpha_deg']:.1f} | "
                     f"{r['turn_ratio_baseline']:.2f} | "
                     f"{r['turn_ratio_alpha']:.2f} |")
    lines.append("")

    lines.append("## Severity assessment")
    lines.append("")
    if len(turning) > 0:
        if very_severe:
            lines.append("- **Severe.** Recommend a fix that preserves angular "
                         "change while constraining magnitude (e.g. angle-only "
                         "blending, hybrid alpha that scales magnitude but "
                         "leaves the predicted heading untouched, or a turn-"
                         "aware target magnitude).")
            lines.append("- The current `alpha_adaptive` rescaling preserves "
                         "the *direction* vector of the LSTM output, so the "
                         "loss of turning comes from the LSTM's *direction* "
                         "output itself once the closed-loop magnitude is "
                         "clamped — i.e. the network is producing nearly "
                         "constant-direction predictions when fed its own "
                         "clamped outputs.")
        elif moderate:
            lines.append("- **Moderate.** Worth a targeted experiment "
                         "(angle-only diagnostic, no fix yet).")
        else:
            lines.append("- **Acceptable.** No fix required for turning. "
                         "Stabilized rollout smooths jitter while preserving "
                         "turn shape.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `plots/heading_evolution_pid_*.png` — per-pid unwrapped heading vs step")
    lines.append("- `plots/angular_change_pid_*.png`   — per-pid Δheading per step")
    lines.append("- `plots/trajectory_diagnosis_pid_*.png` — seed + GT/baseline/stabilized panels")
    lines.append("- `plots/heading_evolution_summary.png` — grid across all picked pids")
    lines.append("- `plots/angular_change_summary.png`   — grid across all picked pids")
    lines.append("- `debug/turn_diagnosis_metrics.csv`   — per-pid metrics")
    lines.append("")
    lines.append("_Stop after diagnosis. No fixes applied._")

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

    track_lengths = df.groupby("person_id").size()
    eligible = set(track_lengths[track_lengths >= WINDOW_SIZE + N_ROLLOUT].index.tolist())

    # Pid 310 first, then 2 turning, 2 straight, 2 noisy.
    picks = pick_multi(df, eligible, want=("turning", "straight", "noisy"), per_cat=2)

    plan = [(310, "anchor")]
    for cat in ("turning", "straight", "noisy"):
        for pid in picks.get(cat, []):
            if pid != 310:
                plan.append((pid, cat))
    print(f"[INFO]  Diagnosing {len(plan)} trajectories: {plan}")

    diags = []
    summaries = []
    for pid, cat in plan:
        d = diagnose_pid(pid, cat, model, fs, ts, interp, device, df)
        if d is None:
            print(f"[SKIP]  pid={pid} not long enough")
            continue
        plot_heading_evolution(d, PLOTS / f"heading_evolution_pid_{pid}.png")
        plot_angular_change(d,    PLOTS / f"angular_change_pid_{pid}.png")
        plot_trajectory_panel(d,  PLOTS / f"trajectory_diagnosis_pid_{pid}.png")
        diags.append(d)
        summaries.append(summarise(d))
        s = summaries[-1]
        print(f"[OK]    pid={pid:<4} [{cat:<8}]  "
              f"turn_GT={s['total_turn_gt_deg']:6.1f}°  "
              f"BL={s['total_turn_baseline_deg']:6.1f}° (ratio={s['turn_ratio_baseline']:.2f})  "
              f"ALPHA={s['total_turn_alpha_deg']:6.1f}° (ratio={s['turn_ratio_alpha']:.2f})")

    if not diags:
        print("[ERROR] No diagnosable trajectories.")
        return

    plot_summary_grid(diags, PLOTS / "heading_evolution_summary.png", kind="heading")
    plot_summary_grid(diags, PLOTS / "angular_change_summary.png",   kind="angular")

    pd.DataFrame(summaries).to_csv(DEBUG / "turn_diagnosis_metrics.csv", index=False)
    write_report(summaries, REPORTS / "turning_behavior_diagnosis.md")

    print(f"[OK]    Wrote {len(diags)} per-pid plot triplets to {PLOTS}")
    print(f"[OK]    Metrics: {DEBUG / 'turn_diagnosis_metrics.csv'}")
    print(f"[OK]    Report : {REPORTS / 'turning_behavior_diagnosis.md'}")


if __name__ == "__main__":
    main()
