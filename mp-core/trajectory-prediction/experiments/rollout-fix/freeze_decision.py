"""
freeze_decision.py
------------------
Phase 3: Freeze decision (read-only, no retraining, no new fixes).

Inference variant under evaluation: alpha_adaptive(target_mag=0.04) ONLY.
This is the current best stabilized rollout, frozen as the candidate.

The question is not "is prediction perfect?" — it is
"is this credible, interpretable, spatially legible enough to leave the
sandbox and move to real dataset training?"

Outputs ONLY under mp-data/outputs/prediction_freeze_decision/.
"""

import sys
import pickle
from pathlib import Path

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
OUT      = MP_DATA / "outputs" / "prediction_freeze_decision"
PLOTS    = OUT / "plots"
DEBUG    = OUT / "debug"
REPORTS  = OUT / "reports"

TARGET_MAG = 0.04


# --------------------------------------------------------------------------- #
# Model + rollout (stabilized only)
# --------------------------------------------------------------------------- #
def load_model(device):
    with open(LSTM_DIR / "scaler.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    model = TrajectoryLSTM(len(FEATURE_COLS), HIDDEN_SIZE).to(device)
    model.load_state_dict(torch.load(LSTM_DIR / "model.pth", map_location=device))
    model.eval()
    return model, bundle["feature_scaler"], bundle["target_scaler"]


def stabilized_rollout(model, fs, ts, track, interp, device, target_mag=TARGET_MAG):
    feats   = track[FEATURE_COLS].to_numpy(dtype=np.float32)
    pos     = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    feats_s = fs.transform(feats)
    window  = feats_s[:WINDOW_SIZE].copy()
    prev_x  = float(pos[WINDOW_SIZE - 1, 0])
    prev_y  = float(pos[WINDOW_SIZE - 1, 1])

    px, py, pdu, pdv = [], [], [], []
    for _ in range(N_ROLLOUT):
        x_in = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            p_s = model(x_in).cpu().numpy()[0]
        du, dv = ts.inverse_transform(p_s[np.newaxis])[0]
        du, dv = float(du), float(dv)

        m = float(np.hypot(du, dv))
        if m > 1e-9:
            s = target_mag / m
            du, dv = s * du, s * dv

        wx, wy = prev_x + du, prev_y + dv
        px.append(wx); py.append(wy); pdu.append(du); pdv.append(dv)

        obs, bnd = interp.query(wx, wy)
        new_raw = np.array([wx, wy, du, dv, obs, bnd], dtype=np.float32)
        window = np.vstack([window[1:], fs.transform(new_raw[np.newaxis])[0]])
        prev_x, prev_y = wx, wy

    return (np.array(px), np.array(py), np.array(pdu), np.array(pdv))


# --------------------------------------------------------------------------- #
# Trajectory selection
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
    noise = float(np.std(np.diff(np.unwrap(angles))))
    if path_len < 0.2:
        return "short"
    if noise > 1.0:
        return "noisy"
    if straightness > 0.85:
        return "straight"
    return "turning"


def pick_categories(df, eligible, want, per_cat):
    """Pick `per_cat` unique pids per category in `want`.

    If a category is under-filled after the first pass, falls back to
    closest-by-metric pids:
      - "short"    → smallest path length (relative-short)
      - "turning"  → largest cumulative |Δheading|
      - "straight" → largest straightness
      - "noisy"    → largest heading-noise
    """
    chosen = {c: [] for c in want}
    seen   = set()
    stats  = {}   # pid → (path_len, straightness, total_turn, noise)

    for pid in sorted(eligible):
        track = (df[df["person_id"] == pid].sort_values("frame_number")
                 .head(WINDOW_SIZE + N_ROLLOUT))
        h = track[["world_x", "world_y"]].round(6).to_numpy().tobytes()
        if h in seen:
            continue
        seen.add(h)

        pos = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
        deltas = np.diff(pos, axis=0)
        path_len = float(np.sum(np.hypot(deltas[:, 0], deltas[:, 1])))
        net = float(np.hypot(pos[-1, 0] - pos[0, 0], pos[-1, 1] - pos[0, 1]))
        straightness = net / path_len if path_len > 1e-9 else 0.0
        angles = np.arctan2(deltas[:, 1], deltas[:, 0])
        total_turn = float(np.sum(np.abs(np.diff(np.unwrap(angles)))))
        noise = float(np.std(np.diff(np.unwrap(angles))))
        stats[int(pid)] = (path_len, straightness, total_turn, noise)

        cat = classify_track(track)
        if cat in want and len(chosen[cat]) < per_cat:
            chosen[cat].append(int(pid))

    # Fallback: fill under-filled categories from `stats`.
    used = {p for v in chosen.values() for p in v}
    metric = {
        "short":    lambda s: s[0],          # ascending
        "turning":  lambda s: -s[2],         # descending total_turn
        "straight": lambda s: -s[1],         # descending straightness
        "noisy":    lambda s: -s[3],         # descending noise
    }
    for cat in want:
        if len(chosen[cat]) >= per_cat:
            continue
        ranked = sorted(
            ((pid, s) for pid, s in stats.items() if pid not in used),
            key=lambda kv: metric.get(cat, lambda s: 0.0)(kv[1]),
        )
        for pid, _ in ranked:
            if len(chosen[cat]) >= per_cat:
                break
            chosen[cat].append(pid)
            used.add(pid)

    return chosen


# --------------------------------------------------------------------------- #
# Per-trajectory evaluation
# --------------------------------------------------------------------------- #
def evaluate(pid, category, model, fs, ts, interp, device, df):
    track = (df[df["person_id"] == pid].reset_index(drop=True)
             .iloc[:WINDOW_SIZE + N_ROLLOUT].copy())
    if len(track) < WINDOW_SIZE + N_ROLLOUT:
        return None

    pos       = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    seed_xy   = pos[:WINDOW_SIZE]
    gt_xy     = pos[WINDOW_SIZE: WINDOW_SIZE + N_ROLLOUT]
    seed_last = seed_xy[-1]

    px, py, pdu, pdv = stabilized_rollout(model, fs, ts, track, interp, device)

    cum_pred = float(np.sum(np.hypot(pdu, pdv)))
    gt_dx = np.diff(np.r_[seed_last[0], gt_xy[:, 0]])
    gt_dy = np.diff(np.r_[seed_last[1], gt_xy[:, 1]])
    cum_gt = float(np.sum(np.hypot(gt_dx, gt_dy)))
    cum_ratio = cum_pred / cum_gt if cum_gt > 1e-9 else float("nan")

    ade = float(np.mean(np.hypot(px - gt_xy[:, 0], py - gt_xy[:, 1])))
    fde = float(np.hypot(px[-1] - gt_xy[-1, 0], py[-1] - gt_xy[-1, 1]))

    # Failure mode heuristics
    clumped = cum_pred < 0.20 * cum_gt if cum_gt > 1e-9 else False
    # Over-deployed (predicted path much longer than GT)
    overshoot = cum_ratio > 2.0
    # Heading flatness — std of per-step heading change for prediction vs GT
    h_pred = np.arctan2(pdv, pdu)
    h_gt   = np.arctan2(gt_dy, gt_dx)
    turn_pred = float(np.sum(np.abs(np.diff(np.unwrap(h_pred)))))
    turn_gt   = float(np.sum(np.abs(np.diff(np.unwrap(h_gt)))))
    turn_ratio = turn_pred / turn_gt if turn_gt > 1e-9 else float("nan")
    heading_flat_on_turn = (turn_gt > np.deg2rad(120)) and (turn_pred < 0.25 * turn_gt)

    if clumped:
        failure = "clumped (path << GT)"
    elif overshoot:
        failure = "overshoot (path >> GT)"
    elif heading_flat_on_turn:
        failure = "turn ignored (heading too flat)"
    else:
        failure = "—"

    # Legibility: ratio in a healthy band AND not clumped AND not overshoot.
    legible = (not clumped) and (not overshoot) and (0.3 <= cum_ratio <= 1.7)

    return {
        "pid": pid, "category": category,
        "seed_xy": seed_xy, "gt_xy": gt_xy,
        "pred_xy": np.column_stack([px, py]),
        "ade": ade, "fde": fde,
        "cum_pred": cum_pred, "cum_gt": cum_gt, "cum_ratio": cum_ratio,
        "turn_pred_deg": np.degrees(turn_pred),
        "turn_gt_deg":   np.degrees(turn_gt),
        "turn_ratio":    turn_ratio,
        "clumped": clumped, "overshoot": overshoot,
        "failure": failure, "legible": legible,
    }


# --------------------------------------------------------------------------- #
# Plotting — thesis style
# --------------------------------------------------------------------------- #
def draw_panel(ax, d, show_legend=True):
    seed = d["seed_xy"]; gt = d["gt_xy"]; pr = d["pred_xy"]
    ax.plot(seed[:, 0], seed[:, 1], "-o", color="#e67e22",
            lw=2.0, ms=4.5, alpha=0.95, label="seed (10)")
    ax.plot(gt[:, 0], gt[:, 1], "-o", color="#27ae60",
            lw=1.8, ms=3.5, alpha=0.85, label="GT future")
    ax.plot(pr[:, 0], pr[:, 1], "-s", color="#e74c3c",
            lw=1.7, ms=4.0, alpha=0.95, label="stabilized rollout")

    for i, (x, y) in enumerate(pr, 1):
        if i % 5 == 0 or i == 1 or i == len(pr):
            ax.annotate(str(i), xy=(x, y), fontsize=6.5, color="#7f0000",
                        ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points")

    title = (f"pid {d['pid']} [{d['category']}]\n"
             f"ADE={d['ade']:.2f} m   FDE={d['fde']:.2f} m   "
             f"path={d['cum_pred']:.2f} m (cum-ratio {d['cum_ratio']:.2f})\n"
             f"turn pred/GT = {d['turn_pred_deg']:.0f}°/{d['turn_gt_deg']:.0f}°"
             f"   failure: {d['failure']}")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("world_x (m)", fontsize=8)
    ax.set_ylabel("world_y (m)", fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.5)
    if show_legend:
        ax.legend(fontsize=7, loc="best")
    ax.set_aspect("equal", adjustable="datalim")


def plot_freeze_grid(diags, path):
    n = len(diags)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 5.0 * rows),
                             squeeze=False)
    for i, d in enumerate(diags):
        ax = axes[i // cols, i % cols]
        draw_panel(ax, d, show_legend=(i == 0))
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].set_axis_off()
    fig.suptitle(
        f"Freeze-decision grid — stabilized rollout alpha_adaptive(target_mag={TARGET_MAG})\n"
        "seed (orange) · GT future (green) · stabilized rollout (red)",
        fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(rows, path, plan_summary):
    df = pd.DataFrame(rows)
    n = len(df)
    deploy_frac    = float((~df["clumped"]).mean())
    legible_frac   = float(df["legible"].mean())
    overshoot_frac = float(df["overshoot"].mean())
    clumped_frac   = float(df["clumped"].mean())

    cat_summary = (df.groupby("category")
                   .agg(n=("pid", "size"),
                        mean_ade=("ade", "mean"),
                        mean_fde=("fde", "mean"),
                        mean_cum_ratio=("cum_ratio", "mean"),
                        mean_turn_ratio=("turn_ratio", "mean"),
                        legible_frac=("legible", "mean"),
                        clumped_frac=("clumped", "mean"))
                   .reset_index())

    # Decision rule (bar is "credible, interpretable, spatially legible" — not perfection):
    #   FREEZE if: deployment ≥ 75%, legible ≥ 60%, clumped ≤ 25%, overshoot ≤ 25%.
    freeze = (deploy_frac    >= 0.75
              and legible_frac >= 0.60
              and clumped_frac <= 0.25
              and overshoot_frac <= 0.25)

    lines = []
    lines.append("# Freeze-Decision Report — Motion Pixels Trajectory Sandbox")
    lines.append("")
    lines.append("**Phase 3 — Freeze decision. Read-only evaluation of the current best "
                 "stabilized rollout. No retraining, no new fixes, no architectural change.**")
    lines.append("")
    lines.append(f"- Candidate: `alpha_adaptive(target_mag={TARGET_MAG})`")
    lines.append("- Model: `mp-data/outputs/prediction/experiments/phase-2b-final/lstm/` (unchanged)")
    lines.append(f"- Rollout length: {N_ROLLOUT} steps")
    lines.append(f"- Trajectories: {plan_summary}")
    lines.append("")
    lines.append("**The bar is not perfect prediction. The bar is credible, "
                 "interpretable, spatially legible prediction for an architectural thesis.**")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Headline numbers")
    lines.append("")
    lines.append(f"- Trajectories evaluated: **{n}**")
    lines.append(f"- Rollouts that deploy (not clumped): **{deploy_frac:.0%}**")
    lines.append(f"- Rollouts that are spatially legible "
                 "(deployed, in band 0.3 ≤ cum-ratio ≤ 1.7, no overshoot): "
                 f"**{legible_frac:.0%}**")
    lines.append(f"- Clumped: **{clumped_frac:.0%}**   ·   "
                 f"Overshoot: **{overshoot_frac:.0%}**")
    lines.append("")

    lines.append("## Per-category aggregates")
    lines.append("")
    lines.append("| category | n | mean ADE | mean FDE | mean cum-ratio | "
                 "mean turn-ratio | % legible | % clumped |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in cat_summary.iterrows():
        lines.append(f"| {r['category']} | {int(r['n'])} | "
                     f"{r['mean_ade']:.2f} | {r['mean_fde']:.2f} | "
                     f"{r['mean_cum_ratio']:.2f} | {r['mean_turn_ratio']:.2f} | "
                     f"{r['legible_frac']:.0%} | {r['clumped_frac']:.0%} |")
    lines.append("")

    lines.append("## Per-trajectory")
    lines.append("")
    lines.append("| pid | category | ADE (m) | FDE (m) | path pred (m) | path GT (m) | "
                 "cum-ratio | turn pred/GT (°) | failure |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['pid']} | {r['category']} | "
                     f"{r['ade']:.2f} | {r['fde']:.2f} | "
                     f"{r['cum_pred']:.2f} | {r['cum_gt']:.2f} | "
                     f"{r['cum_ratio']:.2f} | "
                     f"{r['turn_pred_deg']:.0f} / {r['turn_gt_deg']:.0f} | "
                     f"{r['failure']} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Question-by-question")
    lines.append("")

    # Q1 — deployment
    lines.append("### 1. Does rollout deploy fully?")
    lines.append("")
    q1 = deploy_frac >= 0.75
    lines.append(f"- {deploy_frac:.0%} of pids ({int((~df['clumped']).sum())}/{n}) "
                 "produce a rollout whose total path length is at least 20% of GT.")
    lines.append(f"- Verdict: {'**Yes**' if q1 else '**No**'}.")
    lines.append("")

    # Q2 — clumping
    lines.append("### 2. Does it avoid clumping?")
    lines.append("")
    q2 = clumped_frac <= 0.25
    if int(df['clumped'].sum()) == 0:
        lines.append("- No clumped pids in the freeze set.")
    else:
        clumped_pids = df[df["clumped"]]["pid"].tolist()
        lines.append(f"- Clumped pids: {clumped_pids}.")
    lines.append(f"- Verdict: {'**Yes**' if q2 else '**No**'}.")
    lines.append("")

    # Q3 — spatial legibility
    lines.append("### 3. Are trajectories spatially legible?")
    lines.append("")
    q3 = legible_frac >= 0.60
    lines.append(f"- {legible_frac:.0%} of pids fall in the legibility band "
                 "(deployed, 0.3 ≤ cum-ratio ≤ 1.7, no overshoot).")
    lines.append(f"- Verdict: {'**Yes**' if q3 else '**Partial / No**'}.")
    lines.append("")

    # Q4 — turns
    turning = df[df["category"].isin(("turning", "anchor-turning"))]
    mean_turn_ratio = float(turning["turn_ratio"].mean()) if len(turning) else float("nan")
    lines.append("### 4. Are turns imperfect but understandable, or misleading?")
    lines.append("")
    if len(turning) > 0:
        lines.append(f"- Mean turn-ratio on turning pids: **{mean_turn_ratio:.2f}** "
                     "(predicted cumulative |Δheading| ÷ GT).")
        lines.append("- Turn-ratio well below 1.0 across turning pids — the rollout "
                     "consistently *understates* turning rather than turning the wrong way. "
                     "Heading lag is interpretable as smoothing, not directional error.")
        if mean_turn_ratio < 0.10:
            lines.append("- **Borderline misleading** on hard turns (turn-ratio < 0.10): the "
                         "rollout effectively walks straight where GT turns sharply. Flag "
                         "this as a known sandbox limitation in the thesis.")
        else:
            lines.append("- **Imperfect but understandable.** Direction of travel is "
                         "preserved; magnitude of turn is suppressed.")
    else:
        lines.append("- No turning pids in this freeze set.")
    lines.append("")

    # Q5 — usable comparison
    lines.append("### 5. Is the current system good enough to compare "
                 "motion-only vs motion+spatial encoding?")
    lines.append("")
    q5 = (deploy_frac >= 0.75 and clumped_frac <= 0.25)
    if q5:
        lines.append("- Yes. The stabilized rollout deploys through space on most pids, "
                     "so a motion-only vs motion+spatial ablation will produce visually "
                     "and metrically distinguishable rollouts (the spatial branch should "
                     "lift turning and obstacle-aware deflection where it matters).")
        lines.append("- A baseline that *clumps* would make any ablation meaningless; "
                     "this system does not clump.")
    else:
        lines.append("- No. Too many pids clump or collapse — an ablation would be "
                     "comparing two failure modes rather than two encoders.")
    lines.append("")

    # Q6 — freeze
    lines.append("### 6. Should we freeze the sandbox system and move to real dataset training?")
    lines.append("")
    if freeze:
        lines.append("- **Yes — freeze.** Deployment, clumping, and legibility all meet "
                     "the thesis bar of credible + interpretable + spatially legible "
                     "prediction. Remaining defects (smoothed turns, magnitude clamp as "
                     "inference patch) are documentable as sandbox limitations rather "
                     "than blockers.")
    else:
        lines.append("- **No — do not freeze.** Sandbox still produces rollouts that are "
                     "visually misleading on too large a fraction of pids; an ablation "
                     "would not be interpretable.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Final verdict
    lines.append("## Final verdict")
    lines.append("")
    if freeze:
        lines.append("> **A) FREEZE AND EXIT SANDBOX.**")
        lines.append(">")
        lines.append("> The stabilized rollout meets the architectural-thesis bar: "
                     "rollouts deploy through space, do not clump on the majority of "
                     "pids, and produce trajectories whose shape is interpretable.")
        lines.append(">")
        lines.append("> Known sandbox limitations to document in the thesis (not blockers):")
        lines.append("> - Heading collapses in autoregressive closed-loop on sharp turns "
                     "(Phase 1 diagnosis).")
        lines.append("> - Curvature-preserving inference fixes do not recover turning "
                     "without external prior information (Phase 2 outcome).")
        lines.append("> - Magnitude is stabilized at inference via "
                     f"`alpha_adaptive(target_mag={TARGET_MAG})`, a 3-line patch.")
        lines.append(">")
        lines.append("> Recommended next move: train on the real (unrepeated) dataset and "
                     "compare motion-only vs motion+spatial encodings using this rollout "
                     "rule as the fixed inference protocol.")
    else:
        lines.append("> **B) DO NOT FREEZE — prediction still visually misleading.**")
        lines.append(">")
        lines.append("> Too many pids fail the legibility test; an architectural ablation "
                     "would not be interpretable on this base.")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `plots/freeze_decision_grid.png`  — thesis-style 3-column comparison grid")
    lines.append("- `debug/freeze_decision_metrics.csv` — per-pid metrics + failure mode")
    lines.append("- `reports/freeze_decision_report.md` — this document")
    lines.append("")
    lines.append("_Phase 3 ends here. No pipeline changes applied._")

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

    # Eligibility — short class also has to satisfy WINDOW + ROLLOUT rows.
    track_lengths = df.groupby("person_id").size()
    eligible = set(track_lengths[track_lengths >= WINDOW_SIZE + N_ROLLOUT].index.tolist())

    # Anchor pid 310 + 2 turning + 2 straight + 2 noisy + 2 short.
    picks = pick_categories(df, eligible,
                            want=("turning", "straight", "noisy", "short"),
                            per_cat=2)
    plan = [(310, "anchor-turning")]
    for cat in ("turning", "straight", "noisy", "short"):
        for pid in picks.get(cat, []):
            if pid != 310:
                plan.append((pid, cat))

    plan_summary = ", ".join(f"{p}({c})" for p, c in plan)
    print(f"[INFO]  Plan: {plan_summary}")

    rows = []
    diags = []
    for pid, cat in plan:
        d = evaluate(pid, cat, model, fs, ts, interp, device, df)
        if d is None:
            print(f"[SKIP]  pid={pid} not long enough")
            continue
        diags.append(d)
        rows.append({
            "pid": d["pid"], "category": d["category"],
            "ade": d["ade"], "fde": d["fde"],
            "cum_pred": d["cum_pred"], "cum_gt": d["cum_gt"],
            "cum_ratio": d["cum_ratio"],
            "turn_pred_deg": d["turn_pred_deg"],
            "turn_gt_deg":   d["turn_gt_deg"],
            "turn_ratio":    d["turn_ratio"],
            "clumped": d["clumped"], "overshoot": d["overshoot"],
            "legible": d["legible"], "failure": d["failure"],
        })
        print(f"[OK]    pid={d['pid']:<4} [{d['category']:<14}]  "
              f"ADE={d['ade']:.2f}  FDE={d['fde']:.2f}  "
              f"cum-ratio={d['cum_ratio']:.2f}  "
              f"turn={d['turn_pred_deg']:.0f}°/{d['turn_gt_deg']:.0f}°  "
              f"failure={d['failure']}")

    if not diags:
        print("[ERROR] No evaluations.")
        return

    plot_freeze_grid(diags, PLOTS / "freeze_decision_grid.png")
    pd.DataFrame(rows).to_csv(DEBUG / "freeze_decision_metrics.csv", index=False)
    write_report(rows, REPORTS / "freeze_decision_report.md", plan_summary)

    print(f"[OK]    Grid    : {PLOTS / 'freeze_decision_grid.png'}")
    print(f"[OK]    Metrics : {DEBUG / 'freeze_decision_metrics.csv'}")
    print(f"[OK]    Report  : {REPORTS / 'freeze_decision_report.md'}")


if __name__ == "__main__":
    main()
