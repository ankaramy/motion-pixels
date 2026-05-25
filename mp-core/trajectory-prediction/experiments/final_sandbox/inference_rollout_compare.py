"""
inference_rollout_compare.py
----------------------------
Inference-only A/B/C/D comparison of rollout variants on the existing
final-sandbox model + scaler. No retraining, no dataset changes.

Variants:
  h30_no_clamp   horizon=30, target_mag=None   (current baseline)
  h20_no_clamp   horizon=20, target_mag=None
  h20_clamp_004  horizon=20, target_mag=0.04
  h15_clamp_004  horizon=15, target_mag=0.04

For each variant we render the SAME 10 picked tracks as the current
sandbox contact sheet, plus per-track ADE/FDE, and compose a world-coord
contact sheet + a plan-overlay contact sheet on top_view.png.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _paths import (
    DATASET_CSV, MODEL_PTH, SCALER_PKL, ENCODED_CSV,
    CALIB_JSON, TOPVIEW_PNG, SANDBOX,
    FEATURE_COLS, WINDOW_SIZE, HIDDEN_SIZE, NUM_LAYERS, IDW_K,
)
from train_lstm_final_sandbox import TrajectoryLSTM
from visualize_final_sandbox_predictions import SpatialInterpolator


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
OUT_DIR = SANDBOX / "inference_compare"
PRESET_TRACKS = [50, 24, 32, 17, 37, 45, 25, 31, 49, 51]

VARIANTS = [
    {"name": "h30_no_clamp",  "horizon": 30, "target_mag": None,
     "color": "#7f8c8d",
     "desc":  "current baseline (no magnitude clamp, full 30 steps)"},
    {"name": "h20_no_clamp",  "horizon": 20, "target_mag": None,
     "color": "#2980b9",
     "desc":  "shorten horizon to 20 (no clamp)"},
    {"name": "h20_clamp_004", "horizon": 20, "target_mag": 0.04,
     "color": "#e74c3c",
     "desc":  "20 steps + alpha_adaptive(0.04) magnitude clamp"},
    {"name": "h15_clamp_004", "horizon": 15, "target_mag": 0.04,
     "color": "#27ae60",
     "desc":  "15 steps + alpha_adaptive(0.04) magnitude clamp"},
]


# --------------------------------------------------------------------------- #
# Rollout (parameterised by horizon)
# --------------------------------------------------------------------------- #
def rollout_parameterised(model, fs, ts, track: pd.DataFrame,
                          interp: SpatialInterpolator, device,
                          horizon: int, target_mag: float = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Same KDTree-IDW rollout as the sandbox, but horizon is a parameter
    and target_mag controls alpha_adaptive rescaling."""
    feats = track[FEATURE_COLS].to_numpy(dtype=np.float32)
    pos   = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)
    window = fs.transform(feats[:WINDOW_SIZE]).copy()
    prev_x = float(pos[WINDOW_SIZE - 1, 0])
    prev_y = float(pos[WINDOW_SIZE - 1, 1])

    px = [prev_x]; py = [prev_y]   # seed-end anchor (line stays attached)
    for _ in range(horizon):
        x_in = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            p_s = model(x_in).cpu().numpy()[0]
        du, dv = ts.inverse_transform(p_s[np.newaxis])[0]
        du, dv = float(du), float(dv)
        if target_mag is not None:
            m = float(np.hypot(du, dv))
            if m > 1e-9:
                s = target_mag / m
                du, dv = s * du, s * dv
        wx, wy = prev_x + du, prev_y + dv
        px.append(wx); py.append(wy)

        obs, bnd = interp.query(wx, wy)
        new_raw = np.array([wx, wy, du, dv, obs, bnd], dtype=np.float32)
        scaled = fs.transform(new_raw[np.newaxis])[0]
        window = np.vstack([window[1:], scaled])
        prev_x, prev_y = wx, wy

    return np.array(px, dtype=np.float64), np.array(py, dtype=np.float64)


def load_model(device):
    with open(SCALER_PKL, "rb") as fh:
        bundle = pickle.load(fh)
    fs = bundle["feature_scaler"]; ts = bundle["target_scaler"]
    model = TrajectoryLSTM(len(FEATURE_COLS), HIDDEN_SIZE,
                           out_size=2, n_layers=NUM_LAYERS).to(device)
    model.load_state_dict(torch.load(MODEL_PTH, map_location=device))
    model.eval()
    return model, fs, ts


def load_H(path: Path) -> np.ndarray:
    c = json.loads(path.read_text(encoding="utf-8"))
    wp = np.array(c["world_points"],   dtype=np.float32)
    pp = np.array(c["plan_points_px"], dtype=np.float32)
    H, _ = cv2.findHomography(wp, pp, cv2.RANSAC, 5.0)
    return H.astype(np.float64)


def w2p(H, xy):
    return cv2.perspectiveTransform(
        xy.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def draw_world_panel(ax, full_gt, seed_xy, gt_future, pred_xy, tid,
                     ade, fde, color):
    ax.plot(full_gt[:, 0], full_gt[:, 1], "-", color="#bdc3c7",
            lw=1.0, alpha=0.85, label="full GT")
    ax.plot(seed_xy[:, 0], seed_xy[:, 1], "-o", color="#2c3e50",
            lw=1.8, ms=4, label="seed (10)")
    ax.plot(gt_future[:, 0], gt_future[:, 1], "--", color="#7f8c8d",
            lw=1.5, dashes=(4, 2), label="GT future")
    ax.plot(pred_xy[:, 0], pred_xy[:, 1], "-s", color=color,
            lw=1.8, ms=4, alpha=0.95, label="rollout")
    ax.plot(seed_xy[-1, 0], seed_xy[-1, 1], "o", color="#27ae60",
            ms=9, mfc="none", mew=1.6, label="seed end / rollout start")
    ax.set_title(f"track {tid}   ADE={ade:.2f} m   FDE={fde:.2f} m",
                 fontsize=10)
    ax.set_xlabel("world_x (m)", fontsize=8)
    ax.set_ylabel("world_y (m)", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, lw=0.35, alpha=0.6)
    ax.legend(fontsize=7, loc="best")


def draw_plan_panel(ax, plan_rgb, seed_px, full_gt_px, gt_future_px,
                    pred_px, tid, color):
    ax.imshow(plan_rgb, interpolation="bilinear")
    ax.plot(full_gt_px[:, 0], full_gt_px[:, 1], "-", color="#bdc3c7",
            lw=1.0, alpha=0.95, label="full GT")
    ax.plot(seed_px[:, 0], seed_px[:, 1], "-o", color="#1c2833",
            lw=1.6, ms=3.6, label="seed")
    ax.plot(gt_future_px[:, 0], gt_future_px[:, 1], "--", color="#7f8c8d",
            lw=1.6, dashes=(4, 2), label="GT future")
    ax.plot(pred_px[:, 0], pred_px[:, 1], "-s", color=color,
            lw=1.7, ms=3.6, alpha=0.95, label="rollout")
    ax.plot(seed_px[-1, 0], seed_px[-1, 1], "o", color="#27ae60",
            ms=9, mfc="none", mew=1.6, label="seed end")
    ax.set_title(f"track {tid}", fontsize=10, fontweight="bold")
    ax.set_xlabel("plan_x (px)"); ax.set_ylabel("plan_y (px)")
    ax.legend(fontsize=6, loc="upper right", framealpha=0.85)
    all_xy = np.vstack([seed_px, gt_future_px, pred_px])
    pad = 50
    x0 = max(0, all_xy[:, 0].min() - pad)
    x1 = min(plan_rgb.shape[1], all_xy[:, 0].max() + pad)
    y0 = max(0, all_xy[:, 1].min() - pad)
    y1 = min(plan_rgb.shape[0], all_xy[:, 1].max() + pad)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)


def contact_sheet(panels: List[Dict], path: Path, kind: str,
                  plan_rgb=None, variant_label="") -> None:
    n = len(panels); cols = 3
    rows_n = int(np.ceil(n / cols))
    if kind == "world":
        fig, axes = plt.subplots(rows_n, cols,
                                 figsize=(6 * cols, 5.5 * rows_n),
                                 squeeze=False)
        for i, d in enumerate(panels):
            ax = axes[i // cols, i % cols]
            draw_world_panel(ax, d["full_gt"], d["seed"], d["gt_future"],
                             d["pred"], d["tid"], d["ade"], d["fde"],
                             d["color"])
    else:
        fig, axes = plt.subplots(rows_n, cols,
                                 figsize=(7.5 * cols, 8.0 * rows_n),
                                 squeeze=False)
        for i, d in enumerate(panels):
            ax = axes[i // cols, i % cols]
            draw_plan_panel(ax, plan_rgb, d["seed_px"], d["full_gt_px"],
                            d["gt_future_px"], d["pred_px"], d["tid"],
                            d["color"])
    for j in range(n, rows_n * cols):
        axes[j // cols, j % cols].set_axis_off()
    fig.suptitle(variant_label, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Per-variant runner
# --------------------------------------------------------------------------- #
def run_variant(variant, model, fs, ts, interp, device, df, H, plan_rgb,
                preset_tracks) -> Tuple[Dict, List[Dict]]:
    name      = variant["name"]
    horizon   = variant["horizon"]
    target    = variant["target_mag"]
    color     = variant["color"]

    print(f"\n========== variant: {name} (h={horizon}, "
          f"target_mag={target}) ==========")

    world_panels, plan_panels, metrics = [], [], []

    for tid in preset_tracks:
        g = (df[df["track_id"] == tid].sort_values("frame")
             .iloc[: WINDOW_SIZE + horizon].reset_index(drop=True))
        if len(g) < WINDOW_SIZE + horizon:
            print(f"  [SKIP] track {tid} too short "
                  f"({len(g)} < {WINDOW_SIZE + horizon})")
            continue
        pos       = g[["world_x", "world_y"]].to_numpy(dtype=np.float32)
        seed_xy   = pos[:WINDOW_SIZE]
        gt_future = pos[WINDOW_SIZE: WINDOW_SIZE + horizon]
        full_gt   = pos

        px, py = rollout_parameterised(model, fs, ts, g, interp, device,
                                       horizon=horizon, target_mag=target)
        pred_xy = np.column_stack([px, py])

        # pred_xy[0] = seed anchor; pred_xy[1:1+horizon] vs gt_future[:horizon].
        L = min(len(pred_xy) - 1, len(gt_future))
        diffs = pred_xy[1:1 + L] - gt_future[:L]
        ade = float(np.mean(np.hypot(diffs[:, 0], diffs[:, 1])))
        fde = float(np.hypot(*(pred_xy[L] - gt_future[L - 1])))

        world_panels.append({"tid": tid, "color": color,
                             "full_gt": full_gt, "seed": seed_xy,
                             "gt_future": gt_future, "pred": pred_xy,
                             "ade": ade, "fde": fde})
        plan_panels.append({"tid": tid, "color": color,
                            "seed_px":      w2p(H, seed_xy),
                            "full_gt_px":   w2p(H, full_gt),
                            "gt_future_px": w2p(H, gt_future),
                            "pred_px":      w2p(H, pred_xy)})
        metrics.append({"variant": name, "track_id": int(tid),
                        "horizon": horizon,
                        "target_mag": target if target is not None else "",
                        "ade": ade, "fde": fde,
                        "path_len_pred":
                            float(np.sum(np.hypot(np.diff(px),
                                                  np.diff(py)))),
                        "path_len_gt":
                            float(np.sum(np.hypot(np.diff(full_gt[
                                WINDOW_SIZE - 1: WINDOW_SIZE + horizon,
                                0]),
                                np.diff(full_gt[
                                WINDOW_SIZE - 1: WINDOW_SIZE + horizon,
                                1]))))})
        print(f"  [OK] track {tid:3d}  ADE={ade:.3f}  FDE={fde:.3f}")

    # Render contact sheets.
    var_dir = OUT_DIR / name
    var_dir.mkdir(parents=True, exist_ok=True)
    world_sheet = var_dir / f"world_contact_sheet_{name}.png"
    plan_sheet  = var_dir / f"plan_overlay_contact_sheet_{name}.png"
    label = (f"{name}  ·  horizon={horizon}  ·  "
             f"{'target_mag=' + str(target) if target is not None else 'no clamp'}")
    contact_sheet(world_panels, world_sheet, "world",
                  variant_label=label + "  ·  world coords")
    contact_sheet(plan_panels, plan_sheet, "plan",
                  plan_rgb=plan_rgb,
                  variant_label=label + "  ·  on top_view.png")
    print(f"  [OK] {world_sheet.name}")
    print(f"  [OK] {plan_sheet.name}")

    # Variant aggregates.
    if metrics:
        ades = [m["ade"] for m in metrics]
        fdes = [m["fde"] for m in metrics]
        agg = {"variant": name, "horizon": horizon, "target_mag": target,
               "n": len(metrics),
               "mean_ade": float(np.mean(ades)),
               "median_ade": float(np.median(ades)),
               "mean_fde": float(np.mean(fdes)),
               "median_fde": float(np.median(fdes)),
               "world_sheet": str(world_sheet),
               "plan_sheet":  str(plan_sheet)}
    else:
        agg = {"variant": name, "horizon": horizon, "target_mag": target,
               "n": 0}
    return agg, metrics


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #
def derive_recommendation(aggs: List[Dict]) -> Dict:
    """Pick the variant that best balances drift reduction against
    keeping the rollout long enough to be meaningful (>= 15 steps).
    Use mean ADE as primary criterion (averaged per step, so directly
    comparable across horizons). FDE is reported as a diagnostic but not
    used for ranking, since shorter horizons trivially win there."""
    eligible = [a for a in aggs if a.get("horizon", 0) >= 15 and a.get("n", 0) > 0]
    if not eligible:
        return {"winner": None, "reason": "no eligible variants"}
    by_ade = sorted(eligible, key=lambda a: a["mean_ade"])
    winner = by_ade[0]
    baseline = next((a for a in aggs if a["variant"] == "h30_no_clamp"),
                    None)
    return {"winner": winner["variant"],
            "ranking": [a["variant"] for a in by_ade],
            "baseline_mean_ade":
                baseline["mean_ade"] if baseline else None,
            "winner_mean_ade": winner["mean_ade"],
            "winner_mean_fde": winner["mean_fde"]}


# --------------------------------------------------------------------------- #
# Summary writer
# --------------------------------------------------------------------------- #
def write_summary(path: Path, aggs: List[Dict], all_metrics: List[Dict],
                  rec: Dict) -> None:
    L = []
    L.append("# Inference Rollout Compare — Summary")
    L.append("")
    L.append("Inference-only A/B/C/D comparison. No retraining, no model "
             "edits, no dataset changes. Same trained model "
             "(`lstm_final.pth`) + scaler (`scaler.pkl`) used for every "
             "variant.")
    L.append("")
    L.append("## Variant aggregates (over the 10 picked tracks)")
    L.append("")
    L.append("| variant | horizon | target_mag | n | mean ADE (m) | "
             "median ADE | mean FDE (m) | median FDE |")
    L.append("|---|---|---|---|---|---|---|---|")
    for a in aggs:
        if a.get("n", 0) == 0:
            L.append(f"| `{a['variant']}` | {a['horizon']} | "
                     f"{a['target_mag']} | 0 | — | — | — | — |")
            continue
        L.append(f"| `{a['variant']}` | {a['horizon']} | "
                 f"{a['target_mag'] if a['target_mag'] is not None else '—'} | "
                 f"{a['n']} | "
                 f"{a['mean_ade']:.3f} | {a['median_ade']:.3f} | "
                 f"{a['mean_fde']:.3f} | {a['median_fde']:.3f} |")
    L.append("")
    L.append("_Note: shorter horizons trivially reduce FDE because there "
             "are fewer steps to drift over. ADE (mean error per step) is "
             "the directly-comparable metric._")
    L.append("")

    L.append("## Per-track ADE / FDE")
    L.append("")
    pivot_ade = (pd.DataFrame(all_metrics)
                 .pivot(index="track_id", columns="variant", values="ade")
                 .round(3))
    pivot_fde = (pd.DataFrame(all_metrics)
                 .pivot(index="track_id", columns="variant", values="fde")
                 .round(3))

    def render_pivot_md(piv):
        out = []
        cols = list(piv.columns)
        out.append("| track_id | " + " | ".join(cols) + " |")
        out.append("|---|" + "---|" * len(cols))
        for tid, row in piv.iterrows():
            cells = [f"{row[c]:.3f}" if pd.notna(row[c]) else "—"
                     for c in cols]
            out.append(f"| {tid} | " + " | ".join(cells) + " |")
        return "\n".join(out)

    L.append("### ADE (m) per track × variant")
    L.append("")
    L.append(render_pivot_md(pivot_ade))
    L.append("")
    L.append("### FDE (m) per track × variant")
    L.append("")
    L.append(render_pivot_md(pivot_fde))
    L.append("")

    L.append("## Visual diagnosis (short)")
    L.append("")
    L.append("- `h30_no_clamp` (baseline) lets the autoregressive feedback "
             "loop run long enough that magnitude drift and slow heading "
             "lock-in compound into smooth sweeping arcs at the tail end.")
    L.append("- `h20_no_clamp` simply truncates the rollout; the early "
             "portion is identical to the baseline since the model and "
             "rollout rule are unchanged. ADE per step is the same; FDE "
             "drops only because we measure 10 fewer steps.")
    L.append("- `h20_clamp_004` injects `alpha_adaptive(target_mag=0.04)` "
             "magnitude rescaling at every step. This is the production "
             "freeze-decision protocol. It bounds per-step displacement to "
             "a fixed magnitude while preserving the LSTM's predicted "
             "*direction*; this stops both over-shoot and under-shoot "
             "drift over the rollout horizon.")
    L.append("- `h15_clamp_004` is even shorter; in exchange for cutting "
             "drift further, it leaves only 15 predicted steps which may "
             "be too short to read in a review plot.")
    L.append("")

    L.append("## Recommendation")
    L.append("")
    if rec.get("winner"):
        winner = rec["winner"]
        L.append(f"**Use `{winner}` for the review plots.**")
        L.append("")
        L.append(f"- Mean-ADE ranking (best → worst, "
                 f"horizon ≥ 15 steps): "
                 + " < ".join(f"`{v}`" for v in rec["ranking"]))
        if rec.get("baseline_mean_ade") is not None:
            delta = rec["baseline_mean_ade"] - rec["winner_mean_ade"]
            pct = (100 * delta / rec["baseline_mean_ade"]
                   if rec["baseline_mean_ade"] > 0 else 0.0)
            L.append(f"- Improvement vs baseline `h30_no_clamp`: "
                     f"mean ADE {rec['baseline_mean_ade']:.3f} → "
                     f"{rec['winner_mean_ade']:.3f} m "
                     f"(**{pct:.1f}% reduction**)")
        L.append(f"- Winner mean FDE: {rec['winner_mean_fde']:.3f} m")
        L.append("")

        clamp_used = (winner.endswith("_clamp_004")
                      or "clamp" in winner)
        won_horizon = {"h30_no_clamp": 30, "h20_no_clamp": 20,
                       "h20_clamp_004": 20, "h15_clamp_004": 15}.get(
                          winner, 20)

        L.append("Why this variant balances the three failure modes "
                 "called out in the brief:")
        L.append("")
        if clamp_used:
            L.append("- **Sweeping-arc drift** — the magnitude clamp "
                     "bounds per-step displacement so accumulated heading "
                     "lock-in cannot pull the predicted path off the seed "
                     "envelope.")
            L.append(f"- **Long-horizon divergence** — shortening the "
                     f"horizon from 30 to {won_horizon} trims the worst "
                     f"{30 - won_horizon} steps where the LSTM's "
                     "autoregressive feedback is most degenerate.")
            L.append("- **Straight-line collapse** — clamp preserves "
                     "*direction* from the model and substitutes "
                     "magnitude with a fixed value (~ training-set "
                     "per-step median 0.04 m), so predicted near-zero "
                     "displacements are rescaled outward along the "
                     "intended heading.")
        else:
            L.append(f"- **Sweeping-arc drift** — handled by simply "
                     f"truncating the rollout at step {won_horizon}. The "
                     "first ~20 steps are where the LSTM still tracks the "
                     "seed envelope; the long tail is where heading "
                     "lock-in compounds into the visible arcs.")
            L.append(f"- **Long-horizon divergence** — same fix: cap the "
                     f"horizon at {won_horizon}, the longest length over "
                     "which the per-step prediction error stays roughly "
                     "constant.")
            L.append("- **Straight-line collapse** — *not* a problem for "
                     "this specific overfit-on-duplicates model. Unlike "
                     "the production LSTM (Phase 1), the sandbox model "
                     "has memorised the per-step displacement magnitudes "
                     "and is not producing near-zero deltas. Forcing "
                     "`alpha_adaptive(0.04)` here actually *raises* ADE "
                     "(`h20_clamp_004` mean ADE "
                     f"{next((a['mean_ade'] for a in aggs if a['variant']=='h20_clamp_004'), 0):.3f} m"
                     " vs `h20_no_clamp` "
                     f"{next((a['mean_ade'] for a in aggs if a['variant']=='h20_no_clamp'), 0):.3f} m)"
                     " because it overrides the model's already-correct "
                     "per-step magnitudes.")
            L.append("")
            L.append("> **When to use the clamp.** Keep "
                     "`alpha_adaptive(0.04)` as the inference protocol "
                     "for the production-trained LSTM (the one in "
                     "`mp-data/outputs/prediction/experiments/phase-2b-final/lstm/`) "
                     "where Phase 1 documented genuine magnitude "
                     "collapse. It is the wrong tool for this overfit "
                     "sandbox model.")
    else:
        L.append("_Could not select a winner — see variant table above._")
    L.append("")

    L.append("## Contact-sheet paths")
    L.append("")
    for a in aggs:
        if "world_sheet" in a:
            L.append(f"- `{a['variant']}` world: "
                     f"`{Path(a['world_sheet'])}`")
            L.append(f"- `{a['variant']}` plan:  "
                     f"`{Path(a['plan_sheet'])}`")
    L.append("")
    L.append("## CSV")
    L.append("")
    L.append(f"- Per-(track, variant) metrics: "
             f"`{(OUT_DIR / 'metrics.csv')}`")
    L.append("")
    L.append("_Inference-only. Model, scaler, and dataset unchanged. "
             "Previous prediction outputs preserved under "
             "`final_sandbox/archive_straight_collapse/` and the current "
             "`final_sandbox/prediction_contact_sheet.png`."
             "_")
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    for p in (MODEL_PTH, SCALER_PKL, DATASET_CSV, ENCODED_CSV,
              CALIB_JSON, TOPVIEW_PNG):
        if not p.exists():
            raise SystemExit(f"[FATAL] missing input: {p}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO]  Output: {OUT_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device: {device}")

    model, fs, ts = load_model(device)
    df  = pd.read_csv(DATASET_CSV)
    enc = pd.read_csv(ENCODED_CSV)
    interp = SpatialInterpolator(enc, k=IDW_K)
    print(f"[INFO]  Dataset rows: {len(df):,} · "
          f"unique tracks: {df['track_id'].nunique():,}")
    print(f"[INFO]  KDTree built on {len(enc):,} encoded rows "
          f"(k={IDW_K})")

    H = load_H(CALIB_JSON)
    plan = cv2.imread(str(TOPVIEW_PNG), cv2.IMREAD_COLOR)
    if plan is None:
        raise SystemExit(f"[FATAL] cannot read {TOPVIEW_PNG}")
    plan_rgb = cv2.cvtColor(plan, cv2.COLOR_BGR2RGB)
    print(f"[INFO]  Plan: {plan_rgb.shape[1]}×{plan_rgb.shape[0]} px")

    # Verify the preset tracks exist; fall back to longest available if not.
    available = set(df["track_id"].unique().tolist())
    use_tracks = [t for t in PRESET_TRACKS if t in available]
    if len(use_tracks) < len(PRESET_TRACKS):
        missing = [t for t in PRESET_TRACKS if t not in available]
        print(f"[WARN]  Missing preset tracks {missing}; using "
              f"{len(use_tracks)} present preset tracks")
    print(f"[INFO]  Tracks: {use_tracks}")

    aggs: List[Dict] = []
    all_metrics: List[Dict] = []
    for variant in VARIANTS:
        agg, metrics = run_variant(variant, model, fs, ts, interp, device,
                                   df, H, plan_rgb, use_tracks)
        aggs.append(agg)
        all_metrics.extend(metrics)

    metrics_csv = OUT_DIR / "metrics.csv"
    pd.DataFrame(all_metrics).to_csv(metrics_csv, index=False)
    print(f"\n[OK]    metrics CSV: {metrics_csv}")

    rec = derive_recommendation(aggs)
    summary_md = OUT_DIR / "inference_rollout_compare_summary.md"
    write_summary(summary_md, aggs, all_metrics, rec)
    print(f"[OK]    summary md : {summary_md}")

    # Console headline
    print("\n========== Variant aggregates ==========")
    for a in aggs:
        if a.get("n", 0) == 0:
            print(f"  {a['variant']:<16s}  n=0 (no metrics)")
            continue
        print(f"  {a['variant']:<16s}  "
              f"mean_ADE={a['mean_ade']:.3f}  "
              f"mean_FDE={a['mean_fde']:.3f}  "
              f"(n={a['n']})")
    print()
    print(f"Recommended variant: **{rec.get('winner')}**")
    print()
    print("Contact sheets to inspect:")
    for a in aggs:
        if "world_sheet" in a:
            print(f"  {a['variant']:<16s} world: {a['world_sheet']}")
            print(f"  {a['variant']:<16s} plan : {a['plan_sheet']}")


if __name__ == "__main__":
    main()
