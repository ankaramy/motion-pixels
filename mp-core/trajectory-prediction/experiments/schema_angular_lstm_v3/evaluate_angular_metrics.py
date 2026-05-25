"""
evaluate_angular_metrics.py
---------------------------
Per-track + aggregate metrics for the angular LSTM rollouts.

Inputs  : rollout_predictions.csv  (from visualize_angular_rollouts.py)
Outputs : metrics.csv              (one row per track + an 'ALL' row)
          summary.md               (short readable report)

Metrics (all in NORMALIZED u/v space)
-------------------------------------
- ADE                  : mean per-step L2 error between pred and GT
- FDE                  : final-step L2 error
- heading_error_deg    : mean absolute angular error between predicted
                         step direction and GT step direction
- turn_rate_error      : mean absolute error of per-step turn rate
                         (in turn_rate_rel units, [-1, 1])
- cum_heading_ratio    : ratio of total heading change pred / gt
- angularity_pred/gt   : fraction of steps with |turn| > 15 deg
- angularity_ratio     : angularity_pred / angularity_gt
- curvature_corr       : Pearson r of per-step |turn| pred vs gt
- path_length_ratio    : pred path length / gt path length
- seed_continuity_error: gap between the LAST seed point and the FIRST
                         predicted point (should be ~one step)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _paths import ROLLOUT_CSV, METRICS_CSV, METRICS_MD, SHIFT_THRESH_DEG


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def per_step_headings(uv: np.ndarray) -> np.ndarray:
    """Heading at each step from successive (u, v)."""
    if len(uv) < 2:
        return np.array([])
    d = np.diff(uv, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def path_length(uv: np.ndarray) -> float:
    if len(uv) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(uv, axis=0), axis=1).sum())


def evaluate_track(roll: pd.DataFrame) -> dict | None:
    seed = roll[roll["kind"] == "seed"].sort_values("step")
    pred = roll[roll["kind"] == "pred"].sort_values("step")
    gt   = roll[roll["kind"] == "gt"  ].sort_values("step")
    if pred.empty or gt.empty:
        return None

    # Truncate to the overlap horizon.
    n = min(len(pred), len(gt))
    p_uv = pred[["u", "v"]].to_numpy()[:n]
    g_uv = gt  [["u", "v"]].to_numpy()[:n]

    step_err = np.linalg.norm(p_uv - g_uv, axis=1)
    ade = float(step_err.mean())
    fde = float(step_err[-1])

    p_head = per_step_headings(p_uv)
    g_head = per_step_headings(g_uv)
    m = min(len(p_head), len(g_head))
    if m > 0:
        diff = wrap(p_head[:m] - g_head[:m])
        heading_error_deg = float(np.mean(np.abs(np.degrees(diff))))
    else:
        heading_error_deg = float("nan")

    # Turn rates from heading diff.
    if m > 1:
        p_turn = wrap(np.diff(p_head[:m]))
        g_turn = wrap(np.diff(g_head[:m]))
        turn_rate_error = float(np.mean(np.abs(p_turn - g_turn)) / np.pi)
        cum_p = float(np.sum(np.abs(p_turn)))
        cum_g = float(np.sum(np.abs(g_turn)))
        cum_heading_ratio = (cum_p / cum_g) if cum_g > 1e-9 else float("nan")

        thresh_rad = np.deg2rad(SHIFT_THRESH_DEG)
        ang_p = float(np.mean(np.abs(p_turn) > thresh_rad))
        ang_g = float(np.mean(np.abs(g_turn) > thresh_rad))
        ang_ratio = (ang_p / ang_g) if ang_g > 1e-9 else float("nan")

        # Pearson r on per-step |turn|.
        sp = np.abs(p_turn); sg = np.abs(g_turn)
        if sp.std() > 1e-9 and sg.std() > 1e-9:
            curvature_corr = float(np.corrcoef(sp, sg)[0, 1])
        else:
            curvature_corr = float("nan")
    else:
        turn_rate_error = float("nan")
        cum_heading_ratio = float("nan")
        ang_p = ang_g = ang_ratio = float("nan")
        curvature_corr = float("nan")

    pl_p = path_length(p_uv)
    pl_g = path_length(g_uv)
    path_length_ratio = (pl_p / pl_g) if pl_g > 1e-9 else float("nan")

    # Seed continuity: gap between last seed point and first prediction.
    if not seed.empty:
        last_seed = seed[["u", "v"]].to_numpy()[-1]
        first_pred = p_uv[0]
        seed_continuity_error = float(np.linalg.norm(first_pred - last_seed))
    else:
        seed_continuity_error = float("nan")

    return {
        "ADE":                   ade,
        "FDE":                   fde,
        "heading_error_deg":     heading_error_deg,
        "turn_rate_error":       turn_rate_error,
        "cum_heading_ratio":     cum_heading_ratio,
        "angularity_pred":       ang_p,
        "angularity_gt":         ang_g,
        "angularity_ratio":      ang_ratio,
        "curvature_corr":        curvature_corr,
        "path_length_ratio":     path_length_ratio,
        "seed_continuity_error": seed_continuity_error,
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if not ROLLOUT_CSV.exists():
        raise SystemExit(f"[FATAL] missing rollout: {ROLLOUT_CSV}")

    roll = pd.read_csv(ROLLOUT_CSV)
    rows = []
    for tid, g in roll.groupby("track_id"):
        m = evaluate_track(g)
        if m is None:
            continue
        m = {"track_id": int(tid), **m}
        rows.append(m)
    if not rows:
        raise SystemExit("[FATAL] no track metrics computed")

    df = pd.DataFrame(rows)

    # Aggregate row.
    agg = {"track_id": "ALL"}
    for c in df.columns:
        if c == "track_id":
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        agg[c] = float(s.mean())
    df_out = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)
    df_out.to_csv(METRICS_CSV, index=False)
    print(f"[OK]    Wrote {METRICS_CSV.name}")

    # Summary markdown.
    md = ["# Angular LSTM v3 — Rollout Metrics", ""]
    md.append(f"Tracks evaluated: **{len(df)}**")
    md.append("")
    md.append("## Aggregate (mean across tracks)")
    md.append("")
    md.append("| metric | value |")
    md.append("|---|---|")
    for k in [
        "ADE", "FDE",
        "heading_error_deg", "turn_rate_error",
        "cum_heading_ratio",
        "angularity_pred", "angularity_gt", "angularity_ratio",
        "curvature_corr", "path_length_ratio",
        "seed_continuity_error",
    ]:
        v = agg.get(k)
        md.append(f"| `{k}` | "
                  + ("—" if v is None or (isinstance(v, float)
                                          and not np.isfinite(v))
                     else f"{v:.4f}") + " |")
    md.append("")
    md.append("## Per-track table")
    md.append("")
    # Render without external `tabulate` dependency.
    cols = list(df_out.columns)
    md.append("| " + " | ".join(cols) + " |")
    md.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df_out.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("—" if not np.isfinite(v) else f"{v:.4f}")
            else:
                cells.append(str(v))
        md.append("| " + " | ".join(cells) + " |")
    METRICS_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK]    Wrote {METRICS_MD.name}")

    print("\nAggregate metrics:")
    for k, v in agg.items():
        if k == "track_id":
            continue
        print(f"  {k:>24}: {v:.4f}" if isinstance(v, float) else
              f"  {k:>24}: {v}")


if __name__ == "__main__":
    main()
