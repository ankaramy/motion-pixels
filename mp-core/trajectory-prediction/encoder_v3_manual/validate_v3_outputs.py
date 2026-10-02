"""
Encoder V3 (manual masks) - validate produced outputs.

    python encoder_v3_manual\\validate_v3_outputs.py --recording placa_catalunya_01
    python encoder_v3_manual\\validate_v3_outputs.py --all

Read-only. Checks that each recording's V3 outputs exist and are well formed:
expected feature columns present, distances/clearances finite for most rows,
inside_walkable high, inside_obstacle low, and surfaces the recorded status.
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v3_lib

REQUIRED_COLS = [
    "dist_to_obstacle_v3_m", "dist_to_walkable_boundary_v3_m",
    "inside_walkable_v3", "inside_obstacle_v3",
    "clearance_forward_v3_m", "clearance_left_v3_m", "clearance_right_v3_m",
    "clearance_asymmetry_v3",
    "obstacle_bearing_sin_v3", "obstacle_bearing_cos_v3",
    "boundary_bearing_sin_v3", "boundary_bearing_cos_v3",
]


def check(recording_id):
    P = v3_lib.paths_for(recording_id)
    outdir = P["outdir"]
    csv = os.path.join(outdir, "trajectories_encoded_v3.csv")
    diagp = os.path.join(outdir, "feature_diagnostics.json")
    r = {"recording_id": recording_id, "exists": os.path.exists(csv),
         "problems": [], "ok": False}
    if not r["exists"]:
        r["problems"].append("trajectories_encoded_v3.csv MISSING - run encoder first.")
        return r

    df = pd.read_csv(csv)
    r["rows"] = len(df)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        r["problems"].append(f"missing feature columns: {missing}")
    r["pct_inside_walkable"] = round(100 * (df.get("inside_walkable_v3", 0) == 1).mean(), 3)
    r["pct_inside_obstacle"] = round(100 * (df.get("inside_obstacle_v3", 0) == 1).mean(), 3)
    do = pd.to_numeric(df.get("dist_to_obstacle_v3_m"), errors="coerce")
    cf = pd.to_numeric(df.get("clearance_forward_v3_m"), errors="coerce")
    r["pct_dist_finite"] = round(100 * np.isfinite(do).mean(), 3)
    r["pct_clear_finite"] = round(100 * np.isfinite(cf).mean(), 3)

    # bearing sanity: sin^2+cos^2 ~ 1 where valid (or 0,0 where flagged)
    bs = pd.to_numeric(df.get("obstacle_bearing_sin_v3"), errors="coerce").to_numpy()
    bc = pd.to_numeric(df.get("obstacle_bearing_cos_v3"), errors="coerce").to_numpy()
    mag = bs**2 + bc**2
    bad_unit = np.isfinite(mag) & (mag > 0.01) & (np.abs(mag - 1) > 0.05)
    if bad_unit.any():
        r["problems"].append(f"{int(bad_unit.sum())} rows have non-unit obstacle bearing.")

    if os.path.exists(diagp):
        r["recorded_status"] = json.load(open(diagp)).get("status")

    if r["pct_dist_finite"] < 50:
        r["problems"].append(f"only {r['pct_dist_finite']}% rows have finite distance.")
    if r["pct_inside_walkable"] < 1:
        r["problems"].append("inside_walkable ~ 0% - likely coordinate mismatch.")
    r["ok"] = not r["problems"]
    return r


def fmt(r):
    L = [f"### {r['recording_id']}"]
    if not r["exists"]:
        L.append("- **MISSING** outputs")
        for p in r["problems"]:
            L.append(f"  - {p}")
        return "\n".join(L) + "\n"
    L += [
        f"- Recorded status: {r.get('recorded_status','?')}",
        f"- Validation: {'OK' if r['ok'] else 'PROBLEMS'}",
        f"- Rows: {r['rows']}",
        f"- Inside walkable: {r['pct_inside_walkable']}%   inside obstacle: {r['pct_inside_obstacle']}%",
        f"- Finite distance: {r['pct_dist_finite']}%   finite clearance: {r['pct_clear_finite']}%",
    ]
    for p in r["problems"]:
        L.append(f"  - PROBLEM: {p}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Validate Encoder V3 outputs")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true")
    g.add_argument("--recording", choices=v3_lib.VALIDATED)
    args = ap.parse_args()

    targets = v3_lib.VALIDATED if args.all else [args.recording]
    results = [check(t) for t in targets]
    for r in results:
        print(fmt(r))
    n_ok = sum(1 for r in results if r["ok"])
    print(f"VALIDATION: {n_ok}/{len(results)} OK")


if __name__ == "__main__":
    main()
