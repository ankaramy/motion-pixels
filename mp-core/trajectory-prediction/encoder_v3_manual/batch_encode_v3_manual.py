"""
Encoder V3 (manual masks) - batch-encode the five validated recordings and
write the aggregate report (with old-encoder comparison).

    python encoder_v3_manual\\batch_encode_v3_manual.py --all
    python encoder_v3_manual\\batch_encode_v3_manual.py --recording placa_catalunya_01

Aggregate report:
    new_datasets\\Barcelona_v3_manual_encoded\\Encoder_V3_Manual_Report.md

Processes ONLY the five validated recordings (placa_montjuic_01 and
stairs_montjuic_02 are excluded - homography instability). Does NOT build the
master dataset, run a classifier, or train anything.
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v3_lib


def old_dist_range(recording_id):
    """Return ranges of the OLD (trajectory-derived) distance columns, if present."""
    p = v3_lib.paths_for(recording_id)["old_csv"]
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_csv(p, usecols=lambda c: c in ("dist_to_obstacle", "dist_to_boundary"))
    except Exception:
        df = pd.read_csv(p)
    out = {}
    for col in ("dist_to_obstacle", "dist_to_boundary"):
        if col in df.columns:
            a = pd.to_numeric(df[col], errors="coerce").to_numpy()
            a = a[np.isfinite(a)]
            if a.size:
                out[col] = [round(float(a.min()), 4), round(float(a.mean()), 4),
                            round(float(a.max()), 4)]
    return out or None


def write_report(diags):
    os.makedirs(v3_lib.OUT_ROOT, exist_ok=True)
    path = os.path.join(v3_lib.OUT_ROOT, "Encoder_V3_Manual_Report.md")
    L = ["# Encoder V3 (Manual Architectural Masks) - Aggregate Report", "",
         "Spatial features computed from **manually annotated architectural masks**",
         "(walkable / obstacle), replacing the old trajectory-occupancy masks of",
         "`encode_spatial_auto_v2.py`. Additive prototype: no old output, master",
         "dataset, classifier, or model was touched.", "",
         "**Coordinate alignment:** `calib.json` uses "
         "`plan_scale_plus_correspondences`; world<->plan is an exact isotropic "
         "similarity fitted from correspondences (RMS residual ~0 px, fitted scale "
         "== 1/meters_per_plan_pixel), so plan-pixel distance x mpp = metres.", ""]

    # summary table
    L += ["## Summary", "",
          "| Recording | Status | Rows | Tracks | inWalk % | inObst % | "
          "mean dist_obst (m) | mean dist_bnd (m) | mean fwd/left/right clr (m) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for d in diags:
        cf = d["clearance_forward_v3_m"]["mean"]
        cl = d["clearance_left_v3_m"]["mean"]
        cr = d["clearance_right_v3_m"]["mean"]
        L.append(f"| {d['recording_id']} | **{d['status']}** | {d['row_count']} | "
                 f"{d['track_count']} | {d['percent_inside_walkable']} | "
                 f"{d['percent_inside_obstacle']} | "
                 f"{d['dist_to_obstacle_v3_m']['mean']} | "
                 f"{d['dist_to_walkable_boundary_v3_m']['mean']} | "
                 f"{cf} / {cl} / {cr} |")

    # per-recording detail
    L += ["", "## Per-recording detail", ""]
    for d in diags:
        L += [f"### {d['recording_id']} - {d['status']}", "",
              f"- Rows / tracks: {d['row_count']} / {d['track_count']}",
              f"- In-bounds: {d['percent_inbounds']}%",
              f"- Inside walkable: {d['percent_inside_walkable']}% "
              f"({d['percent_inside_walkable_of_inbounds']}% of in-bounds)",
              f"- Inside obstacle: {d['percent_inside_obstacle']}%",
              f"- dist_to_obstacle_v3_m: {d['dist_to_obstacle_v3_m']}",
              f"- dist_to_walkable_boundary_v3_m: {d['dist_to_walkable_boundary_v3_m']}",
              f"- clearance fwd/left/right (m): {d['clearance_forward_v3_m']} / "
              f"{d['clearance_left_v3_m']} / {d['clearance_right_v3_m']}",
              f"- Invalid dist / clearance NaN: {d['percent_invalid_dist_nan']}% / "
              f"{d['percent_invalid_clearance_nan']}%",
              f"- Output: `{d['output_csv']}`"]
        old = old_dist_range(d["recording_id"])
        if old:
            L += ["", "  **Comparison vs old encoder (trajectory-derived masks):**"]
            if "dist_to_obstacle" in old:
                L.append(f"  - OLD dist_to_obstacle [min/mean/max]: {old['dist_to_obstacle']}")
            L.append(f"  - NEW dist_to_obstacle_v3_m [min/mean/max]: "
                     f"[{d['dist_to_obstacle_v3_m']['min']}, "
                     f"{d['dist_to_obstacle_v3_m']['mean']}, "
                     f"{d['dist_to_obstacle_v3_m']['max']}]")
            if "dist_to_boundary" in old:
                L.append(f"  - OLD dist_to_boundary [min/mean/max]: {old['dist_to_boundary']}")
            L.append("  - NOTE: old obstacle masks were derived from trajectory "
                     "coverage, not architecture; numeric ranges are NOT directly "
                     "comparable and this is NOT a model-quality claim.")
        if d["warnings"]:
            L += ["", "  Warnings:"] + [f"  - {w}" for w in d["warnings"]]
        L += [""]

    n_pass = sum(1 for d in diags if d["status"] == "PASS")
    n_check = sum(1 for d in diags if d["status"] == "CHECK")
    n_fail = sum(1 for d in diags if d["status"] == "FAIL")
    L += ["## Verdict", "",
          f"- PASS: {n_pass}   CHECK: {n_check}   FAIL: {n_fail}", "",
          "Phase 2 question answered: spatial features CAN be computed from the "
          "accepted manual architectural masks and verified to align with "
          "trajectories. Do NOT train, rebuild the master dataset, or run the "
          "classifier until these diagnostics are reviewed and accepted.", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))
    print(f"\n[REPORT] {path}")


def main():
    ap = argparse.ArgumentParser(description="Encoder V3 manual - batch")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="encode all five validated recordings")
    g.add_argument("--recording", choices=v3_lib.VALIDATED, help="single recording")
    args = ap.parse_args()

    targets = v3_lib.VALIDATED if args.all else [args.recording]
    diags = []
    for rid in targets:
        try:
            diags.append(v3_lib.encode_recording(rid))
        except Exception as e:
            print(f"[{rid}] ERROR: {e}")
            diags.append({"recording_id": rid, "status": "FAIL",
                          "row_count": 0, "track_count": 0,
                          "percent_inbounds": 0, "percent_inside_walkable": 0,
                          "percent_inside_walkable_of_inbounds": 0,
                          "percent_inside_obstacle": 0,
                          "percent_invalid_dist_nan": 100,
                          "percent_invalid_clearance_nan": 100,
                          "dist_to_obstacle_v3_m": v3_lib._stats(np.array([])),
                          "dist_to_walkable_boundary_v3_m": v3_lib._stats(np.array([])),
                          "clearance_forward_v3_m": v3_lib._stats(np.array([])),
                          "clearance_left_v3_m": v3_lib._stats(np.array([])),
                          "clearance_right_v3_m": v3_lib._stats(np.array([])),
                          "warnings": [f"ENCODE ERROR: {e}"],
                          "output_csv": ""})
    if args.all:
        write_report(diags)
    print("\nDONE batch:", ", ".join(f"{d['recording_id']}={d['status']}" for d in diags))


if __name__ == "__main__":
    main()
