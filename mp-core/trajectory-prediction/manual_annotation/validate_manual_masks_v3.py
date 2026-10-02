"""
MOTION PIXELS - Manual Mask Validation (V3)
===========================================

Checks manual walkable/obstacle masks produced by annotate_masks_v3.py.
Read-only: does not modify masks, encoders, datasets, or models.

USAGE
-----
    python manual_annotation/validate_manual_masks_v3.py --recording placa_catalunya_01
    python manual_annotation/validate_manual_masks_v3.py --all

For a single recording the report is written to:
    mp-data/annotations/manual_masks_v3/<recording_id>/validation_report.md
For --all:
    mp-data/annotations/manual_masks_v3/manual_mask_validation_report.md

Checks per recording:
    - masks exist / missing
    - image size
    - walkable coverage %
    - obstacle coverage %
    - overlap pixels (MUST be 0)
    - empty mask warnings
"""

import os
import argparse
from datetime import datetime

import cv2
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_ROOT = os.path.join(REPO_ROOT, "mp-data", "annotations", "manual_masks_v3")

VALIDATED = [
    "stairs_montjuic_01",
    "red_bridge_combined_01",
    "esplanade_espanya_01",
    "placa_espanya_01",
    "placa_catalunya_01",
]


def check(recording_id):
    d = os.path.join(OUT_ROOT, recording_id)
    wp = os.path.join(d, "walkable_mask_v3_manual.png")
    op = os.path.join(d, "obstacle_mask_v3_manual.png")
    r = {"recording_id": recording_id, "walkable_exists": os.path.exists(wp),
         "obstacle_exists": os.path.exists(op), "warnings": [], "ok": False}

    if not (r["walkable_exists"] and r["obstacle_exists"]):
        r["warnings"].append("MISSING mask file(s) - has this recording been annotated?")
        return r

    w = cv2.imread(wp, cv2.IMREAD_GRAYSCALE)
    o = cv2.imread(op, cv2.IMREAD_GRAYSCALE)
    if w is None or o is None:
        r["warnings"].append("Mask file unreadable.")
        return r
    if w.shape != o.shape:
        r["warnings"].append(f"Size mismatch walkable{w.shape} vs obstacle{o.shape}.")
        return r

    h, wd = w.shape
    total = h * wd
    wb, ob = w > 127, o > 127
    wpx, opx = int(wb.sum()), int(ob.sum())
    overlap = int(np.logical_and(wb, ob).sum())

    r.update({
        "image_size": [wd, h],
        "walkable_pixels": wpx,
        "obstacle_pixels": opx,
        "walkable_coverage_percent": round(100 * wpx / total, 3),
        "obstacle_coverage_percent": round(100 * opx / total, 3),
        "annotated_coverage_percent": round(100 * (wpx + opx) / total, 3),
        "overlap_pixels": overlap,
    })
    if overlap > 0:
        r["warnings"].append(f"OVERLAP = {overlap}px (must be 0).")
    if wpx == 0:
        r["warnings"].append("Walkable mask is EMPTY.")
    if opx == 0:
        r["warnings"].append("Obstacle mask is EMPTY.")
    r["ok"] = overlap == 0 and wpx > 0 and opx > 0
    return r


def fmt(r):
    lines = [f"### {r['recording_id']}"]
    if not (r["walkable_exists"] and r["obstacle_exists"]):
        lines.append("- **Status:** MISSING")
        lines.append(f"  - walkable mask present: {r['walkable_exists']}")
        lines.append(f"  - obstacle mask present: {r['obstacle_exists']}")
        for warn in r["warnings"]:
            lines.append(f"  - WARNING: {warn}")
        return "\n".join(lines) + "\n"
    if "image_size" not in r:
        lines.append("- **Status:** UNREADABLE")
        for warn in r["warnings"]:
            lines.append(f"  - WARNING: {warn}")
        return "\n".join(lines) + "\n"

    lines += [
        f"- **Status:** {'PASS' if r['ok'] else 'CHECK'}",
        f"- Image size: {r['image_size'][0]} x {r['image_size'][1]}",
        f"- Walkable coverage: {r['walkable_coverage_percent']}% ({r['walkable_pixels']} px)",
        f"- Obstacle coverage: {r['obstacle_coverage_percent']}% ({r['obstacle_pixels']} px)",
        f"- Annotated total: {r['annotated_coverage_percent']}%",
        f"- Overlap pixels: {r['overlap_pixels']} (must be 0)",
    ]
    for warn in r["warnings"]:
        lines.append(f"  - WARNING: {warn}")
    return "\n".join(lines) + "\n"


def write_report(results, path, title):
    out = [f"# {title}", "", f"_Generated: {datetime.now().isoformat(timespec='seconds')}_", ""]
    n_pass = sum(1 for r in results if r.get("ok"))
    out.append(f"**Summary:** {n_pass}/{len(results)} recordings PASS "
               f"(masks present, non-empty, zero overlap).")
    out.append("")
    # table
    out.append("| Recording | Status | Walkable % | Obstacle % | Overlap | Warnings |")
    out.append("|---|---|---|---|---|---|")
    for r in results:
        if "image_size" in r:
            status = "PASS" if r["ok"] else "CHECK"
            out.append(f"| {r['recording_id']} | {status} | "
                       f"{r['walkable_coverage_percent']} | "
                       f"{r['obstacle_coverage_percent']} | "
                       f"{r['overlap_pixels']} | {len(r['warnings'])} |")
        else:
            out.append(f"| {r['recording_id']} | MISSING | - | - | - | "
                       f"{len(r['warnings'])} |")
    out.append("")
    for r in results:
        out.append(fmt(r))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(out))
    print(f"[REPORT] {path}")


def main():
    ap = argparse.ArgumentParser(description="Validate manual masks (V3).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--recording", help="single recording id")
    g.add_argument("--all", action="store_true", help="validate all five recordings")
    args = ap.parse_args()

    if args.all:
        results = [check(rid) for rid in VALIDATED]
        for r in results:
            print(fmt(r))
        write_report(results, os.path.join(OUT_ROOT, "manual_mask_validation_report.md"),
                     "Manual Mask Validation Report (all validated recordings)")
    else:
        r = check(args.recording)
        print(fmt(r))
        write_report([r], os.path.join(OUT_ROOT, args.recording, "validation_report.md"),
                     f"Manual Mask Validation - {args.recording}")


if __name__ == "__main__":
    main()
