"""
run_schema_ablation_bridge_overfit10x.py
----------------------------------------
OVERFIT10X — not thesis generalization evidence.

Orchestrator for the schema_ablation_bridge_overfit10x experiment.
Runs:

    1. make_overfit10x_dataset.py
    2. run_overfit10x_ablation.py
    3. visualize_overfit10x_ablation.py

Each step is a subprocess so a failure in one is isolated. After the
run, an artefact summary is printed.

Usage:
    python run_schema_ablation_bridge_overfit10x.py
    python run_schema_ablation_bridge_overfit10x.py --skip-data --skip-train
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent

STEPS = [
    ("make_overfit10x_dataset.py",      "data"),
    ("run_overfit10x_ablation.py",      "train"),
    ("visualize_overfit10x_ablation.py", "viz"),
]

ARTEFACTS = [
    ("dataset csv",        HERE / "schema_ablation_bridge_overfit10x_dataset.csv"),
    ("dataset summary",    HERE / "dataset_summary.md"),
    ("schema json",        HERE / "schema_summary.json"),
    ("model A",            HERE / "A_motion_only"             / "best_model.pth"),
    ("model B",            HERE / "B_motion_position"         / "best_model.pth"),
    ("model C",            HERE / "C_motion_position_spatial" / "best_model.pth"),
    ("model D",            HERE / "D_full_relational"         / "best_model.pth"),
    ("model E (opt)",      HERE / "E_full_affordance"         / "best_model.pth"),
    ("scalers",            HERE / "scalers.pkl"),
    ("loss curves",        HERE / "loss_curves.png"),
    ("drift curves",       HERE / "drift_curves.png"),
    ("ablation results",   HERE / "ablation_results.csv"),
    ("ablation summary",   HERE / "ablation_summary.md"),
    ("visuals dir",        HERE / "ablation_visuals"),
    ("overlay dir",        HERE / "ablation_visuals" / "overlay"),
    ("grid dir",           HERE / "ablation_visuals" / "grid"),
    ("error over time",    HERE / "ablation_visuals" / "error_over_time.png"),
    ("angular over time",  HERE / "ablation_visuals" / "angular_error_over_time.png"),
    ("turn rate bar",      HERE / "ablation_visuals" / "turn_rate_comparison.png"),
    ("visual summary md",  HERE / "ablation_visuals" / "overfit10x_visual_summary.md"),
]


def run(script: Path) -> int:
    print("\n" + "=" * 78)
    print(f"[STEP]  {script.name}")
    print("=" * 78)
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(script)], cwd=str(HERE))
    dt = time.time() - t0
    print(f"[STEP]  {script.name} finished in {dt:.1f}s "
          f"(exit code {proc.returncode})")
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data",  action="store_true",
                    help="skip make_overfit10x_dataset.py")
    ap.add_argument("--skip-train", action="store_true",
                    help="skip run_overfit10x_ablation.py")
    ap.add_argument("--skip-viz",   action="store_true",
                    help="skip visualize_overfit10x_ablation.py")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 78)
    print("OVERFIT10X — not thesis generalization evidence")
    print("=" * 78)

    skip_map = {"data": args.skip_data, "train": args.skip_train,
                "viz":  args.skip_viz}

    for fname, tag in STEPS:
        if skip_map.get(tag):
            print(f"[SKIP]  {fname}")
            continue
        rc = run(HERE / fname)
        if rc != 0:
            print(f"\n[FATAL] {fname} exited with code {rc} — stopping.")
            sys.exit(rc)

    print("\n" + "=" * 78)
    print("[SUMMARY] schema_ablation_bridge_overfit10x artefacts "
          "[OVERFIT10X — not thesis generalization evidence]")
    print("=" * 78)
    for label, p in ARTEFACTS:
        mark = "OK " if p.exists() else "—  "
        print(f"  [{mark}] {label:>20}: {p}")
    print()


if __name__ == "__main__":
    main()
