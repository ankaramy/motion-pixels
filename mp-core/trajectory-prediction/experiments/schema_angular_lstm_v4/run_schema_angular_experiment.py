"""
run_schema_angular_experiment.py
--------------------------------
End-to-end orchestrator for the schema_angular_lstm experiment.

Runs, in order:
    1. make_relational_schema.py
    2. train_angular_lstm.py
    3. visualize_angular_rollouts.py
    4. evaluate_angular_metrics.py

Each step is executed as its own Python subprocess so any failure is
isolated and reported clearly. After the run the orchestrator prints a
short summary of every artefact that was produced.

Usage:
    python run_schema_angular_experiment.py
    python run_schema_angular_experiment.py --skip-schema  # reuse CSV
    python run_schema_angular_experiment.py --skip-train   # reuse model
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _paths import (
    SCHEMA_CSV, MODEL_PTH, SCALER_PKL, LOSS_PNG, TRAIN_SUMMARY_CSV,
    ROLLOUT_CSV, CONTACT_PNG, PER_TRACK_DIR,
    METRICS_CSV, METRICS_MD,
)

STEPS = [
    ("make_relational_schema.py",    "schema"),
    ("train_angular_lstm.py",        "train"),
    ("visualize_angular_rollouts.py","rollout"),
    ("evaluate_angular_metrics.py",  "metrics"),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-schema",  action="store_true",
                    help="skip make_relational_schema.py")
    ap.add_argument("--skip-train",   action="store_true",
                    help="skip train_angular_lstm.py")
    ap.add_argument("--skip-rollout", action="store_true",
                    help="skip visualize_angular_rollouts.py")
    ap.add_argument("--skip-metrics", action="store_true",
                    help="skip evaluate_angular_metrics.py")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    skip_map = {
        "schema":  args.skip_schema,
        "train":   args.skip_train,
        "rollout": args.skip_rollout,
        "metrics": args.skip_metrics,
    }

    for fname, tag in STEPS:
        if skip_map.get(tag):
            print(f"[SKIP]  {fname}")
            continue
        rc = run(HERE / fname)
        if rc != 0:
            print(f"\n[FATAL] {fname} exited with code {rc} — stopping.")
            sys.exit(rc)

    # ---- Final artefact summary ------------------------------------------
    print("\n" + "=" * 78)
    print("[SUMMARY] schema_angular_lstm_v4 artefacts")
    print("=" * 78)
    for label, p in [
        ("schema csv",       SCHEMA_CSV),
        ("model",            MODEL_PTH),
        ("scalers",          SCALER_PKL),
        ("loss curve",       LOSS_PNG),
        ("training summary", TRAIN_SUMMARY_CSV),
        ("rollout csv",      ROLLOUT_CSV),
        ("contact sheet",    CONTACT_PNG),
        ("per-track plots",  PER_TRACK_DIR),
        ("metrics csv",      METRICS_CSV),
        ("summary md",       METRICS_MD),
    ]:
        mark = "OK " if p.exists() else "—  "
        print(f"  [{mark}] {label:>16}: {p}")
    print()


if __name__ == "__main__":
    main()
