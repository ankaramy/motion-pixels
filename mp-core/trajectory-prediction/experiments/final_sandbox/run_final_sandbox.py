"""
run_final_sandbox.py
--------------------
Orchestrator. Runs dataset → train → predictions → plan overlay in order.
Skips a step if its primary output already exists, unless --force is given.
Writes the final sandbox summary at the end.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _paths import (
    DATASET_CSV, DATASET_MD, MODEL_PTH, SCALER_PKL, LOSS_PNG, TRAIN_MD,
    PRED_PLOTS_DIR, PRED_SHEET, PLAN_PLOTS_DIR, PLAN_SHEET, FINAL_MD,
    SANDBOX,
)


STEPS = [
    ("dataset",  "make_final_sandbox_dataset",                DATASET_CSV),
    ("train",    "train_lstm_final_sandbox",                  MODEL_PTH),
    ("predict",  "visualize_final_sandbox_predictions",       PRED_SHEET),
    ("overlay",  "overlay_final_sandbox_predictions_on_plan", PLAN_SHEET),
]


def run_module(name: str) -> bool:
    print(f"\n========== {name} ==========")
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u",
                        str(HERE / f"{name}.py")],
                       cwd=str(HERE))
    elapsed = time.time() - t0
    ok = (r.returncode == 0)
    print(f"---------- {name} {'OK' if ok else 'FAILED'} "
          f"({elapsed:.1f}s) ----------")
    return ok


def write_final_summary(state: dict) -> None:
    L = []
    L.append("# Final Sandbox — Summary")
    L.append("")
    L.append("Isolated overfit-style review experiment using the rerun's "
             "calibrated trajectories and v2.1C spatial encoding. Nothing "
             "outside `final_sandbox/` was modified.")
    L.append("")
    L.append("## Pipeline")
    L.append("")
    L.append("| step | script | primary output | status |")
    L.append("|---|---|---|---|")
    for label, mod, out in STEPS:
        ok = "✓" if out.exists() else "✗"
        L.append(f"| {label} | `{mod}.py` | "
                 f"`{out.relative_to(SANDBOX.parent.parent.parent.parent) if out.exists() else out.name}` | {ok} |")
    L.append("")
    L.append("## Headline output paths")
    L.append("")
    L.append(f"- Dataset: `{DATASET_CSV}`")
    L.append(f"- Model:   `{MODEL_PTH}`")
    L.append(f"- Scaler:  `{SCALER_PKL}`")
    L.append(f"- Loss curve: `{LOSS_PNG}`")
    L.append(f"- Training summary: `{TRAIN_MD}`")
    L.append(f"- Prediction plots: `{PRED_PLOTS_DIR}/`")
    L.append(f"- Prediction contact sheet: `{PRED_SHEET}`")
    L.append(f"- Plan-overlay plots: `{PLAN_PLOTS_DIR}/`")
    L.append(f"- Plan-overlay contact sheet: `{PLAN_SHEET}`")
    L.append("")
    L.append("## Acceptance check")
    L.append("")
    checks = [
        ("dataset created from new calibrated Skate 1 data",
         DATASET_CSV.exists()),
        ("trained model + scaler present",
         MODEL_PTH.exists() and SCALER_PKL.exists()),
        ("loss curve PNG saved",  LOSS_PNG.exists()),
        ("prediction contact sheet saved",
         PRED_SHEET.exists()),
        ("plan overlay contact sheet saved",
         PLAN_SHEET.exists()),
    ]
    for desc, ok in checks:
        L.append(f"- [{'x' if ok else ' '}] {desc}")
    L.append("")
    L.append("All artefacts under "
             f"`{SANDBOX.relative_to(SANDBOX.parents[3])}`.")
    FINAL_MD.write_text("\n".join(L), encoding="utf-8")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Re-run every step even if its output exists.")
    args = ap.parse_args()

    SANDBOX.mkdir(parents=True, exist_ok=True)
    print(f"[INFO]  Sandbox: {SANDBOX}")

    overall_ok = True
    for label, mod, out in STEPS:
        if out.exists() and not args.force:
            print(f"\n========== {label} SKIPPED (output exists) ==========")
            print(f"          {out}")
            continue
        ok = run_module(mod)
        if not ok:
            overall_ok = False
            print(f"[FATAL] step {label} failed; stopping.")
            break

    write_final_summary({"ok": overall_ok})

    print("\n========== Final paths ==========")
    print(f"  Loss curve              : {LOSS_PNG}")
    print(f"  Prediction contact sheet: {PRED_SHEET}")
    print(f"  Plan overlay contact sheet: {PLAN_SHEET}")
    print(f"  Final summary           : {FINAL_MD}")


if __name__ == "__main__":
    main()
