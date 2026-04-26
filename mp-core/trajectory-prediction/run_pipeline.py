"""
run_pipeline.py
---------------
Runs the MotionPixels Prediction pipeline.

Tracking is handled upstream by trajectory-extraction/run_pipeline.py, which
produces trajectories_world.csv.  This pipeline picks up from there.

  Step 1 — ENCODE   encode_space.py                        [interactive]
                    Opens the floor plan so you can click obstacles,
                    boundaries, and entrances.  Auto-seeds obstacles
                    from trajectory-extraction bottleneck data if available.
                    Input:   trajectory-extraction/outputs/trajectories_world.csv
                    Output:  outputs/trajectories_encoded.csv

  Step 2 — TRAIN    train_model.py
                    Trains the LSTM on the encoded trajectories.
                    Input:   outputs/trajectories_encoded.csv
                    Output:  outputs/trajectory_model.pth
                             outputs/scaler.pkl
                             outputs/loss_curve.png

  Step 3 — VISUALISE  visualise_prediction.py
                    Runs the trained model on the best-tracked person.
                    Overlays flow field + bottleneck data if available.
                    Input:   outputs/trajectories_encoded.csv
                             outputs/trajectory_model.pth
                             outputs/scaler.pkl
                    Output:  outputs/prediction_visual.png

Prerequisite
------------
Run trajectory-extraction first to generate trajectories_world.csv:

  cd ../trajectory-extraction
  python run_pipeline.py --video input_skate_1.MOV \\
      --calib_json calib.json \\
      --top_view_image YOUR_TOPVIEW_SKATE.png \\
      --run_bottlenecks --run_flow_fields

Usage
-----
  python run_pipeline.py                        # full run
  python run_pipeline.py --skip-encode          # skip encoding, retrain only
  python run_pipeline.py --from-step train      # retrain + visualise
  python run_pipeline.py --from-step visualise  # re-visualise only
  python run_pipeline.py --yes                  # skip confirmation prompt
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE         = Path(__file__).resolve().parent
MP_ROOT      = HERE.parent.parent
MP_DATA      = MP_ROOT / "mp-data"
TRAJ_OUT     = MP_DATA / "outputs" / "tracking"
BEHAVIOR_OUT = MP_DATA / "outputs" / "behavior"
PROCESSED    = MP_DATA / "processed" / "encoded"
PRED_OUT     = MP_DATA / "outputs" / "prediction"

TRAJ_CSVS = [
    TRAJ_OUT / "trajectories_world_space.csv",
    TRAJ_OUT / "trajectories_world.csv",
    PROCESSED / "trajectories_world.csv",
]

ENCODED_CSV = PROCESSED / "trajectories_encoded.csv"
MODEL_PATH  = PRED_OUT  / "trajectory_model.pth"
SCALER_PATH = PRED_OUT  / "scaler.pkl"
VISUAL_PATH = PRED_OUT  / "prediction_visual.png"
LOSS_PATH   = PRED_OUT  / "loss_curve.png"

BN_CSV   = BEHAVIOR_OUT / "bottlenecks" / "bottleneck_cells.csv"
FLOW_CSV = BEHAVIOR_OUT / "flow_fields"  / "flow_field_cells.csv"

STEP_ORDER = ["encode", "train", "visualise"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str, step: int, total: int = 3) -> None:
    bar = "-" * 60
    print(f"\n{bar}")
    print(f"  Step {step}/{total} -- {title}")
    print(f"{bar}")


def run(script: str, label: str) -> bool:
    print(f"\n[RUN] python {script}\n")
    t0     = time.time()
    result = subprocess.run([sys.executable, script], cwd=str(HERE))
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n[OK]  {label} completed in {elapsed:.1f}s")
        return True
    print(f"\n[ERR] {label} exited with code {result.returncode}")
    return False


def file_status(path, label: str) -> None:
    if path and path.exists():
        size = path.stat().st_size // 1024
        print(f"  OK  {label}  ({size:,} KB)")
    else:
        print(f"  --  {label}")


def first_existing(paths):
    return next((p for p in paths if p.exists()), None)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_encode(skip: bool) -> bool:
    section("ENCODE  --  encode_space.py  [interactive]", step=1)

    if skip:
        if ENCODED_CSV.exists():
            print(f"[SKIP] Using existing: {ENCODED_CSV.name}")
            return True
        print("[WARN] --skip-encode set but trajectories_encoded.csv not found — running anyway.")

    traj = first_existing(TRAJ_CSVS)
    if traj is None:
        print("[ERR]  No trajectory CSV found.")
        print("       Run trajectory-extraction/run_pipeline.py first to generate trajectories_world.csv.")
        return False

    print(f"[INFO] Trajectory source: {traj}")
    if BN_CSV.exists():
        print(f"[INFO] Bottleneck data found -- top obstacles will be auto-seeded.")
    else:
        print("[INFO] No bottleneck data found -- click obstacles manually in the window.")

    print("\n[INFO] An OpenCV window will open. Follow the on-screen instructions:")
    print("         Phase 1 -- click OBSTACLE points  (N to continue)")
    print("         Phase 2 -- click BOUNDARY edges   (N to continue)")
    print("         Phase 3 -- click ENTRANCE points  (Q when done)")

    return run("encode_space.py", "Encoding")


def step_train(skip: bool) -> bool:
    section("TRAIN  --  train_model.py", step=2)

    if skip:
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            print(f"[SKIP] Using existing model: {MODEL_PATH.name}")
            return True
        print("[WARN] --skip-train set but model files not found — running anyway.")

    if not ENCODED_CSV.exists():
        print(f"[ERR]  {ENCODED_CSV.name} not found. Run Step 1 first.")
        return False

    return run("train_model.py", "Training")


def step_visualise(skip: bool) -> bool:
    section("VISUALISE  --  visualise_prediction.py", step=3)

    if skip:
        print("[SKIP] Visualisation skipped (--skip-visualise).")
        return True

    missing = [p.name for p in (MODEL_PATH, SCALER_PATH, ENCODED_CSV) if not p.exists()]
    if missing:
        print(f"[ERR]  Missing files: {', '.join(missing)}")
        print("       Run earlier steps first.")
        return False

    if FLOW_CSV.exists():
        print(f"[INFO] Flow field overlay: {FLOW_CSV.name}")
    if BN_CSV.exists():
        print(f"[INFO] Bottleneck overlay: {BN_CSV.name}")

    return run("visualise_prediction.py", "Visualisation")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="MotionPixels Prediction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--skip-encode",    action="store_true",
                    help="Skip Step 1 -- use existing trajectories_encoded.csv")
    ap.add_argument("--skip-train",     action="store_true",
                    help="Skip Step 2 -- use existing trajectory_model.pth")
    ap.add_argument("--skip-visualise", action="store_true",
                    help="Skip Step 3")
    ap.add_argument("--from-step", choices=STEP_ORDER, metavar="STEP",
                    help="Start from this step (encode / train / visualise). "
                         "All earlier steps are skipped.")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="Skip confirmation prompt")
    args = ap.parse_args()

    if args.from_step:
        idx = STEP_ORDER.index(args.from_step)
        if idx > 0: args.skip_encode = True
        if idx > 1: args.skip_train  = True

    # ── Status report ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MotionPixels Prediction Pipeline")
    print("=" * 60)

    print("\nInputs (from trajectory-extraction):")
    traj = first_existing(TRAJ_CSVS)
    file_status(traj,    "trajectories_world.csv")
    file_status(BN_CSV,  "bottleneck_cells.csv  (obstacle auto-seed + overlay)")
    file_status(FLOW_CSV,"flow_field_cells.csv  (direction overlay)")

    print("\nOutputs (outputs/):")
    file_status(ENCODED_CSV, "trajectories_encoded.csv")
    file_status(MODEL_PATH,  "trajectory_model.pth")
    file_status(SCALER_PATH, "scaler.pkl")
    file_status(LOSS_PATH,   "loss_curve.png")
    file_status(VISUAL_PATH, "prediction_visual.png")

    print("\nSteps to run:")
    steps = [
        ("1. ENCODE",    not args.skip_encode),
        ("2. TRAIN",     not args.skip_train),
        ("3. VISUALISE", not args.skip_visualise),
    ]
    for label, active in steps:
        print(f"  {'[RUN] ' if active else '[SKIP]'}  {label}")

    if traj is None and not args.skip_encode:
        print("\n[WARN] No trajectory CSV found.")
        print("       Step 1 will fail unless you run trajectory-extraction first:")
        print("         cd ../trajectory-extraction")
        print("         python run_pipeline.py --video input_skate_1.MOV \\")
        print("             --calib_json calib.json \\")
        print("             --top_view_image YOUR_TOPVIEW_SKATE.png \\")
        print("             --run_bottlenecks --run_flow_fields")

    if not args.yes:
        print()
        try:
            ans = input("Proceed? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[ABORT]")
            sys.exit(0)
        if ans not in ("", "y", "yes"):
            print("[ABORT]")
            sys.exit(0)

    # ── Run ──────────────────────────────────────────────────────────────────
    t_start = time.time()

    results = {
        "encode":    step_encode   (args.skip_encode),
        "train":     step_train    (args.skip_train),
        "visualise": step_visualise(args.skip_visualise),
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    mins, secs = divmod(int(elapsed), 60)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for step, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {step.capitalize()}")

    print(f"\n  Total time : {mins}m {secs}s")
    print("\nOutputs:")
    file_status(ENCODED_CSV, "trajectories_encoded.csv")
    file_status(MODEL_PATH,  "trajectory_model.pth")
    file_status(LOSS_PATH,   "loss_curve.png")
    file_status(VISUAL_PATH, "prediction_visual.png")
    print()

    if not all(results.values()):
        failed = [k for k, v in results.items() if not v]
        print(f"  Failed steps: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
