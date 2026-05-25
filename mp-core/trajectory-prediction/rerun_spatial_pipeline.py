"""
rerun_spatial_pipeline.py
-------------------------
Stateful runner for re-running the Motion Pixels spatial pipeline cleanly
against a NEW plan image, without deleting any existing outputs.

Workflow (idempotent — re-run after each user action):

  Step 0  Inspect repo state (always).
  Step 1  Check inputs/top_view.png exists.       — user must copy plan image.
  Step 2  Check calibration/calib_<name>.json.    — user runs the interactive
                                                    calibration tool (printed).
  Step 3  Apply homography → trajectories_world.csv. — auto.
  Step 4  Run auto v2.1C spatial encoder.         — auto.
  Step 5  Render plan overlays + validation.      — auto.
  Step 6  Write rerun_spatial_pipeline_summary.md and a run_state.json.

Reuses existing scripts unchanged:
  mp-core/trajectory-extraction/calibrate_homography_interactive.py
  mp-core/trajectory-extraction/calibrate_homography.py
  mp-core/trajectory-prediction/encode_spatial_auto_v2.py        (param override)
  mp-core/trajectory-prediction/visualize_auto_v21_on_plan_v2.py (in-process)

Does NOT modify:
  - mp-data/raw/                              (read-only inputs)
  - mp-data/outputs/tracking/                 (reused image-space CSV)
  - mp-data/processed/encoded/auto_v21*/      (previous v21 outputs)
  - any encoder or overlay script

CLI:
  python rerun_spatial_pipeline.py
  python rerun_spatial_pipeline.py --site macba_2026-05-19
  python rerun_spatial_pipeline.py --source-frame mp-data/raw/images/frame.png
  python rerun_spatial_pipeline.py --invert-plan-y
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

HERE     = Path(__file__).resolve().parent
MP_ROOT  = HERE.parent.parent
MP_DATA  = MP_ROOT / "mp-data"
EXTRACT  = MP_ROOT / "mp-core" / "trajectory-extraction"
PREDICT  = MP_ROOT / "mp-core" / "trajectory-prediction"

# Existing assets (inspected and reused).
TRACKED_IMG_CSV      = MP_DATA / "outputs" / "tracking" / "trajectories_image_space.csv"
EXISTING_FRAME_IMG   = MP_DATA / "raw" / "images" / "frame.png"
EXISTING_VIDEO       = MP_DATA / "raw" / "videos" / "input_macba.MOV"
CALIB_INTERACTIVE    = EXTRACT / "calibrate_homography_interactive.py"
HOMOGRAPHY_APPLY     = EXTRACT / "calibrate_homography.py"
ENCODER_V2_SCRIPT    = PREDICT / "encode_spatial_auto_v2.py"
OVERLAY_V2_SCRIPT    = PREDICT / "visualize_auto_v21_on_plan_v2.py"

DEFAULT_SITE = f"macba_{date.today().isoformat()}"


# --------------------------------------------------------------------------- #
# Folder + state
# --------------------------------------------------------------------------- #
def rerun_folder(site: str) -> Path:
    return MP_DATA / "processed" / f"rerun_{site}"


def step_paths(rerun: Path, site: str) -> Dict[str, Path]:
    return {
        "inputs":          rerun / "inputs",
        "top_view":        rerun / "inputs" / "top_view.png",
        "source_frame":    rerun / "inputs" / "source_frame.png",
        "calibration":     rerun / "calibration",
        "calib_json":      rerun / "calibration" / f"calib_{site}.json",
        "calib_preview":   rerun / "calibration" / "calibration_preview.png",
        "trajectories":    rerun / "trajectories",
        "world_csv":       rerun / "trajectories" / "trajectories_world.csv",
        "spatial":         rerun / "spatial_v21C",
        "spatial_csv":     rerun / "spatial_v21C" / "trajectories_encoded.csv",
        "overlays":        rerun / "overlays",
        "overlay_debug":   rerun / "overlays" / "alignment_debug.png",
        "archive":         rerun / "archive",
        "state":           rerun / "run_state.json",
        "summary":         rerun / "rerun_spatial_pipeline_summary.md",
    }


def load_state(path: Path) -> Dict:
    if not path.exists():
        return {"steps_done": [], "log": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"steps_done": [], "log": []}


def save_state(path: Path, state: Dict) -> None:
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def mark(state: Dict, step: str) -> None:
    if step not in state["steps_done"]:
        state["steps_done"].append(step)


def log(state: Dict, msg: str) -> None:
    state["log"].append(msg)
    print(f"        {msg}")


# --------------------------------------------------------------------------- #
# Inspection (Step 0)
# --------------------------------------------------------------------------- #
def inspect_repo() -> Dict[str, Dict]:
    info: Dict[str, Dict] = {}
    info["tracked_image_space_csv"] = {
        "path": str(TRACKED_IMG_CSV.relative_to(MP_ROOT)),
        "exists": TRACKED_IMG_CSV.exists(),
        "rows": (sum(1 for _ in TRACKED_IMG_CSV.open()) - 1
                 if TRACKED_IMG_CSV.exists() else 0),
    }
    info["source_video"] = {
        "path": str(EXISTING_VIDEO.relative_to(MP_ROOT)),
        "exists": EXISTING_VIDEO.exists(),
    }
    info["source_frame_image"] = {
        "path": str(EXISTING_FRAME_IMG.relative_to(MP_ROOT)),
        "exists": EXISTING_FRAME_IMG.exists(),
    }
    info["interactive_calibration_tool"] = {
        "path": str(CALIB_INTERACTIVE.relative_to(MP_ROOT)),
        "exists": CALIB_INTERACTIVE.exists(),
    }
    info["apply_homography_tool"] = {
        "path": str(HOMOGRAPHY_APPLY.relative_to(MP_ROOT)),
        "exists": HOMOGRAPHY_APPLY.exists(),
    }
    info["auto_v21C_encoder_script"] = {
        "path": str(ENCODER_V2_SCRIPT.relative_to(MP_ROOT)),
        "exists": ENCODER_V2_SCRIPT.exists(),
        "note": "v2.1C = v2 with overrides ENVELOPE_DILATE_M=4.0, "
                "WALKABLE_CLOSE_M=1.5, DBSCAN_EPS_M=2.75, "
                "DBSCAN_MIN_SAMPLES=12",
    }
    info["overlay_script"] = {
        "path": str(OVERLAY_V2_SCRIPT.relative_to(MP_ROOT)),
        "exists": OVERLAY_V2_SCRIPT.exists(),
    }
    return info


def print_inspection(info: Dict[str, Dict]) -> None:
    print("Step 0 — Repo inspection")
    print("------------------------")
    for k, v in info.items():
        mark = "✓" if v.get("exists") else "✗"
        extra = ""
        if "rows" in v and v["exists"]:
            extra = f"  ({v['rows']:,} rows)"
        print(f"  {mark} {k:<32s} {v['path']}{extra}")
        if "note" in v:
            print(f"      {v['note']}")
    print()


# --------------------------------------------------------------------------- #
# Step 1 — plan image
# --------------------------------------------------------------------------- #
def step1_check_plan(paths: Dict[str, Path], state: Dict) -> bool:
    print("Step 1 — New plan image")
    print("-----------------------")
    if paths["top_view"].exists():
        size = paths["top_view"].stat().st_size
        print(f"  ✓ Found {paths['top_view'].relative_to(MP_ROOT)} "
              f"({size:,} bytes)")
        mark(state, "step1_plan_image")
        print()
        return True
    print(f"  ✗ MISSING: {paths['top_view'].relative_to(MP_ROOT)}")
    print()
    print("  Action required:")
    print(f"    Copy the new floor-plan / top-down PNG to:")
    print(f"      {paths['top_view']}")
    print("    Then re-run this script.")
    print()
    return False


# --------------------------------------------------------------------------- #
# Step 2 — fresh calibration
# --------------------------------------------------------------------------- #
def step2_check_calibration(paths: Dict[str, Path], site: str,
                            args, state: Dict) -> bool:
    print("Step 2 — Fresh calibration against the new plan")
    print("-----------------------------------------------")
    if paths["calib_json"].exists():
        try:
            c = json.loads(paths["calib_json"].read_text(encoding="utf-8"))
            diag = c.get("diagnostics", {})
            n_pairs = diag.get("n_pairs", "?")
            mean_err = diag.get("mean_reprojection_error_m", "?")
            max_err  = diag.get("max_reprojection_error_m",  "?")
            print(f"  ✓ Found {paths['calib_json'].relative_to(MP_ROOT)}")
            print(f"      pairs: {n_pairs}   "
                  f"mean reprojection error: {mean_err} m   "
                  f"max: {max_err} m")
        except Exception:
            print(f"  ✓ Found {paths['calib_json'].relative_to(MP_ROOT)} "
                  "(could not parse diagnostics; will trust it)")
        mark(state, "step2_calibration")
        print()
        return True

    print(f"  ✗ MISSING: {paths['calib_json'].relative_to(MP_ROOT)}")
    print()
    # Pick a source frame for click correspondences.
    if args.source_frame and Path(args.source_frame).exists():
        src_arg = f"--frame_image \"{Path(args.source_frame).resolve()}\""
        src_desc = f"static image  {args.source_frame}"
    elif paths["source_frame"].exists():
        src_arg = f"--frame_image \"{paths['source_frame'].resolve()}\""
        src_desc = f"static image  {paths['source_frame']}"
    elif EXISTING_FRAME_IMG.exists():
        src_arg = f"--frame_image \"{EXISTING_FRAME_IMG.resolve()}\""
        src_desc = f"existing      {EXISTING_FRAME_IMG.relative_to(MP_ROOT)}"
    elif EXISTING_VIDEO.exists():
        src_arg = (f"--video \"{EXISTING_VIDEO.resolve()}\" "
                   f"--frame_index 0")
        src_desc = (f"video frame 0 from "
                    f"{EXISTING_VIDEO.relative_to(MP_ROOT)}")
    else:
        src_arg = "--frame_image <PATH_TO_CAMERA_FRAME>"
        src_desc = "**you need to choose a source frame**"

    invert_flag = " --invert_plan_y" if args.invert_plan_y else ""

    print("  Action required (interactive — 5–10 min):")
    print()
    print(f"    Source frame:  {src_desc}")
    print(f"    Top-view:      {paths['top_view'].relative_to(MP_ROOT)}")
    print()
    print("  Command:")
    print()
    print(f"    python {CALIB_INTERACTIVE.relative_to(MP_ROOT)} \\")
    print(f"      {src_arg} \\")
    print(f"      --top_view_image \"{paths['top_view'].resolve()}\" \\")
    print(f"      --out_json     \"{paths['calib_json'].resolve()}\" \\")
    print(f"      --out_preview  \"{paths['calib_preview'].resolve()}\""
          f"{invert_flag}")
    print()
    print("  Interactive workflow inside the tool:")
    print("    STEP 1  Click TWO points on the top-view image, then type the")
    print("            real-world distance between them in metres.")
    print("    STEP 2  Alternate clicks: a point on the camera frame ↔ the")
    print("            matching point on the top-view. Need ≥ 6 well-spread")
    print("            pairs. Prefer:")
    print("              - plaza corners (where pavement meets a wall)")
    print("              - curb intersections")
    print("              - bases of fixed columns")
    print("              - stair edges")
    print("              - permanent building corners")
    print("            AVOID shadows, trees, temporary barriers, and any")
    print("            pedestrian (they move between frames).")
    print("    STEP 3  Press S to solve and save calib JSON.")
    print()
    print("  Quality bar (printed by the tool):")
    print("    GOOD < 0.30 m   OK < 1.00 m   POOR ≥ 1.00 m")
    print()
    print("  Then re-run this script.")
    print()
    return False


# --------------------------------------------------------------------------- #
# Step 3 — apply homography → world CSV
# --------------------------------------------------------------------------- #
def step3_apply_homography(paths: Dict[str, Path], state: Dict) -> bool:
    print("Step 3 — Apply homography to tracking CSV")
    print("-----------------------------------------")
    if paths["world_csv"].exists():
        rows = sum(1 for _ in paths["world_csv"].open()) - 1
        print(f"  ✓ Found {paths['world_csv'].relative_to(MP_ROOT)} "
              f"({rows:,} rows)")
        mark(state, "step3_apply_homography")
        print()
        return True

    if not TRACKED_IMG_CSV.exists():
        print(f"  ✗ Image-space tracking CSV not found at "
              f"{TRACKED_IMG_CSV.relative_to(MP_ROOT)}")
        print("    Re-run the tracking stage first:")
        print(f"      python {EXTRACT.relative_to(MP_ROOT)}/track_people.py "
              "--video <your_video> --out_dir <out>")
        print()
        return False

    if not HOMOGRAPHY_APPLY.exists():
        print(f"  ✗ Homography apply tool missing: "
              f"{HOMOGRAPHY_APPLY.relative_to(MP_ROOT)}")
        return False

    print("  Reusing image-space tracking:")
    print(f"    {TRACKED_IMG_CSV.relative_to(MP_ROOT)}")
    print(f"  Calibration:")
    print(f"    {paths['calib_json'].relative_to(MP_ROOT)}")
    print(f"  Output:")
    print(f"    {paths['world_csv'].relative_to(MP_ROOT)}")
    print()
    cmd = [
        sys.executable, str(HOMOGRAPHY_APPLY),
        "--traj_csv",  str(TRACKED_IMG_CSV),
        "--calib_json", str(paths["calib_json"]),
        "--out_csv",   str(paths["world_csv"]),
        "--out_plot",  str(paths["trajectories"] /
                           "trajectories_world_plot.png"),
    ]
    print("  Running:")
    print("    " + " ".join(cmd))
    print()
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        print(f"  ✗ FAILED (exit {r.returncode})")
        log(state, f"step3 failed: {r.stderr[:200]}")
        return False
    if not paths["world_csv"].exists():
        print(f"  ✗ Tool succeeded but {paths['world_csv'].name} not "
              "produced")
        return False
    rows = sum(1 for _ in paths["world_csv"].open()) - 1
    print(f"  ✓ Wrote {paths['world_csv'].relative_to(MP_ROOT)} "
          f"({rows:,} rows)")
    mark(state, "step3_apply_homography")
    log(state, f"step3 wrote {rows} rows")
    print()
    return True


# --------------------------------------------------------------------------- #
# Step 4 — auto v2.1C encoder
# --------------------------------------------------------------------------- #
def step4_run_encoder(paths: Dict[str, Path], state: Dict) -> bool:
    print("Step 4 — Run auto v2.1C spatial encoder")
    print("---------------------------------------")
    if paths["spatial_csv"].exists():
        print(f"  ✓ Found {paths['spatial_csv'].relative_to(MP_ROOT)}")
        mark(state, "step4_encoder")
        print()
        return True

    if not ENCODER_V2_SCRIPT.exists():
        print(f"  ✗ Encoder script missing: "
              f"{ENCODER_V2_SCRIPT.relative_to(MP_ROOT)}")
        return False

    # Import the v2 encoder in-process and override constants for v2.1C.
    sys.path.insert(0, str(PREDICT))
    import importlib
    # Reload in case it was imported earlier with different constants.
    if "encode_spatial_auto_v2" in sys.modules:
        v2 = importlib.reload(sys.modules["encode_spatial_auto_v2"])
    else:
        v2 = importlib.import_module("encode_spatial_auto_v2")

    print("  Overriding v2 constants for v2.1C and redirecting I/O:")
    print(f"    SRC_CSV  ← {paths['world_csv'].relative_to(MP_ROOT)}")
    print(f"    AUTO_V2  → {paths['spatial'].relative_to(MP_ROOT)}")
    print(f"    ENVELOPE_DILATE_M  = 4.0")
    print(f"    WALKABLE_CLOSE_M   = 1.5")
    print(f"    DBSCAN_EPS_M       = 2.75")
    print(f"    DBSCAN_MIN_SAMPLES = 12")
    print()

    v2.SRC_CSV = paths["world_csv"]
    v2.AUTO_V2 = paths["spatial"]
    v2.ENVELOPE_DILATE_M = 4.0
    v2.WALKABLE_CLOSE_M  = 1.5
    v2.DBSCAN_EPS_M      = 2.75
    v2.DBSCAN_MIN_SAMPLES = 12

    try:
        v2.main()
    except SystemExit as e:
        if e.code not in (0, None):
            print(f"  ✗ Encoder failed (exit {e.code})")
            return False
    except Exception as e:
        print(f"  ✗ Encoder raised {type(e).__name__}: {e}")
        return False

    # The encoder writes 'trajectories_encoded_auto_v2.csv' — rename to spec.
    raw_csv = paths["spatial"] / "trajectories_encoded_auto_v2.csv"
    if raw_csv.exists():
        raw_csv.rename(paths["spatial_csv"])
    # Also rename summary md for clarity.
    raw_summary = paths["spatial"] / "auto_v2_summary.md"
    final_summary = paths["spatial"] / "summary.md"
    if raw_summary.exists():
        raw_summary.replace(final_summary)

    if not paths["spatial_csv"].exists():
        print(f"  ✗ Encoded CSV not produced")
        return False
    print(f"  ✓ Wrote {paths['spatial_csv'].relative_to(MP_ROOT)}")
    mark(state, "step4_encoder")
    print()
    return True


# --------------------------------------------------------------------------- #
# Step 5 — plan overlay + validation
# --------------------------------------------------------------------------- #
def step5_run_overlay(paths: Dict[str, Path], site: str, state: Dict) -> Dict:
    print("Step 5 — Plan overlay + geometric validation")
    print("--------------------------------------------")
    if paths["overlay_debug"].exists():
        print(f"  ✓ Found {paths['overlay_debug'].relative_to(MP_ROOT)}")
        mark(state, "step5_overlay")
        return {"ok": True, "passed": None,
                "note": "pre-existing overlay artefacts (re-use)"}

    if not OVERLAY_V2_SCRIPT.exists():
        print(f"  ✗ Overlay script missing: "
              f"{OVERLAY_V2_SCRIPT.relative_to(MP_ROOT)}")
        return {"ok": False, "passed": False, "note": "overlay script missing"}

    # Import overlay v2 in-process and override paths for this rerun.
    sys.path.insert(0, str(PREDICT))
    import importlib
    if "visualize_auto_v21_on_plan_v2" in sys.modules:
        ov = importlib.reload(sys.modules["visualize_auto_v21_on_plan_v2"])
    else:
        ov = importlib.import_module("visualize_auto_v21_on_plan_v2")

    print("  Overriding overlay paths:")
    print(f"    CALIB    ← {paths['calib_json'].relative_to(MP_ROOT)}")
    print(f"    TOPDOWN  ← {paths['top_view'].relative_to(MP_ROOT)}")
    print(f"    output   → {paths['overlays'].relative_to(MP_ROOT)}")
    print()

    # The overlay script's main() reads --variant CLI; replicate its logic
    # against our rerun spatial folder + override hard-coded paths.
    ov.CALIB = paths["calib_json"]
    ov.TOPDOWN = paths["top_view"]
    # ENCODED is used to resolve the variant; point it at the rerun root.
    ov.ENCODED = paths["spatial"].parent
    ov.DEFAULT_VARIANT = paths["spatial"].name

    import cv2, numpy as np   # noqa: E402
    import pandas as pd       # noqa: E402

    # The overlay v2 expects the variant folder to contain
    # 'trajectories_encoded.csv'; that's already what we produced in step 4.
    # Its load_variant() points to that filename, good.
    try:
        plan = cv2.imread(str(ov.TOPDOWN), cv2.IMREAD_COLOR)
        if plan is None:
            print(f"  ✗ Cannot read plan image: {ov.TOPDOWN}")
            return {"ok": False, "passed": False,
                    "note": "plan image unreadable"}
        plan_rgb = cv2.cvtColor(plan, cv2.COLOR_BGR2RGB)
        plan_h, plan_w = plan_rgb.shape[:2]

        calib = ov.load_calib_and_build_H(ov.CALIB)
        var = ov.load_variant(paths["spatial"])
        val = ov.validate(calib["H"], calib, var["df"], var["ali"],
                          plan_rgb.shape)
        passed, reason = ov.passes(val)

        # Write overlay outputs into rerun overlays folder.
        out_dir = paths["overlays"]
        out_dir.mkdir(parents=True, exist_ok=True)

        out_debug = out_dir / "alignment_debug.png"
        ov.plot_alignment_debug(out_debug, plan_rgb, calib, val)

        outputs = [out_debug]
        if passed:
            ww = ov.warp_mask(var["walk"], calib["H"], var["x_min"],
                              var["y_min"], plan_w, plan_h)
            wo = ov.warp_mask(var["obs"],  calib["H"], var["x_min"],
                              var["y_min"], plan_w, plan_h)
            wb = ov.warp_mask(var["bnd"],  calib["H"], var["x_min"],
                              var["y_min"], plan_w, plan_h)
            comp = ov.composite_overlay(plan_rgb, ww, wo, wb)
            out_masks = out_dir / "plan_overlay_masks.png"
            ov.plot_overlay_masks(out_masks, comp, calib, var, val)
            out_contact = out_dir / "plan_overlay_contact_sheet.png"
            ov.plot_contact_sheet(out_contact, plan_rgb, comp, calib,
                                  var, val)
            outputs += [out_masks, out_contact]

        out_summary = out_dir / "overlay_summary.md"
        ov.write_summary(out_summary, paths["spatial"], calib, val,
                         passed, reason, outputs)
        outputs.append(out_summary)

        print(f"  Calib residual: mean={val['calib_residual_mean']:.2f} px"
              f"  max={val['calib_residual_max']:.2f} px")
        print(f"  Trajectories inside image: "
              f"{val['n_traj_inside']}/{val['n_traj_sampled']}")
        print(f"  Hull area / image area: "
              f"{val['hull_area_fraction']:.1%}")
        print(f"  Decision: {'PASS' if passed else 'FAIL'} ({reason})")
        print()
        mark(state, "step5_overlay")
        log(state, f"overlay {'PASS' if passed else 'FAIL'} ({reason})")
        return {"ok": True, "passed": passed, "reason": reason,
                "validation": val, "outputs": outputs,
                "calib_diagnostics": calib["diagnostics"]}
    except Exception as e:
        print(f"  ✗ Overlay raised {type(e).__name__}: {e}")
        return {"ok": False, "passed": False, "note": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------- #
# Step 6 — write summary
# --------------------------------------------------------------------------- #
def write_summary(summary_path: Path, paths: Dict[str, Path], site: str,
                  info: Dict, state: Dict,
                  overlay_result: Optional[Dict], args) -> None:
    L: List[str] = []
    L.append(f"# Spatial Pipeline Rerun — `{site}`")
    L.append("")
    L.append("Read-only rerun against a new plan image. No existing outputs "
             "deleted; all rerun artefacts live under "
             f"`mp-data/processed/rerun_{site}/`.")
    L.append("")
    L.append("## Folder structure")
    L.append("")
    L.append("```")
    L.append(f"mp-data/processed/rerun_{site}/")
    L.append( "├── inputs/")
    L.append( "│   ├── top_view.png             ← new plan image (user-supplied)")
    L.append( "│   └── source_frame.png         ← optional override camera frame")
    L.append( "├── calibration/")
    L.append(f"│   ├── calib_{site}.json        ← produced interactively")
    L.append( "│   └── calibration_preview.png")
    L.append( "├── trajectories/")
    L.append( "│   ├── trajectories_world.csv   ← produced by calibrate_homography.py")
    L.append( "│   └── trajectories_world_plot.png")
    L.append( "├── spatial_v21C/")
    L.append( "│   ├── trajectories_encoded.csv")
    L.append( "│   ├── walkable_mask.png  obstacle_mask.png  boundary_mask.png")
    L.append( "│   ├── entry_exit_points.csv")
    L.append( "│   ├── distance_to_{obstacle,boundary,entrance}_m.png")
    L.append( "│   └── summary.md")
    L.append( "├── overlays/")
    L.append( "│   ├── alignment_debug.png")
    L.append( "│   ├── plan_overlay_masks.png")
    L.append( "│   ├── plan_overlay_contact_sheet.png")
    L.append( "│   └── overlay_summary.md")
    L.append( "├── archive/                     ← reserved (nothing copied yet)")
    L.append( "├── run_state.json")
    L.append( "└── rerun_spatial_pipeline_summary.md   ← this file")
    L.append("```")
    L.append("")

    L.append("## Repo inspection (Step 0)")
    L.append("")
    L.append("| asset | path | status |")
    L.append("|---|---|---|")
    for k, v in info.items():
        m = "✓ present" if v.get("exists") else "✗ missing"
        L.append(f"| `{k}` | `{v['path']}` | {m} |")
    L.append("")

    L.append("## Commands run (or to run)")
    L.append("")
    L.append("**Step 1 — copy the new plan image** (user action):")
    L.append("")
    L.append("```bash")
    L.append(f"# Linux/macOS")
    L.append(f"cp /path/to/your_new_plan.png \\")
    L.append(f"   {paths['top_view'].relative_to(MP_ROOT)}")
    L.append(f"")
    L.append(f"# Windows PowerShell")
    L.append(f"Copy-Item C:\\path\\to\\your_new_plan.png "
             f"{paths['top_view']}")
    L.append("```")
    L.append("")
    L.append("**Step 2 — fresh interactive calibration** "
             "(user action, ~5–10 min):")
    L.append("")
    L.append("```bash")
    L.append(f"python {CALIB_INTERACTIVE.relative_to(MP_ROOT)} \\")
    if args.source_frame:
        L.append(f"  --frame_image \"{args.source_frame}\" \\")
    elif EXISTING_FRAME_IMG.exists():
        L.append(f"  --frame_image \"{EXISTING_FRAME_IMG}\" \\")
    else:
        L.append(f"  --video        \"{EXISTING_VIDEO}\" --frame_index 0 \\")
    L.append(f"  --top_view_image \"{paths['top_view']}\" \\")
    L.append(f"  --out_json       \"{paths['calib_json']}\" \\")
    L.append(f"  --out_preview    \"{paths['calib_preview']}\""
             + (" \\\n  --invert_plan_y" if args.invert_plan_y else ""))
    L.append("```")
    L.append("")
    L.append("Quality thresholds the tool prints itself: "
             "GOOD < 0.30 m · OK < 1.00 m · POOR ≥ 1.00 m.")
    L.append("")
    L.append("**Step 3 — apply homography** (auto-run by this script):")
    L.append("")
    L.append("```bash")
    L.append(f"python {HOMOGRAPHY_APPLY.relative_to(MP_ROOT)} \\")
    L.append(f"  --traj_csv   \"{TRACKED_IMG_CSV}\" \\")
    L.append(f"  --calib_json \"{paths['calib_json']}\" \\")
    L.append(f"  --out_csv    \"{paths['world_csv']}\" \\")
    L.append(f"  --out_plot   \"{paths['trajectories']}/trajectories_world_plot.png\"")
    L.append("```")
    L.append("")
    L.append("**Step 4 — run v2.1C encoder** (auto-run; constants overridden "
             "in-process, original encoder file untouched).")
    L.append("")
    L.append("**Step 5 — plan overlay + validation** (auto-run; same.)")
    L.append("")

    L.append("## State")
    L.append("")
    if state["steps_done"]:
        for s in state["steps_done"]:
            L.append(f"- ✓ {s}")
    else:
        L.append("- _no steps completed yet_")
    L.append("")

    # Calibration block
    L.append("## Calibration result")
    L.append("")
    if not paths["calib_json"].exists():
        L.append("_Not yet produced._")
    else:
        try:
            c = json.loads(paths["calib_json"].read_text(encoding="utf-8"))
            d = c.get("diagnostics", {})
            L.append(f"- pairs: **{d.get('n_pairs', '?')}**, inliers: "
                     f"**{d.get('n_inliers', '?')}**")
            L.append(f"- mean reprojection error: **{d.get('mean_reprojection_error_m', '?')} m**")
            L.append(f"- median: {d.get('median_reprojection_error_m', '?')} m   "
                     f"max: {d.get('max_reprojection_error_m', '?')} m")
            mean_err = d.get("mean_reprojection_error_m")
            if mean_err is not None:
                quality = ("GOOD" if mean_err < 0.30 else
                           "OK"   if mean_err < 1.00 else "POOR")
                L.append(f"- quality band (tool's bar): **{quality}**")
        except Exception as e:
            L.append(f"_Could not parse calibration JSON: {e}_")
    L.append("")

    # Overlay block
    L.append("## Overlay validation result")
    L.append("")
    if overlay_result is None:
        L.append("_Not yet run._")
    elif not overlay_result.get("ok"):
        L.append(f"- **Failed to render**: {overlay_result.get('note', '—')}")
    else:
        v = overlay_result.get("validation")
        if v is not None:
            L.append(f"- Calibration self-reprojection: mean "
                     f"**{v['calib_residual_mean']:.2f} px**, max "
                     f"**{v['calib_residual_max']:.2f} px**")
            L.append(f"- Trajectories inside plan image: "
                     f"**{v['n_traj_inside']}/{v['n_traj_sampled']}** "
                     f"({v['frac_traj_inside']:.1%})")
            L.append(f"- Convex hull of projected trajectories: "
                     f"**{v['hull_area_fraction']:.1%}** of plan area")
        passed = overlay_result.get("passed")
        L.append(f"- Decision: "
                 f"**{'PASS' if passed else 'FAIL'}** "
                 f"({overlay_result.get('reason', '—')})")
        L.append("")
        L.append("**Visual verification (the hard part):** open "
                 f"`{paths['overlay_debug'].relative_to(MP_ROOT)}` and "
                 "confirm the green trajectory points sit on the actual "
                 "pedestrian plaza in the new image — not on roofs, roads, "
                 "or empty pavement. Numerical pass alone is insufficient.")
    L.append("")

    # Next-action recommendation.
    L.append("## Next recommended action")
    L.append("")
    if not paths["top_view"].exists():
        L.append("→ **Copy the new plan image** to "
                 f"`{paths['top_view'].relative_to(MP_ROOT)}` and re-run "
                 "this script.")
    elif not paths["calib_json"].exists():
        L.append("→ **Run the interactive calibration command from Step 2** "
                 "above, then re-run this script.")
    elif not paths["world_csv"].exists():
        L.append("→ Re-run this script; it will auto-execute Step 3 "
                 "(homography apply).")
    elif not paths["spatial_csv"].exists():
        L.append("→ Re-run this script; it will auto-execute Step 4 "
                 "(encoder).")
    elif overlay_result is None or not overlay_result.get("ok"):
        L.append("→ Re-run this script; it will auto-execute Step 5 "
                 "(overlay).")
    elif overlay_result.get("passed"):
        L.append("→ **Visually verify** `alignment_debug.png` against the "
                 "new plan image. If trajectories sit on the actual plaza, "
                 "the geometry rerun is complete and you can proceed to "
                 "downstream prediction work using the new world CSV at "
                 f"`{paths['world_csv'].relative_to(MP_ROOT)}`.")
    else:
        L.append("→ Overlay validation **failed**. Inspect "
                 "`alignment_debug.png`. Most likely the new calibration's "
                 "correspondence points are too sparse or too clustered; "
                 "re-run Step 2 with more spread-out anchor points.")
    L.append("")
    L.append("## Constraints honoured")
    L.append("")
    L.append("- No existing CSVs or outputs were deleted.")
    L.append("- `calib_macba.json` was NOT reused — Step 2 produces a fresh JSON.")
    L.append("- The encoder script "
             "`encode_spatial_auto_v2.py` and overlay script "
             "`visualize_auto_v21_on_plan_v2.py` were imported in-process "
             "with their module-level constants overridden — neither file "
             "was edited.")
    L.append("- No prediction model has been (or will be) trained as part "
             "of this rerun. Geometry + encoding validation only.")
    L.append("")
    L.append("_End of rerun summary._")

    summary_path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=DEFAULT_SITE,
                    help=f"Site/run name (folder suffix). "
                         f"Default: {DEFAULT_SITE}")
    ap.add_argument("--source-frame", default=None,
                    help="Optional explicit camera-frame image for "
                         "calibration. If omitted, prefer "
                         "mp-data/raw/images/frame.png, else extract from "
                         "input_macba.MOV frame 0.")
    ap.add_argument("--invert-plan-y", action="store_true",
                    help="Pass --invert_plan_y to the interactive "
                         "calibration tool (when the new plan image has y "
                         "increasing upward).")
    args = ap.parse_args()

    rerun = rerun_folder(args.site)
    rerun.mkdir(parents=True, exist_ok=True)
    paths = step_paths(rerun, args.site)
    for k in ("inputs", "calibration", "trajectories", "spatial", "overlays",
              "archive"):
        paths[k].mkdir(parents=True, exist_ok=True)

    print(f"=== Spatial pipeline rerun — site: {args.site} ===")
    print(f"=== Folder: {rerun} ===")
    print()

    state = load_state(paths["state"])
    info  = inspect_repo()
    print_inspection(info)

    overlay_result: Optional[Dict] = None
    if step1_check_plan(paths, state):
        if step2_check_calibration(paths, args.site, args, state):
            if step3_apply_homography(paths, state):
                if step4_run_encoder(paths, state):
                    overlay_result = step5_run_overlay(paths, args.site, state)

    save_state(paths["state"], state)
    write_summary(paths["summary"], paths, args.site, info, state,
                  overlay_result, args)

    print(f"Summary written: {paths['summary']}")
    print()
    # Final next-action banner.
    if not paths["top_view"].exists():
        print("NEXT: copy the new plan image to inputs/top_view.png, then "
              "re-run this script.")
    elif not paths["calib_json"].exists():
        print("NEXT: run the interactive calibration command printed in "
              "Step 2, then re-run this script.")
    elif overlay_result and overlay_result.get("passed"):
        print("NEXT: open alignment_debug.png and visually verify "
              "trajectories land on the actual plaza in the new image.")
    elif overlay_result and overlay_result.get("ok") and not \
            overlay_result.get("passed"):
        print("NEXT: validation FAILED — re-do Step 2 with better-spread "
              "calibration anchor points.")
    else:
        print("NEXT: re-run this script after the missing input is in "
              "place.")


if __name__ == "__main__":
    main()
