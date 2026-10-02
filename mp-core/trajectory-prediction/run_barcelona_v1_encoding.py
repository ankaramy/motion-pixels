"""
run_barcelona_v1_encoding.py
----------------------------
Batch driver that runs the FROZEN production spatial encoder
(encode_spatial_auto_v2.py, v2.1C configuration) over the 5 approved
Barcelona_v1_filtered_250m recordings.

This driver does NOT contain any encoding logic of its own. It reuses the
frozen encoder unchanged, exactly the way the canonical runner
`rerun_spatial_pipeline.py` (Step 4) does: it imports the module in-process
and overrides its module-level I/O + tunable constants for the v2.1C
configuration, then calls `v2.main()`. No encoder file is edited.

v2.1C overrides (identical to rerun_spatial_pipeline.py step4_run_encoder):
    ENVELOPE_DILATE_M  = 4.0
    WALKABLE_CLOSE_M   = 1.5
    DBSCAN_EPS_M       = 2.75
    DBSCAN_MIN_SAMPLES = 12

Input  (per recording, frozen, read-only):
    <DATASET>/<rec>/filtered_250m/trajectories_world_filtered_250m.csv
Output (new namespace, never overwrites prior experiments):
    <DATASET>/Barcelona_v1_encoded/<rec>/spatial_v21C/
        trajectories_encoded.csv   (renamed from trajectories_encoded_auto_v2.csv)
        walkable_mask.png  obstacle_mask.png  boundary_mask.png
        distance_to_obstacle_m.png  distance_to_boundary_m.png
        distance_to_entrance_m.png
        entry_exit_points.csv
        debug_stages.png  debug_trajectory_sampling.png
        summary.md                 (renamed from auto_v2_summary.md)
"""
from __future__ import annotations

import importlib
import json
import sys
import traceback
from pathlib import Path

HERE    = Path(__file__).resolve().parent
PREDICT = HERE
DATASET = Path(r"C:\Users\OWNER\Desktop\new_datasets")
OUT_ROOT = DATASET / "Barcelona_v1_encoded"

# Common ancestor of both the frozen inputs and the new outputs, so the
# encoder's display-only `path.relative_to(MP_ROOT)` calls never raise.
DISPLAY_ROOT = Path(r"C:\Users\OWNER\Desktop")

RECORDINGS = [
    "esplanade_espanya_01",
    "stairs_montjuic_01",
    "red_bridge_combined_01",
    "placa_catalunya_01",
    "placa_espanya_01",
]


def encode_one(v2, rec: str) -> dict:
    src = DATASET / rec / "filtered_250m" / "trajectories_world_filtered_250m.csv"
    out_dir = OUT_ROOT / rec / "spatial_v21C"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        return {"recording": rec, "ok": False, "error": f"missing input {src}"}

    # --- Redirect I/O + apply v2.1C config on the frozen encoder module ---
    v2.MP_ROOT = DISPLAY_ROOT          # display-only (relative_to) safety
    v2.SRC_CSV = src
    v2.AUTO_V2 = out_dir
    v2.MANUAL_CSV = out_dir / "__no_manual_benchmark__.csv"  # forces clean skip

    v2.ENVELOPE_DILATE_M  = 4.0
    v2.WALKABLE_CLOSE_M   = 1.5
    v2.DBSCAN_EPS_M       = 2.75
    v2.DBSCAN_MIN_SAMPLES = 12

    print(f"\n{'='*70}\n[ENCODE] {rec}\n  src -> {src}\n  out -> {out_dir}\n{'='*70}")
    try:
        v2.main()
    except SystemExit as e:
        if e.code not in (0, None):
            return {"recording": rec, "ok": False,
                    "error": f"encoder SystemExit({e.code})"}
    except Exception as e:
        traceback.print_exc()
        return {"recording": rec, "ok": False,
                "error": f"{type(e).__name__}: {e}"}

    # --- Rename to spatial_v21C spec (mirrors rerun_spatial_pipeline step4) ---
    raw_csv = out_dir / "trajectories_encoded_auto_v2.csv"
    final_csv = out_dir / "trajectories_encoded.csv"
    if raw_csv.exists():
        raw_csv.replace(final_csv)
    raw_summary = out_dir / "auto_v2_summary.md"
    final_summary = out_dir / "summary.md"
    if raw_summary.exists():
        raw_summary.replace(final_summary)

    ok = final_csv.exists()
    result = {"recording": rec, "ok": ok, "out_dir": str(out_dir),
              "encoded_csv": str(final_csv) if ok else None,
              "error": None if ok else "encoded CSV not produced"}
    return result


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    sys.path.insert(0, str(PREDICT))
    v2 = importlib.import_module("encode_spatial_auto_v2")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for rec in RECORDINGS:
        results.append(encode_one(v2, rec))

    print("\n\n" + "#" * 70)
    print("# BATCH ENCODING RESULTS")
    print("#" * 70)
    for r in results:
        status = "OK " if r["ok"] else "FAIL"
        print(f"  [{status}] {r['recording']:24s} {r.get('error') or r['out_dir']}")

    (OUT_ROOT / "_encoding_batch_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n{n_ok}/{len(results)} recordings encoded successfully.")
    sys.exit(0 if n_ok == len(results) else 2)


if __name__ == "__main__":
    main()
