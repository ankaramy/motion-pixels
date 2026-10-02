# Motion Pixels — Dataset Expansion Resume Point (stopped night of 2026-06-04 → 06-05)

Tracking phase paused intentionally. **Encoding NOT started. Training NOT started.**

## Completed & approved recordings (orientation-aware, do NOT rerun)

| Recording | Tracks | Rows | Projection | Upright overlay |
|---|---:|---:|---|---|
| esplanade_espanya_01 | 568 | 283,875 | valid | complete (H.264 CRF18) |
| stairs_montjuic_01 | 392 | 128,259 | valid | complete (H.264 CRF18) |
| red_bridge_combined_01 | 716 | 87,076 | valid | complete (H.264 CRF18) |

**Total so far: 1,676 tracks · 499,210 rows.**

Each lives in `new_datasets/<rec>/tracking/` with raw-coord CSVs, world CSV, behavior outputs, upright overlay + static png. yolov8s baselines archived under `tracking_yolov8s_archive/`; esplanade pre-orientation under `tracking_pre_orientation_archive/`.

## Current blocker — placa_espanya_01 (run stopped before completion)

- **Symptom:** at Stage 1 the video's rotation/timebase metadata `1000/240117` is rejected by the mp4v codec → `Failed to initialize VideoWriter`. `tracked.mp4` was created at **0 bytes** and never grew.
- **Implication:** `draw_trajectories.py` uses the SAME `cv2.VideoWriter(mp4v, fps)` → the **upright overlay (Stage 5) would fail the same way**. So even a completed tracking run would not yield the overlay deliverable without a fix.
- **Secondary issue:** extreme slowness — `conf=0.10` on the very crowded Plaça Espanya at imgsz 1920 pushes CPU-side NMS to its ~2s/frame cap (`WARNING NMS time limit 2.050s exceeded`). Run was alive (CPU climbing, GPU 34–45%) but grinding; ~1h40m elapsed, still in Stage 1.
- **Process:** PID 15692, intentionally terminated via TaskStop (task `bcasvyb0x`). Confirmed no python running afterward.
- **Partial outputs (preserved, not deleted):** `mp-data/outputs/tracking/tracked.mp4` (0 bytes). No CSV (written only at end of tracking). `new_datasets/placa_espanya_01/` still has only calibration/plan/raw_video (no tracking/ folder).
- **Task log:** `C:\Users\OWNER\AppData\Local\Temp\claude\c--Users-OWNER-Desktop-IAAC-IAAC-Thesis-motion-pixels\882be8ee-cab1-4ce0-9269-f6ca50768e3e\tasks\bcasvyb0x.output` (574 bytes, preserved).

## Remaining recordings queue (all have video + calib.json + plan; READY)
1. placa_espanya_01  ← fix VideoWriter first, then re-run
2. placa_catalunya_01
3. placa_montjuic_01
4. stairs_montjuic_02

## Tomorrow's first task
1. **Fix VideoWriter / fps / timebase compatibility.** Likely cause: cv2 returns an absurd/odd `CAP_PROP_FPS` (~240.117) from this clip; mp4v's timebase denominator caps at 65535. Candidate fixes (visualization writers only — `track_people.py` tracked.mp4 writer + `draw_trajectories.py` overlay writer): sanitize fps before `VideoWriter` (e.g. `fps = fps if 1 < fps <= 120 else 30`), and/or switch fourcc to `avc1`/`H264`, and/or guard `writer.isOpened()`. Verify the CSV/projection path is unaffected (it is — those don't use the writer).
2. Confirm overlay generation is reliable on this clip (test a short render).
3. Re-run placa_espanya_01 (frozen config: yolov8m / imgsz 1920 / conf 0.10 / bytetrack_high_recall.yaml; orientation-aware; upright H.264 CRF18 overlay).
4. Continue placa_catalunya_01 → placa_montjuic_01 → stairs_montjuic_02, preserving + verifying each.

## Config reminders (frozen)
- Detector/tracker FROZEN: yolov8m, imgsz=1920, conf=0.10, bytetrack_high_recall.yaml. No detector experiments.
- Orientation-aware tracking is integrated (track_people.py, default on): rotate for inference, map detections back to raw coords; CSVs stay in raw OpenCV space; calib.json untouched.
- Export defaults (run_pipeline.py): H.264 CRF18, full resolution, orientation-aware upright overlay.
- Do NOT modify calib.json / homography / projection. Do NOT start encoding or training.
