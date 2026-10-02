# Motion Pixels — Final Archival Report

**Date:** 2026-10-02 · **Scope:** final archival pass that turns this repository into the single canonical Motion Pixels project. Historical, large and generated material was moved to a separate local preservation archive, which is not part of Git. Nothing was permanently deleted.

## Canonical Final Pipeline

```text
video → YOLOv8m + ByteTrack (bytetrack_high_recall.yaml) → image→plan homography (calib.json)
      → world trajectories → behavioral analysis (speed, dwell/linger, flow, bottlenecks, heatmaps)
      → manual architectural masks → Encoder V3 spatial features → Barcelona_v3_manual dataset
      → MODEL_X → MODEL_XR → MODEL_XC → MODEL_XC_B_CURV_LIGHT (FINAL)
      → evaluation (H10–H400) → visualizations → official thesis prototype
```

Entry points are listed in [README.md](README.md).

## Final Model

**`MODEL_XC_B_CURV_LIGHT`**: 2-layer LSTM (hidden 128), 10 input features, 10 observed steps, autoregressive rollout, loss = MSE + 1.0·magnitude + 0.1·curvature. Checkpoint, scalers and sigma are in `mp-core/trajectory-prediction/MODEL_XC/checkpoints/MODEL_XC_B_CURV_LIGHT/`.

Verification during this pass:
- **Exact reproduction.** The frozen checkpoint, run through the updated code path on the full dataset via `MP_DATA_ROOT`, gives test **ADE 0.332295 m / FDE 0.533285 m over 90,310 windows**. This matches `eval_base.json` exactly.
- **Persisted world bounds.** `experiments/MODEL_X/config/recording_world_bounds.json` stores the full-dataset bounds. With them, predictions on the committed sample are identical to the full-dataset predictions (max difference 0.0 m). The previous behaviour, which recomputed bounds from the loaded data, would have shifted sample predictions by up to 5.36 m.
- **Sample inference.** `MODEL_XC/predict_example.py` runs on the committed sample on CPU and on CUDA: H10 ADE 0.258 m on the 25 sample tracks.

The runtime chain (`MODEL_XC → MODEL_XR/xr_common.py → experiments/MODEL_X/model_x_lib.py` + config + split + bounds) is fully committed.

## What Is Now On GitHub

| Area | Content |
|---|---|
| Extraction | `mp-core/trajectory-extraction/`: final high-recall tracking, calibration UI, reconstruction, behavior metrics, overlays |
| Encoding and data | `encoder_v3_manual/`, `manual_annotation/`, `build_barcelona_v3_master.py`; `mp-data/recordings/<rec>/` (7 calibrations + plans), `mp-data/annotations/manual_masks_v3/` (7 recordings), `mp-data/datasets/` (manifests, schema, feature ranges, split, encoder metadata), `mp-data/sample/` |
| Prediction | MODEL_X library/config/split/baseline, MODEL_XR, MODEL_XC (final checkpoint, reports, horizon results, `predict_example.py`), horizon sweep, research experiment code, `frozen_model_C/` (historical) |
| Visualization | `final_metrics/`, `plots_wassim/`, `behavior_maps/` (final maps, recipes, stills, generators), `ready_animations/` (outputs), `presentation_visuals/`, hero visuals, template spec, presentation animation scripts |
| Prototype | `mp-visualization/prototype-official-v1/` (official frozen thesis prototype, imported from the Desktop); `platform-prototype/` (Vite source only) |
| Documentation | `README.md`, `PROJECT_CONTEXT.md`, `docs/DATA_AVAILABILITY.md`, `CITATION.cff`, `THESIS/` (text, chapters, figures, research), `publication_final/` (booklet build + design system) |
| Environment | `requirements.txt` (core; CPU/CUDA torch instructions), `requirements-publication.txt`, `requirements-freeze-py310-cu121.txt` (original freeze) |

Commits in this pass (on top of `f2618ac`):

| Commit | Message |
|---|---|
| `99df1cc` | Final cleanup: stop tracking generated and local development files |
| `ca99f8d` | Finalize trajectory extraction and high-recall tracking pipeline |
| `3eb6db0` | Add Encoder V3 manual spatial encoding and Barcelona dataset metadata |
| `39fc7ac` | Add final MODEL_XC trajectory prediction pipeline and checkpoints |
| `58b8a40` | Add final Motion Pixels visualization pipeline |
| `72082f3` | Add official thesis prototype (OFFICIAL_MOTION_PIXELS_V1) |
| (docs commit) | Add final thesis documentation and reproducibility guide |

The unpushed local commit `5956946`, which had added 2,264 `node_modules` files, was **not** pushed. The branch was reset onto `origin/main` without touching the working tree, and the commit remains reachable through the local branch `backup/pre-final-cleanup-2026-10-02`.

Portability fixes, applied only to the canonical pipeline:
- The dataset path in `model_x_config.json` is now relative and resolved through `MP_DATA_ROOT`.
- Encoder V3, mask annotation, the V3 master builder and the final behavior-map generator use `MP_DATA_ROOT`.
- `plots_wassim` uses repo-relative paths, with output going to `MP_PLOTS_OUT`.
- The calib.json path fields are now relative, and the plan image is resolved relative to the calib file.

## What Was Moved To Local Archive

The local archive is `MotionPixels_Archive/` on the author's machine: **59.5 GB, 99,398 files**. It contains a per-move manifest (`FINAL_CLEANUP_MOVE_MANIFEST.txt`: source, destination, size, reason), a README and the cleanup tooling.

From this repository: 1,154 moves, 4.86 GB.

| Category | Material |
|---|---|
| `datasets_large/repo/` | Sandbox-era `mp-data/processed` (~750 MB); large evaluation tables (`per_window_metrics.csv` ×6, horizon-sweep window inventory, curvature metrics, experiment dataset CSVs) |
| `generated_outputs/repo/` | `mp-data/outputs` (1.8 GB), behavior-map MP4/GIF media (~1 GB), OSM cache, extraction scratch |
| `old_experiments/repo/` | Experiment outputs (MP_X, phase3–6, schema_angular_lstm*, horizon-sweep audit figures, curvature audit), mask overlays, hero-visual intermediates |
| `old_models/repo/` | Incomplete `MODEL_XC_E_CURV_DIR_LIGHT` |
| `old_publication_builds/repo/` | Generated thesis/booklet PDFs, render caches, `publication_rebrand` |
| `thesis_production/repo/` | Earlier thesis draft (`THESIS/archive_codex`) |
| `raw_videos/repo/` | `mp-data/raw/videos`, raw frame images |
| `miscellaneous_review/repo/` | `tmp/`, stray empty files (`=`, `config,py`), screenshot, the pre-cleanup audit (`GITHUB_FINAL_AUDIT.md`, `GITHUB_FINAL_MANIFEST.txt`) |

Kept locally but ignored by Git (not moved): `mp-env/`, `node_modules/`, `dist/`, YOLO weights, agent state, `__pycache__/`.

## Large Data Excluded From Git

226 files over 25 MB are preserved in the archive and none are in Git. The largest file in Git is 7.5 MB. The main items are:

- `Barcelona_v3_manual_master_dataset/master_dataset.csv`: 379 MB, md5 `173ba156f52774ec33f3962e7c99f38e`. This is the final model's data.
- `model_C_dataset.csv`: 311 MB.
- Encoder V3 encoded CSVs: ~90 MB each.
- Per-recording `trajectories_world*.csv`: 20–84 MB each.
- Barcelona_v1 datasets.
- Large evaluation tables.
- The 527 MB prototype demo video.

What each item is, how it was produced and where to place it: [docs/DATA_AVAILABILITY.md](docs/DATA_AVAILABILITY.md).

## Raw Video Excluded From Git

**36 video files** (~14 GB) are preserved in `MotionPixels_Archive/raw_videos/`: the original iPhone `.MOV` recordings, normalized copies, tracked overlays and metric overlays. They are excluded for privacy, because they show identifiable pedestrians in public space, and for size. `.gitignore` blocks `.mov`/`.m4v`/`.avi` everywhere, plus MP4 footage under `mp-data/` and in tracked-overlay files.

## Desktop Folders Archived

88 Desktop/Downloads items (54.7 GB) were moved:

- **Datasets:** `new_datasets` → `datasets_large/new_datasets` (footage split into `raw_videos/new_datasets`).
- **Prototypes** → `old_prototypes/`: `mp-prototype-final` (including the original `OFFICIAL_MOTION_PIXELS_V1`), `mp-prototype-codex-final`, `MotionPixels_Codex_*` (7), `MotionPixels_Claude_HTML_Base`, `MotionPixels_Platform_Prototype{,_STUDIO_V2}` and the prototype revision brief.
- **Experiments** → `old_experiments/`: `MotionPixels_AngularFix_V1`, `MotionPixels_Barcelona_Training_Visuals`, `MotionPixels_Encoding_Calibration_Audit`, `MotionPixels_Thesis_Outputs`.
- **Thesis production** → `thesis_production/`: `BOOKLET`, `BOOKLET_images`, the thesis review/brief/outline PDFs.
- **Presentation material** → `presentation_material/`: `mp-presentation`, `slidedeck`, `mp-video.mp4`, and Motion-Pixels-named Downloads (mp-finals zips, presentation pptx/pdf, hero images).
- **Generated outputs:** `Plots_Wassim` → `generated_outputs/`.
- **Legacy material** → `desktop_legacy/`: the early `IAAC_Thesis` folders (`SecondYOLOtest`, `mp-old-tests`, `mp-midterm`, `MotionPixels_Prediction`, empty `mp-data`) and the hand-off folders (`forclaude`, `forclaude.zip`, `00-placaespanya`).

Before archiving, the final material was copied into the repository: 103 files, covering calibrations, plans, dataset and encoder metadata, the official prototype, presentation scripts and the phase-4 helper scripts.

## Remaining Unresolved Files

- **Desktop (left in place, not confidently Motion Pixels):** loose images (`10f3c501…png`, `7256f314…png`, `e9c866cd…png`, `Comp 1_01068.png`, `Frame 36.png`, `Frame 37.png`, `Phase1.png`), four screenshots from May–June 2026, and `files.zip`. Also a Codex agent workspace under `Documents/Codex/2026-06-08/`.
- **No LICENSE yet.** The author has to choose one; until then all rights are reserved.
- **Historical paths.** Provenance JSON/reports, run logs and historical experiment scripts still contain the original absolute local paths, kept verbatim as provenance. Non-canonical visual generators (prediction hero, H400 hero, design studies, publication build) still expect the original data layout.
- **Already-published history.** `origin/main` still contains the sandbox-era large CSVs in commits made before this pass; history was intentionally not rewritten.
- **A reference that was never in the frozen prototype.** `prototype-official-v1` references `demo_assets/placa_espanya/tracking/trajectories_image_space.csv`, which was already missing from the frozen original build.

## Known Missing Historical Files

- `mp-visualization/motion_pixels_animation_test/generate_d3_style_animation.py` is the drawing helper imported by `ready_animations/generate_ready_animations.py`. It is **lost**: it is not in Git history, the project folders or the archive, and it has **not** been reconstructed. The surviving outputs and the specification ([MOTION_PIXELS_VISUALIZATION_TEMPLATE.md](mp-visualization/MOTION_PIXELS_VISUALIZATION_TEMPLATE.md)) are preserved; see [ready_animations/README.md](mp-visualization/ready_animations/README.md).

## Final Repository Size

1,034 tracked files, about 266 MB of tracked content. The largest components are the thesis figures (88 MB), the official prototype demo assets (57 MB), manual masks with their source annotations (24 MB), and behavior-map stills (22 MB). No tracked file exceeds 7.5 MB. Clone size is larger because pre-pass history still contains the sandbox-era data.

## Archive Size

59.5 GB / 99,398 files in `MotionPixels_Archive/` (local only):

| Category | Size |
|---|---|
| old_prototypes | 19.6 GB |
| raw_videos | 14.0 GB |
| desktop_legacy | 9.2 GB |
| datasets_large | 5.4 GB |
| old_experiments | 5.0 GB |
| generated_outputs | 2.8 GB |
| presentation_material | 2.8 GB |
| old_publication_builds | 0.4 GB |
| miscellaneous_review | 0.2 GB |
| thesis_production | 0.1 GB |

## Final Git Commit

Content of the pass is complete at `fc566e2` ("Add final thesis documentation and reproducibility guide"). The commit that records this push status follows it. A fresh `git clone` of the committed state was verified to run the final model (`predict_example.py`) without any external data.

## GitHub Push Status

**Not pushed by the archival session.** `git push origin main` (a fast-forward of 8 commits onto `origin/main` `f2618ac`; no force or history rewrite needed) failed with an authentication error:

```text
remote: Invalid username or token. Password authentication is not supported for Git operations.
fatal: Authentication failed for 'https://github.com/ankaramy/motion-pixels.git/'
```

The non-interactive session had no valid GitHub credential: the stored Git Credential Manager token is invalid or expired, and no browser sign-in was possible. No destructive workaround was attempted. To publish, run this from an authenticated terminal (for example VS Code, or after `git credential-manager github login`):

```bash
git push origin main
```
