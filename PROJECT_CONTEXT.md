# Motion Pixels — Project Context

## Project status

**Motion Pixels is a completed IAAC thesis project.**

The repository should now be treated as an archival research and demonstration codebase. Future work may document, reproduce, present, maintain, or extend the thesis, but must not describe the thesis itself as unfinished or imply that additional experiments are required for its completion.

## Project in one sentence

Motion Pixels investigates how ordinary video recordings of public space can be transformed into spatial data and pedestrian trajectory predictions that help architects understand how people move through, occupy, and respond to the built environment.

## Thesis proposition

Conventional architectural drawings describe intended geometry but do not directly capture lived movement. Motion Pixels connects computer vision, spatial calibration, behavioral analysis, and recurrent neural networks to turn observed pedestrian motion into an additional layer of architectural evidence.

The project asks whether movement extracted from video can be:

1. reconstructed in architectural/world coordinates;
2. interpreted through spatial and behavioral metrics; and
3. used to predict plausible near-future pedestrian trajectories.

The value of the work lies in the complete workflow and its architectural interpretation, not only in achieving a single prediction score.

## System overview

The project implements this general pipeline:

```text
site video
  -> pedestrian detection and tracking
  -> image-to-plan homography calibration
  -> world-space trajectory reconstruction
  -> spatial and behavioral feature encoding
  -> trajectory-model training and rollout
  -> metrics, maps, animations, and interface visualizations
```

### 1. Trajectory extraction

The extraction tools in `mp-core/trajectory-extraction/` detect and track pedestrians, calibrate footage against an architectural plan, reconstruct trajectories in world coordinates, and derive outputs such as:

- speed and motion traces;
- heatmaps;
- flow fields;
- linger or dwell zones;
- bottleneck indicators; and
- plan-based trajectory drawings.

### 2. Dataset and spatial encoding

The data layer in `mp-data/` contains raw references, calibration information, manual annotations, processed trajectories, spatial encodings, recording manifests, and experiment datasets.

A recording is one continuous site-video session with its own calibration, plan, tracking output, and spatial encoding. Dataset splits should be made at recording or track level as documented by each experiment, never by randomly mixing overlapping windows in a way that leaks spatial or trajectory identity between training and evaluation.

The selected Model C feature concept combines:

- ego-motion: displacement, speed, heading, and turn rate;
- normalized position in the calibrated site; and
- proximity to obstacles and boundaries.

This supports the thesis interpretation of movement as a relationship between a pedestrian's recent motion, their location, and nearby spatial constraints.

### 3. Trajectory prediction

The prediction work in `mp-core/trajectory-prediction/` uses recurrent trajectory models, primarily a two-layer LSTM, to predict future displacement autoregressively.

The sandbox studies established that the system was **data-bound rather than architecture-bound**: the architecture could fit the available signal, while honest held-out performance was constrained by dataset diversity and scale. Spatial obstacle and boundary features were retained in the selected Model C schema because they produced the strongest capacity-check result and matched the architectural research question.

Later controlled experiments investigate known prediction limitations, especially regression toward the mean, underprediction of displacement magnitude, heading error, curvature loss, and degradation over long rollout horizons. These successors are part of the completed research record; they do not make the thesis status provisional.

#### Final model

The **final model** — the checkpoint used to produce the curated visuals in the final presentation — is **`MODEL_XC_B_CURV_LIGHT`** (the curvature-aware LSTM successor, λ_mag=1.0 / λ_curv=0.1). It is the last stage of the MODEL_X → MODEL_XR → MODEL_XC arc (baseline → magnitude fix → curvature fix) and won the MODEL_XC variant sweep as the best all-round model (recovers trajectory shape, shape_ratio 0.05→0.33, and beats MODEL_XR_B_MAG on ADE 0.332 vs 0.362, at a modest direction cost).

- Frozen checkpoint: `mp-core/trajectory-prediction/MODEL_XC/checkpoints/MODEL_XC_B_CURV_LIGHT/model_best.pth`
- Curated still visuals: `mp-core/trajectory-prediction/presentation_visuals/` (see `PRESENTATION_VISUAL_SELECTION_REPORT.md`) — MODEL_XC is the hero (bold magenta) line, overlaid against MODEL_X / MODEL_XR for comparison.
- Curated animated + still visuals (best / worst / stress-test): `mp-visualization/ready_animations/` (see `READY_ANIMATIONS_REPORT.md`) — a deterministic replay of this frozen checkpoint.

When presentation or derived material refers to "the model," it refers to `MODEL_XC_B_CURV_LIGHT`. The earlier stages (MODEL_X, MODEL_XR) are preserved and presented as the diagnostic arc that leads to it, not as replacements.

### 4. Visualization and demonstrator

`mp-visualization/` contains the project's architectural communication layer: plan overlays, behavioral maps, hero visuals, evaluation graphics, animations, and a Vite/React platform prototype.

The platform prototype demonstrates the intended experience:

```text
upload footage and plan
  -> calibrate
  -> process
  -> inspect observed and predicted movement
```

It is a front-end demonstrator with mocked data rather than a production application or connected backend. It communicates how the research workflow could become an accessible architectural analysis tool.

## Final evidence and interpretation

The final evaluation materials show a clear horizon-dependent limitation:

- short-range endpoint placement is substantially stronger than full path reconstruction;
- prediction quality remains most useful through approximately H100, corresponding to about 5 m of ground-truth travel in the evaluated dataset;
- degradation becomes clearly visible around H200 and beyond;
- long-range outputs should therefore be read as coarse behavioral context rather than precise paths; and
- absolute endpoint success can appear relatively high while normalized path accuracy remains low, because the model tends to undershoot movement magnitude and simplify trajectory shape.

These limitations are findings of the thesis, not omissions to conceal. Claims made in presentations or derived material should preserve the distinction between endpoint tolerance, ADE/FDE, heading accuracy, path fidelity, and qualitative architectural usefulness.

## Repository map

| Path | Role |
|---|---|
| `mp-core/trajectory-extraction/` | Tracking, calibration, reconstruction, and behavioral metrics |
| `mp-core/trajectory-prediction/` | Encoding (Encoder V3), training, experiments, evaluation, and rollout prediction; final model in `MODEL_XC/` |
| `mp-data/` | Recording calibrations and plans, manual masks, dataset metadata, and a small sample (large data: `docs/DATA_AVAILABILITY.md`) |
| `mp-visualization/` | Maps, plots, animations, reports; official frozen prototype in `prototype-official-v1/`, earlier Vite demonstrator in `platform-prototype/` |
| `docs/` | Research decisions, workflow documentation, data availability, reviews, and final briefs |

Generated outputs, superseded datasets and raw video were moved to the author's local, non-Git archive in the final archival pass (2026-10-02); see `FINAL_ARCHIVAL_REPORT.md`.

## Canonical project principles

- Treat the thesis as finished.
- Preserve frozen checkpoints and reference experiments; create successors instead of overwriting them.
- Distinguish sandbox/capacity experiments from honest held-out evidence.
- Keep coordinate systems, calibration assumptions, units, frame rates, and prediction horizons explicit.
- Do not present the mocked platform prototype as a connected production system.
- Do not overstate predictive precision, especially at long horizons.
- Prefer reproducible scripts and generated reports over manually altered result files.
- Preserve provenance between a visual, its dataset, its model/checkpoint, and its evaluation split.
- Treat the repository's many generated artifacts and uncommitted research files cautiously; inspect Git state before cleanup or broad refactoring.

## How to describe the project

### Short description

Motion Pixels is an architectural research project that uses computer vision and trajectory prediction to translate pedestrian video into calibrated movement data, behavioral maps, and forecasts of how people may move through public space.

### Portfolio description

Motion Pixels is a completed IAAC thesis exploring movement as a design input. It combines pedestrian tracking, plan calibration, spatial feature encoding, LSTM-based trajectory prediction, and architectural visualization in a reproducible video-to-analysis workflow. The project demonstrates both the opportunity and the limits of predictive movement models: they provide useful short-range spatial tendencies and behavioral context, but become less reliable as precise path predictors over longer horizons.

## Context for future contributors and AI assistants

Before changing the project:

1. Read this file and the relevant component-level README or report.
2. Check the working tree and preserve unrelated research artifacts.
3. Identify whether the requested work concerns thesis documentation, reproduction, presentation, maintenance, or a post-thesis extension.
4. Keep completed-thesis claims separate from new experimental claims.
5. Verify numerical claims against generated CSV/JSON reports rather than relying on filenames or presentation graphics alone.

Useful starting documents include:

- `docs/sandbox_exit_brief.md`
- `docs/multi_recording_ingest.md`
- `docs/dataset_expansion_plan.md`
- `mp-visualization/final_metrics/HORIZON_ACCURACY_REPORT.md`
- `mp-core/trajectory-prediction/MODEL_XR/README.md`
- `mp-core/trajectory-prediction/MODEL_XC/README.md`
- `docs/DATA_AVAILABILITY.md`
- `mp-visualization/prototype-official-v1/README.md`
- `mp-visualization/platform-prototype/README.md`
