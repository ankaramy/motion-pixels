# Real-Video Dataset Expansion — Scope & Plan

**Date:** 2026-05-21
**Status:** Plan only. No code changes in this document. Companion to `docs/sandbox_exit_brief.md`.

---

## 1. Why

The sandbox exit brief established that the held-out limitation is data volume, not architecture. The next phase lifts the pipeline from "one calibrated recording" (MACBA rerun, 52 tracks) to a **registered set of calibrated recordings**, so that:

- We can hold out *whole recordings*, not just tracks within one recording.
- The schema ablation (A/B/C/D — currently sub-noise on 9 test tracks) becomes resolvable.
- The angular-error floor can be re-evaluated under genuine train/test separation.

Concretely: enough material exists in `mp-data/raw/videos/` to multiply the dataset roughly **×3–4** without recording anything new:

```
input_macba.MOV     ← already processed to v21C (52 tracks)
input_skate_1.MOV   ← old pipeline only, needs reprocessing under new schema
input_skate_2.MOV   ← unprocessed
input_raval.MOV     ← unprocessed
input_video.mp4     ← unknown / candidate
input_video1.mp4    ← unknown / candidate
input_video2.mp4    ← unknown / candidate
```

`mp-data/raw/calibration/` currently only contains `calib_macba.json` and `calib_skate1.json`. Each new recording needs its own calibration + plan image.

---

## 2. Out of scope (this phase)

These are deliberately deferred so the expansion does not turn into a research project of its own:

- **No encoder changes.** v21C stays. We do not add new spatial channels, do not bring back `openness_lr_asymmetry`, do not redesign the KDTree-IDW rollout.
- **No architecture changes.** Frozen Model C (10 features, hidden 128 × 2, MSE-only) is the model that runs against the expanded dataset. The point of the expansion is to *test* C honestly, not to tune it.
- **No loss-function changes.** The angular-error question (auxiliary heading loss) is parked until C-on-expanded numbers are in.
- **No new tracking pipeline.** Trajectory extraction stays as the upstream producer. We accept its current detection/ID-switching errors as a fixed input.

If any of these become necessary during expansion, that is a signal to stop and re-scope, not to change them silently.

---

## 3. The unit of expansion: a "recording"

A recording is everything needed to turn one video into a v21C-encoded CSV. Defined as:

```
<recording_id>/
├── video.MOV / .mp4         ← from mp-data/raw/videos/
├── calib.json               ← per-recording, NOT shared across recordings
├── plan.png                 ← per-recording top-down plan image
├── landmarks/
│   ├── obstacles.json       ← clicked in encode_space.py, persisted
│   ├── boundaries.json
│   └── entrances.json
└── trajectories_raw.csv     ← from trajectory-extraction
```

This is the lossless input to a deterministic v21C encoding step. Everything downstream (the encoded CSV, the dataset CSVs, the trained models) can be regenerated from these inputs.

`landmarks/*.json` is new — currently `encode_space.py` is interactive-only and does not persist clicked points. Persisting them is a small change that makes the per-recording encode reproducible without re-clicking. It is the only meaningful upstream change this phase requires.

---

## 4. World-frame harmonisation — the key design question

Different recordings have **different coordinate frames**. MACBA rerun lives in `x ∈ [0.27, 13.19]`, `y ∈ [−26.25, 24.41]`. Raval will not.

Two viable strategies, with different implications:

**Option A — Per-recording normalisation.**
Each recording is min-max normalised to its own bounds before going into the trainer. `u, v ∈ [0, 1]` is *relative to that recording's plan image*. Models learn "where you are in this room" rather than "where you are in the world."
- Pro: simple, no cross-recording calibration tricks.
- Pro: matches how `u, v` already work in v21C.
- Con: spatial features `dist_to_obstacle_norm` etc. are already percentile-ranked per recording, so they are already implicitly per-recording. This option just makes the convention explicit.
- Con: a model trained this way cannot zero-shot reason about absolute scale.

**Option B — Shared world frame.**
Pick a global frame and reproject every recording into it. `u, v` becomes "where in the global frame."
- Pro: the model can in principle learn metric absolute behaviour.
- Con: requires a shared reference each recording can be aligned to. We don't have that — MACBA, Raval, Skate are physically different sites.
- Con: defeats the whole point of `dist_to_*_norm` being percentile-ranked.

**Recommended:** Option A. It is what v21C already does in spirit; making it explicit costs nothing. Document loudly that `u, v` are per-recording, not global.

---

## 5. Train / val / test partition

The sandbox partitioned by *track* within one recording. That is what produced the 9-track noise floor. Going forward:

- **Split by recording**, not by track.
- One recording fully held out as the test set. Another (smaller, optional) held out as val.
- This means each split is at least one full recording in size, which is the actual unit of generalization we care about.

Concrete first pass (subject to ingest order):

| Split | Recordings | Approx tracks |
|---|---|---|
| train | macba_rerun + skate_1 + (one of: skate_2, raval) | depends on yields |
| val | one held-out recording | depends |
| test | one held-out recording | depends |

We will not freeze the split mapping until we know each recording's track yield. The principle is fixed; the assignment is not.

---

## 6. Manifest format

A single JSON file at `mp-data/processed/recordings/manifest.json` registers every recording the project knows about. Strawman:

```json
{
  "version": 1,
  "recordings": [
    {
      "id": "macba_rerun_2026-05-19",
      "video": "mp-data/raw/videos/input_macba.MOV",
      "calib": "mp-data/raw/calibration/calib_macba.json",
      "plan_image": "mp-data/raw/images/top-down.png",
      "trajectories_raw": "mp-data/processed/rerun_macba_2026-05-19/spatial_v21C/trajectories_encoded.csv",
      "world_bounds": {"xmin": 0.271, "xmax": 13.190, "ymin": -26.250, "ymax": 24.413},
      "n_tracks": 52,
      "n_rows": 18014,
      "split_default": "train",
      "notes": "existing sandbox recording; v21C encoded"
    }
  ]
}
```

The training script reads the manifest, loads each recording's encoded CSV, normalises per-recording, concatenates, and applies split assignments. **No code in this document; this is the shape we will land on.**

---

## 7. Step-by-step plan

Numbered so progress is legible.

1. **Persist landmark clicks in `encode_space.py`.** Save `obstacles.json`, `boundaries.json`, `entrances.json` next to the recording. Re-run idempotent if the JSONs already exist. *Only upstream change.*
2. **Calibrate the next recording (Skate 2).** Produce `calib_skate2.json` and the matching plan image. Sanity check via the reprojection-error block already in `calib_macba.json`.
3. **Encode Skate 2 to v21C** using the same script that produced the MACBA rerun encoding. No script changes; new inputs only.
4. **Write the manifest** at `mp-data/processed/recordings/manifest.json` registering MACBA rerun + Skate 2. Keep `version: 1` until something forces a bump.
5. **Add a multi-recording dataset builder** that reads the manifest, applies per-recording normalisation (Option A from §4), and emits a single concatenated training CSV with a `recording_id` column. Lives in `mp-core/trajectory-prediction/` alongside the existing `build_dataset_v2.py`. Old single-recording path stays for reference.
6. **Smoke test.** Re-train frozen Model C on the manifest with `split = train: macba, test: skate_2`. Compare held-out ADE/FDE/ang_err against the sandbox bridge numbers. **Do not retune C** at this stage — the point is to see what the honest number is.
7. **Repeat steps 2–3 for Raval** and (optionally) Skate 1 under the new schema.
8. **Re-train Model C** with three recordings train / one held out. Report.
9. **Re-run the A/B/C/D ablation** under the multi-recording split *only if* step 8 lands a meaningful held-out signal. Otherwise the ablation is still noise-dominated and we need more recordings before it is worth running again.

Steps 1–4 are infrastructure. Steps 5–6 are the first real test. Steps 7–9 are scaling.

---

## 8. Acceptance criteria for "expansion phase complete"

The phase is done when **all** of the following hold:

- [ ] At least 3 recordings registered in the manifest, each fully reproducible from `(video, calib, plan, landmarks)`.
- [ ] Multi-recording dataset builder produces a deterministic, version-tagged training CSV.
- [ ] Frozen Model C re-trained on the manifest, with a per-recording held-out test set, and the held-out metrics are reported next to the sandbox numbers in an updated brief.
- [ ] A/B/C/D ablation re-run on the expanded dataset (only if step 8 above shows a usable signal).
- [ ] Angular-error floor either: (a) closes meaningfully under more data, or (b) is shown not to — at which point auxiliary angular loss becomes the next phase's lead.

---

## 9. Open questions to resolve before step 1

1. **Skate 1 reprocessing.** The legacy pipeline already produced a Skate 1 dataset. Do we (a) reprocess it under v21C for consistency, or (b) treat the legacy outputs as a separate, lower-trust recording? Recommendation: (a) — consistency matters more than the cost of one re-click.
2. **Recording IDs.** Strawman is `<site>_<YYYY-MM-DD>` (e.g., `macba_rerun_2026-05-19`, `skate_2_2026-05-21`). Confirm or replace before the manifest is written, because the ID will appear in every downstream artefact.
3. **Where to put per-recording artefacts.** Strawman is `mp-data/processed/<recording_id>/spatial_v21C/`. This matches the existing MACBA rerun layout. Confirm before we duplicate the pattern for Skate 2.
4. **Tracker fidelity per recording.** Trajectory-extraction's performance is not the same on every video. If Raval or Skate 2 yields fewer than ~30 clean tracks, that recording probably should not enter the manifest until tracking is improved. Need a per-recording "track health" threshold; placeholder: ≥ 30 tracks with ≥ 11 frames each.

These are blockers for step 1 in the sense that committing to them late means manifest churn. Resolve before, not after.

---

## 10. What this plan does *not* commit to

- A specific number of additional recordings.
- A specific training schedule for the expanded dataset.
- A specific decision about `entrance_affinity_norm` — that is parked until the expanded dataset exists.
- A specific decision about auxiliary angular loss — same.
- A new model. Model C is frozen for the duration of this phase. If we end up needing a different architecture, that is a *separate* phase.

The plan is deliberately narrow: more recordings, same model, honest held-out. Anything beyond that is the next phase's problem.
