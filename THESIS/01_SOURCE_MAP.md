# 01 — Source Map (research archive audit)

Every research file is classified and routed to a thesis section. The repository is treated as a research
**archive**: Motion Pixels changed significantly during development, so an older presentation, experiment,
model, dataset, or method is **not** assumed to represent the final research. Conflicts are flagged, never
silently resolved to the convenient version. Read-only audit — no dataset, checkpoint, or result changed.

**Classification legend:**
`FINAL` = final thesis evidence · `EXPLORATORY` = investigative, not a final result · `SUPERSEDED` =
replaced by later work · `FAILED` = negative result kept for the record · `CAPACITY` = capacity / sanity
check (not generalisation) · `FUTURE` = future-work material · `REFERENCE` = external reference material ·
`VISUAL` = visual/communication artifact · `EXCLUDED` = present in workspace but not thesis evidence.

---

## A. Evidence chronology (oldest → final)

The final research is the **end** of this chain. Do not quote an early link as if it were the result.

| # | Material | Location | Class | Note |
|---|---|---|---|---|
| 1 | Extraction / tracking documentation | `docs/MP_prediction_review.md`, `docs/MP_tracking_review*`, `mp-core/trajectory-extraction/*` | REFERENCE / historical | Implementation history; check against current scripts for changed behaviour. |
| 2 | Early single-layer, absolute-coordinate prediction | `docs/MP_prediction_review.md` | SUPERSEDED | Not the final architecture. |
| 3 | MACBA schema iterations + straight-collapse archive | `mp-data/processed/rerun_macba_2026-05-19/final_sandbox/archive_straight_collapse/` | FAILED / EXPLORATORY | Straight-line collapse; kept as negative record. |
| 4 | May-21 sandbox exit + schema ablation | `docs/sandbox_exit_brief.md`, `experiments/schema_ablation_bridge/`, `frozen_model_C/` | CAPACITY / EXPLORATORY | Real track-held-out run is a small **sandbox** reference. Does **not** establish citywide generalisation. |
| 5 | Overfit 10× duplicate run | `experiments/schema_ablation_bridge_overfit10x/` | CAPACITY | Memorisation / capacity check only; no new samples. |
| 6 | Encoder V2/V3 + coverage-derived masks | `docs/encoder_v3_recording_inventory.md`, `encoder_v2_dev/`, `encoder_v3_manual/` | SUPERSEDED (masks) | Coverage masks replaced by **manual architectural masks** (`mp-data/annotations/manual_masks_v3/`). |
| 7 | Barcelona V3 master dataset | external `Barcelona_v3_manual_master_dataset/` + `experiments/MODEL_X/data_audit/dataset_audit.md` | **FINAL dataset** | 1,064,379 rows, **3,534 tracks, 5 recordings**. Embedded recording split ≠ MODEL_X training split. |
| 8 | MODEL_X baseline (H10) | `experiments/MODEL_X/reports/model_x_report.md`, `splits/model_x_track_split.csv` | **FINAL baseline** | Mixed-recording **track-held-out**, **not** unseen-site. |
| 9 | Horizon sweep H20–H400 | `experiments/MODEL_X_HORIZON_SWEEP/` | **FINAL horizon evidence** | **Separate per-horizon training/checkpoint** — not one frozen H10 model (final_metrics report mislabels this). |
| 10 | MODEL_XR (magnitude-aware loss) | `MODEL_XR/reports/`, variant CSVs | **FINAL controlled successor** | Best = B_MAG. Presented as objective-fix, alongside baseline. |
| 11 | MODEL_XC (curvature-aware loss) | `MODEL_XC/reports/MODEL_XC_REPORT.md`, variant CSVs, checkpoints | **FINAL model** | Best = **B_CURV_LIGHT** → **the final model** used for the curated presentation visuals (see `PROJECT_CONTEXT.md` "Final model"). Variant **E incomplete → excluded**. |
| 12 | Behavioural maps | `mp-visualization/behavior_maps/behaviormaps_final/` + recipes | **FINAL visuals** | Communication artifacts; stylised animation ≠ literal time replay. |
| 13 | Curated presentation visuals (still) | `mp-core/.../presentation_visuals/` | **FINAL visuals** | MODEL_X / XR / **XC (hero)** overlays; 30 curated stills. |
| 14 | Curated animations (animated + still, best/worst/stress) | `mp-visualization/ready_animations/` | **FINAL visuals** | Deterministic replay of frozen **MODEL_XC_B_CURV_LIGHT**. |
| 15 | Platform prototype | `mp-visualization/platform-prototype/` | **FINAL demonstrator (mocked)** | Front-end only; mocked processing/predictions. |
| 16 | `PROJECT_CONTEXT.md` | repo root | REFERENCE (framing) | Archival framing; "completed thesis" status; names the final model. |

---

## B. Section-by-section source routing

### Front matter
- **Abstract / Conclusion:** drafted last against finished chapters. Sources = the FINAL rows above.
- **Acknowledgments / AI declaration:** author-supplied text in the outline (verbatim). No repo source.

### Chapter 1 — The flaw in the plan
- REFERENCE: Hillier & Hanson, *The Social Logic of Space* (1984); Hillier, *Space is the Machine* (1996);
  Whyte, *The Social Life of Small Urban Spaces* (1980) + PPS seating interpretation.
- Author outline supplies the argument, RQ, and hypothesis text.
- **Do not claim** the space↔behaviour relationship was *never* quantified (Space Syntax already did, in
  part) — outline wording softened in prior draft; keep that correction.
- Whyte "bench myth" precise attribution = **unresolved** (see `04_OPEN_ISSUES.md`).

### Chapter 2 — Tools + Instruments for Behavioral Analysis
- Space Syntax: same Hillier references as Ch.1.
- Literature Review corpus: the outline table entries (Giuliari/Transformer, Social-LSTM, AgentFormer,
  Social GAN, SoPhie, TCN, occupancy Transformer, ISPRS radar, ACADIA). **Several are abbreviated /
  incomplete → author must supply full records; no fabricated citations.**
- Practical Review: **Bentley OpenPaths + LEGION**, **Autodesk InfraWorks** — official vendor docs.
- Pipeline: `mp-core/trajectory-extraction/*` (YOLOv8 + ByteTrack, `bytetrack_high_recall.yaml`,
  `calibrate_homography_interactive.py`, `track_people.py`, `run_pipeline.py`); encoding scripts; horizon
  definitions from the sweep. Note: model `speed` = **metres per step**, behaviour-map speed uses elapsed
  seconds — keep the two unit systems distinct.
- Ethics: GDPR **Article 89(1)** text (`THESIS/research/GDPR_research_safeguards_excerpts.txt`);
  EDPS / European Commission guidance. Qualify the categorical-permission slide claim.

### Chapter 3 — Motion Pixels, learning movement from Barcelona
- **MACBA sandbox:** rows 3–5 above. `schema_ablation_bridge/` (ablation A/B/C/D), `frozen_model_C/`,
  `overfit10x/`, `phase-2b-final/` model comparison (LSTM/GRU/TCN). MACBA institutional context = museum
  website (REFERENCE).
- **Prediction-model comparison (LSTM/GRU/TCN/Transformer):** `mp-data/outputs/prediction/experiments/
  phase-2b-final/` has LSTM/GRU/TCN. **Original 4-model figure incl. Transformer not fully reconciled →
  open issue.**
- **Building the dataset:** row 7 (V3). `encoder_v3_recording_inventory.md`, `dataset_audit.md`,
  `manual_masks_v3/`. **Outline counts (8 rec / 7,000 / 500) ≠ final V3 (5 rec / 3,534).** Reconcile.
- **Behavioural maps:** `behaviormaps_final/` + `FINAL_BEHAVIOR_MAP_RECIPES.md`. Outline names Heatmap,
  Flow Field, Speed Control, Prediction Projections; repo finals = **flow_fields, speed_population,
  bottleneck_density** (+ `prediction_hero`, `trace_dust_hero`). Map outline names → repo families (see
  `04_OPEN_ISSUES.md`; a literal "Heatmap" still is not in `behaviormaps_final/`).
- **Horizon rollouts (best/worst + stress):**
  - Baseline ADE-ranked panels: `MODEL_X_HORIZON_SWEEP/H*/best_6_overall`, `worst_6_overall` (MODEL_X).
  - **Curated best/worst (final):** `presentation_visuals/` (still, XC hero) + `ready_animations/`
    (animated, MODEL_XC_B_CURV_LIGHT). **Decision needed:** which set illustrates the chapter — MODEL_X
    sweep panels or MODEL_XC curated set. Given MODEL_XC is the final model, prefer the curated set for the
    "best/worst" figures and keep the sweep panels for the horizon-metrics table.
  - **Stress test 40/60/100 m:** the repo stress artifacts are **H800/H1000/H1200** (~40/50/60 m), not the
    outline's 40/60/100 m. **No verified 40/60/100 m outputs located → open issue.**
- **Prototype:** `platform-prototype/README.md` + `src/`. Mocked. No screenshots exist as files yet.

### Chapter 4 — What comes next?
- Findings from rows 8–11 (learned behaviour, limitations). `MODEL_XC_REPORT.md`, horizon comparison,
  `HORIZON_ACCURACY_REPORT.md`. Distinguish measured findings from proposed future studies.
- The "data-bound not architecture-bound" claim is a **capacity** finding (row 4/5); the later objective
  changes (XR/XC) mean data is **not** the exclusive cause — qualify accordingly.

---

## C. Material inspected / excluded from scope
- **Inspected:** repository Markdown + experiment inventory, the six outline images, extraction docs,
  sandbox brief, frozen-model material, V3 schema/training reports, horizon sweep, XR/XC reports,
  behaviour-map recipes, prototype docs, `PROJECT_CONTEXT.md`, and the prior `THESIS/` draft + meta-files.
- **Not sources:** generated caches (`cache/`, `node_modules/.cache/gh-pages/…` — a full mirror of the
  repo, ignore it), the Python env (`mp-env/`), `node_modules/`. Raw videos are not reprocessed.
- **Not located:** a full external literature library; any earlier authored thesis manuscript beyond the
  `THESIS/` draft.
- **Codex first draft is void:** `THESIS/MASTER_THESIS.md`, `THESIS/chapters/`, `THESIS/front_matter/`,
  `THESIS/BIBLIOGRAPHY.md`, and the `THESIS_*.md` / `research/` meta-files were produced by a prior
  (OpenAI Codex) pass and are **not** used as authority. They may be mined for verified numbers only; the
  fresh Anthropic draft supersedes them.

---

## D. Conflicts flagged (do not silently pick one)
1. **Dataset counts:** outline "8 recordings / ~7,000 extracted / ~500 trained" vs final V3 "5 recordings /
   3,534 tracks." Acquisition-vs-retention reconciliation is an author item.
2. **Sandbox split arithmetic:** brief says 36/8/9 of 52; code produces **36/7/9**. Use code.
3. **Unseen-site label:** MODEL_X comparison table calls frozen MACBA "unseen-site"; it is a **within-site
   track split.**
4. **One frozen model vs horizon family:** `final_metrics` calls it a frozen H10 model; the sweep uses
   **per-horizon retraining/checkpoints.**
5. **Spatial-feature handling:** sandbox **refreshes** spatial features during rollout; MODEL_X **freezes**
   last-observed clearances. Different mechanisms — don't merge.
6. **Speed units:** model `speed` = m/step; behaviour-map speed = m/s. Keep separate.
7. **Horizon labels:** H = steps/frames with approximate metre labels — **never minutes.**
8. **"Normalized Path Accuracy":** computed from endpoint error + net displacement, **not** full path shape.
9. **Stress test:** outline asks 40/60/100 m; repo has H800/H1000/H1200. Mismatch.
10. **Final-model figure provenance:** MODEL_X sweep panels vs MODEL_XC curated visuals for "best/worst."
11. **Article 89:** a safeguards provision, not standalone lawful basis for recording identifiable people.
12. **Behaviour-map naming:** outline "Heatmap / Speed Control" vs repo "bottleneck_density /
    speed_population" — confirm the mapping; a literal heatmap still is not in the finals folder.
