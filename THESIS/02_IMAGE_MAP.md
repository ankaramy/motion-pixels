# 02 — Image Map (figures → thesis sections)

Maps existing repository images to thesis sections with a proposed figure number and caption topic. **Only
images that actually exist on disk are listed as available.** Images the outline calls for but that are
**not** in the repo are listed under "Placeholders / to source." No image is invented. `status` = FINAL
(publication-ready evidence), EXPLORATORY (investigative), or NEEDS-EDIT (crop/relabel/redesign).

Columns: **Fig** · **File** · **Shows** · **Relevance** · **Status** · **Caption topic**

---

## Chapter 1 — The flaw in the plan
| Fig | File | Shows | Relevance | Status | Caption topic |
|---|---|---|---|---|---|
| 1.1 | *Placeholder — "beginning of presentation" / "flaw in the plan" slide* | Opening concept art | Outline requests it | NEEDS-EDIT (to source, likely from a Motion Pixels presentation PDF — not in repo) | The designed path vs the desired path |
| 1.2 | *Placeholder — `research_question`* | RQ graphic | Frames the RQ | NEEDS-EDIT (not in repo) | Can trajectories be predicted from space↔behaviour? |
| 1.3 | *Placeholder — `hypothesis`* | Hypothesis graphic | Frames the hypothesis | NEEDS-EDIT (not in repo) | Space shapes behaviour → data → prediction |

> Chapter 1 has **no** existing repo figures. All three are author graphics to be produced/sourced.

## Chapter 2 — Tools + Instruments for Behavioral Analysis
| Fig | File | Shows | Relevance | Status | Caption topic |
|---|---|---|---|---|---|
| 2.1 | *Placeholder — `space_syntax`* | Space Syntax axial/visibility example | Space Syntax subsection | NEEDS-EDIT (not in repo) | Space Syntax as configurational analysis |
| 2.2 | *Placeholder — `PRACTICAL_REVIEW`* | Bentley/Autodesk tool screenshots | Practical review | NEEDS-EDIT (not in repo; vendor imagery → permissions) | Commercial crowd-sim / mobility tools |
| 2.3 | *Placeholder — `gaps`* | Positioning diagram (explains/measures/simulates → predicts) | Gap analysis | NEEDS-EDIT (author diagram, not in repo) | Where Motion Pixels sits |
| 2.4 | *Placeholder — `computational_pipeline`* | Full pipeline diagram | Core methods figure | NEEDS-EDIT (exists only inside the outline PDF; needs a clean standalone export) | Video → detection/tracking → homography → encoding → LSTM → horizons → outputs/maps |
| 2.5 | `mp-data/annotations/manual_masks_v3/<rec>/overlay_manual.png` | Manual walkable/obstacle mask overlay per recording | Real calibration + spatial-encoding evidence for the pipeline | FINAL | Manual architectural masking feeding spatial features |
| 2.6 | GDPR Art. 89(1) + `motionpixels` anonymised-tracking panel (in outline) | Legal text + boxed pedestrians | Ethics subsection | NEEDS-EDIT (compose from GDPR excerpt + a tracking frame; qualify claim) | Anonymous trajectories under Article 89 safeguards |

## Chapter 3 — Motion Pixels, learning movement from Barcelona

### 3.1 MACBA sandbox
| Fig | File | Shows | Relevance | Status | Caption topic |
|---|---|---|---|---|---|
| 3.1 | `mp-data/raw/images/frame.png` | A frozen tracking frame | Sandbox site / tracking | FINAL (verify it is MACBA/esplanade) | MACBA esplanade sandbox frame |
| 3.2 | `mp-data/outputs/prediction/experiments/phase-2b-final/phase2b_final_spatial_model_comparison.png` | Model comparison plot | LSTM vs GRU vs TCN choice | FINAL | Why LSTM was selected |
| 3.3 | `.../phase-2b-final/phase2b_final_multi_plan_view.png` | Predictions on plan | Sandbox rollout quality | FINAL | Sandbox predictions on the plan |
| 3.4 | `.../phase-2b-final/phase2b_final_top10_worst_plan_view.png` | Worst-case sandbox rollouts | Honest failure view | FINAL | Sandbox worst cases |
| 3.5 | `mp-visualization/overfit10x_replots/overfit10x_modelC_highlight_collage.png` | Model C highlighted vs alternatives, 10× overfit | **Capacity** result — angular recovery | FINAL (CAPACITY — label as memorisation, normalised u/v axes, not a metric site plan) | 10× capacity check: schema can fit angular motion |
| 3.6 | `experiments/schema_ablation_bridge_overfit10x/ablation_visuals/angular_error_over_time.png` | Angular error curves A/B/C/D | Feature ablation | EXPLORATORY | Ablation: which features recover turning |
| 3.7 | Model C feature schema table (from `MODEL_XC_REPORT.md` / scalers order) | 10-feature schema | "Dataset structure agreed on" | FINAL (rebuild as a clean table) | The agreed Model C feature schema |

### 3.2 Building the dataset + maps + horizons + prototype
| Fig | File | Shows | Relevance | Status | Caption topic |
|---|---|---|---|---|---|
| 3.8 | `manual_masks_v3/<rec>/annotation_rgb.png` + `walkable_mask_v3_manual.png` | Per-site masks | Dataset sites & encoding | FINAL | The five/seven Barcelona sites and their masks |
| 3.9 | `behaviormaps_final/flow_fields/<rec>_flow_fields_still.png` (+ `.gif`/`.mp4`) | Flow field map | "Flow Field" map | FINAL VISUAL (stylised — not literal heading replay) | Flow fields across a plaza |
| 3.10 | `behaviormaps_final/speed_population/<rec>_speed_population_still.png` (+ anim) | One dot per pedestrian by speed | "Speed Control" map | FINAL VISUAL | Speed population (fast/medium/slow) |
| 3.11 | `behaviormaps_final/bottleneck_density/<rec>_bottleneck_density_still.png` (+ anim) | Congestion cells | Bottleneck / density map | FINAL VISUAL (precomputed scores, not a verified failure diagnosis) | Bottleneck density |
| 3.12 | *"Heatmap"* | occupancy heatmap | Outline names a Heatmap | NEEDS-EDIT (no literal heatmap still in `behaviormaps_final/`; nearest = bottleneck_density) | Occupancy heatmap |
| 3.13 | `mp-core/.../presentation_visuals/H100/H100_collage.png` (also H20/H60/H200/H400) | Curated best paths, XC hero line vs MODEL_X/XR | **Prediction Projections** / best-of horizon | FINAL VISUAL (MODEL_XC_B_CURV_LIGHT hero) | Curated horizon predictions (H100 lead) |
| 3.14 | `presentation_visuals/H*/individual/H*_best_0{1..6}.png` | Individual curated cases | Best-per-horizon | FINAL VISUAL | Readable per-horizon predictions |
| 3.15 | `ready_animations/best/*.gif` + `_preview.png` (9) | Animated best rollouts | **Best** graphs per prediction step | FINAL VISUAL (animated + still, MODEL_XC) | Best rollout animations |
| 3.16 | `ready_animations/worst/*.gif` + `_preview.png` (5) | Animated worst rollouts | **Worst** graphs per prediction step | FINAL VISUAL | Worst rollout animations |
| 3.17 | `ready_animations/stress_test/*.gif` (7: H800/H1000/H1200) | Long-horizon extrapolation | Stress test | FINAL VISUAL (**H800/1000/1200 ≈ 40/50/60 m, NOT the outline's 40/60/100 m**) | Stress-test extrapolation beyond validated range |
| 3.18 | `MODEL_X_HORIZON_SWEEP/H*/best_6_overall.png` & `worst_6_overall.png` | ADE-ranked panels (MODEL_X) | Horizon-metric evidence | FINAL (MODEL_X, ADE-ranked — different selection than 3.13–3.16) | Baseline best/worst by ADE per horizon |
| 3.19 | `mp-visualization/hero_visuals/h400_dual_placas/hero_H400_*.png` | H400 hero over dual plazas | Hero prediction shot | FINAL VISUAL | Long-horizon hero prediction |
| 3.20 | Prototype UI | upload→calibrate→process→inspect | "Building the prototype" | **MISSING** (live Vite app; no screenshot on disk — capture needed; label mocked) | The Motion Pixels demonstrator |

## Chapter 4 — What comes next?
| Fig | File | Shows | Relevance | Status | Caption topic |
|---|---|---|---|---|---|
| 4.1 | `MODEL_XC_REPORT.md` result tables (rebuild as chart) | shape/magnitude/direction tradeoff X→XR→XC | "What the model learned" | FINAL (rebuild) | The X→XR→XC objective-fix arc |
| — | (No new evidentiary figures required; Ch.4 is interpretive) | | | | |

---

## Selection caveats (carry into captions)
- **"Best/worst" ≠ representative.** Curated sets (`presentation_visuals/`, `ready_animations/`) are chosen
  for **readability**, not lowest error; the sweep panels (`best_6_overall`) are **ADE-ranked**. State which
  criterion a figure uses.
- **Final model is MODEL_XC_B_CURV_LIGHT** for the curated visuals; the sweep panels are MODEL_X. Don't
  imply a single model produced both.
- Behaviour-map animations are **stylised** density/particle representations, not literal measured-heading
  replays.
- Capacity collage (3.5) uses **normalised u/v axes** and is **memorisation** evidence only.

## Placeholders / to source (exist in outline, not in repo)
`beginning-of-presentation` slide, `research_question`, `hypothesis`, `space_syntax`, `PRACTICAL_REVIEW`,
`gaps`, standalone `computational_pipeline` export, GDPR ethics composite, a literal `Heatmap` still, and
**prototype screenshots**. The referenced Motion Pixels **presentation PDF** that several of these come from
was **not located** in the repo. → `04_OPEN_ISSUES.md`.
