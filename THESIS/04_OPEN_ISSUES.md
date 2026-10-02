# 04 — Open Issues (needs author input / unresolved / contradictory)

Items that are unclear, contradictory, unsupported, or need author decisions **before** thesis prose is
finalised. None of these make the completed research project "unfinished" — they are documentation and
editorial decisions. Nothing here is filled with invented material.

---

## A. Author input required (blocking specific text/figures)
1. **Thesis title & subtitle** — not fixed in the outline. Title page needs a searchable title + optional
   subtitle. *(Front matter 2A)*
2. **Cover / title-page administrative details** — thesis cluster name, exact degree wording, submission
   date confirmation, required cover boilerplate, NEW IAAC logo asset. *(Cover, back cover)*
3. **AI-use declaration & acknowledgments** — supplied text is used verbatim; confirm final wording and
   that all named people/roles are correct (Chronis, Markopolou, Karafali, family, Elias & Evangelo).
4. **Literature-review corpus** — several outline table entries are abbreviated/incomplete (Pérez/
   Transformer 2020, Zhang flow 2022, Transformer+CVAE 2021, Social Graph Transformer 2023, Chalmers 2024,
   Social-LSTM 2024, SocialMP 2025, Su occupancy 2023, ISPRS radar 2025, ACADIA 2024). Supply full
   bibliographic records. **No citation will be fabricated.** *(Ch.2)*
5. **Ethics / legal operational facts** — data controller, lawful basis, institutional review,
   participant-information, retention period, access controls, anonymisation/publication-review. Needed to
   replace the qualified Article-89 paragraph. *(Ch.2 ethics)*
6. **Original 4-model comparison figure** (LSTM/GRU/TCN/**Transformer**) — identify the figure, dataset,
   checkpoints, and numbers. Repo `phase-2b-final/` has LSTM/GRU/TCN only. *(Ch.3 sandbox)*
7. **Stress-test 40/60/100 m** — outline asks for these exact ranges; repo has **H800/H1000/H1200
   (≈40/50/60 m)**. Provide the 40/60/100 m outputs + checkpoint identity + eligible tracks + selection
   criteria, or approve re-labelling the section to the existing stress horizons. *(Ch.3 horizons)*

## B. Contradictions to resolve (do not silently pick one)
8. **Dataset counts.** Outline "**8 recordings / ~7,000 extracted / ~500 trained**" vs final V3
   "**5 recordings / 3,534 tracks**." Decide whether Ch.3 narrates acquisition (8/7,000/500) *and* final
   retention (5/3,534) explicitly, or drops the acquisition figures. The two are not interchangeable.
9. **Best/worst figure provenance.** The curated best/worst visuals (`presentation_visuals/`,
   `ready_animations/`) are **MODEL_XC_B_CURV_LIGHT** (the final model); the horizon-sweep panels
   (`best_6_overall`, `worst_6_overall`) are **MODEL_X**, ADE-ranked. Decide which set illustrates the
   "best/worst graphs from each prediction step" — recommended: MODEL_XC curated set for the qualitative
   figures, MODEL_X sweep for the horizon-metrics table. Don't imply one model produced both.
10. **Behaviour-map naming.** Outline lists **Heatmap, Flow Field, Speed Control, Prediction Projections**;
    repo finals are **flow_fields, speed_population, bottleneck_density** (+ `prediction_hero`,
    `trace_dust_hero`). Confirm mapping (Speed Control→speed_population; Heatmap→bottleneck_density or a new
    occupancy heatmap; Prediction Projections→presentation_visuals/hero). **No literal "Heatmap" still
    exists** in `behaviormaps_final/`.
11. **"Data-bound, not architecture-bound."** Stated as a headline finding, but it is a **capacity** result
    (sandbox 10× memorisation). The later XR/XC objective changes *improved* geometry, so data is **not**
    the exclusive cause. Keep the qualification the prior draft added.

## C. Provenance / labelling cautions
12. **Sandbox split** — use code-derived **36/7/9**, not the brief's 36/8/9.
13. **"Unseen-site"** — frozen MACBA is a **within-site track split**; MODEL_X is mixed-recording
    track-held-out. Neither is unseen-site transfer.
14. **Horizon family** — per-horizon retraining/checkpoints, not one frozen H10 model (correct the
    final_metrics wording).
15. **Spatial-feature handling** — sandbox refreshes spatial features during rollout; MODEL_X freezes
    last-observed clearances. Describe separately.
16. **Units** — model `speed` = m/step; behaviour-map speed = m/s. Horizon H = frames/steps with metre
    labels, **never minutes.**
17. **"Normalized Path Accuracy"** — endpoint-based (FDE / net displacement), not full-trajectory shape.
    Quote the source label and explain the formula; don't imply path fidelity.
18. **red_bridge_combined_01 = "Passeig de Colom"** — geographic attribution not independently verified.
19. **Whyte "bench myth"** — exact anecdote/page unresolved; use a sourced seating discussion.

## D. Missing assets (outline calls for them; not in repo)
20. **Presentation graphics**: `beginning-of-presentation` slide, `research_question`, `hypothesis`,
    `space_syntax`, `PRACTICAL_REVIEW`, `gaps`, a standalone `computational_pipeline` export, the GDPR
    ethics composite, and a literal `Heatmap`. Several are meant to come from a **Motion Pixels
    presentation PDF that was not located** in the repo. Provide that presentation or approve producing
    new graphics.
21. **Prototype screenshots** — the platform is a live Vite/React app with **mocked** data; no screenshot
    exists on disk. Capture stills (and label them as a mocked demonstrator) for Ch.3 "Building the
    prototype."
22. **NEW IAAC logo** — required on cover + back cover; asset not in repo.

## E. Scope / process notes
23. **Prior draft is void (decided).** `THESIS/MASTER_THESIS.md` (~14,145 words) + `chapters/` +
    `front_matter/` + `THESIS_*.md` + `research/*` were produced by OpenAI Codex and are **neglected**. A
    full fresh Anthropic draft will be written against these 00–04 files. Verified numbers may be reused,
    but no Codex prose is authoritative. Recommend the Codex files be deleted (or moved to an
    `archive_codex/` folder) once the new draft begins, to avoid confusion.
24. **Conclusion word budget** — outline shows a "Conclusion" heading with no target; editorial allocation
    ~600 words (prior draft 582). Confirm.
25. **Caches are not sources** — `cache/`, `mp-env/`, and the `node_modules/.cache/gh-pages/…` full repo
    mirror are ignored. Raw videos are not reprocessed.

## Marker tally
Literal `[AUTHOR INPUT REQUIRED]` / `[CITATION REQUIRED]` markers this audit would place in prose: the
7 items in section A. No missing source has been filled with invented content.
