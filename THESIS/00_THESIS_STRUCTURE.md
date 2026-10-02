# 00 — Thesis Structure (authoritative reconstruction)

**Project:** Motion Pixels
**Author:** Ramy Anka
**Supervisor:** Dr. Wassim Jabi
**Institution:** Institute for Advanced Architecture of Catalonia (IAAC)
**Programme:** MaAI — Master in AI for Architecture & the Built Environment
**Location / date:** Barcelona, June 2026
**Academic cycle:** 2025–2026

**Source of this structure:** reconstructed *exactly* from the six outline images in
`Motion_Pixels_Thesis_Review.pdf` (the author-supplied outline), which are authoritative. Where the
outline gives no explicit number, it is marked **[allocation — editorial]** and can be changed by the
author. Nothing in the outline was merged, renamed, or re-ordered. A prior full draft already exists in
`THESIS/MASTER_THESIS.md` (≈14,145 words); this file re-derives the plan from the outline and flags where
that draft and the outline diverge (see `04_OPEN_ISSUES.md`).

Word-count discipline: aim within **±5%** of each stated target. Counts are narrative prose only —
headings, tables, captions, and placeholders excluded.

---

## Front matter

### 1. Cover
- A4 vertical. Must include: thesis **title**, author(s), supervisor(s), **thesis cluster**, academic cycle
  "2025 – 2026", and the **NEW IAAC logo**.
- Not counted in word budget.

### 2. Title page (1 page)
- **A.** Title and subtitle — keep as few as possible, "searchable", avoid uncommon abbreviations / acronyms
  / codes / symbols / formulas.
- **B.** Author: **Ramy Anka**
- **C.** Supervisor: **Dr. Wassim Jabi**
- **D.** Institution: **Institute for Advanced Architecture of Catalonia**
- **E.** Master programme: **MaAI — Master in AI for Architecture & the Built Environment**
- **F.** Location and date: **Barcelona, June 2026**
- *Open:* final thesis title + subtitle not fixed in the outline (see `04_OPEN_ISSUES.md`).

### 3. Abstract
- **≤ 500 words, 1 page.** Brief version of the thesis: problem, research question, hypothesis,
  experiments, results.
- Must include **≤ 5 keywords** (before or after the abstract).
- *Note:* prior draft abstract is ~303 words — within the ≤500 ceiling; keyword line to confirm.

### 4. Preface / prologue (3 pages max — *optional*)
- **A.** Introductory text functioning as a user's manual / reader's guide.
- **B.** Acknowledgments — supplied names: Dr. Wassim Jabi; Angelos Chronis and Areti Markopolou (programme
  initiation + full scholarship); Eleni Karafali (programme coordinator); parents Milad and Marie and
  sister Zeina; Elias and Evangelo.
- **Declaration of AI Use (in preface)** — supplied text: generative AI tools used to support writing,
  editing, coding, debugging, and organising research material; all research decisions, methodological
  development, computational experiments, analysis, visual work, and final editorial decisions undertaken
  and reviewed by the author.

---

## Chapters

### Chapter 1 — The flaw in the plan · target **±2,000 words**
| Subsection | Target (words) | Figures expected | Notes from outline |
|---|---:|---|---|
| Introduction to topic | 850 [allocation] | "beginning of presentation" + "section flaw in the plan from pdf" | Architects design for humans, but people choose the *desired* path, not the *designed* one; design and behaviour are negotiated; the architecture↔behaviour relationship has been theorised but never properly quantified; many prior attempts, so this thesis needs an anchoring case to build a methodology. |
| Early references | 500 [allocation] | — | **Space Syntax, Bill Hillier**; **William Whyte and his "bench myth."** |
| Research Question | 350 [allocation] | `research_question` (to source) | Premise: a direct correlation exists between architectural/urban characteristics and spatial user behaviour; weak coupling → misaligned spatial cues, wayfinding failures, bottleneck formation. **RQ: Can pedestrian trajectories be predicted from the relationship between human behaviour and architectural/urban space?** |
| Hypothesis | 300 [allocation] | `hypothesis` (to source) | Space inhibits/shapes user behaviour; capture that behaviour → data; through that data predict user trajectories. |

### Chapter 2 — Tools + Instruments for Behavioral Analysis · target **±3,000 words**
| Subsection | Target (words) | Figures expected | Notes from outline |
|---|---:|---|---|
| Space Syntax | 450 [allocation] | `space_syntax` (to source) | What Space Syntax is + Hillier's philosophy; the many attempts to use it; how it correlates to this topic and where Motion Pixels stands against it. |
| Literature Review | 700 [allocation] | corpus table (from outline) | Corpus table: paper / model used / dataset-setting / why it matters for Motion Pixels. Entries incl. Transformer nets for pedestrian prediction (Pérez et al. 2020), flow prediction GNN (Zhang et al. 2022), Transformer+CVAE (2021), Social Graph Transformer (2023), Chalmers deep-sequence (2024), Social-LSTM variant (2024), SocialMP (2025), building-occupancy Transformer (Su et al. 2023), ISPRS radar (2025), ACADIA motion-capture (2024). **Several entries are incomplete — do not fabricate citations.** |
| Practical Review | 350 [allocation] | `PRACTICAL_REVIEW` (to source) | Commercial tools: **Bentley — OpenPaths + LEGION** (agent-based crowd sim); **Autodesk — InfraWorks** (infrastructure/mobility planning). |
| The gaps and positioning | 350 [allocation] | `gaps` (to source) | Section A Space Syntax → *explains* movement (lacks real-world observation); Section B Literature → *measures* movement (gap: generalisation); Section C Simulation software → *simulates* movement (lacks predictive layer). **Motion Pixels predicts movement.** |
| Computational Pipeline | 800 [allocation] | `computational_pipeline` diagram | Transcribe the pipeline diagram into detailed text: detection + tracking (YOLOv8 + ByteTrack), homography/calibration, spatial encoding, metrics (speed/direction/distance/density/stops/dwell), dataset, LSTM, horizon predictions (H20≈1 m, H60≈3 m, H100≈5 m, H200≈10 m, H400≈20 m), outputs (2D/3D plots, charts, dataset), behaviour maps (bottlenecks, flow field, speed). Explain the AI-model options tested and why many horizons. |
| The Ethical Question | 350 [allocation] | GDPR Art. 89(1) + anonymised-tracking figure | Read GDPR **Article 89(1)** and argue the research is legally grounded: personal data may be processed for scientific research with appropriate safeguards; system extracts **anonymous** trajectories/behavioural patterns — faces/names/identities not part of the analysis. **Article 89 is a safeguards provision, not blanket permission — qualify the "Yes, categorically" slide claim.** |

### Chapter 3 — Motion Pixels, learning movement from Barcelona · target **±6,000 words**
Outline explicitly splits this chapter: **MACBA sandbox ≈2,000 words**, **the rest ≈4,000 words**.

**3.1 MACBA, the sandbox experiment (≈2,000 words)**
| Sub-part | Target | Figures expected | Notes |
|---|---:|---|---|
| Intro | 350 [allocation] | sandbox / frozen frames | MACBA = Museu d'Art Contemporani de Barcelona; the esplanade in front is a skater gathering spot — chosen because skateboards give varied speeds, angularities and trajectories. One MACBA→esplanade video became the sandbox to test every digital tool from the pipeline. |
| Behavioral Metrics | 450 [allocation] | metric-testing figures | Parameters per test run: active (1–2), paused, dwelling people; direction shifts; lingering zones; speed as average / max / stops; parametrised fast / medium / slow. |
| Prediction Models | 500 [allocation] | model comparison (phase-2b) | Report on models tried — **LSTM, GRU, TCN, small Transformer** (from the lit review); explain what was picked (**LSTM**) and why. |
| Overfit Test + Feature Ablation | 500 [allocation] | overfit-test frames, ablation panels | 10× overfit **capacity test**: hybrid schema = normalised position (u,v) + metric motion (du,dv) + relational spatial encodings (obstacle / boundary distance, entrance affinity). Best config = **Model C (motion + position + spatial context)** recovers angular behaviour → conclusion: **data-bound, not architecture/schema-bound.** Also discuss feature ablation → leads to the agreed dataset. |
| Dataset structure agreed on | 200 [allocation] | schema table | Final Model C feature schema table. |

**3.2 The rest (≈4,000 words)**
| Subsection | Target | Figures expected | Notes |
|---|---:|---|---|
| Building the dataset | 1,000 [allocation] | site frozen frames / plans | Dataset expanded across Barcelona by site-hunting for variety: MACBA + esplanade espanya; staircases/angularity at Montjuïc stairs; open plazas (Plaça Montjuïc, Plaça Espanya, Plaça Catalunya); circular movement at the red bridge (Passeig de Colom). **Outline says: across 8 recordings, ~7,000 trajectories extracted, ~500 make it to training after culling.** *Conflict:* final V3 training set = **5 recordings, 3,534 tracks** (see `03_FACT_CHECK.md`, `04_OPEN_ISSUES.md`). |
| The Behavioral Maps | 950 [allocation] | Heatmap, Flow Field, Speed Control, Prediction Projections | Explain what each map does, its legend, and which dataset feature built it. |
| Predictive Layer: Horizon Rollouts | 1,000 (outline-stated) | best/worst per horizon + stress | Horizons 1 m / 3 m / 5 m / 10 m / 20 m (H20/H60/H100/H200/H400) **+ stress test 40/60/100 m**. Discuss best/worst graph per prediction step, what each graph offers, what a stress test is and why. Outline pins this at ~1,000 words within the 4,000. |
| Building the prototype | 800 [allocation] | prototype frozen frames | Purpose of the prototype: wraps the research into a functional, informative platform for architects / urban designers (upload → calibrate → process → inspect observed & predicted movement). **State it is a mocked front-end demonstrator, not a connected production system.** |

### Chapter 4 — What comes next? · target **±2,500 words**
| Subsection | Target | Figures expected | Notes (author bullets from outline) |
|---|---:|---|---|
| What did the model learn | 650 [allocation] | — | Recent movement = strongest short-term predictor; spatial context (obstacles/boundaries) helps when enough data; direction & curvature harder than position; model was **data-bound, not architecture-bound.** |
| Limitations of research | 650 [allocation] | — | Limited number/diversity of real trajectories; tracking errors & occlusions; calibration + spatial encoding still manual; longer predictions accumulate error and go too straight; results not yet generalisable across all urban space types. |
| The meaning of it all | 600 [allocation] | — | Prediction is **not** the final objective — understanding space through behaviour is; movement as architectural information/representation; connects designed space with real use; Observe → Understand space → Inform design. |
| Future research directions | 600 [allocation] | — | More sites/typologies; better heading/curvature/long-horizon; pedestrian-to-pedestrian interactions; richer architectural features + 3D context; design feedback loop Observe → Predict → Modify → Evaluate; ultimately a behaviour-informed architectural design tool. |

### Conclusion
- Outline shows a standalone **"Conclusion"** heading (after Chapter 4) with **no stated word count**.
- **[allocation — editorial] ~600 words.** Prior draft = 582 words. Not a fifth content chapter; a closing.

---

## Back matter

### 8. Bibliography
- **Harvard style**, **minimum 20 references** (`citethisforme.com/harvard-referencing`).
- Not counted in word budget.

### 9. Back cover
- A4 vertical, **NEW IAAC logo.**

---

## Word-count roll-up

| Block | Target |
|---|---:|
| Chapter 1 | 2,000 |
| Chapter 2 | 3,000 |
| Chapter 3 | 6,000 |
| Chapter 4 | 2,500 |
| **Chapters subtotal** | **13,500** |
| Abstract | ≤500 |
| Conclusion | ~600 [editorial] |
| **Expected narrative total** | **≈14,100–14,600** |

Front matter (cover, title, preface/acknowledgments/AI declaration) and back matter (bibliography, back
cover) are **not** in the narrative count.

**Is the outline fully understandable?** Yes — the six images give a complete, unambiguous chapter/section
structure with word targets for all four chapters. The only genuinely under-specified items are: the exact
thesis **title/subtitle**, the **Conclusion** word budget, and several **incomplete literature-table
citations**. These are recorded in `04_OPEN_ISSUES.md`, not invented here.
