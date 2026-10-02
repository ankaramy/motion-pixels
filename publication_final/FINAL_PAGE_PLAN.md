# Motion Pixels — Final Page Plan

91 A4 pages, front cover → back cover, composed as facing spreads (left = even, right = odd). Real thesis
text and real research images only. Built by `build.py`; QA rasters in `qc/`, spread previews in `spreads/`.
Chapter openers land on left pages (10, 20, 34, 76). Dark spreads: 18–19 (conceptual) and 64–65 (prediction
hero) plus the covers. Everything else light.

## Front matter (2–9)
| Pg | Side | Content | Source | Type |
|--|--|--|--|--|
| 1 | R | Front cover, full-bleed | `front_cover.png` | Cover |
| 2 | L | Half-title, void, MOTIONPIXELS mark | — | Void |
| 3 | R | Title page, full metadata | master front matter | Title |
| 4 | L | Abstract I | Abstract | Reading |
| 5 | R | Abstract II + keywords | Abstract | Reading |
| 6 | L | Contents | structure | List |
| 7 | R | Preface + Declaration of AI Use | Preface / AI decl | Reading |
| 8 | L | Acknowledgments | Acknowledgments | Reading (quiet) |
| 9 | R | Prologue image, full-bleed | `beginning_4.jpg` | Hero image |

## Chapter 1 — Observation · "The Flaw in the Plan" (10–19)
| 10 | L | Opener **1** / OBSERVATION / The Flaw in the Plan | — | Opener |
| 11 | R | §1.1 Introduction, lead + body | Ch1 | Reading |
| 12 | L | Body cont. + supporting photo | `beginning_1.jpg` | Text+image |
| 13 | R | Desire lines — Fig 1.1 | `beginning_4`, `beginning_3` | Image |
| 14 | L | Body — intuition, prior attempts, the medium | Ch1 | Reading |
| 15 | R | Early references, Whyte — Fig 1.2 | `whyte.jpg`, `whyte_2.jpg` | Text+image |
| 16 | L | Whyte method / between two references | Ch1 | Reading |
| 17 | R | Research Question, pull quote + hypothesis lead | Ch1 | Reading |
| 18 | L | **Dark** research-question diagram — Fig 1.3 | `research_question_crop` | Dark |
| 19 | R | **Dark** hypothesis diagram — Fig 1.4 | `hypothesis_crop` | Dark |

## Chapter 2 — Instrument · "Tools and Instruments" (20–33)
| 20 | L | Opener **2** / INSTRUMENT | — | Opener |
| 21 | R | Chapter intro + Space Syntax lead | Ch2 | Reading |
| 22 | L | Space Syntax body | Ch2 | Reading (2-col) |
| 23 | R | Space Syntax reading — Fig 2.1 | `space_syntax.jpg` | Image |
| 24 | L | Literature review body | Ch2 | Reading (2-col) |
| 25 | R | Literature review cont. | Ch2 | Reading (2-col) |
| 26 | L | Practical review body | Ch2 | Reading (2-col) |
| 27 | R | Commercial tools — Fig 2.2 | `bentley…`, `infraworks…` | Image+text |
| 28 | L | Gaps & positioning body | Ch2 | Reading (2-col) |
| 29 | R | Positioning diagram — Fig 2.3 | `gaps_crop` | Image (centred) |
| 30 | L | Computational pipeline body | Ch2 | Reading (2-col) |
| 31 | R | **Pipeline, primary large** — Fig 2.4 | `pipeline_crop` | Primary |
| 32 | L | The ethical question (GDPR) | Ch2 | Reading |
| 33 | R | Ethical cont. + transition | Ch2 | Reading (quiet) |

## Chapter 3 — Evidence · "Learning Movement from Barcelona" (34–75) — visual core
| 34 | L | Opener **3** / EVIDENCE | — | Opener |
| 35 | R | Chapter intro | Ch3 | Reading |
| 36 | L | MACBA the sandbox body | Ch3 | Reading |
| 37 | R | MACBA context — Fig 3.1 | `MACBA_2011`, `source_bcncolours` | Image |
| 38 | L | Sandbox rationale + face-blur | Ch3 | Reading |
| 39 | R | Sandbox tracking — Fig 3.2 | `skate_1/2_tracking` | Image (large) |
| 40 | L | Behavioural metrics body | Ch3 | Reading (2-col) |
| 41 | R | Early behaviour maps — Fig 3.3 | sandbox bottleneck/flow/linger | Grid |
| 42 | L | Prediction models body | Ch3 | Reading (2-col) |
| 43 | R | Model comparison — Fig 3.4 | `phase2b…comparison` | Image |
| 44 | L | Overfit + ablation body | Ch3 | Reading (2-col) |
| 45 | R | Capacity test — Fig 3.5 | `overfit10x…collage` | Image (wide) |
| 46 | L | Feature ablation — Fig 3.6 | `feature_ablation/*` | Grid+text |
| 47 | R | Dataset schema — Fig 3.7 | `figure_3_7…`, `datastructure` | Image+text |
| 48 | L | Building the dataset body | Ch3 | Reading (2-col) |
| 49 | R | Site plans — Fig 3.8 | `sites_mass_plan/*` (5) | Grid |
| 50 | L | Dataset body + recording table | Ch3 | Text+table |
| 51 | R | Tracking frames — Fig 3.9 | `dataset/*_tracking` (5) | Grid |
| 52 | L | Culling / masks body | Ch3 | Reading |
| 53 | R | Mask overlay — Fig 3.10 | `manual_masks…overlay` | Image |
| 54 | L | Behavioural maps intro + flow | Ch3 | Reading |
| 55 | R | Flow field — Fig 3.11 | `…flow_fields_still` | Image |
| 56 | L | Speed maps text — atlas intro | Ch3 | Reading |
| 57 | R | **Speed atlas** all sites | `dataset/*_speedmap` (7) | Atlas grid |
| 58 | L | Density maps text | Ch3 | Reading |
| 59 | R | **Density atlas** all sites | `dataset/*_heatmap` (7) | Atlas grid |
| 60 | L | Speed highlight — Fig 3.12 | 3 speed maps | Grid |
| 61 | R | Density highlight — Fig 3.13 | 3 heatmaps | Grid |
| 62 | L | Prediction projections intro | Ch3 | Reading |
| 63 | R | Rollouts over plans — Fig 3.15 | `dataset/*_predictionrollout` (3) | Stacked |
| 64 | L | **Dark hero** dual-placas (left) — Fig 3.14 | `hero_H400_dual` | Hero |
| 65 | R | **Dark hero** dual-placas (right) | `hero_H400_dual` | Hero |
| 66 | L | Horizon rollouts body | Ch3 | Reading (2-col) |
| 67 | R | Horizon body + error table | Ch3 | Text+table |
| 68 | L | Horizon sequence A — Fig 3.16 | H20/H60/H100 best+worst | Grid |
| 69 | R | Horizon sequence B — Fig 3.17 | H200/H400 best+worst | Grid |
| 70 | L | H400 best/worst large + stress test | H400 rollouts | Image+text |
| 71 | R | Prototype intro | Ch3 | Reading |
| 72 | L | Platform start + calibrate — Fig 3.18 | `platform/initiation,calibration` | Image |
| 73 | R | Prototype body + dashboard — Fig 3.19 | `platform/dashboard` | Image+text |
| 74 | L | Prototype status (mocked) | Ch3 | Reading |
| 75 | R | Export — Fig 3.20 + transition | `platform/save` | Image+text |

## Chapter 4 — Reflection · "What Comes Next" (76–85) + Conclusion/end matter (86–91)
| 76 | L | Opener **4** / REFLECTION | — | Opener |
| 77 | R | What the model learned | Ch4 | Reading |
| 78 | L | Body cont. | Ch4 | Reading (2-col) |
| 79 | R | Direction/curvature/under-reach + pull | Ch4 | Reading |
| 80 | L | Limitations body | Ch4 | Reading |
| 81 | R | **Limitations diagram, large** — Fig 4.1 | `limitations_crop` | Primary |
| 82 | L | The meaning of it all | Ch4 | Reading |
| 83 | R | Meaning cont. + pull quote | Ch4 | Reading |
| 84 | L | Future research directions | Ch4 | Reading (2-col) |
| 85 | R | Future cont. | Ch4 | Reading |
| 86 | L | Conclusion (reflective, void) | Conclusion | Reading |
| 87 | R | Conclusion cont. + final pull quote | Conclusion | Reading |
| 88 | L | Bibliography I | Bibliography | List (2-col) |
| 89 | R | Bibliography II | Bibliography | List (2-col) |
| 90 | L | Colophon / quiet close | — | Void |
| 91 | R | Back cover, full-bleed | `back_cover.png` | Cover |
