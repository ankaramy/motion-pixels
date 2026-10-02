# Content Final Revision Report

Revision of the Motion Pixels thesis against the author's three-part review sheet (author-input answers,
image comments, text comments). Output: `output/pdf/MOTION_PIXELS_CONTENT_FINAL.pdf` (46 pages). The prior
revision `output/pdf/Motion_Pixels_Thesis_Draft.pdf` is left in place. Written content is now treated as
frozen pending the design phase.

## Author-input items resolved (the 11 numbered red flags)

| # | Location | Answer given | Action taken |
|---|---|---|---|
| 1 | Ch1, Whyte bench | keep citation, do not reference | Marker removed; sentence kept; no reference added |
| 2 | Ch2, practical review figure | refer to BOOKLET_Images | Placed Bentley OpenPaths + Autodesk InfraWorks screenshots |
| 3 | Ch2, ethics (controller/lawful basis) | confirmed | Marker removed; reworded as arrangements addressed under the research safeguards |
| 4 | Ch2, GDPR composite figure | ignore that figure and remove it | Figure removed entirely |
| 5 | Ch3, MACBA frame | frame is not the esplanade; see figure comments | Replaced with correct MACBA sandbox tracking frames from BOOKLET_Images |
| 6 | Ch3, model-comparison figure | transformer not in image; remove from title | Caption reworded to LSTM/GRU/TCN; transformer dropped from the caption |
| 7 | Ch3, feature-schema table | refer to BOOKLET_Images | Placed `figure_3_7_model_c_feature_schema.png` |
| 8 | Ch3, occupancy heatmap | no additional heatmap; final result | Marker removed; density map stated as the final occupancy reading |
| 9 | Ch3, stress test range | stick to 40, 50 and 60 | Text changed to 40, 50 and 60 m |
| 10 | Ch3, prototype screenshots | refer to BOOKLET_Images | Placed platform screenshots, described from supplied notes |
| 11 | Ch4, limitations diagram | refer to BOOKLET_Images | Placed `limitations.png` in the Limitations section |

No `AUTHOR INPUT REQUIRED` or `CITATION REQUIRED` markers remain in the manuscript (verified). No red
correction boxes appear in the output.

## Images added (29 figures total, from 11 before)

**Chapter 1 (4).** Fig 1.1 desire-line stock photos (3-up, credited as stock); Fig 1.2 William Whyte
observing plazas (2-up, credited Project for Public Spaces); Fig 1.3 research question; Fig 1.4 hypothesis.

**Chapter 2 (4).** Fig 2.1 Space Syntax (credited after Hillier, Space is the Machine); Fig 2.2 practical
review, Bentley OpenPaths + InfraWorks (2-up, credited); Fig 2.3 gaps and positioning; Fig 2.4 computational
pipeline (placed first in the pipeline section).

**Chapter 3 (20).** MACBA context photos (2-up, credited MACBA); MACBA sandbox tracking frames (2-up);
early sandbox behaviour maps, bottleneck heatmap + flow field + linger zones (3-up); model comparison;
capacity collage; feature-ablation panels (3-up); Model C schema table; site mass plans (5-up); per-site
tracking frames (5-up); site mask; flow field; speed maps (3-up); density/heatmap (3-up); dual-plaza hero;
prediction projections (3-up); horizon best/worst at H100 and at H400 (with purple-best / green-worst
legend in the captions); three platform screenshots.

**Chapter 4 (1).** Fig 4.1 limitations diagram.

Multi-image figures are used for sequences and comparisons (sites, best/worst, map families). No borders or
frames are drawn around any image, per instruction.

## Images replaced

- MACBA tracking frame: `mp-data/raw/images/frame.png` (wrong site) → correct MACBA sandbox tracking frames.
- Behaviour-map singles → multi-site comparisons (speed and density) using the supplied dataset exports.
- Horizon figures: MODEL_X sweep panel and presentation collage → the supplied best/worst rollout previews
  with the purple/green legend.
- Prototype placeholder → the four supplied platform screenshots.

## Images removed

- GDPR / Article 89 composite figure (Ch2), per answer 4.
- Objective-arc placeholder diagram (Ch4): no source existed; the X→XR→XC arc is already covered in Ch3
  text, and the 4.1 slot now holds the limitations diagram.

## Text comments implemented

- **Face blurring**: added to the MACBA sandbox workflow (anonymisation before storage, applied to every
  recording).
- **Prototype screenshots**: described using the supplied per-screenshot notes (initial screen, calibration,
  studio dashboard, export).
- **Future research** expanded: a more developed, better-tuned prediction tool trained on a larger dataset;
  richer behavioural maps that combine features; a 3D reconstruction of the space from video with behaviour,
  including heatmaps, read in three dimensions.
- **Conclusion ending** rewritten to return to spatial design and spatial intelligence, preserving the
  supplied argument in the author voice (behaviour as feedback into design, not a by-product).

## Technical sections shortened

- Ch2 detection/tracking: the orientation debugging narrative was cut to the methodological choice (frames
  rotated upright, high resolution, low confidence threshold, high-recall tracking). Reproducibility detail
  preserved.
- Ch2 ethics: the hedged multi-sentence qualification trimmed to one plain statement.
- Ch3 density/heatmap paragraph: the "what the outline calls a heatmap" aside removed; stated directly as
  the final occupancy reading.
- Preserved in full: calibration, spatial encoding, feature schema (six/eight/ten/eleven counts), model
  architecture (2-layer LSTM, hidden 128, dropout 0.2, window 10), training (AdamW, thirty epochs, batch
  1024, MSE on next displacement), horizons, metrics, and all quantitative results.

## LLM-language and editorial audit

- Zero em dashes (verified across all chapters and front/back matter).
- Zero occurrences of the flagged formulaic words (furthermore, moreover, additionally, ultimately,
  importantly, this highlights/demonstrates/underscores, by leveraging, robust, holistic, etc.).
- Self-conscious hedging ("worth being honest", "I have to be careful", "the honest version") already
  removed in the previous pass; spot-checked again.
- Figures renumbered sequentially per chapter (1.1-1.4, 2.1-2.4, 3.1-3.20, 4.1). No figure is referenced by
  number in prose, so captions are self-contained.

## Remaining citation issues

- Whyte bench observation: per answer 1, the claim is kept without a specific reference. Whyte (1980) and
  the Project for Public Spaces seating primer remain in the bibliography as general support.
- No fabricated citations were introduced. Bibliography stands at 23 verified entries.

## Remaining factual uncertainties

- Red Bridge public name / precise location: still described by its working name; not asserted beyond that.
- Acquisition funnel: the outline's ~8 recordings / ~7,000 extracted vs the final 5 recordings / 3,534
  tracks is presented as a funnel (acquisition vs trained); the exact per-stage counts are not itemised.
- Prediction rollouts in the dataset figures are illustrative 600-step MODEL_X H400 rollouts, labelled as
  illustrative, not validated routes.

## Final chapter word counts (transitions and captions excluded)

| Section | Words | Target | Status |
|---|---:|---|---|
| Abstract | 463 | ≤500 | within |
| Ch 1 The Flaw In The Plan | 1,898 | 2,000 | within 5% |
| Ch 2 Tools And Instruments | 3,515 | 3,000 (cap 5,000) | within raised cap |
| Ch 3 Learning Movement From Barcelona | 5,617 | 6,000 | −6.4% (slightly under) |
| Ch 4 What Comes Next | 2,565 | 2,500 | within 5% |
| Conclusion | 641 | 600 | +6.8% (strong spatial-design ending, as requested) |

Chapter 3 sits a little under target after the technical simplification and the earlier tone cuts; the
argument and all evidence are intact, and the section now carries far more visual material. It can be topped
back to 6,000 with supported content on request.

## Build

Reproducible with `python THESIS/research/build_thesis_pdf.py` (read-only over research data; no training or
inference). Images resolve from `THESIS/figures/booklet/` (a self-contained copy of BOOKLET_Images).
