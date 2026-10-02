# Motion Pixels — Final Publication QA Report

**Output:** `publication_final/MOTION_PIXELS_FINAL.pdf` — 91 A4 pages, 16.3 MB.
**Build:** `build.py` (HTML/CSS → headless Chromium print PDF) → per-page rasters in `qc/`, facing-spread
preview `MOTION_PIXELS_FINAL_SPREAD_PREVIEW.pdf` + `qc/_spreads_contact.png`.
**Source text:** the authoritative thesis `.md` files, used unaltered (no paraphrase, no added prose, no
em dashes introduced, numbers and model names unchanged).

## Method
The approved rebrand test (`publication_rebrand/test/`) was taken as the design baseline and matured into a
full book. Every author correction was applied. Rendered through the same headless-Chromium pipeline the
test used, for print-precise A4.

## Author corrections — applied
1. **Accent colour** shifted from the test's `#E6009B` to a deeper purple-magenta **`#8E1C74`** (lifted
   `#C4238C` on dark grounds). Used only for numbering, labels, ticks, selected emphasis — never body.
2. **Chapter numbering** is plain `1 2 3 4` (no `C.` prefix).
3. **Chapter titles enlarged** to 46 pt (from 30) with the oversized number and large void kept.
4. **Running header** recomposed as a spread: `MOTION PIXELS` far-outer-left on left pages, chapter name
   far-outer-right on right pages, **no rule**, each shown once per spread.
5. **Body justified** with automatic hyphenation (verified: clean hyphenation, no rivers, no clipping).
6. **Visuals centred** where it earns authority (pipeline, gaps, limitations, flow field, mask, atlas).
7. **Accidental void reduced** — text blocks balanced by vertical centring; Chapter 4 kept deliberately
   calmer per the brief.
8. **Visual scale raised** for the pipeline, limitations and behaviour maps (see below).
9. **Behaviour maps kept light** with their own analytical colour and legends; not re-rendered dark.
10. **Source Serif 4 body / Inter structural** contrast kept.
11. **Diagram title cropping** limited to redundant `motionpixels` branding; no data/label/legend/arrow
    removed. Crops in `assets/` (`pipeline_crop`, `limitations_tight`, `gaps_tight`, dark
    `research_question_crop`, `hypothesis_crop`).

## Key spreads / decisions
- **Pipeline (p31)** — full-width diagram + a six-stage numbered explanatory band + horizon key; now a
  primary object (correction 23).
- **Limitations (p81)** — re-cropped to its content bounds and enlarged to fill the page (correction 37).
- **Behaviour atlas (p57, p59)** — every available speed map and density map (7 + 7) as comparative
  plates, plus 3-site highlights (p60, p61); legends preserved (correction 27/28).
- **Prediction climax** — horizon sequence H20→H400 (p66–p70) and the **dark dual-plaza hero spread**
  (p64–p65). Only two dark spreads in the book: the Ch1 conceptual RQ/hypothesis spread (p18–p19) and the
  hero, plus the covers — dark kept exceptional (correction 32).
- **Covers** used exactly as supplied, full-bleed (p1 front, p91 back).

## Checks performed
- **Per-page raster review** of all 91 pages (`qc/_contact_all.png` + zoom): no clipped/overflowing text,
  no missing or distorted images, no caption stranded from its figure, legends legible.
- **Spread review** (`qc/_spreads_contact.png`): facing pairs judged as compositions; rhythm alternates
  text / image / dark / grid without repeated identical templates.
- **Overlap fix** — the desire-line figure (p13) was rebuilt in flow layout after an initial caption/image
  overlap; re-verified.
- **Figure numbering** — removed a duplicate `Fig 3.12/3.13` (atlas plates now carry plate captions; the
  canonical 3.12/3.13 remain on the highlight pages) and the stray `Fig 1.0`.
- **Content integrity** — abstract, preface, AI declaration, acknowledgments, all four chapters and their
  sections, conclusion and full bibliography present in order; figures map to Ch1–Ch4 figure list.
- **File size** — source images downscaled/recompressed at build (`_opt/`, longest side 2000 px, hero
  3400 px, covers q92) → 101 MB reduced to 16 MB with quality preserved (cover/hero spot-checked).

## Known intentional states (not defects)
- Chapter 4 and the conclusion carry more whitespace than Chapter 3 (brief: Ch4 calmer).
- Speed maps read pale — that is the source rendering; evidence is not recoloured.
- The limitations diagram keeps the source's brighter magenta (analytical export, not recoloured).

## Reproduce
`python publication_final/build.py`
