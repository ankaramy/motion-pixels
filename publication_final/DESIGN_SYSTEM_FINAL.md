# Motion Pixels — Final Design System

Authoritative layout system for `MOTION_PIXELS_FINAL.pdf`. It matures the approved rebrand test
(`publication_rebrand/test/`) and applies the author's corrections. Identity comes from the supplied
covers; the interior stays calmer than the covers. Rendered with headless Chromium (Playwright) from
HTML/CSS for print-precise A4, rasterised for QA with PyMuPDF. Body text is the authoritative thesis
prose, unaltered.

## Page and format
- **Size:** A4 portrait, 210 × 297 mm. Every page A4; the book is composed as facing spreads (left = even
  folio, right = odd folio).
- **Margins (mirrored):** top 15 mm, bottom 16 mm, **outer 17 mm, inner/gutter 21 mm**. Nothing critical
  within ~21 mm of the spine.
- **Grid:** 6 columns per page → 12 across a spread, 4 mm gap. Live text width ≈ 172 mm.
- **Bleed:** full-bleed images run to the trim edge.

## Typography
- **Inter** (sans) — chapter numbers, titles, section numbers/titles, captions, labels, running headers,
  folios, tables, technical annotation.
- **Source Serif 4** (serif) — all body reading text, lead paragraphs, abstract, preface, conclusion.
  *(Author correction 10: serif body kept.)*
- **Gugi** — identity only, one small MOTIONPIXELS mark on chapter openers. Never body/caption/header.

### Type scale (pt)
| Role | Font | Size / leading |
|---|---|---|
| Chapter number | Inter 700 | 210 / 0.85 (openers only) |
| Chapter descriptor | Inter 500 caps tracked | 8.5 |
| Chapter title | Inter 600 | 46 / 48 *(correction 3: enlarged from 30)* |
| Section number | Inter 600 | 11 (accent) |
| Section title | Inter 600 | 18 / 22 |
| Body serif | Source Serif 4 400 | 9.5 / 15 **justified, hyphenated** *(correction 5)* |
| Lead paragraph | Source Serif 4 400 | 12.5 / 18.5 (justified) |
| Pull quote | Source Serif 4 400 | 19 / 26 |
| Caption | Inter 400 | 7.8 / 11 (grey; label in accent) *(correction 43: strengthened)* |
| Figure label | Inter 700 | 7.8 (accent) |
| Running header | Inter 700 (brand) / 400 (chapter) | 7.5 tracked |
| Folio | Inter 400 | 8 |
| Technical label / legend | Inter 500 caps +tracking | 7 |

## Colour
- **Ink** `#141416` on **paper** `#ffffff` (the majority of pages).
- **Dark ground** `#0b0b0d`, text `#f2f1ef` — prediction heroes and the single Ch1 conceptual spread only.
- **Accent — deep purple-magenta `#8E1C74`** *(correction 1/7: darker, purpler than the test's `#E6009B`)*.
  On dark grounds a lifted `#C4238C` keeps labels legible. Accent used for section/figure numbers, key
  rules, small square ticks, selected emphasis. **Never body text.**
- **Greys:** `#6b6b6b` secondary/captions, `#c9c9c9` hairline, `#efefef` fill.
- Research graphics keep their own colour systems and legends; the book never recolours evidence
  *(correction 9/11)*. Behaviour maps stay light with their warm density scale.

## Running header *(correction 4 — spread composition, no rule)*
- **Left page, far outer left:** `MOTION PIXELS` (Inter 700, small).
- **Right page, far outer right:** current chapter name (Inter 400, light).
- No horizontal rule. Each piece appears once per spread, not on both pages.
- Omitted on: covers, chapter openers, full-bleed heroes, dark spreads.

## Folios
Bottom outer corner only (left → bottom-left, right → bottom-right). Never near the gutter. Omitted on
covers, openers, full-bleed heroes.

## Chapter openers *(correction 2/3)*
Left page. Very large plain number **1 2 3 4** (no "C." prefix). Small conceptual descriptor top
(OBSERVATION / INSTRUMENT / EVIDENCE / REFLECTION). **Enlarged** chapter title low on the page. Large
intentional void between number and title. One accent tick + small Gugi MOTIONPIXELS mark. No header/folio.

## Image system
- **Hierarchy:** L1 supporting · L2 explanatory · L3 primary (near full-page) · L4 hero (full spread).
  Scale follows importance; original research outputs get authority *(correction 8/17)*.
- **No frames/borders.** Separation by whitespace and alignment.
- **Centre** standalone maps/diagrams where it earns authority; do not force-left everything *(correction 6/16)*.
- **Cropping:** photographs may `object-fit: cover`. Analytical graphics kept whole with legends/axes/scale
  bars intact. Baked "motionpixels" titles cropped **only** where they duplicate publication furniture and
  no data/label/legend/relationship is lost *(correction 11)*; crops live in `publication_final/assets/`.
- **Grids:** comparable maps share cell size and gap; concise consistent labels.

## Behaviour atlas & prediction
- The full set of behaviour maps is shown (density + speed across all sites) as comparative grids at
  substantial scale *(correction 27/28)*. Legends preserved; purple/warm meanings unchanged.
- Prediction is the visual climax: a horizon sequence H20→H400 and dark hero spreads (correction 29–32).
- Dark spreads remain exceptional — reserved for prediction plus one Ch1 conceptual spread.

## Void, rhythm, tables, captions
- Intentional void at openers/conceptual pauses; accidental void reduced by giving research material the
  space *(correction 7/15)*. The template is a language, not one repeated layout *(correction 20)*.
- **Tables:** editorial, minimal — one accent hairline under the header row, hairline row separators, strong
  left alignment, generous padding, no cell boxes *(correction 44)*.
- **Captions:** `Fig n.n` label in accent bold + grey text, attached to the figure, never stranded.

## Fonts / build
Fonts: `publication_rebrand/fonts/{Inter,SourceSerif4,Gugi}.ttf`. Build: `publication_final/build.py`
→ `MOTION_PIXELS_FINAL.pdf` + `qc/` page rasters + `spreads/` spread previews.
