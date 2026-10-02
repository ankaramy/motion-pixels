# Motion Pixels â€” Final Polish Pass (13)

Second refinement-only pass. Same files (`app.js`, `render-methods.js`, `styles.css`).

### Landing
- Removed the "IAAC Â· Architectural Research" eyebrow.
- "Start Analysis" â†’ **"Start New Study"** (Open Last Study unchanged).
- Logo still wired to `mp-stilllogo.png` (placeholder shipped â€” see Assets note).

### Profile panel
- Identity changed to **Antoni GaudÃ­ Â· antoni.gaudi@mp.com**.
- Generic dots replaced with **real plan thumbnails** in Recent Projects & Saved Studies.

### Studio
- **Artifact cleanup:** removed `buildBehaviorAccents` from the scene â€” the raw-generation
  green flow arrows, isolated cyan speed dots, and orange bottleneck patches are gone.
  Only the final cleaned map overlays remain (verified 0 accent nodes).
- **Saved / Layers / Export rail:** replaced the absolute-positioned sliding panels (which
  overlapped) with a docked, scrollable column of three independent, non-overlapping
  cards (verified: each section top â‰¥ previous bottom).
- **Export Drawing** (renamed from "Export Studio Drawing"): opens an export modal with
  **PNG** and **SVG** download cards. SVG serializes the live drawing; PNG rasterizes the
  plan drawing at 2400 px. Both trigger real downloads.
- **Save Analysis:** button background is now white (was cream); it opens a **Save** modal â€”
  Select Project (from the library, with thumbnails) / Create New Project / Save Study.
- Project cards in the rail now show plan thumbnails.

### Calibration
- Camera-view video now `autoPlay + loop` so it is visible immediately (no black frame),
  centered in its portrait frame.

### Pipeline / Processing
- Removed the faint placeholder grid from the build animation; the progressive
  Speed â†’ Flow â†’ Bottlenecks â†’ Predictions map build-up remains.

### Assets still required from you (Desktop paths aren't reachable here)
- **`mp-stilllogo.png`** â€” real logo (a labeled placeholder is in place; drop your file at
  the project root to replace it).
- **`new_datasets/<location>/plan/â€¦`** â€” per-location plan images for distinct project
  thumbnails. Currently every study uses the one Placa Espanya plan we have. Attach the
  `new_datasets` folder and I'll wire each project to its own plan.
- Calibration uses the existing `tracked.mp4`; the dedicated
  `placa_espanya_01/tracking/tracked.mp4` and the `behavior_maps_final` clips weren't
  reachable â€” the existing real behavior-map clips are used in their place.

---

# Motion Pixels â€” UI Refinement Pass (12)

Refinement-only pass over `11_final_2d_polish`. No redesign, no workflow/navigation
changes. A pristine snapshot of the source files is preserved in
`12_pre_refinement_backup/`.

Files touched: `app.js`, `render-methods.js`, `styles.css`, plus a new
placeholder `mp-stilllogo.png`.

---

## P1 â€” Studio compositing (highest priority)

**Root cause found:** the behavior-map and prediction videos are glow-on-**black**
renders, but the Studio composited them with `mix-blend-mode: multiply`. Multiplying a
black ground over the light plan darkened the whole canvas and stacked multiple semi-
opaque videos on top of each other â€” the "muddy", "leaking", and "stacked footage"
artifacts.

- **Single coherent layer / multi-video overlay fix** (`app.js â†’ buildBehaviorAssetOverlay`):
  each map is now normalized with `filter: invert(1) hue-rotate(180deg) â€¦` which flips its
  black ground to white (white disappears under `multiply`) **while preserving the data
  hues**, so each map paints as one clean layer with no stacked-footage edges.
  `foreignObject` corrected from `800Ã—460` (letterbox edge artifact) to full `800Ã—520`,
  `object-fit: contain â†’ cover`.
- **Artifact removal / toggles fully control visibility:** verified that with all layers
  off the scene contains **0** map videos and **0** accent overlays; each layer renders
  only its own single video when enabled. Nothing residual remains.
- **Prediction visibility:** opacity raised `0.44 â†’ 0.92`, plus higher saturation/contrast
  and a slight brightness trim in the normalization filter, so the prediction layer reads
  clearly while staying elegant.

## P1 â€” Studio palette (cleaner white)

- Canvas base `#fcfcfb â†’ #ffffff`; the real plan image dropped to `opacity 0.58` with a
  desaturate/brighten filter and a cool high-key wash, replacing the muddy cream with a
  clean cool-white architectural plan (`app.js â†’ buildPlan`).
- Studio chrome shifted to a cool neutral: stage `#f6f8f9`, shell `#ffffff`, rail
  `#f6f8f9` (`styles.css`). No dark UI introduced.

## P2 â€” Landing page

- Removed the header tagline "Mapping Out Spatial Intelligence".
- Center text wordmark replaced with an `<img src="mp-stilllogo.png">` lockup, scaled via
  `.home-logo` (max-width ~560px). **A placeholder image is shipped** â€” drop your real
  `mp-stilllogo.png` at the project root to replace it. If the file is missing the Gugi
  text wordmark is shown as a graceful fallback.
- Nav links + eyebrow set to title case.

## P2 â€” Profile system

- Profile button enlarged (`28â†’36px`, rounded) with a clear hover state (`styles.css`).
- Clicking it opens a **Profile drawer** (`render-methods.js â†’ renderProfile`) with three
  sections: **Recent Projects**, **Saved Studies**, **Export History**. Click-outside or
  the Ã— closes it. Data wired in `app.js â†’ renderVals.profileSections`.

## P2 â€” Typography

- Removed **all** `text-transform: uppercase` (25 rules) across the platform.
- Rewrote affected labels to proper title case: Start Analysis, Open Last Study,
  Pair Movement With Space, Continue, Confirm Calibration, Run Pipeline, Context Layers,
  Saved Studies, Behavior Layers, Export Controls, Video Tracking, Metrics, etc.
- Gugi reserved for branding; Space Grotesk for UI (unchanged).
- Onboarding explanatory copy given more weight/contrast (`color #4b4e53`, weight 450,
  line-height 1.78) and headings strengthened to weight 700.

## P3 â€” Onboarding centering & hierarchy

- Upload, Calibration, Spatial Encoding and Processing screens now center their workflow
  card vertically and horizontally (`justify-content: center`).
- Calibration & Encoding workspaces scaled up slightly
  (`calib-panels 54vh`, `encode-layout 62vh`).
- Section titles / metadata contrast improved.

## P4 â€” Processing animation

- Replaced the single inverted flow-field clip with a **progressive build**
  (`app.js â†’ buildProcPreview`): Speed â†’ Flow Fields â†’ Bottlenecks â†’ Predictions layer in
  on top of each other on a dark stage with `screen` compositing, a construction scan
  sweep, and a live "+ layer" readout â€” the dataset visibly becoming a behavioral
  intelligence surface.

---

## Notes

- The original raw footage `placa-espanya-30fps.mp4` exceeded the import size limit, so the
  Calibration "camera view" falls back to `tracked.mp4`. Swap the path back in
  `app.js â†’ buildPlacaDataset` once the raw clip is available locally.
- `mp-stilllogo.png` is a labeled placeholder â€” replace with the real logo.
- Static screen captures of the video-compositing screens (Studio / Processing) can't be
  produced by the DOM-rerender screenshot tool; they render correctly with live `<video>`
  in a real browser.
