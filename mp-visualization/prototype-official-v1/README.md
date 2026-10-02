# Motion Pixels â€” Final Prototype

Spatial behavior studio prototype. Turns pedestrian video + an architectural plan into a
calibrated, layered behavioral-intelligence drawing (tracking â†’ calibration â†’ spatial
encoding â†’ pipeline â†’ studio), built with React (no build step) over a set of real
Placa Espanya demo assets.

---

## How to run locally

The app is plain static files (React is vendored, no bundler, no npm install). It **must**
be served over HTTP â€” opening `index.html` via `file://` will fail because the code
`fetch()`es CSVs/JSON and loads `<video>`/`<image>` assets that browsers block on the file
protocol.

Pick any static server:

```bash
# from inside this folder

# Python 3
python3 -m http.server 8000

# or Node
npx serve .

# or PHP
php -S localhost:8000
```

Then open **http://localhost:8000/** in a Chromium-based browser (Chrome/Edge recommended â€”
the studio compositing uses CSS `mix-blend-mode` + `<foreignObject>` video and is tuned
for Chromium).

No API keys, no environment variables, no build.

### Deep links (handy for development)
The app reads URL params on load (`applyReviewRoute`):
- `?screen=home|upload|calibrate|encode|process|workspace`
- `#workspace` jumps straight to the studio
- `?layers=all` enables speed/flow/bottlenecks
- `?progress=0.5` sets the timeline position

Example: `http://localhost:8000/?screen=workspace&layers=all`

---

## Folder structure

```
MotionPixels_Final_Prototype_Export/
â”œâ”€â”€ index.html                 # entry point (loads vendor + app scripts)
â”œâ”€â”€ app.js                     # main React component: state, logic, all SVG/scene builders
â”œâ”€â”€ render-methods.js          # screen render methods (mixed into the component)
â”œâ”€â”€ styles.css                 # all styling (layered passes; refinement passes appended)
â”œâ”€â”€ mp-stilllogo.png           # landing logo (PLACEHOLDER â€” replace with real lockup)
â”œâ”€â”€ vendor/
â”‚   â”œâ”€â”€ react.production.min.js
â”‚   â””â”€â”€ react-dom.production.min.js
â”œâ”€â”€ demo_assets/
â”‚   â”œâ”€â”€ manifest.json
â”‚   â””â”€â”€ placa_espanya/
â”‚       â”œâ”€â”€ plan/placa-espanya.png                  # top-down plan (studio base + thumbs)
â”‚       â”œâ”€â”€ video/tracked.mp4                        # tracked footage (detect + calibration)
â”‚       â”œâ”€â”€ calibration/  calib.json, calib_preview.png, calib_plot.png
â”‚       â”œâ”€â”€ tracking/     tracks_summary.csv, speed_per_track.csv, trajectories_world_sample.csv
â”‚       â”œâ”€â”€ behavior_maps/                           # speed / flow / bottleneck maps (.mp4 + _still.png + .csv)
â”‚       â””â”€â”€ prediction_maps/prediction_anim_placa_espanya_01_plain.mp4
â”œâ”€â”€ CHANGELOG.md               # refinement history (passes 12 + 13)
â”œâ”€â”€ screenshots/               # reference captures (landing, profile, export/save modals)
â””â”€â”€ README.md                  # this file
```

## Main files

| File | Responsibility |
|------|----------------|
| `index.html` | Loads React (vendored), `render-methods.js`, then `app.js`; mounts into `#root`. |
| `app.js` | The `MotionPixelsApp` class â€” all state, pipeline flow, homography math, dataset loading, SVG scene/plan/behavior builders, export & save handlers. `window.__motionPixelsApp` is exposed for debugging. |
| `render-methods.js` | `window.MotionPixelsRenderMethods` â€” per-screen `render*` methods, `Object.assign`ed onto the component prototype. Edit screen markup here. |
| `styles.css` | All CSS. Built up in dated passes; the two most recent blocks are `FINAL UI REFINEMENT PASS (12)` and `FINAL POLISH PASS (13)` and override earlier rules by source order. |

Architecture note: there is **one** React component. `app.js` holds logic + the heavy
`React.createElement` scene/graphic builders; `render-methods.js` holds the page-level
layout for each screen. They are joined at the bottom of `app.js` via
`Object.assign(MotionPixelsApp.prototype, window.MotionPixelsRenderMethods)`.

---

## Current limitations

- **Logo is a placeholder.** `mp-stilllogo.png` is a labeled placeholder; drop the real
  logo at the project root (same filename) to replace it. The landing falls back to a Gugi
  text wordmark if the image is missing.
- **Single dataset.** Only the Placa Espanya demo dataset ships. All "projects" in the
  studies rail / profile / save dialog reuse the one Placa plan image as their thumbnail.
  Per-location plans were not available in this environment.
- **Calibration camera view uses `tracked.mp4`.** The original raw `placa-espanya-30fps.mp4`
  exceeded the import size limit, so calibration falls back to the tracked clip. Swap the
  path back in `app.js â†’ buildPlacaDataset` (`demo_video_path`) once the raw clip is local.
- **Behavior/prediction overlays are pre-rendered videos**, not live computation. They are
  composited over the plan with `invert + hue-rotate + multiply` (maps are glow-on-black
  renders; this normalizes the black ground to white so each layer reads as one clean
  layer). This is Chromium-tuned; Safari/Firefox blend support varies.
- **Export PNG** rasterizes the *plan* drawing (videos can't be serialized into an SVG
  image); **Export SVG** serializes the live scene minus `<foreignObject>` video layers.
- **Static screenshot tools** can't capture the studio/pipeline screens because of the live
  `<video>` compositing â€” they render correctly in a real browser.
- `trajectories_image_space.csv` (large, and unused by the loader) is intentionally omitted;
  the app only fetches `trajectories_world_sample.csv`.

---

## Next development notes (for Codex)

1. **Real per-location datasets.** Generalize `buildPlacaDataset()` into a dataset registry
   keyed by location id; give each project its own `plan`, `tracked.mp4`, behavior maps and
   calibration. Wire the studies rail / profile / save library to those real entries instead
   of the shared Placa plan.
2. **Wire `Start New Study` to a real upload.** `renderUpload` already has file inputs and
   `handleVideoSelected` / `handlePlanSelected`; persist the chosen files and drive the
   pipeline from them rather than from the demo dataset.
3. **Live behavior computation.** Replace the pre-rendered map `.mp4`s with on-the-fly
   rendering from the CSVs already parsed in `loadRealDatasetData()` (`realData.flow`,
   `.bottlenecks`, `.tracks`) so layers reflect the actual loaded study. `buildBehaviorAccents`
   (removed from the studio scene in pass 13) shows the vector approach if you want crisp
   SVG layers back.
4. **Real prediction model hookup.** The pipeline + horizon controls (`H40â€¦H400`) are UI-only;
   connect them to an actual forecast service and feed the prediction layer.
5. **Persist saved studies.** `commitSave` currently appends to in-memory `state.savedStudies`;
   back it with localStorage or a real store, and reflect saves in the profile panel.
6. **Export fidelity.** For a true studio-drawing PNG/SVG (including overlays), render the
   behavior layers as vector/canvas instead of video so they can be serialized/rasterized.
7. **Optional build step.** If you move to a bundler, the component is plain ES â€” split
   `app.js` scene builders into modules and keep `render-methods` as the view layer.
