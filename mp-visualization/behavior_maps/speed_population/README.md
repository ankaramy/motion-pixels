# Speed Population

A new map in the Motion Pixels **Speed Map** family, deliberately distinct from
Flow Fields. Every moving dot is **one tracked pedestrian**, plotted at their real
world position and coloured by their **real instantaneous speed**
(red = slow · yellow = medium · green = fast). Pedestrians enter, move, slow
down / speed up, and exit through a clean *arctic* architectural plan with a
subtle metric grid.

| | Flow Fields | **Speed Population** |
|---|---|---|
| Shows | collective movement field | individual pedestrians |
| Aesthetic | dark, glowing strings | light, cartographic, arctic |
| Colour | flow density | pedestrian speed |
| Elements | strings + dust + bloom | discrete dots, no trails/glow |

**Visualization only.** A rendering of existing tracked pedestrian trajectories.
No retracking, recalibration, model inference, or source-data modification.

## Versions
- **V3 (current)** — `generate_speed_population_v3.py`. Full-frame arctic plan
  with the legend as a bottom **frosted HUD overlay** (no separate band), Flow
  Fields-matched layout, **slower** dot motion (≈14.4× real-time, ~41% slower
  than V2) for perceivable travel, **longer fading traces** (3/5/7 s tested,
  5 s default), and two **elegant palettes** (warm coral→amber→teal, cool
  violet→amber→cyan). **Recommended: warm palette, 5 s trace.**
- **V2** — `generate_speed_population_v2.py`. Deeper arctic background (edge AO +
  soft drop shadow), Flow-Fields-style light footer band, denser field via honest
  temporal persistence (slow lingers 1.8 s / fast 0.5 s), red/yellow/green dots,
  global-palette delta-encoded GIF.
- **V1** — `generate_speed_population_v1.py`. Original instantaneous dots,
  continuous + discrete colour variants.

## Run

```bash
cd mp-visualization/behavior_maps/speed_population
python generate_speed_population_v3.py            # current (V3)
python generate_speed_population_v3.py --preview  # quick look
python generate_speed_population_v3.py --no-tests # skip trace-length test GIFs
python generate_speed_population_v2.py            # V2
python generate_speed_population_v1.py            # original V1
```

## Data
- Trajectories: `new_datasets/placa_catalunya_01/filtered_250m/trajectories_world_filtered_250m.csv`
  (same dataset as the Flow Fields animation).
- Plan + calibration: reused via the existing `warp_plan_to_world` homography
  (no recalibration).

## What it does
- Loads each pedestrian track and measures real speed from world coordinates and
  the `time_s` axis (lightly denoised position, then central difference).
- Compresses the full source time range (~5:17) uniformly into a 12 s
  presentation, so **relative speed is preserved** — fast pedestrians visibly
  move faster than slow ones. The footer timer shows **source-video** time.
- Renders dots over an arctic-styled plan (desaturated, brightened, cool-tinted,
  low-contrast) with a subtle 5 m / 10 m grid.

## Outputs (`outputs/`)
V3 (current):
- `placa_catalunya_speed_population_v3_palette_warm.gif` / `.mp4` / `_still.png` — **recommended**
- `placa_catalunya_speed_population_v3_palette_cool.gif` / `.mp4` / `_still.png`
- `placa_catalunya_speed_population_v3_trace_{3,5,7}s.gif` — trace-length tests
- `placa_catalunya_speed_population_v3_report.md` — V3 method report

V2:
- `placa_catalunya_speed_population_v2_high_quality.gif` — primary GIF
- `placa_catalunya_speed_population_v2.mp4` — MP4 backup
- `placa_catalunya_speed_population_v2_still.png` — still
- `placa_catalunya_speed_population_v2_persistence_{1,2,3}s.gif` — persistence tests
- `placa_catalunya_speed_population_v2_report.md` — full method report

V1:
- `placa_catalunya_speed_population_v1.gif` — primary GIF (discrete)
- `placa_catalunya_speed_population_v1.mp4` — MP4 backup
- `placa_catalunya_speed_population_v1_still.png` — still
- `placa_catalunya_speed_population_v1_{discrete,continuous}.gif` — colour variants
- `placa_catalunya_speed_population_v1_report.md` — V1 method report

See the report for all parameters and the continuous-vs-discrete recommendation
(**discrete** is the recommended default).
