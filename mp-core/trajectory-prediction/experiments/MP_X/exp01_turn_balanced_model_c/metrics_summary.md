# exp01 — Turn-balanced Model C — metrics summary

Dataset: `model_C_dataset.csv` — 1,064,379 rows · 3,534 tracks.

## Sampler

WeightedRandomSampler weights: {'straight': 1.0, 'mild_30_90': 8.0, 'sharp_90_150': 12.0, 'uturn_150_180': 16.0}.  Label horizon = 20 steps.  Best epoch 2 (val MSE 0.49480), 17 epochs in 229.1s.

## Training-window turn bins

- train windows (824,917): straight=814012, mild_30_90=3042, sharp_90_150=2316, uturn_150_180=5547
- val windows (124,265): straight=124210, mild_30_90=12, sharp_90_150=13, uturn_150_180=30

## Held-out evaluation full-horizon window bins (val+test)

- straight=188496, mild_30_90=15, sharp_90_150=12, uturn_150_180=23  (total 188,546)
- genuine-turn windows rolled out: 50

## Held-out: all windows vs genuine turns

| subset | n | ADE (m) | FDE (m) | ang err (°) | GT Δhead (°) | pred Δhead (°) | GT path (m) | pred path (m) | angularity | TCR |
|---|---|---|---|---|---|---|---|---|---|---|
| all_windows | 4000 | 0.337 | 0.726 | 83.8 | 74.8 | 48.2 | 0.44 | 0.71 | 20.88 | 0.58 |
| genuine_turns | 50 | 1.418 | 2.692 | 89.8 | 124.2 | 38.0 | 2.64 | 1.22 | 0.37 | 0.60 |

## Held-out: per turn bin

| bin | n | ADE (m) | FDE (m) | ang err (°) | GT Δhead (°) | pred Δhead (°) | angularity | TCR |
|---|---|---|---|---|---|---|---|---|
| straight | 3999 | 0.337 | 0.726 | 83.9 | 74.8 | 48.2 | 20.89 | 0.58 |
| mild_30_90 | 15 | 1.326 | 2.641 | 79.3 | 49.6 | 24.0 | 0.56 | 0.53 |
| sharp_90_150 | 12 | 1.569 | 2.939 | 100.8 | 124.5 | 34.0 | 0.28 | 0.42 |
| uturn_150_180 | 23 | 1.400 | 2.596 | 90.8 | 172.7 | 49.1 | 0.28 | 0.74 |

## Train (capacity reference — model trained on these)

genuine-turn windows: 2052.  Full-horizon bins: straight=769961, mild_30_90=2925, sharp_90_150=2253, uturn_150_180=5455.

| subset | n | ADE (m) | ang err (°) | pred Δhead (°) | angularity | TCR |
|---|---|---|---|---|---|---|
| all_windows | 4000 | 0.559 | 75.1 | 43.9 | 19.70 | 0.50 |
| genuine_turns | 2052 | 1.241 | 71.3 | 49.2 | 0.47 | 0.53 |

Plots in `plots/` (best/worst/collage genuine turns from held-out (val+test); heading_change_comparison.png).
