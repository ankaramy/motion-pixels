# MODEL_X H60 — metrics summary

Approx distance ~**3.0 m**. Best epoch 10 (val ADE 0.974).
Single-step training objective; 60-step autoregressive rollout at eval; spatial features frozen.

## Data availability
- need >= 70 frames; qualifying tracks: train 1695 / val 221 / test 211
- dropped 1407 short tracks; train windows 796736; test windows 77438

## Test metrics (all windows)
- ADE **1.011 m** (median 0.647)
- FDE **1.811 m** (median 1.219)
- angular err 79.4° mean / 61.9° median
- GT len median 3.20 m | pred len median 1.16 m | ratio **0.35**
- pred len p75 1.48 m (GT p75 7.70 m)
- **collapse rate (pred<25% GT): 38.8%**

## Per-recording

| recording | n | ADE | FDE |
|---|---:|---:|---:|
| esplanade_espanya_01 | 25984 | 1.040 | 1.888 |
| placa_catalunya_01 | 24342 | 0.611 | 1.206 |
| placa_espanya_01 | 15687 | 2.049 | 3.344 |
| red_bridge_combined_01 | 3982 | 0.319 | 0.625 |
| stairs_montjuic_01 | 7443 | 0.404 | 0.925 |