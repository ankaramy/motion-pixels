# MODEL_X H100 — metrics summary

Approx distance ~**5.0 m**. Best epoch 11 (val ADE 1.397).
Single-step training objective; 100-step autoregressive rollout at eval; spatial features frozen.

## Data availability
- need >= 110 frames; qualifying tracks: train 1386 / val 184 / test 175
- dropped 1789 short tracks; train windows 772805; test windows 69718

## Test metrics (all windows)
- ADE **1.419 m** (median 0.922)
- FDE **2.565 m** (median 1.730)
- angular err 79.7° mean / 63.7° median
- GT len median 5.31 m | pred len median 1.88 m | ratio **0.35**
- pred len p75 3.02 m (GT p75 12.92 m)
- **collapse rate (pred<25% GT): 38.0%**

## Per-recording

| recording | n | ADE | FDE |
|---|---:|---:|---:|
| esplanade_espanya_01 | 24442 | 1.323 | 2.423 |
| placa_catalunya_01 | 21815 | 0.961 | 1.911 |
| placa_espanya_01 | 13873 | 2.829 | 4.620 |
| red_bridge_combined_01 | 3039 | 0.809 | 1.633 |
| stairs_montjuic_01 | 6549 | 0.598 | 1.354 |