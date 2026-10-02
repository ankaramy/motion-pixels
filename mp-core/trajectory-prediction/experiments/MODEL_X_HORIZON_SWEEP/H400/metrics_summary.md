# MODEL_X H400 — metrics summary

Approx distance ~**20.0 m**. Best epoch 15 (val ADE 3.994).
Single-step training objective; 400-step autoregressive rollout at eval; spatial features frozen.

## Data availability
- need >= 410 frames; qualifying tracks: train 501 / val 62 / test 63
- dropped 2908 short tracks; train windows 581319; test windows 38639

## Test metrics (all windows)
- ADE **5.079 m** (median 4.530)
- FDE **10.015 m** (median 8.896)
- angular err 83.8° mean / 70.2° median
- GT len median 22.98 m | pred len median 4.86 m | ratio **0.27**
- pred len p75 8.77 m (GT p75 48.48 m)
- **collapse rate (pred<25% GT): 48.3%**

## Per-recording

| recording | n | ADE | FDE |
|---|---:|---:|---:|
| esplanade_espanya_01 | 16475 | 5.717 | 11.457 |
| placa_catalunya_01 | 11739 | 3.027 | 6.166 |
| placa_espanya_01 | 7066 | 8.518 | 16.022 |
| red_bridge_combined_01 | 150 | 2.285 | 4.330 |
| stairs_montjuic_01 | 3209 | 1.873 | 3.729 |