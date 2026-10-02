# MODEL_X H200 — metrics summary

Approx distance ~**10.0 m**. Best epoch 19 (val ADE 2.496).
Single-step training objective; 200-step autoregressive rollout at eval; spatial features frozen.

## Data availability
- need >= 210 frames; qualifying tracks: train 958 / val 122 / test 120
- dropped 2334 short tracks; train windows 711368; test windows 55505

## Test metrics (all windows)
- ADE **2.595 m** (median 1.916)
- FDE **5.016 m** (median 3.663)
- angular err 83.7° mean / 79.3° median
- GT len median 10.68 m | pred len median 3.22 m | ratio **0.32**
- pred len p75 5.92 m (GT p75 25.95 m)
- **collapse rate (pred<25% GT): 43.3%**

## Per-recording

| recording | n | ADE | FDE |
|---|---:|---:|---:|
| esplanade_espanya_01 | 21322 | 2.346 | 4.488 |
| placa_catalunya_01 | 16915 | 1.896 | 3.889 |
| placa_espanya_01 | 10710 | 5.186 | 9.669 |
| red_bridge_combined_01 | 1512 | 1.059 | 2.374 |
| stairs_montjuic_01 | 5046 | 0.950 | 1.944 |