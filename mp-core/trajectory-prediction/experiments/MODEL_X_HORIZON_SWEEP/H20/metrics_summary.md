# MODEL_X H20 — metrics summary

Approx distance ~**1.0 m**. Best epoch 25 (val ADE 0.416).
Single-step training objective; 20-step autoregressive rollout at eval; spatial features frozen.

## Data availability
- need >= 30 frames; qualifying tracks: train 2269 / val 296 / test 286
- dropped 683 short tracks; train windows 817830; test windows 87320

## Test metrics (all windows)
- ADE **0.446 m** (median 0.214)
- FDE **0.708 m** (median 0.325)
- angular err 74.1° mean / 54.6° median
- GT len median 1.03 m | pred len median 0.44 m | ratio **0.45**
- pred len p75 0.65 m (GT p75 2.52 m)
- **collapse rate (pred<25% GT): 25.9%**

## Per-recording

| recording | n | ADE | FDE |
|---|---:|---:|---:|
| esplanade_espanya_01 | 27740 | 0.484 | 0.760 |
| placa_catalunya_01 | 27691 | 0.195 | 0.338 |
| placa_espanya_01 | 17964 | 1.018 | 1.572 |
| red_bridge_combined_01 | 5502 | 0.132 | 0.245 |
| stairs_montjuic_01 | 8423 | 0.129 | 0.219 |