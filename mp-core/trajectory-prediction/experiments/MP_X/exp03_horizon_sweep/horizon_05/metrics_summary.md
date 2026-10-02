# exp03 horizon_05 — metrics summary

Turn-only Model C, horizon **5**, displacement floor **0.5 m** (heading>30°, max-step<0.6m fixed). Best epoch 13 (val MSE 2.66414), 18.2s. Trained on 49,439 genuine-turn windows.

Held-out full-horizon eval window bins: straight=199462, mild_30_90=364, sharp_90_150=247, uturn_150_180=462 (genuine rolled: 1073).

## Held-out: all windows vs genuine turns

| subset | n | ADE (m) | FDE (m) | ang err (°) | GT Δhead (°) | pred Δhead (°) | angularity | TCR |
|---|---|---|---|---|---|---|---|---|
| all_windows | 4000 | 0.117 | 0.205 | 77.6 | 75.8 | 27.5 | 10.28 | 0.42 |
| genuine_turns | 1073 | 0.418 | 0.674 | 78.0 | 120.9 | 43.1 | 0.45 | 0.58 |

## Held-out: per turn bin

| bin | n | ADE (m) | FDE (m) | ang err (°) | pred Δhead (°) | angularity | TCR |
|---|---|---|---|---|---|---|---|
| straight | 3979 | 0.115 | 0.203 | 77.5 | 27.4 | 10.33 | 0.41 |
| mild_30_90 | 364 | 0.394 | 0.632 | 69.1 | 38.4 | 0.75 | 0.57 |
| sharp_90_150 | 247 | 0.414 | 0.685 | 78.7 | 35.3 | 0.30 | 0.54 |
| uturn_150_180 | 462 | 0.440 | 0.702 | 84.6 | 50.9 | 0.30 | 0.60 |