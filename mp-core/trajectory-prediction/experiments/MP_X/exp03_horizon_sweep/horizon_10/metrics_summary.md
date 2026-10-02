# exp03 horizon_10 — metrics summary

Turn-only Model C, horizon **10**, displacement floor **1.0 m** (heading>30°, max-step<0.6m fixed). Best epoch 7 (val MSE 1.94092), 9.8s. Trained on 32,326 genuine-turn windows.

Held-out full-horizon eval window bins: straight=195856, mild_30_90=158, sharp_90_150=79, uturn_150_180=205 (genuine rolled: 442).

## Held-out: all windows vs genuine turns

| subset | n | ADE (m) | FDE (m) | ang err (°) | GT Δhead (°) | pred Δhead (°) | angularity | TCR |
|---|---|---|---|---|---|---|---|---|
| all_windows | 4000 | 0.270 | 0.524 | 84.9 | 79.5 | 43.1 | 13.86 | 0.52 |
| genuine_turns | 442 | 0.724 | 1.278 | 83.0 | 122.3 | 52.5 | 0.55 | 0.67 |

## Held-out: per turn bin

| bin | n | ADE (m) | FDE (m) | ang err (°) | pred Δhead (°) | angularity | TCR |
|---|---|---|---|---|---|---|---|
| straight | 3989 | 0.268 | 0.522 | 84.9 | 43.0 | 13.90 | 0.52 |
| mild_30_90 | 158 | 0.686 | 1.209 | 76.0 | 43.6 | 0.84 | 0.57 |
| sharp_90_150 | 79 | 0.695 | 1.276 | 89.4 | 61.5 | 0.53 | 0.76 |
| uturn_150_180 | 205 | 0.765 | 1.332 | 86.0 | 55.8 | 0.33 | 0.71 |