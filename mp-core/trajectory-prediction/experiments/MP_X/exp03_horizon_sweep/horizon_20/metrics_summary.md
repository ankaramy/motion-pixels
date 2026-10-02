# exp03 horizon_20 — metrics summary

Turn-only Model C, horizon **20**, displacement floor **2.0 m** (heading>30°, max-step<0.6m fixed). Best epoch 26 (val MSE 1.06474), 6.2s. Trained on 10,905 genuine-turn windows.

Held-out full-horizon eval window bins: straight=188496, mild_30_90=15, sharp_90_150=12, uturn_150_180=23 (genuine rolled: 50).

## Held-out: all windows vs genuine turns

| subset | n | ADE (m) | FDE (m) | ang err (°) | GT Δhead (°) | pred Δhead (°) | angularity | TCR |
|---|---|---|---|---|---|---|---|---|
| all_windows | 4000 | 0.402 | 0.827 | 80.4 | 74.8 | 50.6 | 32.05 | 0.75 |
| genuine_turns | 50 | 1.300 | 2.484 | 88.0 | 124.2 | 49.4 | 0.60 | 0.78 |

## Held-out: per turn bin

| bin | n | ADE (m) | FDE (m) | ang err (°) | pred Δhead (°) | angularity | TCR |
|---|---|---|---|---|---|---|---|
| straight | 3999 | 0.402 | 0.827 | 80.4 | 50.6 | 32.06 | 0.75 |
| mild_30_90 | 15 | 1.211 | 2.423 | 80.1 | 57.0 | 1.28 | 0.73 |
| sharp_90_150 | 12 | 1.372 | 2.591 | 89.3 | 42.8 | 0.37 | 0.67 |
| uturn_150_180 | 23 | 1.321 | 2.468 | 92.5 | 48.0 | 0.28 | 0.87 |