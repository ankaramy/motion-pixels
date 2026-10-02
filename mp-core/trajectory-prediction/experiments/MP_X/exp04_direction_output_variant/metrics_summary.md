# exp04 — Direction-output variant — metrics summary

4-output head, loss position_mse + 0.2*heading_mse, turn-only training. Best epoch 22 (val 1.37853), 6.1s.

## Held-out: all windows vs genuine turns

| subset | n | ADE (m) | FDE (m) | ang err (°) | GT Δhead (°) | pred Δhead (°) | angularity | TCR |
|---|---|---|---|---|---|---|---|---|
| all_windows | 4000 | 0.364 | 0.770 | 77.1 | 74.8 | 56.2 | 28.08 | 0.82 |
| genuine_turns | 50 | 1.291 | 2.447 | 89.0 | 124.2 | 46.3 | 0.45 | 0.82 |

## Held-out: per turn bin

| bin | n | ADE (m) | FDE (m) | ang err (°) | GT Δhead (°) | pred Δhead (°) | angularity | TCR |
|---|---|---|---|---|---|---|---|---|
| straight | 3999 | 0.364 | 0.770 | 77.1 | 74.8 | 56.2 | 28.08 | 0.82 |
| mild_30_90 | 15 | 1.119 | 2.204 | 74.2 | 49.6 | 35.4 | 0.72 | 0.73 |
| sharp_90_150 | 12 | 1.384 | 2.601 | 92.2 | 124.5 | 40.0 | 0.34 | 0.83 |
| uturn_150_180 | 23 | 1.356 | 2.524 | 97.0 | 172.7 | 56.6 | 0.33 | 0.87 |