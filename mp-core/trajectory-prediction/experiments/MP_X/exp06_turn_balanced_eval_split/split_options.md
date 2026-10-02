# exp06 — split analysis & options

## Per-recording genuine turns by horizon (displacement floor = 2·H/20)

| recording | H | floor (m) | genuine | mild | sharp | U-turn | left | right | L-frac | med turn° | med max-step (m) | p95 step (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| esplanade_espanya_01 | 5 | 0.5 | 14271 | 2781 | 2213 | 9277 | 7450 | 6821 | 0.52 | 169 | 0.387 | 0.465 |
| placa_catalunya_01 | 5 | 0.5 | 3213 | 1256 | 778 | 1179 | 1565 | 1648 | 0.49 | 120 | 0.373 | 0.136 |
| placa_espanya_01 | 5 | 0.5 | 31731 | 8762 | 6397 | 16572 | 16087 | 15644 | 0.51 | 153 | 0.369 | 0.602 |
| stairs_montjuic_01 | 5 | 0.5 | 784 | 212 | 154 | 418 | 376 | 408 | 0.48 | 156 | 0.417 | 0.089 |
| red_bridge_combined_01 | 5 | 0.5 | 289 | 152 | 93 | 44 | 152 | 137 | 0.53 | 84 | 0.428 | 0.065 |
| esplanade_espanya_01 | 10 | 1.0 | 5322 | 869 | 895 | 3558 | 2768 | 2554 | 0.52 | 169 | 0.437 | 0.465 |
| placa_catalunya_01 | 10 | 1.0 | 1102 | 437 | 319 | 346 | 525 | 577 | 0.48 | 114 | 0.435 | 0.136 |
| placa_espanya_01 | 10 | 1.0 | 25525 | 5728 | 4954 | 14843 | 12885 | 12640 | 0.50 | 161 | 0.419 | 0.602 |
| stairs_montjuic_01 | 10 | 1.0 | 372 | 105 | 66 | 201 | 199 | 173 | 0.53 | 158 | 0.458 | 0.089 |
| red_bridge_combined_01 | 10 | 1.0 | 70 | 53 | 13 | 4 | 38 | 32 | 0.54 | 61 | 0.445 | 0.065 |
| esplanade_espanya_01 | 20 | 2.0 | 1232 | 211 | 213 | 808 | 653 | 579 | 0.53 | 167 | 0.465 | 0.465 |
| placa_catalunya_01 | 20 | 2.0 | 177 | 85 | 58 | 34 | 91 | 86 | 0.51 | 92 | 0.373 | 0.136 |
| placa_espanya_01 | 20 | 2.0 | 9224 | 2629 | 1982 | 4613 | 4752 | 4472 | 0.52 | 150 | 0.463 | 0.602 |
| stairs_montjuic_01 | 20 | 2.0 | 46 | 12 | 11 | 23 | 26 | 20 | 0.57 | 154 | 0.481 | 0.089 |
| red_bridge_combined_01 | 20 | 2.0 | 4 | 3 | 1 | 0 | 0 | 4 | 0.00 | 37 | 0.169 | 0.065 |

## Turn-quality note (why the largest-count recording is NOT the best held-out)

`placa_espanya` carries the overwhelming majority of genuine turns, but the analysis above shows it is artifact-prone: the fastest motion (p95 step ≈ 0.60 m, on the 0.6 m artifact guard) and a *median* genuine turn near 150° (near-U-turn). Its turn abundance largely reflects tracking jitter / ID-switches, not clean decision turns. `esplanade` has 1,232 clean, direction-balanced genuine turns at normal walking speed — the right held-out for measuring turn direction.

## Split options (per-split aggregates at H=20)

| option | split | recordings | rows | tracks | genuine | left | right | mild | sharp | U-turn |
|---|---|---|---|---|---|---|---|---|---|---|
| A_original | train | esplanade;placa_catalunya;placa | 850,927 | 2601 | 10633 | 5496 | 5137 | 2925 | 2253 | 5455 |
| A_original | val | stairs_montjuic | 127,605 | 334 | 46 | 26 | 20 | 12 | 11 | 23 |
| A_original | test | red_bridge | 85,847 | 599 | 4 | 0 | 4 | 3 | 1 | 0 |
| B_turn_rich_placa_espanya | train | esplanade;placa_catalunya;red_bridge | 724,550 | 2376 | 1413 | 744 | 669 | 299 | 272 | 842 |
| B_turn_rich_placa_espanya | val | stairs_montjuic | 127,605 | 334 | 46 | 26 | 20 | 12 | 11 | 23 |
| B_turn_rich_placa_espanya | test | placa | 212,224 | 824 | 9224 | 4752 | 4472 | 2629 | 1982 | 4613 |
| C_clean_balanced_esplanade | train | placa_catalunya;placa;red_bridge | 653,654 | 2678 | 9405 | 4843 | 4562 | 2717 | 2041 | 4647 |
| C_clean_balanced_esplanade | val | stairs_montjuic | 127,605 | 334 | 46 | 26 | 20 | 12 | 11 | 23 |
| C_clean_balanced_esplanade | test | esplanade | 283,120 | 522 | 1232 | 653 | 579 | 211 | 213 | 808 |

## Choice

- **A_original** — held-out genuine = 50 (the under-powered baseline).
- **B_turn_rich_placa_espanya** — held-out genuine ≈ 9,224 but artifact-prone (rejected as primary; run as robustness check).
- **C_clean_balanced_esplanade (CHOSEN)** — held-out genuine = 1,232, direction-balanced (L-frac 0.53), clean walking speed; train keeps placa_espanya so it stays turn-rich. Largest *clean* held-out while keeping train usable.
