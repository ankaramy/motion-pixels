# exp02 — Turn-only diagnostic Model C

## Purpose

Train Model C on **genuine-turn windows only** (no straight dilution, no sampler) to test whether the displacement-only architecture can learn turn DIRECTION when it sees nothing but turns. Early stopping tracks genuine-turn validation windows so the checkpoint is not pulled back to straight collapse.

## Dataset & filter

`model_C_dataset.csv` — 1,064,379 rows · 3,534 tracks. Training windows kept iff GT net displacement > 2 m AND GT heading change > 30° AND max single-step < 0.6 m.

## Windows & turn-bin counts

- Train (genuine-turn only): 10,905 windows — straight=0, mild_30_90=3042, sharp_90_150=2316, uturn_150_180=5547
- Val (genuine-turn only, early-stop signal): 55 — straight=0, mild_30_90=12, sharp_90_150=13, uturn_150_180=30
- Held-out full-horizon eval windows: straight=188496, mild_30_90=15, sharp_90_150=12, uturn_150_180=23 (genuine rolled: 50; train genuine rolled: 2052)

## Training command

```
python run_exp02.py
```

No WeightedRandomSampler. Best epoch 26, val MSE 1.06474, 41 epochs, 7.6s, seed 42.

## Main results (held-out val+test)

- All windows (n=4000): ADE 0.402 m, FDE 0.827 m, angular 80.4°.
- Genuine turns (n=50): ADE 1.300 m, FDE 2.484 m, angular 88.0°, predicted Δheading 49° (GT 124°), angularity 0.60, **TCR 0.78**.
- Train capacity (n=2052 genuine): ADE 1.262 m, angular 74.0°, TCR 0.73.

## exp02 vs exp01 (held-out genuine turns)

| metric | exp01 balanced | exp02 turn-only | Δ |
|---|---|---|---|
| ADE (m) | 1.418 | 1.300 | -0.118 |
| FDE (m) | 2.692 | 2.484 | -0.208 |
| angular err (°) | 89.8 | 88.0 | -1.758 |
| pred Δhead (°) | 38 | 49 | +11.5 |
| angularity | 0.37 | 0.60 | +0.233 |
| TCR | 0.60 | 0.78 | +0.180 |

## Verdict

**1. Does turn-only training increase predicted heading change?**
Predicted heading change 38° (exp01) → 49° (exp02), INCREASED by +11°; TCR 0.60→0.78.

**2. Does it improve angular direction accuracy?**
Angular error 89.8° → 88.0° (-1.8°); ~unchanged (still ~chance ~90°).

**3. Does it reduce ADE/FDE on genuine turns?**
Genuine-turn ADE 1.418 → 1.300 m (-0.118); FDE 2.692 → 2.484 m. REDUCED.

**4. Where is the next bottleneck?**
Angularity appears but turn DIRECTION stays wrong (≈chance ~90°). Removing straight dilution did not teach direction, so the next bottleneck is most likely the **displacement-only output head** (no explicit heading target to commit to) and/or **insufficient turn diversity** (few, repetitive turn geometries). → motivates exp04 (explicit heading_sin/cos output + heading loss).

## Plots

`plots/best_6_genuine_turns/` + `.png`, `plots/worst_6_genuine_turns/` + `.png`, `plots/collage_9_genuine_turns.png`, `plots/heading_change_comparison.png` (from held-out (val+test)).

## Files

`metrics.csv`, `metrics_all_vs_genuine_heldout.csv`, `metrics_per_bin_heldout.csv`, `metrics_train_capacity.csv` (+ train variants), `metrics_summary.md`, `training/` (config.json, epoch_log.csv, training_loss.csv, validation_loss.csv), `models/best_model_exp02.pth`, `models/scalers_exp02.pkl`.
