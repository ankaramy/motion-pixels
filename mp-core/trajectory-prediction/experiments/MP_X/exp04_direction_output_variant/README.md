# exp04 — Direction-output variant of Model C

## Purpose

Test whether the turn-direction bottleneck (exp01/exp02 produced turn magnitude but ~90° angular error) is the **displacement-only output head**. The head is widened 2→4 to also predict `future_heading_sin/cos`; loss = position_mse + 0.2·heading_mse; trained turn-only. At rollout the explicit predicted heading drives the heading input channels.

## Dataset, targets & filter

`model_C_dataset.csv` — 1,064,379 rows · 3,534 tracks. Targets = `target_du, target_dv, future_heading_sin, future_heading_cos` (heading target = unit next-step displacement). Trained only on genuine-turn windows (disp>2m, head>30°, max-step<0.6m), genuine-turn validation for early stopping.

## Windows & turn-bin counts

- Train (turn-only): 10,905 — straight=0, mild_30_90=3042, sharp_90_150=2316, uturn_150_180=5547
- Val (turn-only, early-stop): 55 — straight=0, mild_30_90=12, sharp_90_150=13, uturn_150_180=30
- Held-out full-horizon eval windows: straight=188496, mild_30_90=15, sharp_90_150=12, uturn_150_180=23 (genuine rolled: 50; train genuine rolled: 2052)

## Training command

```
python run_exp04.py
```

Loss `position_mse + 0.2*heading_mse`. Best epoch 22, val total 1.37853, 37 epochs, 6.1s, seed 42.

## Main results (held-out val+test)

- All windows (n=4000): ADE 0.364 m, FDE 0.770 m, angular 77.1°.
- Genuine turns (n=50): ADE 1.291 m, FDE 2.447 m, angular 89.0°, predicted Δheading 46° (GT 124°), angularity 0.45, **TCR 0.82**.
- Train capacity (n=2052 genuine): ADE 1.234 m, angular 71.8°, TCR 0.75.

## Three-way comparison (held-out genuine turns)

| metric | exp01 balanced | exp02 turn-only | exp04 dir-output |
|---|---|---|---|
| ADE (m) | 1.418 | 1.300 | 1.291 |
| FDE (m) | 2.692 | 2.484 | 2.447 |
| angular err (°) | 89.8 | 88.0 | 89.0 |
| pred Δhead (°) | 38 | 49 | 46 |
| angularity | 0.37 | 0.60 | 0.45 |
| TCR | 0.60 | 0.78 | 0.82 |

## Verdict

**1. Does explicit heading output reduce angular error?**
Angular error 88.0° (exp02) → 89.0° (exp04), +1.0°. No — essentially unchanged (still ≈chance ~90°).

**2. Does it preserve/increase angularity?**
Angularity 0.60 → 0.45 (-0.15); predicted Δheading 49° → 46°. Decreased.

**3. Does it improve Turn Capture Rate?**
TCR 0.78 → 0.82 (+0.04). Improved.

**4. Does it damage ADE/FDE?**
Genuine-turn ADE 1.300 → 1.291 m (-0.009); FDE 2.484 → 2.447 m. No meaningful damage.

**5. Magnitude vs direction — where does this leave us?**
Explicit heading output improves turn MAGNITUDE/expression but NOT direction (angular error stays ≈chance ~90°). The remaining issue is therefore **not** the output representation but **insufficient directional context / ambiguity in the input window**: a 10-frame seed of mostly-straight approach motion does not determine which way the pedestrian will turn at a decision point. Likely fixes are longer/again-richer context, decision-point-aware spatial features (which way is open), or multi-modal (mixture) outputs rather than a single mean heading.

## Plots

`plots/best_6_genuine_turns/` + `.png`, `plots/worst_6_genuine_turns/` + `.png`, `plots/collage_9_genuine_turns.png`, `plots/heading_change_comparison.png`, `plots/final_heading_scatter.png` (GT vs predicted final direction; from held-out (val+test)).

## Files

`metrics.csv`, `metrics_all_vs_genuine_heldout.csv`, `metrics_per_bin_heldout.csv`, `metrics_train_capacity.csv` (+ train variants), `metrics_summary.md`, `training/` (config.json, epoch_log.csv, training_loss.csv, validation_loss.csv), `models/best_model_exp04.pth`, `models/scalers_exp04.pkl`.
