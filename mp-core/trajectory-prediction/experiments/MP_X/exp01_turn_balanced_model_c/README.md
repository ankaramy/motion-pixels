# exp01 — Turn-balanced Model C

## Purpose

Test whether Model C's straight-line collapse is caused by turn scarcity / class imbalance rather than a fundamental incapacity. Same frozen Model C architecture and recipe; the only change is a `WeightedRandomSampler` that oversamples turn windows.

## Dataset

`C:\Users\OWNER\Desktop\new_datasets\Barcelona_v3_manual_master_dataset\model_C_dataset.csv` — 1,064,379 rows · 3,534 tracks (Barcelona_v3_manual_master, recording-level split).

## Windows & turn-bin counts

- Train windows (824,917): straight=814012, mild_30_90=3042, sharp_90_150=2316, uturn_150_180=5547
- Val windows (124,265): straight=124210, mild_30_90=12, sharp_90_150=13, uturn_150_180=30
- Held-out full-horizon eval windows: straight=188496, mild_30_90=15, sharp_90_150=12, uturn_150_180=23 (genuine rolled: 50; train genuine rolled: 2052)

## Sampler weights

`{'straight': 1.0, 'mild_30_90': 8.0, 'sharp_90_150': 12.0, 'uturn_150_180': 16.0}` (straight / mild_30_90 / sharp_90_150 / uturn_150_180).

## Training command

```
python run_exp01.py
```

Best epoch 2, val MSE 0.49480, 17 epochs, 229.1s, seed 42.

## Main results (held-out val+test)

- All windows (n=4000): ADE 0.337 m, FDE 0.726 m, angular 83.8°.
- Genuine turns (n=50): ADE 1.418 m, FDE 2.692 m, angular 89.8°, angularity ratio 0.37, **Turn Capture Rate 0.60**.
- Per genuine bin (held-out): mild_30_90 n=15 ang 79° TCR 0.53; sharp_90_150 n=12 ang 101° TCR 0.42; uturn_150_180 n=23 ang 91° TCR 0.74.

## Visible angularity improved?

**Partial success — the straight-line collapse is broken, but turn DIRECTION is still wrong.**

Turn-balancing made Model C produce **visibly curved rollouts** and commit to a turn on
the majority of genuine turns: held-out **Turn Capture Rate 0.60** with predicted heading
change **38–49°** on genuine turns, vs the Phase-4 baseline's collapse (mean predicted
turn rate ≈ 0.10 rad/step → essentially straight). The best/worst/collage panels show the
orange predicted paths now bending instead of running straight. So **turn scarcity / class
imbalance was a real and major cause of the collapse** — the architecture is *not*
fundamentally incapable (consistent with the OVERFIT10X capacity result).

**However**, the model **under-rotates and mis-directs**: held-out angular error stays at
~90° (chance), angularity ratio ≈ 0.28–0.56 (predicts only ~⅓ of the GT heading change),
and ADE on genuine turns rises to ~1.4 m (vs 0.34 m on straight windows). U-turns get the
highest capture (TCR 0.74) but still resolve to ~49° of turn, not 173°. Held-out genuine
turns are also tiny (n=50; stairs_montjuic val has only 55 genuine-turn windows total), so
these numbers are noisy — the train capacity reference (n=2052) tells the same story
(TCR 0.53, angular 71°, angularity 0.47).

**Verdict:** balancing recovers angular *expression* (the rollouts curve again) but not
angular *accuracy* (direction at decision points). Necessary, not sufficient. Next:
exp02 (turn-only diagnostic — can it fit direction with zero straight dilution?) and
exp04 (explicit heading output — is the bottleneck the displacement-only head?).

## Plots

- `plots/best_6_genuine_turns/` and `best_6_genuine_turns.png`
- `plots/worst_6_genuine_turns/` and `worst_6_genuine_turns.png`
- `plots/collage_9_genuine_turns.png`
- `plots/heading_change_comparison.png`

## Files

`metrics.csv` (per held-out window), `metrics_all_vs_genuine_heldout.csv`, `metrics_per_bin_heldout.csv`, `metrics_train_capacity.csv` (+ train variants), `metrics_summary.md`, `training/` (config.json, epoch_log.csv, training_loss.csv, validation_loss.csv), `models/best_model_exp01.pth`, `models/scalers_exp01.pkl`.
