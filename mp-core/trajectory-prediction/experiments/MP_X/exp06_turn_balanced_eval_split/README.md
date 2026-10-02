# exp06 — Turn-balanced evaluation split

## Purpose

Re-cut the recording-level split so the HELD-OUT set has many, direction-balanced genuine turns, then re-run the exp02 turn-only Model C and test whether the ~88-90° held-out angular error was a weak-sample artifact. Schema and architecture unchanged; only recording→split assignment changes (in memory).

## Chosen split (data-driven)

`C_clean_balanced_esplanade`: **test = esplanade** (1,232 clean, balanced genuine turns), val = stairs, train = placa_catalunya + placa_espanya + red_bridge (stays turn-rich). placa_espanya (9,224 turns) was REJECTED as the held-out because it is artifact-prone (p95 step 0.60 m on the artifact guard, median turn 150°); it is run separately as a robustness check. See `split_options.md` / `split_analysis.csv`.

## Result (held-out genuine turns)

See `comparison_vs_exp02.md`. exp02 n=50 ang 88.0° -> exp06 esplanade n=1278 ang 93.4°; robustness placa_espanya n=2106 ang 84.3°.

## Verdict

**1. Was the previous 88-90° angular error caused by a weak held-out sample?**
The original held-out set was n=50 turns (92% one recording). On the 1278-turn, direction-balanced, CLEAN esplanade held-out the angular error is 93.4° vs exp02's 88.0° (Δ +5.4°). So NO — the error is essentially unchanged on a 20x-larger clean balanced held-out, so the previous 88-90° was NOT just a weak-sample artifact.

**2. Does direction improve on a more balanced held-out turn set?**
Predicted Δheading 47° (GT 144°), angularity 0.40, TCR 0.66, angular error 93.4°. Direction is still ≈chance (~90°) — the model expresses turns but does not commit to the correct side.

**3. Do we now have enough evidence to claim direction failure, or not?**
YES — with a 20x-larger, direction-balanced, clean held-out turn set the angular error is still ≈chance, AND the robustness run (placa_espanya) agrees. The direction failure is now WELL-MEASURED and real, not a sampling artifact. This justifies moving to representational fixes (decision-point context, multi-modal heading output) rather than more data-split tweaks.

## Files

`split_analysis.csv`, `split_options.md/.csv`, `metrics_all_vs_genuine_heldout.csv`, `metrics_per_bin_heldout.csv`, `metrics.csv`, `comparison_vs_exp02.md`, `models/`, `training/`, `plots/` (best_6 / worst_6 / collage / heading_change_comparison), `robustness_placa_espanya/`.
