# MP_X / schema_audit

Schema truth audit of the Model C dataset (the exp01-04 input), checking for dataset/schema/coordinate/leakage/rollout bugs behind the turn-direction failure.

## Run
```
python run_schema_audit.py
```

## Verdict
**DATA/SPLIT IMBALANCE LIKELY**

Every schema check is CLEAN — motion (du/dv match world backward-diff to ~1e-17 m), targets (forward t->t+1 to ~1e-17 m), turn_rate (circular wrap, corr 0.99), coordinate convention (normal atan2(dy,dx), all recordings), normalization (per-recording, no split leakage), leakage (no target/future inputs, recording-level split, no cross-split duplicates), turn labels (reproduce exp01-04), and the rollout motion updater (matches the dataset to machine precision on moving steps; only a benign stationary-step heading convention differs). The angular-error metric is also convention-invariant, so a coordinate flip cannot produce it. The decisive issue is the HELD-OUT TURN SAMPLE itself: TEST genuine turns=4, held-out total=50 (92% one recording); train L-frac 0.52 vs test 0.00. The test recording contributes almost no genuine turns and they are one-directional, so the 'near-chance angular error on held-out turns' is measured on a tiny, single-recording, direction-imbalanced sample and is statistically unreliable. This is a data/split sparsity confound, NOT a code/schema bug. Fix by adding turn-rich, direction-balanced recordings to the held-out/test set (and to training), then re-evaluate turn direction. The representational-limitation hypothesis from exp01-04 remains plausible but is currently UNDER-MEASURED on held-out data.

## Key outputs

- `schema_audit_report.md` — full 13-section report + final verdict table.
- `schema_audit_summary.csv` — per-category PASS/WARNING/FAIL.
- `top_5_most_likely_causes.md`, `recommended_next_action.md`.
- `suspicious_rows.csv`, `suspicious_tracks.csv`, plus per-section CSVs.
- `plots/` — heading/turn_rate/target/coordinate/leakage/turn_label/rollout/visual panels.

## Trust exp01-04?

**With caveats** — see the verdict; address flagged rows first.