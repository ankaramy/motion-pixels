# Recommended next action

The schema, targets, coordinates, normalization, leakage and rollout are all CLEAN (see the audit table). Do NOT keep hunting for a coordinate/alignment bug. The actionable problem is that the **held-out turn set is too small and one-sided to measure turn direction**: the test recording (red_bridge) has only ~4 genuine turns (all one direction) and ~92% of held-out genuine turns come from a single recording (stairs_montjuic). Priority actions:

1. **Fix the evaluation set first.** Add turn-rich, direction-balanced recordings to the held-out / test split (e.g. the placa_espanya roundabout, which holds the overwhelming majority of genuine turns, is currently in TRAIN). Without this, no turn-direction conclusion on held-out data is statistically meaningful — exp01-04 held-out turn numbers (n=50) are under-powered.
2. **Re-run exp01-04 turn-direction evaluation** on the rebalanced split before concluding the limitation is representational. The representational hypothesis is plausible but currently under-measured.
3. **Then, if direction still fails on a proper held-out turn set**, pursue the representational fixes: decision-point context features (which way is open / goal bearing) and a multi-modal / mixture output head instead of single mean-heading regression.
4. **Report exp01-04 honestly:** Model C can EXPRESS turns (visible curvature, TCR up to 0.82); turn DIRECTION is unresolved AND under-measured on held-out data due to test-set turn sparsity — not a schema or rollout bug.

### Flagged rows

- **Recording/split balance** [WARNING]: TEST genuine turns=4, held-out total=50 (92% one recording); train L-frac 0.52 vs test 0.00 → add turn-rich, direction-balanced recordings to the held-out/test set before judging turn direction