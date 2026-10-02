# Top 5 most likely causes of the turn-direction failure

Audit verdict: **DATA/SPLIT IMBALANCE LIKELY**

## 1. Recording/split balance  [WARNING]
- Evidence: TEST genuine turns=4, held-out total=50 (92% one recording); train L-frac 0.52 vs test 0.00
- Suggested fix: add turn-rich, direction-balanced recordings to the held-out/test set before judging turn direction

## 2. Representational ambiguity at decision points (exp01-04)  [HYPOTHESIS]
- Evidence: turns produced but direction ≈chance; invariant to balance/turn-only/heading-head; only horizon modestly helps (78° at H=5)
- Suggested fix: richer decision-point context or multi-modal/mixture outputs

## 3. Spatial channel carries ~no turn signal (prior audits)  [HYPOTHESIS]
- Evidence: dist_to_obstacle/boundary classifiers at chance; rollout approximates them via KDTree-IDW
- Suggested fix: decision-point-aware spatial features (which way is open)
