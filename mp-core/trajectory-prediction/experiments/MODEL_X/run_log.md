# MODEL_X — Run Log

## 2026-06-11

**Dataset audit.** Located candidate masters. Only two exist, both 5 recordings (NOT 7):
`Barcelona_v1_master_dataset` and `Barcelona_v3_manual_master_dataset`, each 1,064,379 rows /
3,534 tracks. `placa_montjuic_01` + `stairs_montjuic_02` never encoded (excluded: bad homography).
All 12 required features + targets present in both. Details: `data_audit/dataset_audit.md`.

**User decisions.** (1) Train on the 5 available recordings. (2) Use **v3 manual** master dataset.

**Architecture confirmed.** `TrajectoryLSTM` from `schema_ablation_bridge/run_bridge_ablation.py`
(hidden 128, 2 layers, dropout 0.2, head→2, MSE, seed 42). Copied verbatim into `model_x_lib.py`;
source not edited.

**Split built** (`splits/model_x_track_split.csv`) — mixed-recording track-level, 80/10/10, seed 42:
- train 2,827 tracks / 850,758 rows / 822,488 train-windows
- val   353 tracks / 116,787 rows / 110,214 eval-windows
- test  354 tracks / 96,834 rows / 90,310 eval-windows
- every recording contributes to train/val/test; no track shared.

**Smoke test** of windowing → scaler → batched rollout → metrics: PASS.

**Training:** launched `training/train_model_x.py` (AdamW, lr 1e-3, wd 1e-5, batch 256, MSE,
grad-clip 1.0, ≤80 epochs, early-stop patience 12 on val ADE; per-epoch val ADE/FDE on a 3,000-window
10-step rollout subsample).

**Training DONE.** 47 epochs (early-stopped), 11.3 min on CUDA. Best epoch **35**:
val ADE **0.2945 m**, val FDE **0.4718 m** (ADE & FDE best at same epoch → unambiguous final
checkpoint `best_by_val_ADE.pth`). Single-step val MSE drifted up after ~ep10 but rollout val ADE kept
improving — the rollout metric is what we select on. Curves: `training/loss_curve.png`,
`training/ade_fde_curve.png`.

**Evaluation DONE** (full test split, 90,310 rollout windows):
- All-window: ADE **0.305 m** / FDE **0.480 m** (median 0.136 / 0.198). Test ≈ val → clean
  generalization on the mixed split, no rollout-metric overfit.
- Per-recording ADE: red_bridge 0.065, stairs 0.092, catalunya 0.131, esplanade 0.343,
  **placa_espanya 0.691** (hardest, turn-heavy, dominates the mean).
- Genuine turns (2,858): ADE 0.730 / FDE 1.225 / TCR **0.381**.
- **Honest weakness:** pred/GT length ratio median **0.41** (0.30 on turns) — MSE regression-to-mean
  undershoots path length; conservative but plausible, not fabricated, not collapsed stubs.

**Plots DONE.** best/worst 6, best-6 genuine turns, representative 9-panel, clean presentation collage
(`plots/collages/model_x_best_predictions_presentation.png` — verified visually), path-length
distribution, GT-vs-pred scatter, per-recording ADE/FDE bar.

**Report:** `reports/model_x_report.md`. Comparison vs Model C / Exp03 / Exp06 included with the
explicit caveat that MODEL_X's mixed split is easier than their held-out splits (not apples-to-apples).
