# Overfitting Experiment — overfit-10x

## What this is

A **model-capacity and training sanity check**.

The trajectory dataset is duplicated 10× (giving each copy unique person IDs) and
both an LSTM and a GRU are trained on it for an extended number of epochs with no
regularisation. If the model is working correctly it should memorise the training
set and drive loss toward zero.

This experiment answers one question:
> "Does this model architecture have enough capacity to fit the training distribution?"

## What this is NOT

**This is not a generalisation test.**

- Duplicated trajectories share identical spatial paths and feature values.
- There is no held-out validation set — the model is expected to overfit.
- Loss curves from this experiment **must not be cited as thesis performance evidence**.
- Reported metrics here are meaningless for real-world pedestrian prediction quality.

## Inputs

| File | Source |
|------|--------|
| `trajectories_encoded.csv` | `mp-data/processed/encoded/` |

## Outputs

All outputs go to `mp-data/outputs/prediction/experiments/overfit-10x/`:

```
overfit-10x/
├── trajectories_overfit.csv     # 10x duplicated encoded dataset
├── lstm/
│   ├── overfit_lstm.pth
│   ├── overfit_lstm_scaler.pkl
│   └── loss_curve_lstm.png
├── gru/
│   ├── overfit_gru.pth
│   ├── overfit_gru_scaler.pkl
│   └── loss_curve_gru.png
└── comparison_loss_curves.png
```

## Run order

```bash
# 1. Create the duplicated dataset
python make_overfit_dataset.py

# 2. Train both architectures
python train_lstm_overfit.py
python train_gru_overfit.py

# 3. Compare results
python compare_results.py
```

## Expected results

- Both models should achieve very low final loss (< 0.001 normalised MSE).
- If loss does not converge, the training loop or feature scaling has a bug.
- GRU should train faster; LSTM may reach a slightly lower final loss.

## Key differences from the main pipeline

| | Main pipeline | This experiment |
|---|---|---|
| Dataset | Real encoded CSV (1×) | Duplicated 10× |
| Epochs | 100 | 500 |
| Batch size | 64 | 32 |
| Hidden size | 128 | 256 |
| Goal | Generalisation | Memorisation |
| Min displacement filter | 0.3 m | disabled |
