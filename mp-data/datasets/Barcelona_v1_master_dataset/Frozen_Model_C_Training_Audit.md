# Frozen Model C — Training-Readiness Audit (Barcelona_v1)

**Type:** READ-ONLY audit. No training, no code/dataset changes performed.
**Date:** 2026-06-06
**Dataset under test:** `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_master_dataset\model_C_dataset.csv`
**Frozen recipe source:** `mp-core/trajectory-prediction/experiments/schema_ablation_bridge/run_bridge_ablation.py`
**Frozen checkpoints:** `mp-core/trajectory-prediction/frozen_model_C/{held_out,overfit10x}/`

---

## Architecture

`TrajectoryLSTM` (`run_bridge_ablation.py:128`), confirmed by `frozen_model_C/README.md`:

| Field | Frozen value |
|---|---|
| Model type | `nn.LSTM` → `nn.Linear` head |
| Input features | **10** |
| Hidden size | **128** |
| Num layers | **2** |
| Dropout | **0.2** (applied between LSTM layers, since `num_layers > 1`) |
| Output head | `Linear(128 → 2)` |
| Output | `(target_du, target_dv)` in metres/step |
| Forward | takes last timestep hidden state `out[:, -1, :]` → head |
| Seed | 42 |

✅ Matches the training-phase brief's stated architecture (LSTM, hidden 128, 2 layers, dropout 0.2, 10→2) **exactly**.

---

## Features

Frozen `FEATURE_SETS["C_motion_position_spatial"]` (`run_bridge_ablation.py:87`) = 10 features, in this order:
```
du, dv, speed, heading_sin, heading_cos, turn_rate, u, v,
dist_to_obstacle_norm, dist_to_boundary_norm
```
Targets (`TARGET_COLS`): `target_du, target_dv`.

- Order is **referenced by name** in `make_windows` (`g[feat_cols]`), so CSV column ordering is irrelevant — only presence matters.
- `entrance_affinity_norm` is the **Model D** feature (`D_full_relational`), correctly **absent** from Model C.

✅ All 10 features + 2 targets present in `model_C_dataset.csv`. Match is exact.

---

## Windowing

| Field | Frozen value (`run_bridge_ablation.py`) |
|---|---|
| Window length | **10** (`WINDOW_SIZE`) |
| Window construction | `make_windows`: per trajectory, sorted by `timestep`; `X = feat[i:i+10]`, `Y = target[i+9]` |
| Target semantics | the next-step displacement read at the **last** window frame (no future leak — `target_du/dv` is already the next-step displacement precomputed in the dataset) |
| Stride | **1** (sliding window, `for i in range(n - WINDOW_SIZE)`) |
| Min track length | **implicit ≥ 11** (a track needs `n > WINDOW_SIZE` to yield ≥1 window). The dataset already enforced `MIN_TRACK_LEN = 11`, so this is consistent. |
| Rollout horizon | **20** (`N_ROLLOUT`), fully autoregressive — used for **evaluation only**, not training |

✅ Compatible. The dataset's frozen-recipe track filter (≥11) guarantees every retained track produces at least one window.

---

## Scaling

| Field | Frozen value |
|---|---|
| Scaler | `ColumnScaler` — column-wise StandardScaler (mean/std), `std < 1e-9 → 1.0` |
| Fit scope | **train split only** (`f_sc.fit(train_df, …)`, `run_bridge_ablation.py:670-671`) — no leakage |
| Applied to features | yes |
| Applied to targets | **yes** (`Y_tr = t_sc.transform(Y_tr)`; predictions inverse-transformed at rollout) |
| Persisted | yes → `scalers.pkl` as a per-model dict of `feat_means/feat_stds/feat_cols/tgt_means/tgt_stds/tgt_cols/needs_kdt` |

⚠️ Note: `frozen_model_C/README.md` shows a load snippet `feat_scaler, tgt_scaler = pickle.load(...)[name]` (tuple form), but the actual `scalers.pkl` blob written by the code is the **dict** form above. Minor doc drift; the dict is authoritative.

✅ StandardScaler, fit on train only, features + targets, stored to disk. Correct and leakage-free.

---

## Optimization

| Field | Frozen value | Brief said | Action |
|---|---|---|---|
| Optimizer | **AdamW**, `weight_decay=1e-4` | "Adam" | follow frozen (AdamW) |
| Learning rate | **1e-3** | — | follow frozen |
| LR scheduler | **StepLR(step=25, gamma=0.35)** | not mentioned | follow frozen |
| Gradient clip | **clip_grad_norm_ = 1.0** | not mentioned | follow frozen |
| Batch size | **512** | 64 | follow frozen (512) — brief allows "unless production code uses another frozen default" |
| Max epochs | **80** | 100 | follow frozen (80) |
| Early stopping patience | **15** | 15 | ✅ agree |
| Loss | **MSE** on `(target_du, target_dv)` only | MSE | ✅ agree |
| Seed | 42 | — | follow frozen |

✅ Recipe is internally consistent. The brief's 64 / 100 / "Adam" differ from frozen 512 / 80 / AdamW+wd; per the brief's own rule ("follow the frozen code") and the training brief's "unless existing production code uses another frozen default", **the frozen values win**. These differences are reported, not silently changed.

---

## Checkpointing

Frozen behaviour (`train_one`):
- Tracks best **validation** MSE; keeps `best_state` (deep-copied CPU weights).
- On early-stop or max-epoch, loads best state and saves **`best_model.pth`** only.
- Loss histories returned in-memory and plotted to `loss_curves.png`; **no per-epoch CSV** and **no separate final checkpoint** are written by the frozen script.

⚠️ The training brief additionally requires: `final_model_C_barcelona.pt`, `training_config.json`, `epoch_log.csv`, learning-rate/epoch-time curves. These are **additive artifacts** the frozen script does not currently emit — they must be produced by a thin training wrapper, not by editing the frozen recipe.

---

## Barcelona Compatibility

**Column / schema check** — `model_C_dataset.csv` vs frozen expectations:

| Requirement | Status |
|---|---|
| `trajectory_id`, `timestep` present (windowing keys) | ✅ |
| 10 Model C feature columns present | ✅ |
| `target_du`, `target_dv` present | ✅ |
| Column ordering | ✅ irrelevant (selected by name) |
| NaN / Inf | ✅ none (verified in `dataset_validation.json`) |
| `entrance_affinity_norm` excluded | ✅ |
| Recording-level `split` column present | ✅ (`train`/`val`/`test`) |

**Behavioural incompatibilities if `run_bridge_ablation.py` is run *verbatim* on Barcelona** (these are the real findings):

1. **🔴 Split logic conflict (must override).** `split_ids()` (`:592`) does a **random by-trajectory 70/15/15** shuffle (seed 42) and **ignores both the `split` and `recording_id` columns**. Run as-is on Barcelona this would scatter all 5 recordings' tracks across train/val/test — violating the absolute recording-level-split rule and leaking across sites. The Barcelona `split` column must be used instead.
2. **🟠 Trains A/B/C/D in a loop.** `main()` iterates all `FEATURE_SETS`. The brief says **C only — no A/B/D ablation yet**. The wrapper must restrict to `C_motion_position_spatial`.
3. **🟠 Single global `world_bounds`.** The autoregressive **rollout evaluation** reads one `schema_summary.world_bounds_used` to convert `u,v ↔ world_x,y`. Barcelona is multi-recording with **per-recording** bounds (in `manifest.json`). Training (windows + scaling + loss) does **not** use world bounds and is unaffected; only the rollout/ADE/FDE eval needs per-recording bounds wiring. Using one global bound would corrupt rollout geometry.
4. **🟠 KDTree spatial refresh is built from `train_df` globally.** For per-recording rollouts the KDTree (`build_kdt`) should be scoped to the recording being rolled out (it maps `u,v → dist_*_norm`), since `u,v` are per-recording normalized and not comparable across sites.
5. **🟡 No `schema_summary.json` in the Barcelona master folder.** The frozen `main()` requires `--schema`. The Barcelona equivalent is `manifest.json` (per-recording bounds). The wrapper should read bounds from there.

**Conclusion:** the **architecture, windowing, scaler, optimizer, and loss are fully compatible and should be reused unchanged** (import `TrajectoryLSTM`, `ColumnScaler`, `make_windows`, `WINDOW_SIZE`, hyperparameters from the frozen module). The only things that must be adapted — at the **wrapper level, without editing frozen code** — are the data-split source (use the `split` column), restricting to Model C, and per-recording world-bounds/KDTree for rollout evaluation.

---

## Risks

- **Cross-site generalization is the real test.** Frozen Model C's honest held-out number is ADE 0.532 m / FDE 0.981 m on **9** MACBA tracks (small-N; A/B/C/D within noise — the documented data-volume bottleneck). Barcelona gives a far larger, multi-site held-out set (`red_bridge_combined_01`, 599 tracks), but it is a **different geometry** than any training site — expect higher error than the sandbox and possibly straight-line bias.
- **Per-recording normalization changes feature meaning across sites.** `u,v` and `dist_*_norm` are percentile/MinMax **within each recording**. The model must learn site-relative behaviour; this is intended, but rollout bounds/KDTree must be kept per-recording (Risk #3/#4) or geometry breaks.
- **Velocity outliers.** A few rows have |du| up to ~25 m/step (far-field ID swaps). StandardScaler is fit on train, so extreme tails slightly inflate std; consistent with the frozen recipe (no clipping). Monitor train loss stability.
- **Single val recording (`stairs_montjuic_01`).** Early stopping/best-checkpoint selection rests on one site (334 tracks, 1 entrance cluster). Acceptable per the brief's split, but val loss is a single-geometry signal.
- **Environment OK.** torch 2.7.0+cu118, CUDA available (RTX 4070 Laptop), sklearn 1.7, scipy 1.15 — all present. Full A/B/C/D took ~80 epochs on far less data; C-only on ~1.06M rows on GPU is well within reach.

---

## Recommendation

Proceed to training **via a thin training wrapper** placed in the new output tree (`MotionPixels_Barcelona_Training_Visuals/`), which:

1. Imports the frozen `TrajectoryLSTM`, `ColumnScaler`, `make_windows`, `WINDOW_SIZE`, and all hyperparameters from `run_bridge_ablation.py` — **no edits to frozen code**.
2. Builds `train_ids/val_ids/test_ids` from the **`split` column** (not `split_ids()`).
3. Trains **only** `C_motion_position_spatial` (10 features).
4. Fits `ColumnScaler` on **train recordings only**; persists `scaler.pkl`.
5. For rollout evaluation, uses **per-recording** world bounds (from `manifest.json`) and a **per-recording** KDTree.
6. Additionally emits `best`+`final` checkpoints, `training_config.json`, `epoch_log.csv` (brief's required artifacts).

Keep frozen defaults: window 10, hidden 128 × 2 layers, dropout 0.2, AdamW lr 1e-3 wd 1e-4, StepLR(25, 0.35), grad-clip 1.0, batch 512, max 80 epochs, patience 15, MSE loss, seed 42 — and **report** that these (notably batch 512 / 80 epochs / AdamW) differ from the prompt's 64 / 100 / Adam wording.

---

## Final verdict

### ✅ SAFE TO TRAIN — via the thin wrapper described above.

The frozen Model C architecture and recipe are **fully schema-compatible** with `model_C_dataset.csv` (correct features, targets, no NaN/Inf, windowing valid). The **only** blocking issue is operational, not architectural: `run_bridge_ablation.py` run **verbatim** would use a random by-trajectory split and train A/B/C/D — which would violate the recording-level-split rule and the "C-only" rule. Substituting the `split` column, restricting to C, and wiring per-recording rollout bounds removes the blocker without touching frozen code or data.

> ⚠️ **BLOCKED if run unmodified** (`python run_bridge_ablation.py --dataset model_C_dataset.csv`): wrong split + trains all four models. Use the wrapper.

**No training was started. Awaiting go-ahead to proceed to Phase 1 (dataset audit plots) and Phase 2 (training).**
