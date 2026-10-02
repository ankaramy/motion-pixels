# MODEL_XC — PAUSE STATUS

> **Superseded (historical note).** The aggregate was run afterwards; see `reports/MODEL_XC_REPORT.md`. Final model = `MODEL_XC_B_CURV_LIGHT`.

**Paused:** 2026-06-12. Chain task `bmy4ib20k` stopped via TaskStop. No new training jobs running.
All existing artifacts preserved (nothing deleted: checkpoints, reports, figures, logs intact).

## Control gate
PASSED. `MODEL_XC_A_BMAG_CONTROL` reproduces MODEL_XR_B_MAG at H10:
ADE 0.3624 (ref 0.362), FDE 0.6028 (0.603), step1_sm 1.036, nd 0.935, cosine 0.956. Verified.

## Variant status at pause
| variant | trained | eval_base.json | horizon.json | notes |
|---|---|---|---|---|
| MODEL_XC_A_BMAG_CONTROL | ✅ complete | ✅ | ✅ | control, reproduced B_MAG |
| MODEL_XC_B_CURV_LIGHT (λ_curv 0.10) | ✅ complete | ✅ | ✅ | done |
| MODEL_XC_C_CURV_MED (λ_curv 0.25) | ✅ complete | ✅ | ✅ | done |
| MODEL_XC_D_CURV_STRONG (λ_curv 0.50) | ✅ complete | ✅ | ✅ | done |
| MODEL_XC_E_CURV_DIR_LIGHT (λ_curv 0.25,λ_dir 0.05) | ⚠️ INTERRUPTED ~epoch 23 | ❌ | ❌ | optional variant; `model_best.pth` exists (from ep23) but `train_summary.json` missing → treat as INCOMPLETE/undertrained; not evaluated |

**Aggregate (`aggregate_xc.py`) has NOT been run yet** → no `reports/model_xc_*.csv`, no figures,
no `MODEL_XC_REPORT.md` yet.

## Remaining work
- (Optional) finish **E** (retrain to convergence + eval + horizon), OR drop E (it is the optional
  variant).
- Run **`aggregate_xc.py`** to produce the 7 CSVs, 11 figures, and pick the best variant.
- Write `reports/MODEL_XC_REPORT.md` (the markdown report — authored manually after aggregate, as in
  prior phases).

## Locations
- **Logs:** `MODEL_XC/xc_run.log` (full chain stdout); per-variant training logs in
  `MODEL_XC/checkpoints/<variant>/train_log.csv`.
- **Checkpoints:** `MODEL_XC/checkpoints/<variant>/model_best.pth` (+ `scalers.json`, `sigma.json`,
  `train_summary.json` for A–D).
- **Per-variant metrics:** `MODEL_XC/experiments/<variant>/{eval_base.json, horizon.json}` (A–D).
- **Configs:** `MODEL_XC/configs/*.json`.

## Exact commands to resume
Run from `mp-core/trajectory-prediction/MODEL_XC`.

**Option 1 — aggregate now on A–D only (E excluded automatically; fastest):**
```
python aggregate_xc.py
```
(`aggregate_xc.py` only includes variants that have `experiments/<variant>/eval_base.json`, so E is
skipped cleanly. Best variant chosen among B/C/D.)

**Option 2 — finish optional E first, then aggregate on all five:**
```
python train_model_xc.py    --config configs/E_curv_dir_light.json
python evaluate_model_xc.py --config configs/E_curv_dir_light.json
python run_horizon_xc.py    --config configs/E_curv_dir_light.json
python aggregate_xc.py
```

**If a single A–D variant ever needs a rerun (e.g. CUDA fault):**
```
python train_model_xc.py    --config configs/<variant>.json
python evaluate_model_xc.py --config configs/<variant>.json
python run_horizon_xc.py    --config configs/<variant>.json
```
(rerun only the failed variant; do not run concurrent training jobs.)

## Notes
- Do NOT overwrite MODEL_X or MODEL_XR.
- E's interrupted `model_best.pth` is from ~ep23 and undertrained — if E is used, retrain it (Option 2)
  rather than evaluating the partial checkpoint.
- Paused by user request; will NOT restart automatically.
