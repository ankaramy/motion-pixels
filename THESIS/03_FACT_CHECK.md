# 03 — Fact Check (technical & numerical claim ledger)

Each important technical/numerical claim, its value, its source file, and a confidence note. **Confidence
describes traceability to a stored source, not statistical certainty.** Read-only audit — nothing retrained
or recomputed here beyond reading stored reports/CSVs. Where a number was independently recomputed in the
prior audit (`research/verify_evidence.py` → `research/evidence_verification.json`), that is noted.

---

## A. Dataset & schema
| Claim | Value | Source | Confidence |
|---|---|---|---|
| MACBA sandbox base | 52 tracks / 18,014 rows | `schema_ablation_bridge/…schema_summary.json`, `dataset_summary.md` | High |
| Sandbox split (actual) | **36 train / 7 val / 9 test** (TRAIN=.70, VAL=.15) | `run_bridge_ablation.py` split_ids | High — code-derived; brief's "36/8/9" is wrong |
| Ablation feature counts A/B/C/D | 6 / 8 / 10 / 11 | `schema_ablation_bridge/ablation_results.csv` | High |
| Model C feature identity (10) | du, dv, speed, heading_sin, heading_cos, turn_rate, u, v, obstacle-clearance, boundary-clearance | `MODEL_X/models/scalers.json` (scaler order) | High |
| Duplicated (overfit) dataset | ×10 → 520 distinct tracks / 180,140 rows | `schema_ablation_bridge_overfit10x/dataset_summary.md` | High — no new samples |
| Accepted V3 dataset | **1,064,379 rows / 3,534 tracks / 5 recordings** | `Barcelona_v3_manual_master_dataset/dataset_schema_report.md`; `MODEL_X/data_audit/dataset_audit.md` | High (full row recount not redone) |
| Per-recording (rows / tracks) | esplanade 283,120/522 · catalunya 355,583/1,255 · espanya 212,224/824 · stairs 127,605/334 · red_bridge 85,847/599 | `MODEL_X/reports/model_x_report.md` | High |
| Excluded recordings | 2 — placa_montjuic_01, stairs_montjuic_02 | `encoder_v3_recording_inventory.md`; `dataset_audit.md` | High |
| MODEL_X track split | **2,827 / 353 / 354** (3,534 IDs) | `MODEL_X/splits/model_x_track_split.csv` | High — independently counted |
| Split meaning | mixed-recording **track-held-out** (every recording in every partition) | same split + report | High — **not** unseen-site |

## B. Models & architecture
| Claim | Value | Source | Confidence |
|---|---|---|---|
| Sandbox / MODEL_X arch | 2-layer LSTM, hidden 128, dropout 0.2, observation window 10 | `frozen_model_C/README.md`; `model_x_lib.py`; MODEL_X report | High |
| MODEL_X optimisation | AdamW, lr 1e-3, batch 256, checkpoint epoch 35, scaled next-displacement MSE | `MODEL_X/reports/model_x_report.md` | High |
| Prediction-model family tested | LSTM, GRU, TCN (+ small Transformer per outline) | `phase-2b-final/` (lstm/gru/tcn dirs, `multi_eval_results.csv`) | High for LSTM/GRU/TCN; **Transformer figure not reconciled** |
| Horizon models | **separate per-H training + checkpoint selection** (e.g. H20 best_epoch 25 vs H10 epoch 35) | `MODEL_X_HORIZON_SWEEP/run_horizon.py` | High — code-verified |
| **Final model** | **MODEL_XC_B_CURV_LIGHT** (curvature-aware LSTM, λ_mag=1.0, λ_curv=0.1) | `MODEL_XC/reports/MODEL_XC_REPORT.md`; `checkpoints/MODEL_XC_B_CURV_LIGHT/model_best.pth`; `PROJECT_CONTEXT.md` | High — used for `presentation_visuals/` + `ready_animations/` |
| XC variant E | incomplete (interrupted ~epoch 23) → **excluded** from rankings | `MODEL_XC_REPORT.md` | High |

## C. Capacity check (sandbox 10×) — memorisation, not generalisation
| Claim | Value | Source | Confidence |
|---|---|---|---|
| Model A held-out ADE / FDE | 0.4871 / 0.8998 m | `schema_ablation_bridge/ablation_results.csv` | High |
| Model C held-out ADE / FDE | 0.5315 / 0.9809 m | same CSV | High |
| Model C held-out curvature corr | ≈ −0.032 | same CSV | High |
| Model C capacity ADE / FDE | 0.0535 / 0.1096 m | `…overfit10x/ablation_results.csv` | High |
| Capacity curvature corr / cum-heading ratio | 0.4961 / 0.9959 | same CSV | High |
| Model D capacity ADE | 0.0611 m | same CSV | High |

## D. Baseline & objective-successor metrics (H10, test)
| Claim | Value | Source | Confidence |
|---|---|---|---|
| MODEL_X H10 ADE / FDE | 0.305 / 0.480 m | `model_x_report.md` (corroborated by XR control CSV 0.30495/0.47994) | High |
| MODEL_X median raw path-length ratio | 0.413 | `model_x_report.md` | High — jitter-sensitive |
| XR net-displacement ratio | control 0.395 → **B_MAG 0.935** | `MODEL_XR/reports/model_xr_variant_comparison.csv` | High |
| XR ADE | control 0.305 → B_MAG 0.362 m | same CSV | High |
| XC (B_CURV_LIGHT) ADE | 0.332 m | `MODEL_XC/reports/model_xc_variant_comparison.csv` | High |
| XC shape ratio | control 0.053 → **light 0.331** (≈6×) | same CSV | High — distinct from sandbox curvature corr |
| XC direction cosine | control 0.956 → **light 0.884** | same CSV | High — the expected direction cost |

## E. Horizon sweep (independently recomputed in prior audit)
| Horizon | ADE (m) | FDE (m) | Angle err (°) | Windows | Tracks |
|---|---|---|---|---|---|
| H20 | 0.446 | 0.708 | 74.1 | 87,320 | 286 |
| H60 | 1.011 | 1.811 | 79.4 | 77,438 | 211 |
| H100 | 1.419 | 2.565 | 79.7 | 69,718 | 175 |
| H200 | 2.595 | 5.016 | 83.7 | 55,505 | 120 |
| H400 | 5.079 | 10.015 | 83.8 | 38,639 | 63 |
Source: `MODEL_X_HORIZON_SWEEP/H*/per_window_metrics.csv`. Confidence High (recomputed). Missing angular
values per horizon (1,582 / 1,518 / 1,478 / 1,359 / 1,141) were **excluded from means, not zero-filled**.

| Claim | Value | Source | Confidence |
|---|---|---|---|
| Horizon distance labels | H20/60/100/200/400 ≈ 1/3/5/10/20 m | sweep comparison + accuracy report | High **as labels only — never minutes** |
| H100 median lengths | pred 1.88 m vs recorded 5.31 m | `comparison_summary.csv`, `comparison_report.md` | High — jitter caveat |
| Endpoint tolerance thresholds | 1/2/3/5/10 m | `final_metrics/compute_horizon_accuracy.py` | High |
| Endpoint success % (H20→H400) | 82.7 / 72.3 / 71.5 / 64.0 / 55.2 | per-window CSV + threshold | High — recomputed |
| Endpoint-normalized "path accuracy" | 18.2%–26.3% | `clamp(1−FDE/GT_net_disp,0,1)` | High — **endpoint-based, not path shape** |
| Strict collapse onset | already exceeds threshold by **H60**; H100 is a *qualified* directional-context ceiling | `comparison_report.md` vs accuracy report | High — depends on metric + jitter |

## F. Maps, units, ethics
| Claim | Value | Source | Confidence |
|---|---|---|---|
| Speed-map category thresholds | 0.8 and 1.6 m/s | `FINAL_BEHAVIOR_MAP_RECIPES.md` | High — not universal categories |
| Model `speed` unit | **metres per step** (≠ m/s) | `dataset_schema_report.md` | High |
| GDPR basis | Article **89(1)** = safeguards/derogations for scientific-research processing | `GDPR_research_safeguards_excerpts.txt` | High — **not** standalone lawful basis |

## G. Claims that DID NOT verify (do not report as completed results)
| Claim (from outline) | Status |
|---|---|
| "8 recordings, ~7,000 extracted, ~500 trained" | **Unreconciled** with final V3 (5 rec / 3,534 tracks). Acquisition≠retention. Author item. |
| Stress test "40 / 60 / 100 m" | **No verified output found** at those exact ranges. Repo stress artifacts are H800/H1000/H1200 (≈40/50/60 m). |
| Original 4-model comparison incl. **Transformer** | phase-2b has LSTM/GRU/TCN; a complete 4-model benchmark figure is **not reconciled**. |
| "Yes, categorically" legal permission | Overstated. Article 89 gives safeguards, not blanket permission; actual controller/lawful-basis/retention arrangements unconfirmed. |
| red_bridge = Passeig de Colom | Attribution **not independently established**; do not assert without author confirmation. |
| Whyte "bench myth" exact attribution/page | Unresolved; use a sourced seating discussion instead. |

## Limits of this fact check
Documentary audit + read of stored metrics only. No retraining, re-annotation, calibration validation, or
legal review. Exact numbers can still describe noisy measurements. No claim of confidence intervals,
statistical significance, universal model superiority, or validated design improvement is made.
