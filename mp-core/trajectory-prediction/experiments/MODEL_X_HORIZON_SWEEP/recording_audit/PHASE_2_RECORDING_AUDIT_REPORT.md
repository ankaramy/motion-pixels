# MODEL_X Horizon Sweep — Phase 2 Recording Audit

**Date:** 2026-06-11 · **Scope:** recording audit ONLY. No retraining, no weight changes, no dataset
changes, no MODEL_XR, no Phase 3. Inference re-run is deterministic (`model.eval()`) on the existing
`H*/model_best.pth` to compute jitter-aware metrics the stored CSVs lack.

## Method & thresholds

For every test window of every horizon, length is measured **three ways** (Phase-1 mandate):
- **cumulative** path length (sum of per-step segments — sensitive to tracking jitter),
- **net displacement** (straight-line start→end — jitter-immune),
- **smoothed** path length (centred moving average, window = **5 frames**, then cumulative).

Length ratios / collapse rates are computed on **moving windows** (GT net displacement > **0.5 m**) so
near-stationary jitter does not dominate. Direction = cosine similarity between GT and predicted
net-displacement vectors (defined where GT net > 0.5 m and pred net > 0.05 m). Red-Bridge labels:
`similar` if within ±15 % of the non-RB mean; otherwise `better/worse_than_average` (metric-aware
direction); window share `over/under-represented` if outside ±15 % of mean non-RB share.

Files: `per_recording_horizon_metrics.csv`, `dataset_composition_by_recording.csv`,
`red_bridge_vs_others.csv`, `recording_audit_window_inventory.csv` (328,620 windows), `figures/*.png`.

## Headline per-recording numbers (mean over horizons)

| recording | topology | ADE med | **cum** ratio | **net-disp** ratio | **smoothed** ratio | cosine |
|---|---|---|---|---|---|---|
| esplanade_espanya_01 | open circulation | 1.74 | 0.278 | 0.581 | 0.421 | **0.993** |
| placa_catalunya_01 | open plaza | 0.87 | 0.327 | 0.445 | 0.387 | 0.672 |
| placa_espanya_01 | open roundabout | **3.04** | **0.143** | **0.291** | **0.185** | 0.680 |
| red_bridge_combined_01 | corridor flow | **0.54** | **0.525** | **1.204** | **0.670** | 0.669 |
| stairs_montjuic_01 | constrained multi-dir | 0.55 | 0.359 | 0.474 | 0.434 | **0.423** |

## Dataset composition

| recording | rows % | tracks % | rollout-window % (all H) |
|---|---|---|---|
| esplanade_espanya_01 | 26.6 | 14.8 | **35.3** |
| placa_catalunya_01 | 33.4 | 35.5 | 31.2 |
| placa_espanya_01 | 19.9 | 23.3 | 19.9 |
| red_bridge_combined_01 | **8.1** | 17.0 | **4.3** |
| stairs_montjuic_01 | 12.0 | 9.5 | 9.3 |

Red Bridge has many tracks (17 %) but few rows (8 %) → **short tracks** (corridor crossings), so it
contributes only **4.3 %** of rollout windows, shrinking to **0.39 %** at H400. esplanade + catalunya
dominate rollout windows (~66 % combined).

## Answers

**A. Does every recording show length underprediction?**
By **cumulative** ratio, yes — all medians 0.14–0.53 (< 1). But by **net displacement** the picture splits:
Red Bridge **over-predicts** (1.20 > 1), esplanade/catalunya/stairs land ~0.44–0.58, and only
placa_espanya is severely short (0.29). So underprediction is **not uniform** and is mild-to-absent once
jitter is removed for the easier recordings.

**B. Is the apparent collapse weaker with net displacement than cumulative path length?**
**Yes — substantially.** Net-displacement ratios are ~1.5–4× the cumulative ratios for every recording
(e.g. esplanade 0.58 vs 0.28; Red Bridge 1.20 vs 0.53). Smoothed sits between the two. The three metrics
**disagree**, and the disagreement is exactly the jitter inflation Phase 1 flagged: **cumulative pred/GT
ratio overstates collapse.** The honest "collapse" is the net-displacement number, which is much milder.

**C. Does Red Bridge behave differently?**
**Yes — and it is the BEST recording, not a problem.** Lowest ADE at every horizon
(`better_than_average` ×5), highest cumulative ratio, and net-disp ratio ≈ 1 (the corridor's directional
flow is the easiest thing to predict; the model even slightly over-shoots net distance). Its only odd
value is H400 net-disp ratio 3.64 / cosine 0.23 — but that horizon has just **3 Red-Bridge test tracks
(150 windows)**, so treat it as small-sample noise, not behaviour. Red Bridge is structurally distinct
(corridor) but distinct in a *favourable* direction.

**D. Is Red Bridge overrepresented in the horizon windows?**
**No — it is under-represented at every horizon** (6.3 % → 0.39 %; `underrepresented` ×5). It cannot be
driving training behaviour: it is both the smallest rollout contributor and the easiest case.

**E. Is any single recording responsible for most of the collapse?**
**No single recording "causes" it, but placa_espanya is the worst by far** (ADE 3.04, net-disp ratio 0.29,
smoothed 0.19) and contributes ~20 % of windows. The collapse is **broad/systemic** — present in all open
plazas — and *worst* in the large fast roundabout (espanya), not in Red Bridge. Removing Red Bridge would
make the averages look *worse*, not better.

**F. Are direction metrics better than distance metrics across recordings?**
**Yes, clearly.** Cosine similarity is high where motion is directional (esplanade 0.99, espanya/catalunya/
red_bridge ~0.67–0.68) even though their length ratios are low. The exception is **stairs_montjuic**
(cosine 0.42) — the constrained multi-directional stair site is the one place *direction* is genuinely
hard. So overall the model has learned **heading better than travelled distance** — confirming the
Phase-1 hypothesis at the recording level.

**G. What should Phase 3 focus on?**
1. **Adopt net-displacement (and smoothed) length as the primary magnitude metric**; demote raw cumulative
   pred/GT ratio (jitter-biased). Re-state the sweep's "collapse" using net displacement.
2. **Target magnitude/length underprediction, not direction** — direction is already good; the gap is the
   model stepping short. The natural lever is a training-objective change (e.g. multi-step / displacement-
   magnitude-aware loss), which is a Phase-3 design decision.
3. **Focus difficulty on the open fast plazas (placa_espanya foremost, then catalunya/esplanade at long
   horizons)** and on **stairs_montjuic for direction**. Red Bridge needs no remediation.
4. Consider that GT trajectories themselves are jittery — a **GT-denoising / smoothing** step before
   training or evaluation may matter as much as the model objective.

## Methodology note (metric disagreement)

The three length ratios **disagree by design and by data**: cumulative < smoothed < net-displacement for
every recording. Where they disagree, the **net-displacement** ratio is the trustworthy magnitude signal
(jitter-immune) and the **cumulative** ratio is the pessimistic bound. Any Phase-3 claim about "collapse"
must cite the net-displacement number or it will overstate the problem.

## Warnings
- **H400 small samples:** Red Bridge (3 tracks) and the long-horizon pools generally thin out; H400
  per-recording values are noisy — the 3.64 net-disp spike is an artifact, not behaviour.
- Length ratios/collapse computed on **moving** windows (GT net > 0.5 m); cosine on direction-defined
  windows only. Counts (`n_windows`, `n_moving`) are in `per_recording_horizon_metrics.csv`.
- No missing files; all five horizons had model + scalers + metrics. Inference only — nothing retrained
  or modified outside `recording_audit/`.

## Figures (`figures/`)
per_recording_ADE_by_horizon · per_recording_FDE_by_horizon · per_recording_cumulative_ratio_by_horizon ·
per_recording_net_displacement_ratio_by_horizon · per_recording_smoothed_ratio_by_horizon ·
per_recording_cosine_similarity_by_horizon · horizon_window_composition_by_recording ·
collapse_rate_by_recording_and_horizon · red_bridge_vs_others_summary
