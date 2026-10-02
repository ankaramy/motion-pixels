# exp03 — Horizon sweep — comparison

Turn-only Model C (exp02 recipe) at horizons 5 / 10 / 20. Same architecture, 10 inputs, `target_du/dv` output, no sampler, no heading head. Only the horizon (and the horizon-scaled displacement floor) changes.

## Displacement-floor note (why it scales)

A fixed 2 m genuine-turn floor empties the short-horizon turn sets (held-out genuine turns measured at 1 / 2 / 50 for H=5 / 10 / 20). We hold heading-change>30° and max-step<0.6m fixed but scale the displacement floor D(H)=2·H/20 (0.5 / 1.0 / 2.0 m) to keep a constant per-step speed floor and comparable populations (held-out genuine 1073 / 442 / 50).

## Held-out genuine-turn metrics by horizon

| horizon | disp floor (m) | n_turns | ADE (m) | FDE (m) | angular err (°) | GT Δhead (°) | pred Δhead (°) | angularity | TCR |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.5 | 1073 | 0.418 | 0.674 | 78.0 | 120.9 | 43.1 | 0.45 | 0.58 |
| 10 | 1.0 | 442 | 0.724 | 1.278 | 83.0 | 122.3 | 52.5 | 0.55 | 0.67 |
| 20 | 2.0 | 50 | 1.300 | 2.484 | 88.0 | 124.2 | 49.4 | 0.60 | 0.78 |

## Verdict

**1. Does H=5 reduce angular error vs H=20?**
YES — H=5 angular error 78.0° < H=20 88.0° (Δ -10.0°).

**2. Does shorter horizon reduce or preserve predicted angularity?**
Preserved — angularity is actually highest at H=20 (0.45 / 0.55 / 0.60 for H5/H10/H20); predicted Δheading 43° / 52° / 49°. Crucially the GT turn severity is ~constant across horizons (GT Δhead 121° / 122° / 124°), because the filter holds heading-change>30° fixed and the qualifying turns are sharp at every horizon. So angularity (the ratio) is a fair cross-horizon comparison and shorter horizon does NOT buy more angular expression.

**3. Does Turn Capture Rate improve or collapse?**
TCR H5=0.58 / H10=0.67 / H20=0.78 (max at H=20). Improves toward longer horizon.

**4. Is the direction problem horizon-based or present even at short horizon?**
PARTLY horizon-based, but NOT solved by it. Angular error falls monotonically as the horizon shrinks (88° → 83° → 78°), so observability does matter — fewer steps ahead is easier. BUT even at H=5 the error is 78°, only ~12° below chance (90°) and far from a usable direction prediction (a correct turn would be <30–45°). So shortening the horizon is a secondary mitigation, not a cure: the direction failure persists strongly even at the shortest horizon. Combined with exp04 (explicit heading head did not help either), this points to **input/representation ambiguity at the decision point** as the dominant cause, with horizon length a secondary modulator.

**5. Which horizon gives the best thesis visualization?**
H=20 gives the strongest visible turning (max angularity×TCR). For thesis figures, H=20 shows the most visible curvature in absolute terms while H=5 has the lowest angular error — use H=20 for the headline rollout panels.

## Per-horizon outputs

`horizon_05/`, `horizon_10/`, `horizon_20/` — each with model checkpoint, `training/config.json` + loss CSVs, `metrics.csv`, `metrics_summary.md`, and `plots/` (best_6 / worst_6 / collage / heading_change_comparison).
