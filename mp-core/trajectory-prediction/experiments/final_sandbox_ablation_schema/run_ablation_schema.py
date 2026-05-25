"""
run_ablation_schema.py
----------------------
Orchestrate the schema-ablation experiment: dataset → train (4 models) →
visualise → plan overlay → metrics + summary + winner selection.

Skips a step if its primary output already exists, unless --force.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _paths import (
    DATASET_CSV, MODELS_DIR, PLOTS_DIR, PLAN_DIR, EXP,
    METRICS_CSV, FINAL_MD, MODEL_DEFS, MP_ROOT,
)


def run(label, mod_name) -> bool:
    print(f"\n========== {label} ==========")
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-u", str(HERE / f"{mod_name}.py")],
        cwd=str(HERE))
    print(f"---------- {label} {'OK' if r.returncode == 0 else 'FAILED'}"
          f"  ({time.time()-t0:.1f}s) ----------")
    return r.returncode == 0


STEPS = [
    ("dataset",  "make_schema_dataset",         DATASET_CSV),
    ("train",    "train_ablation_models",       MODELS_DIR / "training_summaries.csv"),
    ("visuals",  "visualize_ablation_grid",     EXP / "metrics_world.csv"),
    ("overlay",  "overlay_ablation_on_plan",    PLAN_DIR / "plan_contact_sheet_combined.png"),
]


# --------------------------------------------------------------------------- #
# Aggregation + recommendation
# --------------------------------------------------------------------------- #
def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    g = (metrics.groupby("variant")
                 .agg(n=("track_id", "size"),
                      mean_ade=("ade", "mean"),
                      median_ade=("ade", "median"),
                      mean_fde=("fde", "mean"),
                      median_fde=("fde", "median"),
                      mean_heading_err_deg=("heading_err_deg", "mean"),
                      mean_cum_heading_ratio=("cum_heading_ratio", "mean"),
                      mean_path_len_ratio=("path_len_ratio", "mean"),
                      mean_seed_continuity=("seed_continuity_err", "mean"),
                      mean_angularity_pred=("angularity_pred", "mean"),
                      mean_angularity_gt=("angularity_gt", "mean"),
                      mean_curvature_corr=("curvature_corr", "mean"))
                 .reset_index())
    g["angularity_ratio"] = (g["mean_angularity_pred"]
                              / g["mean_angularity_gt"].replace(0, np.nan))
    return g


def select_winners(agg: pd.DataFrame) -> Dict[str, str]:
    """Three winners:
       - lowest ADE/FDE   → min mean_ade (FDE tiebreak)
       - best angular     → highest score on:
            angular_score = (1 - |1 - cum_heading_ratio|) * 0.5
                          + max(0, mean_curvature_corr) * 0.3
                          + (1 - |1 - path_len_ratio|.clip(0,1)) * 0.2
       - best thesis      → balance: angular_score - 0.5 * (mean_ade / max_mean_ade)
    """
    a = agg.copy()
    a["cum_dev"]  = (1 - (a["mean_cum_heading_ratio"] - 1).abs()).clip(lower=0)
    a["curv_pos"] = a["mean_curvature_corr"].clip(lower=0).fillna(0)
    a["path_ok"]  = (1 - (a["mean_path_len_ratio"] - 1).abs()).clip(lower=0)
    a["angular_score"] = (0.5 * a["cum_dev"]
                          + 0.3 * a["curv_pos"]
                          + 0.2 * a["path_ok"])

    best_ade  = a.sort_values(["mean_ade", "mean_fde"]
                              ).iloc[0]["variant"]
    best_ang  = a.sort_values("angular_score", ascending=False
                              ).iloc[0]["variant"]

    max_ade = a["mean_ade"].max()
    a["thesis_score"] = (a["angular_score"]
                          - 0.5 * (a["mean_ade"] / max(1e-9, max_ade)))
    best_thesis = a.sort_values("thesis_score", ascending=False
                                ).iloc[0]["variant"]
    return {"best_ade": best_ade,
            "best_angular": best_ang,
            "best_thesis": best_thesis,
            "scored": a[["variant", "angular_score", "thesis_score"]]
                       .to_dict(orient="records")}


def label_for(name: str) -> str:
    for m in MODEL_DEFS:
        if m["name"] == name:
            return m["label"]
    return name


# --------------------------------------------------------------------------- #
# Summary writer
# --------------------------------------------------------------------------- #
def write_summary(agg: pd.DataFrame, winners: Dict,
                  metrics: pd.DataFrame) -> None:
    L: List[str] = []
    L.append("# Final Sandbox Ablation — Schema Summary")
    L.append("")
    L.append("Isolated rebuild using the **full Motion Pixels schema** and "
             "explicit angular-continuation prioritisation. All four "
             "ablation models share architecture (LSTM 256×2, window 10) "
             "and training recipe (×10 duplicate, 30 epochs, MSE on "
             "next-step `target_du`/`target_dv`). Only the feature set "
             "changes between A/B/C/D.")
    L.append("")
    L.append("**Rollout recomputation rule.** At every predicted step the "
             "rollout updates `world_x`, `world_y`, `u`, `v`, `delta_x`, "
             "`delta_y`, `speed`, `heading_angle`, `is_stop`, `is_shift`, "
             "`heading_sin`, `heading_cos`, and `turn_rate` from the new "
             "predicted state. Spatial features `dist_to_obstacle`, "
             "`dist_to_boundary`, `dist_to_entrance` are re-queried via "
             "KDTree-IDW (k=5) over the v2.1C encoded CSV at the new "
             "`(world_x, world_y)`. No feature is frozen at seed values.")
    L.append("")

    L.append("## Feature sets")
    L.append("")
    L.append("| model | label | # features | features |")
    L.append("|---|---|---|---|")
    for m in MODEL_DEFS:
        L.append(f"| `{m['name']}` | {m['label']} | {len(m['features'])} | "
                 + ", ".join(f"`{c}`" for c in m["features"]) + " |")
    L.append("")

    L.append("## Per-model metrics (averaged over picked tracks)")
    L.append("")
    L.append("| model | ADE (m) | FDE (m) | heading err (°) | "
             "cum-heading ratio | path-len ratio | angularity pred/GT | "
             "curvature corr | seed-cont err |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in agg.sort_values("variant").iterrows():
        L.append(f"| `{r['variant']}` | "
                 f"{r['mean_ade']:.3f} | {r['mean_fde']:.3f} | "
                 f"{r['mean_heading_err_deg']:.1f} | "
                 f"{r['mean_cum_heading_ratio']:.2f} | "
                 f"{r['mean_path_len_ratio']:.2f} | "
                 f"{r['mean_angularity_pred']:.1f}/"
                 f"{r['mean_angularity_gt']:.1f} | "
                 f"{r['mean_curvature_corr']:.2f} | "
                 f"{r['mean_seed_continuity']:.3f} |")
    L.append("")

    L.append("Glossary:")
    L.append("- **cum-heading ratio** = predicted Σ|Δheading| ÷ GT Σ|Δheading|  "
             "(1.0 = matched, < 1 = under-turning, > 1 = over-turning)")
    L.append("- **angularity** = number of per-step |Δheading| > "
             "15° in the rollout, compared to the GT count")
    L.append("- **curvature corr** = Pearson r between predicted and GT "
             "per-step signed heading deltas")
    L.append("- **seed-cont err** = distance from rollout's first point to "
             "the seed's last point (anchored, should be 0)")
    L.append("")

    L.append("## Per-track metrics")
    L.append("")
    pivot_ade = (metrics.pivot(index="track_id", columns="variant",
                                values="ade").round(3))
    pivot_curv = (metrics.pivot(index="track_id", columns="variant",
                                 values="curvature_corr").round(3))
    def piv_md(piv, label):
        out = [f"### {label}", ""]
        cols = list(piv.columns)
        out.append("| track | " + " | ".join(cols) + " |")
        out.append("|---|" + "---|" * len(cols))
        for tid, row in piv.iterrows():
            cells = [f"{row[c]:.3f}" if pd.notna(row[c]) else "—"
                     for c in cols]
            out.append(f"| {tid} | " + " | ".join(cells) + " |")
        out.append("")
        return out
    L += piv_md(pivot_ade, "ADE per track × model (m)")
    L += piv_md(pivot_curv, "Curvature correlation per track × model")

    L.append("## Visual diagnosis")
    L.append("")
    L.append("Open the per-model contact sheets and the combined plot to "
             "judge angular continuation directly. A model passes the "
             "thesis bar if:")
    L.append("- the predicted path **starts at the green seed-end marker** "
             "(seed-continuity err ≈ 0);")
    L.append("- the predicted curve **bends in the same direction** as the "
             "GT future (curvature corr > 0);")
    L.append("- it does **not** decay into a single smooth arc or a "
             "vertical/horizontal sweep over the 20-step horizon;")
    L.append("- it produces a comparable number of angular changes "
             "(angularity ratio close to 1).")
    L.append("")

    L.append("## Recommendation")
    L.append("")
    L.append(f"- **Lowest ADE/FDE** → `{winners['best_ade']}` "
             f"({label_for(winners['best_ade'])})")
    L.append(f"- **Best angular continuation** → "
             f"`{winners['best_angular']}` "
             f"({label_for(winners['best_angular'])}). Selected by composite "
             "angular score = 0.5·(1 − |cum_heading_ratio − 1|) + "
             "0.3·max(0, curvature_corr) + 0.2·(1 − |path_len_ratio − 1|).")
    L.append(f"- **Best thesis presentation** → "
             f"`{winners['best_thesis']}` "
             f"({label_for(winners['best_thesis'])}). Balances angular "
             "score against normalised ADE.")
    L.append("")
    L.append("### Composite scores (higher = better)")
    L.append("")
    L.append("| variant | angular_score | thesis_score |")
    L.append("|---|---|---|")
    for s in winners["scored"]:
        L.append(f"| `{s['variant']}` | "
                 f"{s['angular_score']:.3f} | "
                 f"{s['thesis_score']:.3f} |")
    L.append("")

    L.append("## Paths to inspect (in this order)")
    L.append("")
    best_t = winners["best_thesis"]
    best_a = winners["best_angular"]
    L.append("1. **Combined plan overlay (all 4 models on calibrated "
             f"`top_view.png`)** — `{PLAN_DIR / 'plan_contact_sheet_combined.png'}`")
    L.append(f"2. **Best-thesis per-model world contact sheet** — "
             f"`{PLOTS_DIR / f'contact_sheet_{best_t}.png'}`")
    L.append(f"3. **Best-angular plan contact sheet** — "
             f"`{PLAN_DIR / f'plan_contact_sheet_{best_a}.png'}`")
    L.append("")
    L.append("Combined-per-track plots (normalised u/v axes, all four "
             "models on one panel per track) live under "
             f"`{PLOTS_DIR / 'combined_all_models/'}`.")
    L.append("")

    L.append("## Acceptance check")
    L.append("")
    checks = [
        ("dataset built from new calibrated Skate 1 data",
         DATASET_CSV.exists()),
        ("4 trained models present",
         all((MODELS_DIR / m["name"] / "model.pth").exists()
             for m in MODEL_DEFS)),
        ("4 per-model loss curves saved",
         all((MODELS_DIR / m["name"] / "loss_curve.png").exists()
             for m in MODEL_DEFS)),
        ("4 per-model world contact sheets saved",
         all((PLOTS_DIR / f"contact_sheet_{m['name']}.png").exists()
             for m in MODEL_DEFS)),
        ("combined per-track plots saved",
         (PLOTS_DIR / "combined_all_models").exists()),
        ("4 plan-overlay contact sheets + combined saved",
         all((PLAN_DIR / f"plan_contact_sheet_{m['name']}.png").exists()
             for m in MODEL_DEFS)
         and (PLAN_DIR / "plan_contact_sheet_combined.png").exists()),
        ("metrics CSV present", METRICS_CSV.exists()),
    ]
    for desc, ok in checks:
        L.append(f"- [{'x' if ok else ' '}] {desc}")
    L.append("")
    L.append("All artefacts under "
             f"`{EXP.relative_to(MP_ROOT)}`. No file outside this "
             "folder was modified.")
    FINAL_MD.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    EXP.mkdir(parents=True, exist_ok=True)
    print(f"[INFO]  Experiment folder: {EXP}")

    ok = True
    for label, mod, out in STEPS:
        if out.exists() and not args.force:
            print(f"\n========== {label} SKIPPED (output exists: "
                  f"{out.name}) ==========")
            continue
        if not run(label, mod):
            ok = False; print(f"[FATAL] step {label} failed; stopping.")
            break

    if not ok:
        print("[STOP] not writing summary because a step failed")
        return

    metrics = pd.read_csv(EXP / "metrics_world.csv")
    metrics.to_csv(METRICS_CSV, index=False)
    agg = aggregate(metrics)
    winners = select_winners(agg)
    write_summary(agg, winners, metrics)

    print("\n========== Aggregates ==========")
    for _, r in agg.sort_values("variant").iterrows():
        print(f"  {r['variant']:<12s}  "
              f"ADE={r['mean_ade']:.3f}  FDE={r['mean_fde']:.3f}  "
              f"head_err={r['mean_heading_err_deg']:.1f}°  "
              f"cum_h_ratio={r['mean_cum_heading_ratio']:.2f}  "
              f"path_ratio={r['mean_path_len_ratio']:.2f}  "
              f"curv_corr={r['mean_curvature_corr']:+.2f}  "
              f"ang(pred/GT)={r['mean_angularity_pred']:.1f}/"
              f"{r['mean_angularity_gt']:.1f}")
    print()
    print(f"  best_ade     → {winners['best_ade']}")
    print(f"  best_angular → {winners['best_angular']}")
    print(f"  best_thesis  → {winners['best_thesis']}")
    print()
    print(f"Summary    : {FINAL_MD}")
    print(f"Metrics CSV: {METRICS_CSV}")


if __name__ == "__main__":
    main()
