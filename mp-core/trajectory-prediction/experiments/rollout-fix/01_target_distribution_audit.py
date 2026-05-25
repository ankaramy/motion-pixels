"""
01_target_distribution_audit.py
-------------------------------
Audits the training-target distribution (delta_x, delta_y) to determine whether
the dataset is dominated by tiny motion steps — which would explain why an
MSE-trained model regresses toward near-zero predictions.

Outputs ONLY into: mp-data/outputs/prediction_rollout_fix/
  reports/target_distribution_audit.md
  plots/target_distribution_histogram.png
  debug/target_distribution_stats.csv

Does NOT touch mp-data/outputs/prediction/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE     = Path(__file__).resolve().parent
MP_ROOT  = HERE.parent.parent.parent.parent
MP_DATA  = MP_ROOT / "mp-data"
SRC_CSV  = MP_DATA / "processed" / "encoded" / "trajectories_encoded.csv"

OUT_ROOT = MP_DATA / "outputs" / "prediction_rollout_fix"
REPORTS  = OUT_ROOT / "reports"
PLOTS    = OUT_ROOT / "plots"
DEBUG    = OUT_ROOT / "debug"

NEAR_ZERO_THRESHOLDS = [0.001, 0.005, 0.01, 0.05, 0.1]


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["person_id", "frame_number"])
    df["delta_x"] = df.groupby("person_id")["world_x"].diff().fillna(0)
    df["delta_y"] = df.groupby("person_id")["world_y"].diff().fillna(0)
    df["delta_mag"] = np.hypot(df["delta_x"], df["delta_y"])
    return df


def channel_stats(arr: np.ndarray, name: str) -> dict:
    return {
        "channel":   name,
        "count":     int(arr.size),
        "mean":      float(np.mean(arr)),
        "std":       float(np.std(arr)),
        "median":    float(np.median(arr)),
        "min":       float(np.min(arr)),
        "max":       float(np.max(arr)),
        "abs_mean":  float(np.mean(np.abs(arr))),
        "abs_median":float(np.median(np.abs(arr))),
        "p25":       float(np.percentile(arr, 25)),
        "p75":       float(np.percentile(arr, 75)),
        "p95_abs":   float(np.percentile(np.abs(arr), 95)),
    }


def near_zero_fractions(arr: np.ndarray, thresholds=NEAR_ZERO_THRESHOLDS) -> dict:
    return {f"frac_<{t}": float(np.mean(np.abs(arr) < t)) for t in thresholds}


def plot_histograms(du, dv, mag, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Target-delta distribution — full encoded dataset",
                 fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.hist(du, bins=120, color="#c0392b", alpha=0.75, edgecolor="white", lw=0.3)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yscale("log")
    ax.set_title(f"delta_x  (mean={du.mean():.4f}  std={du.std():.4f})")
    ax.set_xlabel("delta_x  (m)")
    ax.set_ylabel("count (log)")
    ax.grid(True, lw=0.3, alpha=0.5)

    ax = axes[0, 1]
    ax.hist(dv, bins=120, color="#2980b9", alpha=0.75, edgecolor="white", lw=0.3)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yscale("log")
    ax.set_title(f"delta_y  (mean={dv.mean():.4f}  std={dv.std():.4f})")
    ax.set_xlabel("delta_y  (m)")
    ax.set_ylabel("count (log)")
    ax.grid(True, lw=0.3, alpha=0.5)

    ax = axes[1, 0]
    ax.hist(mag, bins=120, color="#16a085", alpha=0.75, edgecolor="white", lw=0.3)
    ax.set_yscale("log")
    ax.set_title(f"|delta|  (mean={mag.mean():.4f}  median={np.median(mag):.4f})")
    ax.set_xlabel("|delta|  (m)")
    ax.set_ylabel("count (log)")
    ax.grid(True, lw=0.3, alpha=0.5)

    # Cumulative distribution of |delta|
    ax = axes[1, 1]
    sorted_mag = np.sort(mag)
    cdf = np.arange(1, len(sorted_mag) + 1) / len(sorted_mag)
    ax.plot(sorted_mag, cdf, color="#8e44ad", lw=1.5)
    for t in NEAR_ZERO_THRESHOLDS:
        frac = float(np.mean(mag < t))
        ax.axvline(t, color="gray", lw=0.5, ls=":")
        ax.text(t, 0.05, f"<{t}\n{frac*100:.1f}%", fontsize=7,
                ha="left", va="bottom", color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("|delta|  (m, log)")
    ax.set_ylabel("CDF")
    ax.set_title("Cumulative distribution of |delta|")
    ax.grid(True, lw=0.3, alpha=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    DEBUG.mkdir(parents=True, exist_ok=True)

    if not SRC_CSV.exists():
        sys.exit(f"[ERROR] {SRC_CSV} not found")

    print(f"[INFO]  Loading {SRC_CSV.name} ...")
    df = pd.read_csv(SRC_CSV)
    if "track_id" in df.columns and "person_id" not in df.columns:
        df = df.rename(columns={"track_id": "person_id"})
    if "frame_idx" in df.columns and "frame_number" not in df.columns:
        df = df.rename(columns={"frame_idx": "frame_number"})

    print(f"[INFO]  {len(df):,} rows | {df['person_id'].nunique()} persons")

    df = add_deltas(df)
    # First frame of each track has delta=0 by definition — drop to avoid skewing
    valid = df.groupby("person_id").cumcount() > 0
    df_v = df[valid]

    du   = df_v["delta_x"].to_numpy(dtype=np.float64)
    dv   = df_v["delta_y"].to_numpy(dtype=np.float64)
    mag  = df_v["delta_mag"].to_numpy(dtype=np.float64)

    stats_du  = channel_stats(du,  "delta_x")
    stats_dv  = channel_stats(dv,  "delta_y")
    stats_mag = channel_stats(mag, "|delta|")

    nz_du  = near_zero_fractions(du)
    nz_dv  = near_zero_fractions(dv)
    nz_mag = near_zero_fractions(mag)

    # ── Plots ───────────────────────────────────────────────────────────────
    hist_path = PLOTS / "target_distribution_histogram.png"
    plot_histograms(du, dv, mag, hist_path)
    print(f"[OK]    Plot   : {hist_path}")

    # ── CSV with raw stats ──────────────────────────────────────────────────
    csv_path = DEBUG / "target_distribution_stats.csv"
    pd.DataFrame([stats_du, stats_dv, stats_mag]).to_csv(csv_path, index=False)
    print(f"[OK]    CSV    : {csv_path}")

    # ── Verdict ─────────────────────────────────────────────────────────────
    # The model output for trajectory 310 sat near scaled (0.228, 0.387).
    # In real units this corresponds roughly to dataset-median delta.
    median_mag = stats_mag["median"]
    pct_zero_1cm  = nz_mag["frac_<0.01"] * 100
    pct_zero_5mm  = nz_mag["frac_<0.005"] * 100
    pct_zero_1mm  = nz_mag["frac_<0.001"] * 100

    if median_mag < 0.05:
        verdict = (f"**DATASET IS DOMINATED BY TINY MOTIONS.**  Median |delta| "
                   f"is only {median_mag*100:.2f} cm/step.  "
                   f"{pct_zero_1cm:.1f}% of frames have |delta| < 1 cm.  "
                   f"An MSE-trained regressor will inevitably learn to output "
                   f"near-zero deltas because that minimises mean error.")
    else:
        verdict = ("Dataset contains substantial motion in most frames; the "
                   "rollout collapse is unlikely to be a target-distribution "
                   "artefact.")

    # ── Markdown report ─────────────────────────────────────────────────────
    md = []
    md += [
        "# Target Distribution Audit",
        "",
        "**Goal:** determine whether the training-target distribution explains "
        "the observed rollout collapse (predicted |delta| ≈ 0.009 m vs GT 0.039 m).",
        "",
        f"**Dataset:** `{SRC_CSV.name}`  ({len(df):,} rows, "
        f"{df['person_id'].nunique()} persons)",
        f"**Targets:** `delta_x`, `delta_y` computed via "
        f"`groupby('person_id').diff()`  ({len(df_v):,} valid samples after "
        "dropping the zero-by-construction first frame of each track).",
        "",
        "---",
        "",
        "## 1. Channel statistics",
        "",
        "| Stat | `delta_x` | `delta_y` | `|delta|` |",
        "|---|---|---|---|",
        f"| count       | {stats_du['count']:,} | {stats_dv['count']:,} | {stats_mag['count']:,} |",
        f"| mean        | {stats_du['mean']:+.5f} | {stats_dv['mean']:+.5f} | {stats_mag['mean']:+.5f} |",
        f"| std         | {stats_du['std']:.5f} | {stats_dv['std']:.5f} | {stats_mag['std']:.5f} |",
        f"| median      | {stats_du['median']:+.5f} | {stats_dv['median']:+.5f} | {stats_mag['median']:+.5f} |",
        f"| abs-mean    | {stats_du['abs_mean']:.5f} | {stats_dv['abs_mean']:.5f} | — |",
        f"| abs-median  | {stats_du['abs_median']:.5f} | {stats_dv['abs_median']:.5f} | — |",
        f"| p25         | {stats_du['p25']:+.5f} | {stats_dv['p25']:+.5f} | {stats_mag['p25']:.5f} |",
        f"| p75         | {stats_du['p75']:+.5f} | {stats_dv['p75']:+.5f} | {stats_mag['p75']:.5f} |",
        f"| p95(|·|)    | {stats_du['p95_abs']:.5f} | {stats_dv['p95_abs']:.5f} | {stats_mag['p95_abs']:.5f} |",
        f"| min         | {stats_du['min']:+.5f} | {stats_dv['min']:+.5f} | {stats_mag['min']:.5f} |",
        f"| max         | {stats_du['max']:+.5f} | {stats_dv['max']:+.5f} | {stats_mag['max']:.5f} |",
        "",
        "## 2. Fraction of near-zero motion",
        "",
        "| Threshold | `|delta_x|` | `|delta_y|` | `|delta|` |",
        "|---|---|---|---|",
    ]
    for t in NEAR_ZERO_THRESHOLDS:
        md.append(
            f"| < {t:<5} m | {nz_du[f'frac_<{t}']*100:6.2f} % "
            f"| {nz_dv[f'frac_<{t}']*100:6.2f} % "
            f"| {nz_mag[f'frac_<{t}']*100:6.2f} % |"
        )

    md += [
        "",
        "## 3. Visual",
        "",
        f"![target distribution]({(PLOTS / 'target_distribution_histogram.png').as_posix()})",
        "",
        "## 4. Verdict",
        "",
        verdict,
        "",
        f"- Median |delta| per step : **{median_mag*1000:.2f} mm**",
        f"- Frames with |delta| < 1 cm : **{pct_zero_1cm:.1f} %**",
        f"- Frames with |delta| < 5 mm : **{pct_zero_5mm:.1f} %**",
        f"- Frames with |delta| < 1 mm : **{pct_zero_1mm:.1f} %**",
        "",
        "### Implication for the rollout",
        "",
        "If the median per-step displacement is much smaller than the typical "
        "motion you expect to see in deployed predictions, the optimiser is "
        "doing exactly what MSE asks it to: shrink toward the conditional mean. "
        "Inference-time corrections (alpha amplification, velocity persistence, "
        "minimum-motion floor) are the cheapest way to push deployed rollouts "
        "back toward realistic spatial extent without retraining or changing "
        "the thesis architecture.",
        "",
        "---",
        "",
        "*Generated by `experiments/rollout-fix/01_target_distribution_audit.py`*",
    ]

    out = REPORTS / "target_distribution_audit.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK]    Report : {out}")

    print("\n──────── SUMMARY ────────")
    print(f"  median |delta|     : {median_mag*1000:.2f} mm")
    print(f"  frames < 1 cm      : {pct_zero_1cm:.1f} %")
    print(f"  frames < 5 mm      : {pct_zero_5mm:.1f} %")
    print(f"  frames < 1 mm      : {pct_zero_1mm:.1f} %")
    print(f"  delta_x mean / std : {stats_du['mean']:+.4f} / {stats_du['std']:.4f}")
    print(f"  delta_y mean / std : {stats_dv['mean']:+.4f} / {stats_dv['std']:.4f}")


if __name__ == "__main__":
    main()
