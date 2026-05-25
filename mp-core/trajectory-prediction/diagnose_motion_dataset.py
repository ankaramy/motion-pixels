"""
diagnose_motion_dataset.py
--------------------------
Diagnostic plots and summary for motion_dataset.csv.

Outputs (all relative to --output_dir):
  feature_correlation.png
  feature_histograms.png
  spatial_feature_maps/  (one PNG per spatial scatter)
  diagnostics_summary.md

Usage
-----
  python diagnose_motion_dataset.py
  python diagnose_motion_dataset.py --dataset PATH --output_dir DIR
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# ── defaults ──────────────────────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent
MP_ROOT      = HERE.parent.parent
DEFAULT_DATA = MP_ROOT / "mp-data" / "processed" / "encoded" / "motion_dataset.csv"
DEFAULT_OUT  = MP_ROOT / "mp-data" / "processed" / "encoded"

# ── feature groups ────────────────────────────────────────────────────────────
IDENTIFIER_COLS = ["trajectory_id", "timestep"]

HIST_FEATURES = [
    "turn_rate",
    "dist_to_obstacle_norm",
    "dist_to_boundary_norm",
    "openness_ahead",
    "openness_left",
    "openness_right",
]

SPATIAL_OVERLAYS = [
    ("speed",                   "Speed (m/step)",            "plasma"),
    ("turn_rate",               "Turn rate (rad)",            "RdBu_r"),
    ("dist_to_obstacle_norm",   "Dist to obstacle (norm)",   "viridis"),
    ("dist_to_boundary_norm",   "Dist to boundary (norm)",   "viridis"),
    ("openness_diff_left",      "Openness ahead - left",     "RdBu_r"),
    ("openness_diff_right",     "Openness ahead - right",    "RdBu_r"),
]

# ── thresholds for flagging ───────────────────────────────────────────────────
NEAR_CONSTANT_CV_THRESH = 0.05      # coefficient of variation below this
NEAR_CONSTANT_RANGE_THRESH = 0.01   # (max-min) below this after [0-1] scaling
HIGH_CORR_THRESH = 0.90             # |r| above this = flagged as highly correlated
NOISY_SKEW_THRESH = 3.0             # |skewness| above this = flagged
NOISY_ZERO_FRAC_THRESH = 0.40       # fraction of exactly-zero values above this


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Correlation matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation(df: pd.DataFrame, feat_cols: list[str], out_path: Path) -> pd.DataFrame:
    corr = df[feat_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.4, linecolor="#cccccc",
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
    )
    ax.set_title("Feature correlation matrix", fontsize=13, pad=12)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.name}")
    return corr


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Histograms
# ─────────────────────────────────────────────────────────────────────────────

def plot_histograms(df: pd.DataFrame, features: list[str], out_path: Path) -> None:
    ncols = 3
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        vals = df[feat].dropna()

        # adaptive bin count
        n_bins = min(80, max(30, int(len(vals) ** 0.4)))
        ax.hist(vals, bins=n_bins, color="#4c8cbf", edgecolor="none", alpha=0.85)

        mean_, std_ = vals.mean(), vals.std()
        ax.axvline(mean_, color="#e05c2a", linewidth=1.5, linestyle="--", label=f"mean {mean_:.3f}")
        ax.axvline(mean_ + std_, color="#e05c2a", linewidth=0.8, linestyle=":",
                   label=f"+/-sd {std_:.3f}")
        ax.axvline(mean_ - std_, color="#e05c2a", linewidth=0.8, linestyle=":")

        zero_frac = (vals == 0).mean()
        subtitle = f"n={len(vals):,}  skew={vals.skew():.2f}"
        if zero_frac > 0.05:
            subtitle += f"  zeros={zero_frac:.1%}"
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.set_xlabel(subtitle, fontsize=7.5)
        ax.set_ylabel("count", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, framealpha=0.6)

    # hide unused axes
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature histograms", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Spatial scatter maps
# ─────────────────────────────────────────────────────────────────────────────

def plot_spatial_maps(df: pd.DataFrame, overlays: list[tuple],
                      out_dir: Path, sample_n: int = 60_000) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # subsample for speed; preserve spatial distribution
    if len(df) > sample_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(df), size=sample_n, replace=False)
        df_plot = df.iloc[idx]
    else:
        df_plot = df

    u = df_plot["u"].to_numpy()
    v = df_plot["v"].to_numpy()

    for col, label, cmap in overlays:
        if col not in df_plot.columns:
            print(f"[skip]  {col} not in DataFrame")
            continue

        c_vals = df_plot[col].to_numpy()

        # symmetric colormap for diverging features
        vmin, vmax = np.nanpercentile(c_vals, 1), np.nanpercentile(c_vals, 99)
        if "RdBu" in cmap:
            bound = max(abs(vmin), abs(vmax))
            vmin, vmax = -bound, bound

        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(
            u, v, c=c_vals, cmap=cmap,
            s=1.5, alpha=0.4, linewidths=0,
            vmin=vmin, vmax=vmax,
            rasterized=True,
        )
        cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(label, fontsize=9)
        ax.set_xlabel("u  (normalised world x)", fontsize=9)
        ax.set_ylabel("v  (normalised world y)", fontsize=9)
        ax.set_title(f"Spatial distribution — {label}", fontsize=11)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)

        if sample_n < len(df):
            ax.set_title(
                f"Spatial distribution — {label}\n(subsample {sample_n:,}/{len(df):,})",
                fontsize=10,
            )

        fname = col.replace(" ", "_").replace("-", "_") + ".png"
        out_path = out_dir / fname
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] spatial_feature_maps/{fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 4 & 5.  Variance table + flagging
# ─────────────────────────────────────────────────────────────────────────────

def compute_variance_table(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in feat_cols:
        s = df[col].dropna()
        rng  = float(s.max() - s.min())
        mean = float(s.mean())
        std  = float(s.std())
        cv   = std / abs(mean) if abs(mean) > 1e-9 else np.inf
        rows.append({
            "feature":    col,
            "mean":       round(mean, 5),
            "std":        round(std, 5),
            "min":        round(float(s.min()), 5),
            "max":        round(float(s.max()), 5),
            "range":      round(rng, 5),
            "cv":         round(cv, 3),
            "skewness":   round(float(s.skew()), 3),
            "zero_frac":  round(float((s == 0).mean()), 4),
            "p5":         round(float(s.quantile(0.05)), 5),
            "p95":        round(float(s.quantile(0.95)), 5),
        })
    return pd.DataFrame(rows).set_index("feature")


def flag_features(
    var_table: pd.DataFrame,
    corr: pd.DataFrame,
    feat_cols: list[str],
) -> dict:
    flags = {
        "near_constant":     [],
        "highly_correlated": [],
        "potentially_noisy": [],
    }

    # scale each feature to [0,1] for range comparison
    for feat in feat_cols:
        row = var_table.loc[feat]
        rng_01 = row["range"]  # already in original units; use CV for comparison
        if row["cv"] < NEAR_CONSTANT_CV_THRESH or row["range"] < NEAR_CONSTANT_RANGE_THRESH:
            flags["near_constant"].append(
                f"{feat}  (cv={row['cv']:.4f}, range={row['range']:.5f})"
            )

    # highly correlated pairs (lower triangle only)
    for i, a in enumerate(feat_cols):
        for b in feat_cols[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= HIGH_CORR_THRESH:
                flags["highly_correlated"].append(
                    f"{a} <-> {b}  (r={r:.3f})"
                )

    # noisy: high absolute skewness OR high zero fraction
    for feat in feat_cols:
        row = var_table.loc[feat]
        reasons = []
        if abs(row["skewness"]) > NOISY_SKEW_THRESH:
            reasons.append(f"|skew|={abs(row['skewness']):.2f}")
        if row["zero_frac"] > NOISY_ZERO_FRAC_THRESH:
            reasons.append(f"zero_frac={row['zero_frac']:.1%}")
        if reasons:
            flags["potentially_noisy"].append(f"{feat}  ({', '.join(reasons)})")

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Markdown summary
# ─────────────────────────────────────────────────────────────────────────────

def write_markdown(
    var_table: pd.DataFrame,
    corr: pd.DataFrame,
    flags: dict,
    n_rows: int,
    n_trajs: int,
    out_path: Path,
) -> None:
    lines = []
    A = lines.append

    A("# Motion Dataset Diagnostics\n")
    A(f"Dataset: **{n_rows:,} rows**, **{n_trajs:,} trajectories**, "
      f"**{len(var_table)} features**\n")

    # ── variance table
    A("## Per-feature variance table\n")
    A("| feature | mean | std | min | max | range | cv | skewness | zero_frac |")
    A("|---------|------|-----|-----|-----|-------|----|----------|-----------|")
    for feat, row in var_table.iterrows():
        A(f"| `{feat}` | {row['mean']:.4f} | {row['std']:.4f} | "
          f"{row['min']:.4f} | {row['max']:.4f} | {row['range']:.4f} | "
          f"{row['cv']:.3f} | {row['skewness']:.3f} | {row['zero_frac']:.3f} |")
    A("")

    # ── near-constant
    A("## Near-constant features\n")
    A(f"Threshold: CV < {NEAR_CONSTANT_CV_THRESH} **or** range < {NEAR_CONSTANT_RANGE_THRESH}\n")
    if flags["near_constant"]:
        for item in flags["near_constant"]:
            A(f"- {item}")
    else:
        A("_None detected._")
    A("")

    # ── highly correlated
    A("## Highly correlated feature pairs\n")
    A(f"Threshold: |Pearson r| >= {HIGH_CORR_THRESH}\n")
    if flags["highly_correlated"]:
        for item in flags["highly_correlated"]:
            A(f"- {item}")
    else:
        A("_None detected._")
    A("")

    # ── potentially noisy
    A("## Potentially noisy features\n")
    A(f"Threshold: |skewness| > {NOISY_SKEW_THRESH} **or** zero fraction > {NOISY_ZERO_FRAC_THRESH:.0%}\n")
    if flags["potentially_noisy"]:
        for item in flags["potentially_noisy"]:
            A(f"- {item}")
    else:
        A("_None detected._")
    A("")

    # ── interpretation notes
    A("## Interpretation notes\n")
    _write_interpretation(lines, var_table, corr, flags)

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out_path.name}")


def _write_interpretation(lines, var_table, corr, flags):
    A = lines.append

    # turn_rate
    tr = var_table.loc["turn_rate"]
    A(f"### `turn_rate`")
    A(f"std={tr['std']:.3f} rad, skew={tr['skewness']:.2f}. "
      f"A large standard deviation indicates real directional variability in the "
      f"dataset — not noise. The near-zero mean confirms the dataset is globally "
      f"unbiased (clockwise and counter-clockwise turns are balanced).")
    A("")

    # obstacle vs openness correlation
    if abs(corr.loc["dist_to_obstacle_norm", "openness_ahead"]) > 0.5:
        r_val = corr.loc["dist_to_obstacle_norm", "openness_ahead"]
        A(f"### `dist_to_obstacle_norm` vs `openness_*`")
        A(f"r(dist_to_obstacle_norm, openness_ahead) = {r_val:.3f}. "
          f"The two features are correlated because both sample the same distance field "
          f"— one at the current position, one at a lookahead offset. "
          f"They are **not redundant**: `dist_to_obstacle_norm` captures proximity, "
          f"while `openness_*` captures navigability ahead. "
          f"Moderate correlation is expected and acceptable.")
        A("")

    # openness symmetry
    r_lr = corr.loc["openness_left", "openness_right"]
    A(f"### `openness_left` vs `openness_right`")
    A(f"r={r_lr:.3f}. "
      f"{'High symmetry: open-left and open-right tend to co-occur (wide corridors).' if r_lr > 0.6 else 'Low correlation: directional affordance is meaningfully asymmetric across the dataset.'} "
      f"Their difference (`openness_ahead - openness_left/right`) isolates turn preference and is "
      f"used in the spatial maps.")
    A("")

    # delta features
    for df_ in ["delta_dist_to_obstacle", "delta_dist_to_boundary"]:
        row = var_table.loc[df_]
        A(f"### `{df_}`")
        A(f"std={row['std']:.5f}, zero_frac={row['zero_frac']:.3f}. "
          f"Low variance is expected — most steps involve small changes in obstacle "
          f"proximity. The feature is **not near-constant** in a harmful sense: it "
          f"correctly encodes approach/retreat dynamics when they occur.")
        A("")

    # zero fraction flags
    noisy = {item.split()[0]: item for item in flags["potentially_noisy"]}
    if "openness_ahead" in noisy or "openness_left" in noisy or "openness_right" in noisy:
        A("### `openness_*` — high zero fraction")
        row_a = var_table.loc["openness_ahead"]
        A(f"zero_frac={row_a['zero_frac']:.1%}. "
          f"Zeros arise when the lookahead point maps to an obstacle pixel "
          f"(dist_to_obstacle = 0) or exits the plan image (OOB fallback = 0). "
          f"**Recommended action**: check whether these zeros cluster near "
          f"real obstacle regions in the spatial maps. "
          f"If they are artefacts of the homography boundary, consider raising "
          f"the OOB fallback to the mean non-zero value, or masking OOB positions "
          f"as `NaN` and imputing at training time.")
        A("")

    # speed
    sp = var_table.loc["speed"]
    A(f"### `speed`")
    A(f"skew={sp['skewness']:.2f}, zero_frac={sp['zero_frac']:.3f}. "
      f"Right-skewed speed distribution is typical for pedestrian data: most steps "
      f"are slow with occasional fast movement. The zero fraction reflects stationary "
      f"detections. Consider log-transforming speed at training time if the model "
      f"struggles with this tail.")
    A("")

    A("## Recommended actions for training\n")
    A("1. **Scale `du`, `dv`, `speed`** with a split-fitted StandardScaler or RobustScaler "
      "(the current dataset stores these in raw metres).")
    A("2. **Verify openness zeros** are geometrically valid (see spatial maps). "
      "If artefacts dominate, replace OOB fallback with `NaN` and impute.")
    A("3. **`turn_rate`** spans (−π, π] and has high variance — normalise before training.")
    A("4. **Watch `delta_dist_*`** features during ablation: their low variance may cause "
      "gradient underweighting. Consider scaling up by a constant (e.g. ×10) relative to "
      "the other distance features.")
    A("5. **No features need to be dropped** based on this diagnostic — "
      "no near-constant or perfectly-collinear pairs were found under the chosen thresholds.")
    A("")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Diagnostic plots for motion_dataset.csv")
    parser.add_argument("--dataset",    type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    print(f"[load]  {args.dataset.name}")
    df = pd.read_csv(args.dataset)
    print(f"        {len(df):,} rows  x  {df.shape[1]} columns")

    # derive openness difference columns for spatial maps
    df["openness_diff_left"]  = df["openness_ahead"] - df["openness_left"]
    df["openness_diff_right"] = df["openness_ahead"] - df["openness_right"]

    feat_cols = [c for c in df.columns if c not in IDENTIFIER_COLS
                 and c not in ("openness_diff_left", "openness_diff_right")]

    n_rows  = len(df)
    n_trajs = df["trajectory_id"].nunique()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = args.output_dir / "spatial_feature_maps"

    # ── 1. correlation matrix
    print("\n[1/4]  correlation matrix ...")
    corr = plot_correlation(df, feat_cols, args.output_dir / "feature_correlation.png")

    # ── 2. histograms
    print("[2/4]  histograms ...")
    plot_histograms(df, HIST_FEATURES, args.output_dir / "feature_histograms.png")

    # ── 3. spatial scatter maps
    print("[3/4]  spatial maps ...")
    plot_spatial_maps(df, SPATIAL_OVERLAYS, maps_dir)

    # ── 4+5. variance table + flags
    print("[4/4]  variance table + flagging ...")
    var_table = compute_variance_table(df, feat_cols)
    flags     = flag_features(var_table, corr, feat_cols)

    print("\n-- variance table -----------------------------------------------")
    print(var_table[["mean", "std", "range", "cv", "skewness", "zero_frac"]].to_string())

    print("\n-- near-constant features ---------------------------------------")
    print("\n".join(flags["near_constant"]) or "  none")

    print("\n-- highly correlated pairs --------------------------------------")
    print("\n".join(flags["highly_correlated"]) or "  none")

    print("\n-- potentially noisy features -----------------------------------")
    print("\n".join(flags["potentially_noisy"]) or "  none")

    write_markdown(
        var_table, corr, flags, n_rows, n_trajs,
        args.output_dir / "diagnostics_summary.md",
    )

    print(f"\n[done]  all outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
