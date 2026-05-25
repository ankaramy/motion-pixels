"""
diagnose_spatial_features.py
-----------------------------
Phase 2C: Spatial encoding diagnostic.

Loads the real (non-duplicated) encoded trajectory dataset and analyses
the three spatial distance features:
  - dist_to_obstacle
  - dist_to_boundary
  - dist_to_entrance

Goal: understand why these features failed to help in Phase 2A.

Outputs  mp-data/outputs/prediction/experiments/phase-2c/
  phase2c_spatial_feature_diagnostics.png   6-panel diagnostic figure
  phase2c_spatial_feature_report.csv        per-feature summary statistics
  phase2c_trajectory_density.png            world-space trajectory density
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

HERE    = Path(__file__).resolve().parent
MP_ROOT = HERE.parent.parent.parent.parent
MP_DATA = MP_ROOT / "mp-data"

SRC_CSV  = MP_DATA / "processed" / "encoded" / "trajectories_encoded.csv"
OUT_DIR  = MP_DATA / "outputs" / "prediction" / "experiments" / "phase-2c"
DIAG_PNG = OUT_DIR / "phase2c_spatial_feature_diagnostics.png"
DENS_PNG = OUT_DIR / "phase2c_trajectory_density.png"
RPT_CSV  = OUT_DIR / "phase2c_spatial_feature_report.csv"

SPATIAL_COLS = ["dist_to_obstacle", "dist_to_boundary", "dist_to_entrance"]
PALETTE      = {"dist_to_obstacle": "#e74c3c",
                "dist_to_boundary": "#2980b9",
                "dist_to_entrance": "#27ae60"}
LABELS       = {"dist_to_obstacle": "dist to obstacle (m)",
                "dist_to_boundary": "dist to boundary (m)",
                "dist_to_entrance": "dist to entrance (m)"}
CMAPS        = {"dist_to_obstacle": "Reds",
                "dist_to_boundary": "Blues",
                "dist_to_entrance": "Greens"}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    if not SRC_CSV.exists():
        raise FileNotFoundError(
            f"Encoded CSV not found: {SRC_CSV}\n"
            "Run encode_space.py first."
        )
    df = pd.read_csv(SRC_CSV)
    if "track_id" in df.columns and "person_id" not in df.columns:
        df = df.rename(columns={"track_id": "person_id"})
    if "frame_idx" in df.columns and "frame_number" not in df.columns:
        df = df.rename(columns={"frame_idx": "frame_number"})
    return df


# ---------------------------------------------------------------------------
# Statistics report
# ---------------------------------------------------------------------------

def build_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in SPATIAL_COLS:
        s = df[col]
        n_missing = s.isna().sum()
        s_valid   = s.dropna()
        rows.append({
            "feature":        col,
            "min":            float(s_valid.min()),
            "max":            float(s_valid.max()),
            "mean":           float(s_valid.mean()),
            "std":            float(s_valid.std()),
            "median":         float(s_valid.median()),
            "p5":             float(s_valid.quantile(0.05)),
            "p95":            float(s_valid.quantile(0.95)),
            "n_missing":      int(n_missing),
            "n_unique":       int(s_valid.nunique()),
            "pct_at_zero":    float((s_valid == 0).mean() * 100),
            "pct_constant":   float((s_valid == s_valid.iloc[0]).mean() * 100)
                              if len(s_valid) > 0 else 100.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Diagnostic figure  (2 rows × 3 cols: histograms + scatter plots)
# ---------------------------------------------------------------------------

def make_diagnostic_figure(df: pd.DataFrame, report: pd.DataFrame):
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#f8f8f8")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0: histograms ──────────────────────────────────────────────────
    for col_i, col in enumerate(SPATIAL_COLS):
        ax = fig.add_subplot(gs[0, col_i])
        s  = df[col].dropna()
        color = PALETTE[col]

        ax.hist(s, bins=80, color=color, alpha=0.78, edgecolor="white",
                linewidth=0.3)
        ax.axvline(s.mean(),   color="#2c3e50", lw=1.4, ls="--",
                   label=f"mean {s.mean():.2f}")
        ax.axvline(s.median(), color="#7f8c8d", lw=1.0, ls=":",
                   label=f"median {s.median():.2f}")
        ax.set_xlabel(LABELS[col], fontsize=9)
        ax.set_ylabel("count", fontsize=9)
        ax.set_title(f"Distribution — {LABELS[col]}", fontsize=10,
                     fontweight="bold", color="#2c3e50")
        ax.legend(fontsize=7.5, framealpha=0.85)
        ax.grid(True, lw=0.35, alpha=0.5)
        ax.set_facecolor("#fafafa")

        # Annotate suspicious stats
        row   = report[report["feature"] == col].iloc[0]
        notes = []
        if row["n_missing"] > 0:
            notes.append(f"{row['n_missing']} missing")
        if row["pct_at_zero"] > 30:
            notes.append(f"{row['pct_at_zero']:.0f}% = 0")
        if row["n_unique"] < 20:
            notes.append(f"only {row['n_unique']} unique vals")
        if row["std"] < 0.01:
            notes.append("near-constant!")
        if notes:
            ax.text(0.98, 0.97, "\n".join(notes),
                    transform=ax.transAxes, fontsize=7.5,
                    va="top", ha="right", color="#c0392b",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef9f9",
                              edgecolor="#e74c3c", alpha=0.9))

    # ── Row 1: scatter plots in world space ───────────────────────────────
    # Subsample for speed (keep ≤ 40K points)
    n_plot = min(40_000, len(df))
    df_s   = df.sample(n_plot, random_state=42)

    for col_i, col in enumerate(SPATIAL_COLS):
        ax    = fig.add_subplot(gs[1, col_i])
        vals  = df_s[col].fillna(df_s[col].median())
        vmin, vmax = float(df[col].quantile(0.02)), float(df[col].quantile(0.98))
        norm  = Normalize(vmin=vmin, vmax=vmax)
        cmap  = CMAPS[col]

        sc = ax.scatter(df_s["world_x"], df_s["world_y"],
                        c=vals, cmap=cmap, norm=norm,
                        s=1.5, alpha=0.55, linewidths=0, rasterized=True)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(LABELS[col], fontsize=7.5)
        cbar.ax.tick_params(labelsize=7)

        ax.set_xlabel("world_x (m)", fontsize=9)
        ax.set_ylabel("world_y (m)", fontsize=9)
        ax.set_title(f"Trajectory space — coloured by\n{LABELS[col]}",
                     fontsize=10, fontweight="bold", color="#2c3e50")
        ax.grid(True, lw=0.3, alpha=0.4, color="#cccccc")
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#555555")

    fig.suptitle(
        "Phase 2C — Spatial Feature Diagnostics\n"
        "Source: trajectories_encoded.csv  (real data, not overfit duplicate)  "
        f"|  {len(df):,} rows  |  {df['person_id'].nunique()} persons",
        fontsize=11, color="#2c3e50", y=1.01, fontweight="bold",
    )
    fig.savefig(DIAG_PNG, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK]  Diagnostics figure : {DIAG_PNG}")


# ---------------------------------------------------------------------------
# Trajectory density figure
# ---------------------------------------------------------------------------

def make_density_figure(df: pd.DataFrame):
    """Hexbin density of all trajectory points in world space."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.patch.set_facecolor("#1a1a2e")

    titles = ["Trajectory density"] + [LABELS[c] for c in SPATIAL_COLS]
    cmaps_all = ["hot"] + [CMAPS[c] for c in SPATIAL_COLS]
    data_c  = [None] + SPATIAL_COLS

    for ax, title, cmap, ccol in zip(axes, titles, cmaps_all, data_c):
        ax.set_facecolor("#0d0d1a")
        if ccol is None:
            hb = ax.hexbin(df["world_x"], df["world_y"],
                           gridsize=60, cmap=cmap, mincnt=1, linewidths=0.1)
            cb = fig.colorbar(hb, ax=ax, shrink=0.85, pad=0.02)
            cb.set_label("point count", fontsize=8, color="white")
        else:
            vals = df[ccol].fillna(df[ccol].median())
            hb = ax.hexbin(df["world_x"], df["world_y"],
                           C=vals, gridsize=60, cmap=cmap,
                           reduce_C_function=np.mean, mincnt=1, linewidths=0.1)
            cb = fig.colorbar(hb, ax=ax, shrink=0.85, pad=0.02)
            cb.set_label(title, fontsize=8, color="white")
        cb.ax.tick_params(colors="white", labelsize=7)
        ax.set_xlabel("world_x (m)", fontsize=9, color="white")
        ax.set_ylabel("world_y (m)", fontsize=9, color="white")
        ax.set_title(title, fontsize=10, color="white", fontweight="bold", pad=6)
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444444")

    fig.suptitle(
        "Phase 2C — Trajectory density vs spatial feature coverage  (hexbin mean)",
        fontsize=11, color="white", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(DENS_PNG, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK]  Density figure      : {DENS_PNG}")


# ---------------------------------------------------------------------------
# Text analysis
# ---------------------------------------------------------------------------

def print_analysis(df: pd.DataFrame, report: pd.DataFrame):
    print("\n" + "=" * 68)
    print("  PHASE 2C — SPATIAL FEATURE DIAGNOSTIC REPORT")
    print("=" * 68)
    print(f"  Dataset : {SRC_CSV.name}  |  {len(df):,} rows  |  "
          f"{df['person_id'].nunique()} persons\n")

    print(f"  {'Feature':<24}  {'min':>7}  {'max':>7}  {'mean':>7}  "
          f"{'std':>7}  {'p5':>7}  {'p95':>7}  {'miss':>5}  {'unique':>7}  {'%=0':>6}")
    print(f"  {'-'*24}  {'-'*7}  {'-'*7}  {'-'*7}  "
          f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*6}")
    for _, r in report.iterrows():
        print(f"  {r['feature']:<24}  {r['min']:>7.3f}  {r['max']:>7.3f}  "
              f"{r['mean']:>7.3f}  {r['std']:>7.3f}  {r['p5']:>7.3f}  "
              f"{r['p95']:>7.3f}  {int(r['n_missing']):>5}  "
              f"{int(r['n_unique']):>7}  {r['pct_at_zero']:>5.1f}%")

    print("\n  Pairwise correlations (Pearson):")
    corr = df[SPATIAL_COLS].corr()
    pairs = [
        ("dist_to_obstacle", "dist_to_boundary"),
        ("dist_to_obstacle", "dist_to_entrance"),
        ("dist_to_boundary", "dist_to_entrance"),
    ]
    for a, b in pairs:
        print(f"    {a} x {b} : {corr.loc[a, b]:+.3f}")

    print("\n" + "-" * 68)
    print("  ANALYSIS")
    print("-" * 68)

    issues = []

    for _, r in report.iterrows():
        feat  = r["feature"]
        label = LABELS[feat]
        flags = []
        if r["n_missing"] > 0:
            flags.append(f"{r['n_missing']} missing values ({r['n_missing']/len(df)*100:.1f}%)")
        if r["std"] < 0.1:
            flags.append(f"near-constant (std={r['std']:.4f}) — almost no variation")
        elif r["std"] < 0.5:
            flags.append(f"low variation (std={r['std']:.3f})")
        if r["pct_at_zero"] > 40:
            flags.append(f"{r['pct_at_zero']:.0f}% of values are exactly 0 — possible encoding error")
        elif r["pct_at_zero"] > 15:
            flags.append(f"{r['pct_at_zero']:.0f}% of values are 0 — many points at boundary")
        if r["n_unique"] < 50:
            flags.append(f"only {r['n_unique']} unique values — feature may be discretised or broken")
        dynamic_range = r["p95"] - r["p5"]
        if dynamic_range < 0.5:
            flags.append(f"tight dynamic range (p5–p95 span = {dynamic_range:.3f} m)")

        if flags:
            print(f"\n  [{feat}]")
            for f in flags:
                print(f"    - {f}")
            issues.append(feat)
        else:
            print(f"\n  [{feat}]  appears healthy (std={r['std']:.3f}, "
                  f"range {r['min']:.2f}–{r['max']:.2f} m, "
                  f"0 missing, {r['n_unique']} unique values)")

    print("\n" + "-" * 68)
    print("  WHY DID SPATIAL FEATURES FAIL IN PHASE 2A?")
    print("-" * 68)

    # Compute correlations with world position
    corr_x = df[SPATIAL_COLS + ["world_x"]].corr()["world_x"][SPATIAL_COLS]
    corr_y = df[SPATIAL_COLS + ["world_y"]].corr()["world_y"][SPATIAL_COLS]
    print("\n  Correlation with world_x / world_y:")
    for col in SPATIAL_COLS:
        print(f"    {col:<28}  x: {corr_x[col]:+.3f}   y: {corr_y[col]:+.3f}")

    # Compute mutual information / variance captured with movement
    print()
    for _, r in report.iterrows():
        col = r["feature"]
        std = r["std"]
        drange = r["p95"] - r["p5"]
        if std < 0.3 or drange < 0.5:
            print(f"  [{col}]  LOW SIGNAL — std={std:.3f}, "
                  f"p5-p95 span={drange:.3f}m. The model sees almost the same value "
                  f"regardless of where the pedestrian is. This feature adds noise, "
                  f"not signal.")
        else:
            print(f"  [{col}]  sufficient variation — "
                  f"std={std:.3f}, span={drange:.3f}m.")

    print("\n" + "=" * 68)
    print("  ANSWERS")
    print("=" * 68)

    all_report = {}
    for _, r in report.iterrows():
        all_report[r["feature"]] = r

    varies = all([all_report[c]["std"] >= 0.3 for c in SPATIAL_COLS])
    print(f"  Do the spatial features vary enough?        "
          f"{'YES' if varies else 'NO — at least one is too flat'}")

    suspicious = [c for c in SPATIAL_COLS
                  if all_report[c]["std"] < 0.1 or all_report[c]["pct_at_zero"] > 40
                  or all_report[c]["n_unique"] < 50]
    print(f"  Suspicious constant/flat values?            "
          f"{'YES: ' + ', '.join(suspicious) if suspicious else 'NO'}")

    missing_any = any(all_report[c]["n_missing"] > 0 for c in SPATIAL_COLS)
    print(f"  Missing values?                             "
          f"{'YES' if missing_any else 'NO'}")

    print(f"  Visual alignment with trajectory space?     "
          f"Check phase2c_spatial_feature_diagnostics.png scatter plots")
    print(f"  Phase 2A failure — most likely cause(s):")
    for col in SPATIAL_COLS:
        r = all_report[col]
        drange = r["p95"] - r["p5"]
        if r["std"] < 0.3:
            print(f"    - {col}: near-constant (std={r['std']:.3f}) "
                  f"— indistinguishable from noise for the model")
        if r["pct_at_zero"] > 15:
            print(f"    - {col}: {r['pct_at_zero']:.0f}% zeros "
                  f"— degenerate mass at boundary dominates the distribution")
        if drange < 1.0:
            print(f"    - {col}: tight range in populated area "
                  f"(p5–p95 only {drange:.2f}m) "
                  f"— insufficient spatial gradient for the model to exploit")

    print("=" * 68 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO]  Loading {SRC_CSV.name} ...")
    df = load_data()
    print(f"[INFO]  {len(df):,} rows  |  {df['person_id'].nunique()} persons  |  "
          f"columns: {list(df.columns)}")

    missing_cols = [c for c in SPATIAL_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing spatial columns: {missing_cols}")

    print("[INFO]  Computing statistics ...")
    report = build_report(df)
    report.to_csv(RPT_CSV, index=False, float_format="%.6f")
    print(f"[OK]   Report CSV         : {RPT_CSV}")

    print("[INFO]  Generating diagnostic figure ...")
    make_diagnostic_figure(df, report)

    print("[INFO]  Generating density figure ...")
    make_density_figure(df)

    print_analysis(df, report)


if __name__ == "__main__":
    main()
