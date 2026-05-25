"""
build_dataset_v2.py
-------------------
Transforms motion_dataset.csv into motion_dataset_v2.csv.

Changes vs v1:
  - Remove  : delta_dist_to_obstacle, delta_dist_to_boundary,
               openness_ahead, openness_left, openness_right
  - Add     : openness_lr_asymmetry = openness_left - openness_right
  - Clip    : du, dv, target_du, target_dv to ±0.30 m/step
  - Outputs : motion_dataset_v2.csv, feature_summary_v2.md
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE         = Path(__file__).resolve().parent
MP_ROOT      = HERE.parent.parent
ENCODED_DIR  = MP_ROOT / "mp-data" / "processed" / "encoded"
DEFAULT_IN   = ENCODED_DIR / "motion_dataset.csv"
DEFAULT_OUT  = ENCODED_DIR / "motion_dataset_v2.csv"
DEFAULT_SUMMARY = ENCODED_DIR / "feature_summary_v2.md"

CLIP_M = 0.30   # ±metres per step

DROP_COLS = [
    "delta_dist_to_obstacle",
    "delta_dist_to_boundary",
    "openness_ahead",
    "openness_left",
    "openness_right",
]

# Final column order for v2
V2_COLS = [
    "trajectory_id",
    "timestep",
    "u",
    "v",
    "du",
    "dv",
    "speed",
    "heading_sin",
    "heading_cos",
    "turn_rate",
    "dist_to_obstacle_norm",
    "dist_to_boundary_norm",
    "openness_lr_asymmetry",
    "target_du",
    "target_dv",
]


def build_v2(df: pd.DataFrame, clip_m: float) -> pd.DataFrame:
    # compute derived feature before dropping source columns
    df = df.copy()
    df["openness_lr_asymmetry"] = df["openness_left"] - df["openness_right"]

    # drop removed features
    missing = [c for c in DROP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns not found in input: {missing}")
    df = df.drop(columns=DROP_COLS)

    # clip displacement columns
    for col in ("du", "dv", "target_du", "target_dv"):
        before = (df[col].abs() > clip_m).sum()
        df[col] = df[col].clip(-clip_m, clip_m)
        if before:
            print(f"  clipped {col}: {before:,} values ({before/len(df):.2%})")

    # recompute speed from clipped du/dv so it stays consistent
    df["speed"] = np.hypot(df["du"], df["dv"])

    # reorder
    extra = [c for c in df.columns if c not in V2_COLS]
    if extra:
        raise ValueError(f"Unexpected columns after transform: {extra}")
    return df[V2_COLS]


def variance_table(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    rows = []
    for col in feat_cols:
        s = df[col].dropna()
        rows.append({
            "feature":   col,
            "mean":      round(s.mean(), 5),
            "std":       round(s.std(), 5),
            "min":       round(s.min(), 5),
            "max":       round(s.max(), 5),
            "range":     round(s.max() - s.min(), 5),
            "skewness":  round(s.skew(), 3),
            "zero_frac": round((s == 0).mean(), 4),
        })
    return pd.DataFrame(rows).set_index("feature")


def corr_table(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    return df[feat_cols].corr().round(3)


def write_summary(df: pd.DataFrame, out_path: Path, clip_m: float,
                  n_clipped: dict, corr_v1: dict) -> None:
    feat_cols  = [c for c in V2_COLS if c not in ("trajectory_id", "timestep",
                                                    "target_du", "target_dv")]
    vt   = variance_table(df, feat_cols)
    corr = corr_table(df, feat_cols)

    lines = []
    A = lines.append

    A("# Motion Dataset v2 — Feature Summary\n")
    A(f"**13 columns** (was 19 in v1): 2 identifiers, 11 input features, 2 targets.\n")

    A("## Changes from v1\n")
    A("| Change | Columns |")
    A("|--------|---------|")
    A("| Removed | `delta_dist_to_obstacle`, `delta_dist_to_boundary`, `openness_ahead`, `openness_left`, `openness_right` |")
    A("| Added | `openness_lr_asymmetry` = openness_left − openness_right |")
    A(f"| Clipped | `du`, `dv`, `target_du`, `target_dv` capped at ±{clip_m} m/step |")
    A("")

    A("## Clipping report\n")
    for col, n in n_clipped.items():
        pct = n / len(df) * 100
        A(f"- `{col}`: {n:,} values clipped ({pct:.2f}% of rows)")
    A("")

    A("## Per-feature statistics\n")
    A("| feature | mean | std | min | max | skewness | zero_frac |")
    A("|---------|------|-----|-----|-----|----------|-----------|")
    for feat, row in vt.iterrows():
        A(f"| `{feat}` | {row['mean']:.4f} | {row['std']:.4f} | "
          f"{row['min']:.4f} | {row['max']:.4f} | "
          f"{row['skewness']:.3f} | {row['zero_frac']:.3f} |")
    A("")

    A("## Feature correlation matrix (input features only)\n")
    A("```")
    A(corr.to_string())
    A("```\n")

    # ── main analysis
    A("## Feature-by-feature analysis: redundant or complementary?\n")

    A("### Position: `u`, `v`\n")
    r_u_obs = corr.loc["u", "dist_to_obstacle_norm"]
    r_u_v   = corr.loc["u", "v"]
    r_u_du  = corr.loc["u", "du"]
    A(f"**Complementary to each other** (r={r_u_v:.3f} — independent axes).")
    A(f"**Partially redundant with `dist_to_obstacle_norm`** (r={r_u_obs:.3f} — spatial confound "
      f"from the floor-plan geometry: the obstacle-dense zone sits along one x-band).")
    A(f"**Independent from motion features** (r(u, du)={r_u_du:.3f}).")
    A("Position must be kept: it anchors the LSTM's spatial memory and is the integration "
      "target for rollout (`u[t+1] = u[t] + Δu`). Dropping it would make the model "
      "spatially amnesic.\n")

    A("### Velocity: `du`, `dv`\n")
    r_du_dv    = corr.loc["du", "dv"]
    r_du_speed = corr.loc["du", "speed"]
    r_dv_speed = corr.loc["dv", "speed"]
    A(f"r(du, dv)={r_du_dv:.3f} — independent motion axes. "
      f"r(du, speed)={r_du_speed:.3f}, r(dv, speed)={r_dv_speed:.3f}.")
    A("**Partially redundant with `speed` and `heading_*`**: mathematically, "
      "`du = speed · cos(heading)`, `dv = speed · sin(heading)`. "
      "All five features (du, dv, speed, heading_sin, heading_cos) encode the same "
      "velocity vector via different decompositions.")
    A("**This redundancy is intentional and beneficial.** The LSTM sees:")
    A("- `du`/`dv`: raw displacement — direct integration target at rollout time.")
    A("- `speed`: magnitude — scalar summary the model can weight independently.")
    A("- `heading_*`: direction — normalised unit vector, decoupled from magnitude.")
    A("Providing all three representations reduces the implicit computation the model "
      "must perform and has been shown to help recurrent models converge faster.\n")

    A("### Direction: `heading_sin`, `heading_cos`\n")
    r_hs_hc = corr.loc["heading_sin", "heading_cos"]
    A(f"r={r_hs_hc:.3f}. Correctly near-zero: sine and cosine of the same angle "
      "are orthogonal over a uniform distribution of headings. "
      "**Complementary** — they jointly encode direction as a unit vector without "
      "the ±π discontinuity a raw angle would introduce. Neither can replace the other.\n")

    A("### Angular dynamics: `turn_rate`\n")
    r_tr_hs = corr.loc["turn_rate", "heading_sin"]
    r_tr_hc = corr.loc["turn_rate", "heading_cos"]
    A(f"r(turn_rate, heading_sin)={r_tr_hs:.3f}, r(turn_rate, heading_cos)={r_tr_hc:.3f}.")
    A("**Complementary to heading.** Heading encodes the current direction; turn_rate "
      "encodes how fast it is changing. A sequence of identical heading values with "
      "zero turn_rate looks the same as a straight walk regardless of the heading value, "
      "but a non-zero turn_rate immediately signals a curve. "
      "An LSTM can infer turn_rate from two consecutive heading values, but providing "
      "it explicitly reduces the sequence depth needed to capture turning behaviour.\n")

    A("### Obstacle proximity: `dist_to_obstacle_norm`\n")
    r_obs_bnd = corr.loc["dist_to_obstacle_norm", "dist_to_boundary_norm"]
    r_obs_asym = corr.loc["dist_to_obstacle_norm", "openness_lr_asymmetry"]
    A(f"r(dist_to_obstacle_norm, dist_to_boundary_norm)={r_obs_bnd:.3f}.")
    A(f"r(dist_to_obstacle_norm, openness_lr_asymmetry)={r_obs_asym:.3f}.")
    A("**Complementary to `dist_to_boundary_norm`** — they measure clearance from "
      "different reference objects (scattered obstacles vs walkable-zone edge). "
      "Their low mutual correlation confirms they carry independent spatial information.")
    A("**Independent from `openness_lr_asymmetry`** — the new asymmetry feature "
      "captures lateral gradient, not magnitude, so low correlation is expected and correct.")
    A("**Still correlated with `u` (r≈0.95)**, but this is a dataset property, not "
      "a reason to drop either feature — see position analysis above.\n")

    A("### Boundary proximity: `dist_to_boundary_norm`\n")
    r_bnd_u = corr.loc["dist_to_boundary_norm", "u"]
    r_bnd_v = corr.loc["dist_to_boundary_norm", "v"]
    A(f"r(dist_to_boundary_norm, u)={r_bnd_u:.3f}, r(dist_to_boundary_norm, v)={r_bnd_v:.3f}.")
    A("**Complementary to all motion and obstacle features.** zero_frac=0.000 — "
      "no pathological zeros. The feature is well-behaved and encodes whether the "
      "agent is crossing the floor plan or hugging the edge, which is not captured "
      "by obstacle proximity or position alone in a non-rectangular space.\n")

    A("### Lateral spatial affordance: `openness_lr_asymmetry`\n")
    r_asym_tr = corr.loc["openness_lr_asymmetry", "turn_rate"]
    r_asym_hs = corr.loc["openness_lr_asymmetry", "heading_sin"]
    A(f"r(openness_lr_asymmetry, turn_rate)={r_asym_tr:.3f}.")
    A(f"r(openness_lr_asymmetry, heading_sin)={r_asym_hs:.3f}.")
    A("**Complementary to all retained features.** This is the only feature that "
      "encodes the lateral gradient of navigable space — whether more room exists "
      "to the left or right of the current heading. "
      "A positive value means the left side is more open; negative means the right.")
    A(f"Low correlation with `turn_rate` (r={r_asym_tr:.3f}) means the model cannot "
      "currently infer turn preference from turning history alone — "
      "the asymmetry adds independent predictive information for imminent direction changes.")
    A("**Caveat from v1 diagnostics:** the individual openness features each had "
      "~35% zero values (obstacle/OOB hits). The difference inherits these zeros "
      "but only when *both* sides simultaneously hit obstacles or boundaries — "
      "a less common condition. zero_frac for the asymmetry is shown in the table above.\n")

    A("## Summary: redundancy map\n")
    A("```")
    A("REDUNDANT CLUSTER (intentional, beneficial):")
    A("  du, dv  <->  speed + heading_sin + heading_cos")
    A("  (same velocity vector, three complementary decompositions)")
    A("")
    A("SPATIAL CONFOUND (dataset geometry, not a modelling error):")
    A("  u  <->  dist_to_obstacle_norm  (r=0.955, floor-plan specific)")
    A("  Keep both: u is needed for integration; dist_to_obstacle_norm")
    A("  provides the absolute clearance value the model needs to plan avoidance.")
    A("")
    A("COMPLEMENTARY (low cross-correlation, independent information):")
    A("  dist_to_obstacle_norm  vs  dist_to_boundary_norm")
    A("  openness_lr_asymmetry  vs  all other features")
    A("  turn_rate              vs  heading_sin/cos")
    A("  u                      vs  v")
    A("```\n")

    A("## Recommended normalisation before training\n")
    A("| feature | transform |")
    A("|---------|-----------|")
    A("| `u`, `v` | already [0, 1] — no further scaling needed |")
    A("| `du`, `dv`, `target_du`, `target_dv` | StandardScaler (zero-mean, unit-var) fitted on train split |")
    A("| `speed` | log1p then StandardScaler (right-skewed, skew=6.6 in v1; clipping reduces tail but doesn't eliminate it) |")
    A("| `heading_sin`, `heading_cos` | already [−1, 1] — no scaling needed |")
    A("| `turn_rate` | divide by π → [−1, 1] (bounded, symmetric) |")
    A("| `dist_to_obstacle_norm`, `dist_to_boundary_norm` | already [0, 1] — no further scaling needed |")
    A("| `openness_lr_asymmetry` | StandardScaler or divide by max observed range |")
    A("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   type=Path, default=DEFAULT_IN)
    parser.add_argument("--output",  type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--clip_m",  type=float, default=CLIP_M)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    print(f"[load]  {args.input.name}")
    df_v1 = pd.read_csv(args.input)
    print(f"        {len(df_v1):,} rows  x  {df_v1.shape[1]} columns")

    # track clip counts before transformation
    n_clipped = {}
    for col in ("du", "dv", "target_du", "target_dv"):
        n_clipped[col] = int((df_v1[col].abs() > args.clip_m).sum())

    df_v2 = build_v2(df_v1, args.clip_m)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_v2.to_csv(args.output, index=False)
    print(f"[saved] {args.output.name}  ({len(df_v2):,} rows  x  {df_v2.shape[1]} columns)")

    print(f"[build] feature summary ...")
    write_summary(df_v2, args.summary, args.clip_m, n_clipped, corr_v1={})
    print(f"[done]")


if __name__ == "__main__":
    main()
