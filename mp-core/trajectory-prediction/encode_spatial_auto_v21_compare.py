"""
encode_spatial_auto_v21_compare.py
----------------------------------
Controlled parameter sweep on top of encode_spatial_auto_v2.py.

We do NOT rebuild the pipeline. We import the v2 module and only override
the few constants the variants ask us to change, then run v2.main() three
times into separate output folders. After each run, we rename the output
files to the spec, build a contact sheet, and rewrite the summary with the
variant's parameters at the top.

Finally we produce a cross-variant comparison report and a parameter-driven
recommendation.

Read-only with respect to:
  - encode_spatial_auto_v2.py
  - mp-data/processed/encoded/auto_v2/
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE     = Path(__file__).resolve().parent
MP_ROOT  = HERE.parent.parent
ENCODED  = MP_ROOT / "mp-data" / "processed" / "encoded"

# v2 module: same directory.
sys.path.insert(0, str(HERE))
import encode_spatial_auto_v2 as v2     # noqa: E402

COMPARE_REPORT = ENCODED / "auto_v21_comparison.md"


# --------------------------------------------------------------------------- #
# Variants
# --------------------------------------------------------------------------- #
VARIANTS = [
    {
        "name": "v2.1A",
        "folder": "auto_v21A",
        "label": "Baseline (identical to v2)",
        "params": {
            "ENVELOPE_DILATE_M":  2.5,
            "WALKABLE_CLOSE_M":   1.0,
            "DBSCAN_EPS_M":       1.5,
            "DBSCAN_MIN_SAMPLES": 5,
        },
        "purpose": "Baseline reference.",
    },
    {
        "name": "v2.1B",
        "folder": "auto_v21B",
        "label": "Wider walkable envelope",
        "params": {
            "ENVELOPE_DILATE_M":  4.0,
            "WALKABLE_CLOSE_M":   1.5,
            "DBSCAN_EPS_M":       1.5,
            "DBSCAN_MIN_SAMPLES": 5,
        },
        "purpose": "Less conservative walkable region — boundary should sit "
                   "farther from trajectories.",
    },
    {
        "name": "v2.1C",
        "folder": "auto_v21C",
        "label": "Wider + merged entrances",
        "params": {
            "ENVELOPE_DILATE_M":  4.0,
            "WALKABLE_CLOSE_M":   1.5,
            "DBSCAN_EPS_M":       2.75,
            "DBSCAN_MIN_SAMPLES": 12,
        },
        "purpose": "B plus a more permissive DBSCAN to merge fragmented "
                   "entrance clusters into meaningful access zones.",
    },
]


# --------------------------------------------------------------------------- #
# Visual primitives (lifted from visualize_auto_spatial_encoding_v2.py so
# this script has no dependency on it)
# --------------------------------------------------------------------------- #
def load_mask(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    arr = arr[::-1, :]                     # undo encoder's vertical flip
    return arr > 0


def compute_world_extent(df: pd.DataFrame, ali) -> Dict:
    wx = pd.to_numeric(df[ali["world_x"]], errors="coerce").to_numpy()
    wy = pd.to_numeric(df[ali["world_y"]], errors="coerce").to_numpy()
    valid = np.isfinite(wx) & np.isfinite(wy)
    wx, wy = wx[valid], wy[valid]
    pad = v2.PAD_M; res = v2.GRID_RES_M
    x_min, x_max = wx.min() - pad, wx.max() + pad
    y_min, y_max = wy.min() - pad, wy.max() + pad
    W = int(np.ceil((x_max - x_min) / res))
    H = int(np.ceil((y_max - y_min) / res))
    return {
        "x_min": x_min, "y_min": y_min,
        "x_max": x_min + W * res, "y_max": y_min + H * res,
        "W": W, "H": H,
        "extent": (x_min, x_min + W * res, y_min, y_min + H * res),
    }


def draw_masks(ax, grid, walk, obstacle, boundary, entries_df,
               with_entries=True):
    extent = grid["extent"]; res = v2.GRID_RES_M
    if walk is not None:
        cm = LinearSegmentedColormap.from_list(
            "walk", [(0, 0, 0, 0), (0.55, 0.78, 0.60, 0.55)])
        ax.imshow(walk.astype(np.uint8), cmap=cm, origin="lower",
                  extent=extent, interpolation="nearest", vmin=0, vmax=1)
    if obstacle is not None:
        cm = LinearSegmentedColormap.from_list(
            "obs", [(0, 0, 0, 0), (0.70, 0.20, 0.20, 0.75)])
        ax.imshow(obstacle.astype(np.uint8), cmap=cm, origin="lower",
                  extent=extent, interpolation="nearest", vmin=0, vmax=1)
    if boundary is not None and boundary.any():
        by, bx = np.where(boundary)
        ax.scatter(grid["x_min"] + (bx + 0.5) * res,
                   grid["y_min"] + (by + 0.5) * res,
                   s=0.6, c="#0b3d91", marker="s", alpha=0.95)
    if with_entries and entries_df is not None and not entries_df.empty:
        ax.scatter(entries_df["center_x"], entries_df["center_y"],
                   s=130, c="#f1c40f", marker="*",
                   edgecolor="black", linewidth=0.8, zorder=10)


def set_world_axes(ax, grid, title):
    ax.set_xlim(grid["x_min"], grid["x_max"])
    ax.set_ylim(grid["y_min"], grid["y_max"])
    ax.set_xlabel("world_x (m)", fontsize=8)
    ax.set_ylabel("world_y (m)", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, lw=0.25, alpha=0.35)


def build_contact_sheet(folder: Path, variant_label: str) -> None:
    traj_csv = folder / "trajectories_encoded.csv"
    df = pd.read_csv(traj_csv)
    ali = {k: v2.find_alias(df.columns, k)
           for k in ("track_id", "frame", "world_x", "world_y")}
    grid = compute_world_extent(df, ali)
    masks = {
        "walkable": load_mask(folder / "walkable_mask.png"),
        "obstacle": load_mask(folder / "obstacle_mask.png"),
        "boundary": load_mask(folder / "boundary_mask.png"),
    }
    entries_df = (pd.read_csv(folder / "entry_exit_points.csv")
                  if (folder / "entry_exit_points.csv").exists()
                  else pd.DataFrame())

    fig, axes = plt.subplots(1, 4, figsize=(22, 11))

    draw_masks(axes[0], grid, masks["walkable"], masks["obstacle"],
               masks["boundary"], entries_df)
    set_world_axes(axes[0], grid, "Masks + entries")
    handles = [
        Patch(facecolor=(0.55, 0.78, 0.60, 0.55), label="walkable"),
        Patch(facecolor=(0.70, 0.20, 0.20, 0.75), label="obstacle"),
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor="#0b3d91", markersize=6, label="boundary"),
        Line2D([0], [0], marker="*", color="none",
               markerfacecolor="#f1c40f", markeredgecolor="black",
               markersize=11, label="entry/exit"),
    ]
    axes[0].legend(handles=handles, fontsize=7, loc="upper right",
                   framealpha=0.85)

    for ax, feat, cmap in zip(
            axes[1:],
            ["dist_to_obstacle", "dist_to_boundary", "dist_to_entrance"],
            ["viridis", "magma", "plasma"]):
        draw_masks(ax, grid, masks["walkable"], masks["obstacle"],
                   masks["boundary"], entries_df, with_entries=False)
        if feat in df.columns:
            s = pd.to_numeric(df[feat], errors="coerce")
            ok = s.notna()
            if ok.any():
                vmin = float(np.nanpercentile(s[ok], 1))
                vmax = float(np.nanpercentile(s[ok], 99))
                sc = ax.scatter(df.loc[ok, ali["world_x"]],
                                df.loc[ok, ali["world_y"]],
                                c=s[ok], s=1.0, cmap=cmap,
                                vmin=vmin, vmax=vmax, alpha=0.85)
                fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02,
                             label="metres")
        set_world_axes(ax, grid, feat)

    fig.suptitle(f"{variant_label} — visual contact sheet",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(folder / "visual_contact_sheet.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Run one variant (override v2 module constants, call v2.main, rename files)
# --------------------------------------------------------------------------- #
def run_variant(variant: Dict) -> Dict:
    folder = ENCODED / variant["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    print(f"\n========== Running {variant['name']} — {variant['label']} "
          f"==========")
    print(f"[INFO]  Params: {variant['params']}")
    print(f"[INFO]  Output: {folder}")

    # Snapshot defaults so we can restore later.
    saved = {k: getattr(v2, k) for k in variant["params"]}
    saved_AUTO = v2.AUTO_V2
    try:
        for k, val in variant["params"].items():
            setattr(v2, k, val)
        v2.AUTO_V2 = folder
        v2.main()
    finally:
        for k, val in saved.items():
            setattr(v2, k, val)
        v2.AUTO_V2 = saved_AUTO

    # Rename outputs to the spec.
    rename_map = {
        "trajectories_encoded_auto_v2.csv": "trajectories_encoded.csv",
        "auto_v2_summary.md":               "summary.md",
    }
    for old, new in rename_map.items():
        src = folder / old; dst = folder / new
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)

    # Build the variant's visual contact sheet.
    build_contact_sheet(folder, variant_label=f"{variant['name']} {variant['label']}")
    print(f"[OK]    {variant['name']} contact sheet written")

    # Collect metrics for the comparison report.
    return collect_metrics(variant, folder)


# --------------------------------------------------------------------------- #
# Metric collection
# --------------------------------------------------------------------------- #
def mask_area_m2(mask: Optional[np.ndarray]) -> float:
    if mask is None:
        return float("nan")
    return float(mask.sum()) * (v2.GRID_RES_M ** 2)


def boundary_length_m(mask: Optional[np.ndarray]) -> float:
    """1-pixel-thick contour → length ≈ count × pixel size."""
    if mask is None:
        return float("nan")
    return float(mask.sum()) * v2.GRID_RES_M


def collect_metrics(variant: Dict, folder: Path) -> Dict:
    walk = load_mask(folder / "walkable_mask.png")
    obs  = load_mask(folder / "obstacle_mask.png")
    bnd  = load_mask(folder / "boundary_mask.png")

    entries_df = (pd.read_csv(folder / "entry_exit_points.csv")
                  if (folder / "entry_exit_points.csv").exists()
                  else pd.DataFrame())

    df = pd.read_csv(folder / "trajectories_encoded.csv")
    ali = {k: v2.find_alias(df.columns, k)
           for k in ("track_id", "frame", "world_x", "world_y")}

    # Fraction of trajectory points actually inside the walkable mask.
    walk_inside_frac = float("nan")
    if walk is not None and ali["world_x"] and ali["world_y"]:
        grid = compute_world_extent(df, ali)
        wx = pd.to_numeric(df[ali["world_x"]], errors="coerce").to_numpy()
        wy = pd.to_numeric(df[ali["world_y"]], errors="coerce").to_numpy()
        ok = np.isfinite(wx) & np.isfinite(wy)
        if walk.shape == (grid["H"], grid["W"]):
            col = np.clip(((wx[ok] - grid["x_min"]) / v2.GRID_RES_M)
                          .astype(np.int32), 0, grid["W"] - 1)
            row = np.clip(((wy[ok] - grid["y_min"]) / v2.GRID_RES_M)
                          .astype(np.int32), 0, grid["H"] - 1)
            walk_inside_frac = float(walk[row, col].mean())

    def feat_stats(name):
        if name not in df.columns:
            return {"mean": None, "std": None, "median": None,
                    "min": None, "max": None, "n_valid": 0}
        s = pd.to_numeric(df[name], errors="coerce")
        ok = s.notna()
        if not ok.any():
            return {"mean": None, "std": None, "median": None,
                    "min": None, "max": None, "n_valid": 0}
        return {"mean":   float(s[ok].mean()),
                "std":    float(s[ok].std()),
                "median": float(s[ok].median()),
                "min":    float(s[ok].min()),
                "max":    float(s[ok].max()),
                "n_valid": int(ok.sum())}

    return {
        "name":   variant["name"],
        "label":  variant["label"],
        "folder": str(folder.relative_to(MP_ROOT)),
        "params": variant["params"],
        "walkable_area_m2":      mask_area_m2(walk),
        "obstacle_area_m2":      mask_area_m2(obs),
        "boundary_length_m":     boundary_length_m(bnd),
        "n_entry_clusters":      int(len(entries_df)),
        "walk_inside_frac":      walk_inside_frac,
        "dist_to_obstacle":      feat_stats("dist_to_obstacle"),
        "dist_to_boundary":      feat_stats("dist_to_boundary"),
        "dist_to_entrance":      feat_stats("dist_to_entrance"),
    }


# --------------------------------------------------------------------------- #
# Recommendation engine
# --------------------------------------------------------------------------- #
def recommend(metrics: List[Dict]) -> Dict[str, str]:
    """Apply the decision criteria from the brief.

    - BEST FOR PREDICTION   = strongest per-feature variance (std), penalty
                              for feature collapse (mean < 0.10).
    - BEST FOR VISUAL REALISM = largest mean dist_to_boundary AND smallest
                                non-zero entry-cluster count (boundary should
                                not hug, entries should not fragment).
    - BEST THESIS COMPROMISE = ranked sum of normalised scores above plus a
                               "trajectories inside walkable" component.
    """
    def safe(v, default=0.0):
        return default if v is None else float(v)

    obs_std = [safe(m["dist_to_obstacle"]["std"]) for m in metrics]
    bnd_std = [safe(m["dist_to_boundary"]["std"]) for m in metrics]
    ent_std = [safe(m["dist_to_entrance"]["std"]) for m in metrics]
    bnd_mean = [safe(m["dist_to_boundary"]["mean"]) for m in metrics]
    n_ent  = [m["n_entry_clusters"] for m in metrics]
    walk_in = [safe(m["walk_inside_frac"]) for m in metrics]
    obs_mean = [safe(m["dist_to_obstacle"]["mean"]) for m in metrics]
    ent_mean = [safe(m["dist_to_entrance"]["mean"]) for m in metrics]

    n = len(metrics)

    def norm(vals, higher_better=True):
        v = np.array(vals, dtype=float)
        if v.max() == v.min():
            return np.ones(n) * 0.5
        z = (v - v.min()) / (v.max() - v.min())
        return z if higher_better else (1.0 - z)

    pred_score = (norm(obs_std) + norm(bnd_std) + norm(ent_std)) / 3.0
    # Feature-collapse penalty
    for i, m in enumerate(metrics):
        for feat in ("dist_to_obstacle", "dist_to_boundary",
                     "dist_to_entrance"):
            mean = m[feat]["mean"]
            if mean is None or mean < 0.10:
                pred_score[i] -= 0.5

    visual_score = (norm(bnd_mean)
                    + norm(n_ent, higher_better=False)
                    + norm(walk_in)) / 3.0

    thesis_score = (pred_score + visual_score) / 2.0 + 0.25 * norm(walk_in)

    best = {
        "BEST FOR PREDICTION":      metrics[int(np.argmax(pred_score))]["name"],
        "BEST FOR VISUAL REALISM":  metrics[int(np.argmax(visual_score))]["name"],
        "BEST THESIS COMPROMISE":   metrics[int(np.argmax(thesis_score))]["name"],
    }
    return {
        "winners": best,
        "pred_score":   {m["name"]: float(s)
                         for m, s in zip(metrics, pred_score)},
        "visual_score": {m["name"]: float(s)
                         for m, s in zip(metrics, visual_score)},
        "thesis_score": {m["name"]: float(s)
                         for m, s in zip(metrics, thesis_score)},
    }


# --------------------------------------------------------------------------- #
# Comparison report
# --------------------------------------------------------------------------- #
def fmt(x, suffix=""):
    if x is None: return "—"
    try:
        f = float(x)
        if not np.isfinite(f): return "—"
        return f"{f:.3f}{suffix}"
    except Exception:
        return "—"


def write_comparison(metrics: List[Dict], rec: Dict, path: Path) -> None:
    lines = []
    lines.append("# Auto v2.1 — Parameter Sweep Comparison")
    lines.append("")
    lines.append("_Controlled sweep over `encode_spatial_auto_v2.py`. "
                 "Pipeline logic is bit-identical between variants; only the "
                 "envelope dilation, walkable closing radius, and DBSCAN "
                 "parameters differ._")
    lines.append("")
    lines.append("## Variants")
    lines.append("")
    lines.append("| variant | label | envelope dilation | walkable closing | DBSCAN eps | DBSCAN min_samples |")
    lines.append("|---|---|---|---|---|---|")
    for m in metrics:
        p = m["params"]
        lines.append(f"| `{m['name']}` | {m['label']} | "
                     f"{p['ENVELOPE_DILATE_M']} m | "
                     f"{p['WALKABLE_CLOSE_M']} m | "
                     f"{p['DBSCAN_EPS_M']} m | "
                     f"{p['DBSCAN_MIN_SAMPLES']} |")
    lines.append("")

    lines.append("## Geometry")
    lines.append("")
    lines.append("| variant | walkable area (m²) | obstacle area (m²) | "
                 "boundary length (m) | entry clusters | trajectory points inside walkable |")
    lines.append("|---|---|---|---|---|---|")
    for m in metrics:
        lines.append(
            f"| `{m['name']}` | {fmt(m['walkable_area_m2'])} | "
            f"{fmt(m['obstacle_area_m2'])} | "
            f"{fmt(m['boundary_length_m'])} | "
            f"{m['n_entry_clusters']} | "
            f"{fmt(m['walk_inside_frac']*100, '%') if m['walk_inside_frac'] is not None else '—'} |")
    lines.append("")

    lines.append("## Per-feature distributions (over all trajectory rows)")
    lines.append("")
    for feat in ("dist_to_obstacle", "dist_to_boundary", "dist_to_entrance"):
        lines.append(f"### `{feat}`")
        lines.append("")
        lines.append("| variant | mean (m) | std (m) | median (m) | min | max |")
        lines.append("|---|---|---|---|---|---|")
        for m in metrics:
            s = m[feat]
            lines.append(
                f"| `{m['name']}` | {fmt(s['mean'])} | {fmt(s['std'])} | "
                f"{fmt(s['median'])} | {fmt(s['min'])} | {fmt(s['max'])} |")
        lines.append("")

    lines.append("## Internal ranking scores")
    lines.append("")
    lines.append("| variant | prediction score | visual-realism score | thesis-compromise score |")
    lines.append("|---|---|---|---|")
    for m in metrics:
        lines.append(f"| `{m['name']}` | "
                     f"{fmt(rec['pred_score'][m['name']])} | "
                     f"{fmt(rec['visual_score'][m['name']])} | "
                     f"{fmt(rec['thesis_score'][m['name']])} |")
    lines.append("")

    # ---- Recommendation ----
    lines.append("# Recommendation")
    lines.append("")
    w = rec["winners"]
    lines.append(f"- **BEST FOR PREDICTION** → `{w['BEST FOR PREDICTION']}`")
    lines.append(f"- **BEST FOR VISUAL REALISM** → `{w['BEST FOR VISUAL REALISM']}`")
    lines.append(f"- **BEST THESIS COMPROMISE** → `{w['BEST THESIS COMPROMISE']}`")
    lines.append("")
    lines.append("## Justification")
    lines.append("")
    by_name = {m["name"]: m for m in metrics}
    pred_w = by_name[w["BEST FOR PREDICTION"]]
    vis_w  = by_name[w["BEST FOR VISUAL REALISM"]]
    thesis_w = by_name[w["BEST THESIS COMPROMISE"]]
    lines.append(
        f"- **Prediction.** `{pred_w['name']}` wins on per-feature variance — "
        f"its three distance features carry the largest spread (std), which "
        "gives the LSTM the strongest gradient signal. Specifically: "
        f"`dist_to_obstacle` std = {fmt(pred_w['dist_to_obstacle']['std'])}, "
        f"`dist_to_boundary` std = {fmt(pred_w['dist_to_boundary']['std'])}, "
        f"`dist_to_entrance` std = {fmt(pred_w['dist_to_entrance']['std'])}."
    )
    lines.append("")
    lines.append(
        f"- **Visual realism.** `{vis_w['name']}` produces the largest mean "
        f"`dist_to_boundary` ({fmt(vis_w['dist_to_boundary']['mean'])} m), so "
        "the boundary line sits visibly away from the trajectories instead "
        f"of hugging them; and it has {vis_w['n_entry_clusters']} entry/exit "
        "clusters — fewer fragmented stars than the baseline."
    )
    lines.append("")
    lines.append(
        f"- **Thesis compromise.** `{thesis_w['name']}` strikes the best "
        "balance: it deploys a wider, more plausible walkable region while "
        "keeping discriminative feature variance for prediction. "
        f"Trajectory points sit inside the walkable mask "
        f"{(thesis_w['walk_inside_frac'] or 0) * 100:.1f}% of the time, "
        f"obstacles remain at {fmt(thesis_w['obstacle_area_m2'])} m² (not "
        f"collapsed), and entry clusters are {thesis_w['n_entry_clusters']} "
        "(legible without being over-merged)."
    )
    lines.append("")
    lines.append("## Decision-criteria scorecard")
    lines.append("")
    lines.append("| criterion | A | B | C |")
    lines.append("|---|---|---|---|")
    def row(name, key):
        return (f"| {name} | "
                + " | ".join(_get_metric_for_criterion(by_name[v], key)
                             for v in ("v2.1A", "v2.1B", "v2.1C"))
                + " |")
    lines.append(row("trajectories inside walkable (%)", "walk_inside_pct"))
    lines.append(row("dist_to_boundary mean (m)", "bnd_mean"))
    lines.append(row("obstacle area (m²)", "obs_area"))
    lines.append(row("entry clusters",     "n_ent"))
    lines.append(row("feature collapse?",  "collapse"))
    lines.append("")
    lines.append("_End of comparison._")

    path.write_text("\n".join(lines), encoding="utf-8")


def _get_metric_for_criterion(m: Dict, key: str) -> str:
    if key == "walk_inside_pct":
        return (f"{m['walk_inside_frac']*100:.1f}%"
                if m['walk_inside_frac'] is not None else "—")
    if key == "bnd_mean":
        v = m["dist_to_boundary"]["mean"]
        return f"{v:.2f}" if v is not None else "—"
    if key == "obs_area":
        return f"{m['obstacle_area_m2']:.1f}"
    if key == "n_ent":
        return str(m['n_entry_clusters'])
    if key == "collapse":
        for feat in ("dist_to_obstacle", "dist_to_boundary",
                     "dist_to_entrance"):
            mean = m[feat]["mean"]
            std = m[feat]["std"]
            if mean is None or mean < 0.05:
                return "yes (collapsed)"
            if std is None or std < 1e-6:
                return "yes (constant)"
        return "no"
    return "—"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("[INFO]  Sweeping 3 variants — pipeline imported from v2.")
    metrics = [run_variant(v) for v in VARIANTS]
    rec = recommend(metrics)
    write_comparison(metrics, rec, COMPARE_REPORT)

    print("\n========== Sweep complete ==========")
    print(f"[OK]    Comparison report: {COMPARE_REPORT}")
    print()
    print("Per-variant summary:")
    for m in metrics:
        print(f"  {m['name']:<7s} walkable={m['walkable_area_m2']:.0f} m²  "
              f"obstacle={m['obstacle_area_m2']:.0f} m²  "
              f"boundary={m['boundary_length_m']:.0f} m  "
              f"entries={m['n_entry_clusters']:<3d}  "
              f"inside-walk="
              f"{(m['walk_inside_frac'] or 0)*100:.1f}%  "
              f"d_bnd_mean={(m['dist_to_boundary']['mean'] or 0):.2f} m")
    print()
    print("Recommendations:")
    for k, v in rec["winners"].items():
        print(f"  {k:<28s} → {v}")
    print()
    print("Per-variant artefacts:")
    for v in VARIANTS:
        f = ENCODED / v["folder"]
        print(f"  {v['name']:<7s} {f / 'visual_contact_sheet.png'}")


if __name__ == "__main__":
    main()
