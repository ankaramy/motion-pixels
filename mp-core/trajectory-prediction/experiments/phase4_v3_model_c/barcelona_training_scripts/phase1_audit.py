"""Phase 1 — dataset audit plots + report (read-only on the datasets)."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common as C

REC_ORDER = list(C.SPLIT_MAP.keys())
SPLIT_ORDER = ["train", "val", "test"]
SPLIT_COLOR = {"train": "#3aaa5e", "val": "#4c8cbf", "test": "#e05c2a"}
REC_COLOR = {r: SPLIT_COLOR[C.SPLIT_MAP[r]] for r in REC_ORDER}


def bar(ax, labels, values, colors, title, ylabel, rot=30):
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=rot, ha="right", fontsize=8)
    ax.set_title(title); ax.set_ylabel(ylabel)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=7)


def main():
    df = C.load_dataset()
    C.D_AUDIT.mkdir(parents=True, exist_ok=True)

    rows_by_rec = [int((df.recording_id == r).sum()) for r in REC_ORDER]
    tracks_by_rec = [int(df[df.recording_id == r].trajectory_id.nunique()) for r in REC_ORDER]
    rows_by_split = [int((df.split == s).sum()) for s in SPLIT_ORDER]
    tracks_by_split = [int(df[df.split == s].trajectory_id.nunique()) for s in SPLIT_ORDER]

    # 1-4 bar charts
    f, a = plt.subplots(figsize=(8, 4.5)); bar(a, REC_ORDER, rows_by_rec, [REC_COLOR[r] for r in REC_ORDER], "Rows by recording", "rows"); f.tight_layout(); f.savefig(C.D_AUDIT/"rows_by_recording.png"); plt.close(f)
    f, a = plt.subplots(figsize=(8, 4.5)); bar(a, REC_ORDER, tracks_by_rec, [REC_COLOR[r] for r in REC_ORDER], "Tracks by recording", "tracks"); f.tight_layout(); f.savefig(C.D_AUDIT/"tracks_by_recording.png"); plt.close(f)
    f, a = plt.subplots(figsize=(5, 4.5)); bar(a, SPLIT_ORDER, rows_by_split, [SPLIT_COLOR[s] for s in SPLIT_ORDER], "Rows by split", "rows", rot=0); f.tight_layout(); f.savefig(C.D_AUDIT/"rows_by_split.png"); plt.close(f)
    f, a = plt.subplots(figsize=(5, 4.5)); bar(a, SPLIT_ORDER, tracks_by_split, [SPLIT_COLOR[s] for s in SPLIT_ORDER], "Tracks by split", "tracks", rot=0); f.tight_layout(); f.savefig(C.D_AUDIT/"tracks_by_split.png"); plt.close(f)

    # 5 feature distributions by split
    feats = C.FEAT_COLS
    f, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, col in zip(axes.flat, feats):
        for s in SPLIT_ORDER:
            v = df.loc[df.split == s, col].to_numpy()
            ax.hist(v, bins=60, density=True, histtype="step", lw=1.5, color=SPLIT_COLOR[s], label=s)
        ax.set_title(col, fontsize=10); ax.set_yticks([])
    axes.flat[0].legend(fontsize=8)
    f.suptitle("Model C feature distributions by split (density)", fontweight="bold")
    f.tight_layout(rect=[0, 0, 1, 0.97]); f.savefig(C.D_AUDIT/"feature_distributions_by_split.png"); plt.close(f)

    # 6 target du/dv distribution
    f, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col in zip(axes, ["target_du", "target_dv"]):
        for s in SPLIT_ORDER:
            ax.hist(df.loc[df.split == s, col], bins=120, density=True, histtype="step", lw=1.4, color=SPLIT_COLOR[s], label=s)
        ax.set_title(col); ax.set_xlim(-3, 3); ax.legend(fontsize=8)
    f.suptitle("Target next-step displacement distribution (clipped to +/-3 m)", fontweight="bold")
    f.tight_layout(rect=[0, 0, 1, 0.95]); f.savefig(C.D_AUDIT/"target_du_dv_distribution.png"); plt.close(f)

    # 7 speed by split
    f, a = plt.subplots(figsize=(7, 4.5))
    for s in SPLIT_ORDER:
        a.hist(df.loc[df.split == s, "speed"], bins=120, density=True, histtype="step", lw=1.5, color=SPLIT_COLOR[s], label=s)
    a.set_xlim(0, 3); a.set_title("Speed distribution by split (clipped to 3 m/step)"); a.set_xlabel("speed (m/step)"); a.legend()
    f.tight_layout(); f.savefig(C.D_AUDIT/"speed_distribution_by_split.png"); plt.close(f)

    # 8 turn_rate by split
    f, a = plt.subplots(figsize=(7, 4.5))
    for s in SPLIT_ORDER:
        a.hist(df.loc[df.split == s, "turn_rate"], bins=120, density=True, histtype="step", lw=1.5, color=SPLIT_COLOR[s], label=s)
    a.set_title("Turn-rate distribution by split"); a.set_xlabel("turn_rate (rad)"); a.legend()
    f.tight_layout(); f.savefig(C.D_AUDIT/"turn_rate_distribution_by_split.png"); plt.close(f)

    # 9 spatial features by recording
    f, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col in zip(axes, C.SPATIAL_COLS):
        for r in REC_ORDER:
            ax.hist(df.loc[df.recording_id == r, col], bins=50, density=True, histtype="step", lw=1.4, color=REC_COLOR[r], label=r)
        ax.set_title(col); ax.set_xlabel("percentile-rank norm")
    axes[0].legend(fontsize=7)
    f.suptitle("Spatial (percentile-rank) features by recording", fontweight="bold")
    f.tight_layout(rect=[0, 0, 1, 0.95]); f.savefig(C.D_AUDIT/"spatial_features_by_recording.png"); plt.close(f)

    # 10 u/v density by recording
    f, axes = plt.subplots(1, 5, figsize=(22, 4.6))
    for ax, r in zip(axes, REC_ORDER):
        sub = df[df.recording_id == r]
        ax.hexbin(sub.u, sub.v, gridsize=45, cmap="viridis", mincnt=1)
        ax.set_title(f"{r}\n({C.SPLIT_MAP[r]})", fontsize=9); ax.set_aspect("equal")
        ax.set_xlabel("u"); ax.set_ylabel("v")
    f.suptitle("Normalized position (u,v) density by recording", fontweight="bold")
    f.tight_layout(rect=[0, 0, 1, 0.93]); f.savefig(C.D_AUDIT/"uv_density_by_recording.png"); plt.close(f)

    # 11 trajectory length distribution
    lens = df.groupby("trajectory_id").size()
    lens_by_split = {s: df[df.split == s].groupby("trajectory_id").size() for s in SPLIT_ORDER}
    f, a = plt.subplots(figsize=(8, 4.5))
    for s in SPLIT_ORDER:
        a.hist(lens_by_split[s], bins=60, histtype="step", lw=1.5, color=SPLIT_COLOR[s], label=f"{s} (median {int(lens_by_split[s].median())})")
    a.set_yscale("log"); a.set_title("Trajectory length distribution (rows per track)"); a.set_xlabel("timesteps"); a.set_ylabel("count (log)"); a.legend()
    f.tight_layout(); f.savefig(C.D_AUDIT/"trajectory_length_distribution.png"); plt.close(f)

    # NaN / Inf
    num = df.select_dtypes(include=[np.number])
    n_nan = int(num.isna().sum().sum())
    n_inf = int(np.isinf(num.to_numpy()).sum())

    # report
    feat_ranges = {c: [float(df[c].min()), float(df[c].max()), float(df[c].mean()), float(df[c].std())] for c in C.FEAT_COLS}
    tgt_ranges = {c: [float(df[c].min()), float(df[c].max())] for c in C.TARGET_COLS}
    rec_splits_ok = bool((df.groupby("recording_id").split.nunique() == 1).all())
    entrance_excluded = "entrance_affinity_norm" not in df.columns

    L = []
    A = L.append
    A("# Phase 1 — Barcelona Model C Dataset Audit\n")
    A(f"- Source: `{C.MODEL_C_CSV}`")
    A(f"- Total rows: **{len(df):,}**")
    A(f"- Total tracks: **{df.trajectory_id.nunique():,}**\n")
    A("## Rows / tracks by recording\n")
    A("| recording | split | rows | tracks |\n|---|---|---|---|")
    for r in REC_ORDER:
        A(f"| {r} | {C.SPLIT_MAP[r]} | {int((df.recording_id==r).sum()):,} | {int(df[df.recording_id==r].trajectory_id.nunique()):,} |")
    A("\n## Rows / tracks by split\n")
    A("| split | recordings | rows | tracks |\n|---|---|---|---|")
    for s in SPLIT_ORDER:
        recs = [r for r in REC_ORDER if C.SPLIT_MAP[r] == s]
        A(f"| {s} | {', '.join(recs)} | {int((df.split==s).sum()):,} | {int(df[df.split==s].trajectory_id.nunique()):,} |")
    A("\n## Feature ranges (Model C, 10 features)\n")
    A("| feature | min | max | mean | std |\n|---|---|---|---|---|")
    for c in C.FEAT_COLS:
        mn, mx, me, sd = feat_ranges[c]
        A(f"| {c} | {mn:.4f} | {mx:.4f} | {me:.4f} | {sd:.4f} |")
    A("\n## Target ranges\n")
    A("| target | min | max |\n|---|---|---|")
    for c in C.TARGET_COLS:
        A(f"| {c} | {tgt_ranges[c][0]:.4f} | {tgt_ranges[c][1]:.4f} |")
    A("\n## Integrity checks\n")
    A(f"- NaNs (numeric cols): **{n_nan}**")
    A(f"- Infinities (numeric cols): **{n_inf}**")
    A(f"- Recording-level split (each recording in exactly one split): **{rec_splits_ok}**")
    A(f"- `entrance_affinity_norm` excluded from Model C matrix: **{entrance_excluded}**")
    A(f"- Columns present: `{list(df.columns)}`\n")
    A("## Plots\n")
    for p in ["rows_by_recording", "tracks_by_recording", "rows_by_split", "tracks_by_split",
              "feature_distributions_by_split", "target_du_dv_distribution", "speed_distribution_by_split",
              "turn_rate_distribution_by_split", "spatial_features_by_recording", "uv_density_by_recording",
              "trajectory_length_distribution"]:
        A(f"- `{p}.png`")
    (C.D_AUDIT/"dataset_audit_report.md").write_text("\n".join(L), encoding="utf-8")

    summary = {"total_rows": int(len(df)), "total_tracks": int(df.trajectory_id.nunique()),
               "rows_by_recording": dict(zip(REC_ORDER, rows_by_rec)),
               "tracks_by_recording": dict(zip(REC_ORDER, tracks_by_rec)),
               "rows_by_split": dict(zip(SPLIT_ORDER, rows_by_split)),
               "tracks_by_split": dict(zip(SPLIT_ORDER, tracks_by_split)),
               "n_nan": n_nan, "n_inf": n_inf, "recording_level_split_ok": rec_splits_ok,
               "entrance_affinity_excluded": entrance_excluded}
    (C.D_AUDIT/"_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Phase 1 done. NaN", n_nan, "Inf", n_inf, "rec_split_ok", rec_splits_ok, "entrance_excluded", entrance_excluded)


if __name__ == "__main__":
    main()
