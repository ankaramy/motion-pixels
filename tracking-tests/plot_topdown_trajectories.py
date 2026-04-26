"""
plot_topdown_trajectories.py
----------------------------
Generate a clean plan-view trajectory diagram from world-coordinate trajectories.

Usage:
    python plot_topdown_trajectories.py \
        --traj_csv outputs/trajectories_world.csv \
        --out_png  outputs/topdown.png \
        --color_by track_id
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_speed_col(df: pd.DataFrame) -> str | None:
    """Return the smoothed speed column name, preferring world (m/s) over image (px/s)."""
    for candidate in ("speed_smooth_m_s", "speed_smooth_px_s", "speed_m_s", "speed_px_s"):
        if candidate in df.columns:
            return candidate
    return None


def segments_from_track(x: np.ndarray, y: np.ndarray):
    """Return an (N-1, 2, 2) array of line segments for LineCollection."""
    pts = np.column_stack([x, y])          # (N, 2)
    return np.stack([pts[:-1], pts[1:]], axis=1)   # (N-1, 2, 2)


# ---------------------------------------------------------------------------
# Colouring strategies
# ---------------------------------------------------------------------------

def plot_by_track_id(ax, df: pd.DataFrame, show_labels: bool):
    track_ids = sorted(df["track_id"].unique())
    cmap = plt.get_cmap("tab20", max(len(track_ids), 1))

    for i, tid in enumerate(track_ids):
        t = df[df["track_id"] == tid].sort_values("frame")
        color = cmap(i % 20)
        ax.plot(t["world_x"], t["world_y"], linewidth=0.9, alpha=0.75, color=color)
        ax.scatter(t["world_x"].iloc[0], t["world_y"].iloc[0], s=14, color=color, zorder=4)
        if show_labels:
            ax.text(t["world_x"].iloc[0], t["world_y"].iloc[0],
                    f" {tid}", fontsize=5, color=color, va="center")


def plot_by_speed(ax, df: pd.DataFrame, speed_col: str):
    vmin = float(np.nanpercentile(df[speed_col].dropna(), 5))
    vmax = float(np.nanpercentile(df[speed_col].dropna(), 95))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("plasma")

    all_segs, all_vals = [], []

    for tid in sorted(df["track_id"].unique()):
        t = df[df["track_id"] == tid].sort_values("frame")
        if len(t) < 2:
            continue
        x = t["world_x"].to_numpy()
        y = t["world_y"].to_numpy()
        speeds = t[speed_col].to_numpy()

        segs = segments_from_track(x, y)         # (N-1, 2, 2)
        vals = (speeds[:-1] + speeds[1:]) / 2.0  # midpoint speed per segment
        all_segs.append(segs)
        all_vals.append(vals)

    if all_segs:
        segs_arr = np.concatenate(all_segs, axis=0)
        vals_arr = np.concatenate(all_vals, axis=0)
        lc = LineCollection(segs_arr, cmap=cmap, norm=norm, linewidth=0.9, alpha=0.85)
        lc.set_array(vals_arr)
        ax.add_collection(lc)
        unit = "m/s" if "m_s" in speed_col else "px/s"
        plt.colorbar(lc, ax=ax, label=f"Speed ({unit})", fraction=0.03, pad=0.02)


def plot_by_dwell(ax, df: pd.DataFrame, show_labels: bool):
    """
    Draw tracks in grey; overlay stopped observations as coloured dots.
    Requires an is_stop column in the CSV (written by compute_metrics.py → stop_flags).
    Falls back to track_id colouring if is_stop is absent.
    """
    if "is_stop" not in df.columns:
        print("[WARN] is_stop column not found — falling back to color_by=track_id")
        plot_by_track_id(ax, df, show_labels)
        return

    for tid in sorted(df["track_id"].unique()):
        t = df[df["track_id"] == tid].sort_values("frame")
        ax.plot(t["world_x"], t["world_y"], linewidth=0.7, alpha=0.45, color="#aaaaaa")

    stopped = df[df["is_stop"] == 1]
    if not stopped.empty:
        ax.scatter(stopped["world_x"], stopped["world_y"],
                   s=8, c="#e63946", alpha=0.6, zorder=4, label="stop")
        ax.legend(fontsize=7, frameon=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Plan-view trajectory diagram in world coordinates")
    ap.add_argument("--traj_csv",   required=True,                    help="World-coordinate trajectory CSV")
    ap.add_argument("--out_png",    default="outputs/topdown.png",    help="Output PNG path")
    ap.add_argument("--color_by",   default="track_id",
                    choices=["track_id", "speed", "dwell"],
                    help="Colouring strategy")
    ap.add_argument("--show_labels", action="store_true",             help="Print track ID next to start point")
    ap.add_argument("--invert_y",   action="store_true",
                    help="Invert the Y axis. Toggle this if the plan view appears "
                         "mirrored top-to-bottom relative to your site. "
                         "Whether to invert depends on how world_points were defined in calib.json.")
    ap.add_argument("--figsize",    type=float, nargs=2, default=[10, 10], metavar=("W", "H"))
    ap.add_argument("--dpi",        type=int, default=150)
    args = ap.parse_args()

    # --- Load ---
    csv_path = Path(args.traj_csv)
    if not csv_path.exists():
        sys.exit(f"[ERROR] traj_csv not found: {csv_path}")

    df = pd.read_csv(csv_path)

    for col in ("world_x", "world_y", "track_id", "frame"):
        if col not in df.columns:
            sys.exit(f"[ERROR] Required column missing: '{col}'. "
                     "Run calibrate_homography.py to add world_x/world_y.")

    if df.empty:
        sys.exit("[ERROR] CSV is empty.")

    n_tracks = df["track_id"].nunique()
    print(f"[INFO] Loaded {len(df)} observations across {n_tracks} tracks")

    # --- Figure ---
    fig, ax = plt.subplots(figsize=args.figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # --- Plot ---
    if args.color_by == "track_id":
        plot_by_track_id(ax, df, show_labels=args.show_labels)

    elif args.color_by == "speed":
        speed_col = find_speed_col(df)
        if speed_col is None:
            print("[WARN] No speed column found — falling back to color_by=track_id. "
                  "Run compute_metrics.py first to add speed columns.")
            plot_by_track_id(ax, df, show_labels=args.show_labels)
        else:
            print(f"[INFO] Colouring by {speed_col}")
            plot_by_speed(ax, df, speed_col)

    elif args.color_by == "dwell":
        plot_by_dwell(ax, df, show_labels=args.show_labels)

    # --- Axes ---
    ax.set_aspect("equal")
    ax.set_xlabel("World X (m)", fontsize=9)
    ax.set_ylabel("World Y (m)", fontsize=9)
    ax.set_title(f"Top-down trajectories  |  {n_tracks} tracks  |  colour: {args.color_by}",
                 fontsize=10, pad=10)
    ax.tick_params(labelsize=7)

    # Invert Y if the plan view appears upside-down relative to the real site.
    # In most setups where world_points increase away from the camera, no inversion is needed.
    if args.invert_y:
        ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#cccccc")
    ax.grid(True, linewidth=0.3, color="#eeeeee")

    # --- Save ---
    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[INFO] Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
