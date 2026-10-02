"""
make_barcelona_v1_contact_sheets.py
-----------------------------------
Builds 5 cross-recording contact sheets from the already-produced
spatial_v21C artifacts (read-only; does not re-encode):

  1. walkable mask        -> _contact_sheets/contact_walkable.png
  2. obstacle mask        -> _contact_sheets/contact_obstacle.png
  3. boundary mask        -> _contact_sheets/contact_boundary.png
  4. distance fields       -> _contact_sheets/contact_distance_fields.png
  5. entrance points       -> _contact_sheets/contact_entrance_points.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded")
OUT = ROOT / "_contact_sheets"
OUT.mkdir(parents=True, exist_ok=True)
RECS = ["esplanade_espanya_01", "stairs_montjuic_01", "red_bridge_combined_01",
        "placa_catalunya_01", "placa_espanya_01"]


def sheet_single(png_name: str, title: str, out_name: str):
    fig, axes = plt.subplots(1, 5, figsize=(24, 6))
    for ax, rec in zip(axes, RECS):
        p = ROOT / rec / "spatial_v21C" / png_name
        if p.exists():
            ax.imshow(mpimg.imread(str(p)), cmap="gray")
        ax.set_title(rec, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / out_name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_name}")


def sheet_distance_fields():
    maps = [("distance_to_obstacle_m.png", "dist_to_obstacle"),
            ("distance_to_boundary_m.png", "dist_to_boundary"),
            ("distance_to_entrance_m.png", "dist_to_entrance")]
    fig, axes = plt.subplots(len(RECS), 3, figsize=(15, 4.2 * len(RECS)))
    for i, rec in enumerate(RECS):
        for j, (png, lbl) in enumerate(maps):
            ax = axes[i, j]
            p = ROOT / rec / "spatial_v21C" / png
            if p.exists():
                ax.imshow(mpimg.imread(str(p)))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(lbl, fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel(rec, fontsize=10)
    fig.suptitle("Distance fields (VIRIDIS: dark=near, bright=far)",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "contact_distance_fields.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  wrote contact_distance_fields.png")


def sheet_entrance_points():
    fig, axes = plt.subplots(1, 5, figsize=(24, 6))
    for ax, rec in zip(axes, RECS):
        d = ROOT / rec / "spatial_v21C"
        df = pd.read_csv(d / "trajectories_encoded.csv",
                         usecols=["world_x", "world_y"])
        ee = pd.read_csv(d / "entry_exit_points.csv")
        ax.scatter(df.world_x, df.world_y, s=0.2, c="#bdc3c7", alpha=0.35)
        if not ee.empty:
            ax.scatter(ee.center_x, ee.center_y, s=160, c="#e67e22",
                       marker="*", edgecolor="black", linewidth=0.8, zorder=3)
            for _, r in ee.iterrows():
                ax.annotate(f"#{int(r.cluster_id)}", (r.center_x, r.center_y),
                            fontsize=7, ha="left", va="bottom",
                            xytext=(3, 3), textcoords="offset points")
        ax.set_title(f"{rec}\n{len(ee)} entrance clusters", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, lw=0.2, alpha=0.4)
        ax.set_xlabel("world_x (m)", fontsize=8)
    fig.suptitle("Entrance / exit clusters (orange ★) over trajectory cloud",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "contact_entrance_points.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  wrote contact_entrance_points.png")


if __name__ == "__main__":
    sheet_single("walkable_mask.png", "Walkable masks (white = walkable)",
                 "contact_walkable.png")
    sheet_single("obstacle_mask.png", "Obstacle masks (white = obstacle)",
                 "contact_obstacle.png")
    sheet_single("boundary_mask.png", "Boundary masks (white = walkable contour)",
                 "contact_boundary.png")
    sheet_distance_fields()
    sheet_entrance_points()
    print(f"\nAll contact sheets -> {OUT}")
