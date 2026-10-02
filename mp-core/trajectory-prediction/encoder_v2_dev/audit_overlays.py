"""Tasks 1,4,7 — plan-image overlays, turn-events-on-obstacles, plan-threshold compare.
READ-ONLY. Uses calib transform to warp frozen masks onto the real plan images."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEV = Path(__file__).resolve().parent
sys.path.insert(0, str(DEV))
import directional_features as DF
import encoding_truth_audit as A
v2 = DF.v2; RES = DF.RES
FIG = A.FIG; ENC = A.ENC
PHASE2 = pd.read_csv(A.PHASE2)


def load_plan(cal):
    im = cv2.imread(cal["plan"]); return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def warp_field(field, grid, cal, shp):
    Hp, Wp = shp[:2]
    yy, xx = np.mgrid[0:Hp, 0:Wp]
    wx = (xx - cal["ox"]) * cal["mpp"]; wy = (yy - cal["oy"]) * cal["mpp"]
    if cal["invert"]: wy = -wy
    col = ((wx - grid["x_min"]) / RES).astype(np.int64); row = ((wy - grid["y_min"]) / RES).astype(np.int64)
    ok = (col >= 0) & (col < grid["W"]) & (row >= 0) & (row < grid["H"])
    out = np.full((Hp, Wp), np.nan)
    out[ok] = field[row[ok], col[ok]]
    return out


def overlay_mask(ax, plan, mask_plan, color, alpha=0.5):
    ax.imshow(plan)
    ov = np.zeros((*mask_plan.shape, 4))
    ov[mask_plan] = (*color, alpha)
    ax.imshow(ov)
    ax.set_xticks([]); ax.set_yticks([])


def full_audit_placa():
    rec = "placa_catalunya_01"; cal = A.load_calib(rec); plan = load_plan(cal)
    R = DF.reconstruct(rec); g = R["grid"]
    enc = pd.read_csv(ENC / rec / "spatial_v21C" / "trajectories_encoded.csv")
    px, py = A.world_to_plan(enc.world_x.to_numpy(), enc.world_y.to_numpy(), cal)
    walkP = A.warp_mask_to_plan(R["walk"], g, cal, plan.shape)
    obsP = A.warp_mask_to_plan(R["obstacle"], g, cal, plan.shape)
    bndP = A.warp_mask_to_plan(R["boundary"], g, cal, plan.shape)
    dobs = warp_field(v2.distance_map_metres(R["obstacle"]), g, cal, plan.shape)

    f, ax = plt.subplots(2, 4, figsize=(26, 12))
    ax[0,0].imshow(plan); ax[0,0].set_title("1. source plan image"); ax[0,0].set_xticks([]); ax[0,0].set_yticks([])
    ax[0,1].imshow(plan); ax[0,1].scatter(px, py, s=0.2, c="#1f3a8a", alpha=0.25); ax[0,1].set_title("2. trajectory cloud"); ax[0,1].set_xticks([]); ax[0,1].set_yticks([])
    overlay_mask(ax[0,2], plan, walkP, (0.1,0.7,0.2)); ax[0,2].set_title("3. walkable mask")
    overlay_mask(ax[0,3], plan, obsP, (0.9,0.2,0.1)); ax[0,3].set_title("4. obstacle mask")
    overlay_mask(ax[1,0], plan, bndP, (0.1,0.1,0.1), 0.9); ax[1,0].set_title("5. boundary mask")
    ax[1,1].imshow(plan); ax[1,1].imshow(dobs, cmap="viridis", alpha=0.6); ax[1,1].set_title("6. distance_to_obstacle (m)"); ax[1,1].set_xticks([]); ax[1,1].set_yticks([])
    ax[1,2].imshow(plan)
    ov = np.zeros((*obsP.shape,4)); ov[obsP]=(0.9,0.2,0.1,0.45); ov[bndP]=(0,0,0,0.9); ax[1,2].imshow(ov)
    ax[1,2].scatter(px, py, s=0.15, c="#1f3a8a", alpha=0.2); ax[1,2].set_title("7. traj + obstacle + boundary"); ax[1,2].set_xticks([]); ax[1,2].set_yticks([])
    ax[1,3].axis("off")
    f.suptitle("Plaça Catalunya — spatial_v21C encoding overlay audit (obstacle mask = edge of walked area, NOT interior furniture)", fontweight="bold", fontsize=13)
    f.tight_layout(rect=[0,0,1,0.96]); f.savefig(FIG/"placa_catalunya_encoding_overlay_audit.png", dpi=130); plt.close(f)
    return cal, plan, R, g


def light_overlays():
    for rec in ["esplanade_espanya_01","placa_espanya_01","red_bridge_combined_01","stairs_montjuic_01"]:
        cal = A.load_calib(rec); plan = load_plan(cal); R = DF.reconstruct(rec); g = R["grid"]
        enc = pd.read_csv(ENC / rec / "spatial_v21C" / "trajectories_encoded.csv")
        px, py = A.world_to_plan(enc.world_x.to_numpy(), enc.world_y.to_numpy(), cal)
        obsP = A.warp_mask_to_plan(R["obstacle"], g, cal, plan.shape)
        f, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].imshow(plan); ax[0].set_title(f"{rec} plan"); ax[0].set_xticks([]); ax[0].set_yticks([])
        overlay_mask(ax[1], plan, obsP, (0.9,0.2,0.1)); ax[1].set_title("obstacle mask")
        ax[2].imshow(plan); ax[2].scatter(px,py,s=0.2,c="#1f3a8a",alpha=0.25)
        ov=np.zeros((*obsP.shape,4)); ov[obsP]=(0.9,0.2,0.1,0.4); ax[2].imshow(ov); ax[2].set_title("traj + obstacle"); ax[2].set_xticks([]); ax[2].set_yticks([])
        f.suptitle(f"{rec} — encoding overlay (light)", fontweight="bold"); f.tight_layout(rect=[0,0,1,0.93])
        f.savefig(FIG/f"{rec}_encoding_overlay_light.png", dpi=120); plt.close(f)


def turn_events_on_obstacles(cal, plan, R, g):
    rec = "placa_catalunya_01"
    sub = PHASE2[(PHASE2.recording_id==rec)&(PHASE2.is_moving)]
    obsP = A.warp_mask_to_plan(R["obstacle"], g, cal, plan.shape)
    f, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, c, col in zip(axes, ["left","straight","right"], ["#d73027","#777777","#2166ac"]):
        s = sub[sub.turn_class==c]
        px,py = A.world_to_plan(s.world_x.to_numpy(), s.world_y.to_numpy(), cal)
        ax.imshow(plan)
        ov=np.zeros((*obsP.shape,4)); ov[obsP]=(0.9,0.2,0.1,0.35); ax.imshow(ov)
        ax.scatter(px,py,s=1.2,c=col,alpha=0.4); ax.set_title(f"{c} turns (n={len(s)})"); ax.set_xticks([]); ax.set_yticks([])
    f.suptitle("Plaça Catalunya turn events over encoded obstacle (red). Turns occur in OPEN plaza, not at encoded obstacles.", fontweight="bold")
    f.tight_layout(rect=[0,0,1,0.94]); f.savefig(FIG/"placa_catalunya_turn_events_on_obstacles.png", dpi=130); plt.close(f)


def plan_threshold_compare(cal, plan, R, g):
    """Plan/image-derived obstacle candidate (dark regions) vs trajectory-derived mask."""
    gray = cv2.cvtColor(plan, cv2.COLOR_RGB2GRAY)
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    dark = gray < thr * 0.9                      # darker-than-pavement = trees/planting/benches/shadow
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5,5),np.uint8)).astype(bool)
    # restrict candidate to inside the walked envelope region (where it would matter)
    envP = A.warp_mask_to_plan(R["walk"] | R["obstacle"], g, cal, plan.shape)
    cand_inside = dark & envP
    obsP = A.warp_mask_to_plan(R["obstacle"], g, cal, plan.shape)
    f, ax = plt.subplots(1, 3, figsize=(20, 7))
    ax[0].imshow(plan); ov=np.zeros((*dark.shape,4)); ov[cand_inside]=(1,0.5,0,0.55); ax[0].imshow(ov)
    ax[0].set_title("plan-derived obstacle CANDIDATE (dark regions inside walked area)"); ax[0].set_xticks([]); ax[0].set_yticks([])
    overlay_mask(ax[1], plan, obsP, (0.9,0.2,0.1)); ax[1].set_title("trajectory-derived obstacle (encoder)")
    ax[2].imshow(plan)
    ov2=np.zeros((*dark.shape,4)); ov2[cand_inside]=(1,0.5,0,0.6); ov2[obsP]=(0.9,0.1,0.1,0.4); ax[2].imshow(ov2)
    ax[2].set_title("overlay: orange=plan candidate, red=encoder"); ax[2].set_xticks([]); ax[2].set_yticks([])
    f.suptitle("Plaça Catalunya — plan/image obstacles the trajectory encoder MISSED (orange inside walked area)", fontweight="bold")
    f.tight_layout(rect=[0,0,1,0.94]); f.savefig(FIG/"plan_based_obstacle_candidate.png", dpi=130); plt.close(f)
    inside_area = cand_inside.sum() * (cal["mpp"]**2)
    print(f"  plan-candidate obstacle area INSIDE walked region (placa) ~ {inside_area:.0f} m2 (encoder interior obstacle = 0 m2)")


def main():
    cal, plan, R, g = full_audit_placa()
    light_overlays()
    turn_events_on_obstacles(cal, plan, R, g)
    plan_threshold_compare(cal, plan, R, g)
    print("[done] overlays + turn events + plan comparison written")


if __name__ == "__main__":
    main()
