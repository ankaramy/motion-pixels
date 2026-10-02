"""
directional_features.py  (Encoder V2 — Phase 2 development module)

Additive directional spatial features built on top of the FROZEN encoder
(encode_spatial_auto_v2.py, v2.1C). Reconstructs masks/grids in memory via the
frozen functions (read-only import) and computes heading-relative directional
features. Does NOT modify the encoder, datasets, or spatial_v21C outputs.

Features:
  clearance_left, clearance_right, clearance_forward   (m, walkable raycast)
  clearance_asymmetry = clearance_right - clearance_left
  obstacle_bearing_sin/cos   (nearest obstacle dir, relative to heading)
  boundary_bearing_sin/cos   (nearest boundary dir, relative to heading)
  entrance_bearing_sin/cos   (nearest entry/exit cluster dir, relative to heading)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

PRED = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRED))
import encode_spatial_auto_v2 as v2  # frozen (read-only)
v2.ENVELOPE_DILATE_M = 4.0
v2.WALKABLE_CLOSE_M = 1.5
v2.DBSCAN_EPS_M = 2.75
v2.DBSCAN_MIN_SAMPLES = 12
RES = v2.GRID_RES_M
MAX_RANGE_M = 12.0

ENC = Path(r"C:\Users\OWNER\Desktop\new_datasets\Barcelona_v1_encoded")


def wrap(a): return np.arctan2(np.sin(a), np.cos(a))


def reconstruct(rec: str) -> dict:
    """Frozen masks/grids/entries + nearest-feature index maps for a recording."""
    df = pd.read_csv(ENC / rec / "spatial_v21C" / "trajectories_encoded.csv")
    ali = {"track_id": "track_id", "frame": "frame", "world_x": "world_x", "world_y": "world_y"}
    grid = v2.build_grid(df, ali)
    masks = v2.build_walkable_and_envelope(grid["occupancy"])
    walk = masks["walkable"]
    obstacle = v2.derive_obstacles(walk, masks["envelope"])
    boundary = v2.derive_boundary(walk)
    entries = v2.derive_entry_exit(df, ali)
    obs_idx = distance_transform_edt(~obstacle, return_indices=True)[1] if obstacle.any() else None
    bnd_idx = distance_transform_edt(~boundary, return_indices=True)[1] if boundary.any() else None
    return dict(grid=grid, walk=walk, obstacle=obstacle, boundary=boundary,
                entries=entries, obs_idx=obs_idx, bnd_idx=bnd_idx)


def _to_rc(px, py, g):
    col = np.clip(((px - g["x_min"]) / RES).astype(np.int64), 0, g["W"] - 1)
    row = np.clip(((py - g["y_min"]) / RES).astype(np.int64), 0, g["H"] - 1)
    return row, col


def raycast(px, py, dx, dy, R, max_m=MAX_RANGE_M):
    """Vectorized clearance through walkable space until obstacle/boundary/edge."""
    g = R["grid"]; walk = R["walk"]; obs = R["obstacle"]; bnd = R["boundary"]
    N = len(px); K = int(max_m / RES)
    clear = np.full(N, max_m, dtype=np.float64)
    active = np.ones(N, dtype=bool)
    for k in range(1, K + 1):
        if not active.any():
            break
        sx = px + k * RES * dx; sy = py + k * RES * dy
        col = ((sx - g["x_min"]) / RES).astype(np.int64)
        row = ((sy - g["y_min"]) / RES).astype(np.int64)
        oob = (col < 0) | (col >= g["W"]) | (row < 0) | (row >= g["H"])
        idx = active & ~oob
        blocked = np.zeros(N, dtype=bool)
        rr = row[idx]; cc = col[idx]
        blocked[idx] = (~walk[rr, cc]) | obs[rr, cc] | bnd[rr, cc]
        stop = active & (oob | blocked)
        clear[stop] = k * RES
        active &= ~stop
    return clear


def nearest_bearing(px, py, heading, idx_map, R):
    """sin/cos of bearing to nearest True-cell (via return_indices) relative to heading."""
    if idx_map is None:
        return np.full(len(px), np.nan), np.full(len(px), np.nan)
    g = R["grid"]; row, col = _to_rc(px, py, g)
    nr = idx_map[0, row, col]; nc = idx_map[1, row, col]
    nwx = g["x_min"] + nc * RES; nwy = g["y_min"] + nr * RES
    rel = wrap(np.arctan2(nwy - py, nwx - px) - heading)
    return np.sin(rel), np.cos(rel)


def entrance_bearing(px, py, heading, R):
    e = R["entries"]
    if e.empty:
        return np.full(len(px), np.nan), np.full(len(px), np.nan)
    cx = e.center_x.to_numpy(); cy = e.center_y.to_numpy()
    d2 = (px[:, None] - cx[None, :]) ** 2 + (py[:, None] - cy[None, :]) ** 2
    j = d2.argmin(axis=1)
    rel = wrap(np.arctan2(cy[j] - py, cx[j] - px) - heading)
    return np.sin(rel), np.cos(rel)


def compute(px, py, heading, R):
    """All directional features for arrays of points + heading (radians).
    Heading-undefined rows (NaN heading) yield NaN features."""
    px = np.asarray(px, float); py = np.asarray(py, float); heading = np.asarray(heading, float)
    ok = np.isfinite(heading)
    out = {k: np.full(len(px), np.nan) for k in
           ["clearance_left", "clearance_right", "clearance_forward", "clearance_asymmetry",
            "obstacle_bearing_sin", "obstacle_bearing_cos",
            "boundary_bearing_sin", "boundary_bearing_cos",
            "entrance_bearing_sin", "entrance_bearing_cos"]}
    if not ok.any():
        return out
    h = heading[ok]; pxo = px[ok]; pyo = py[ok]
    fwd = (np.cos(h), np.sin(h))
    lft = (-np.sin(h), np.cos(h))       # +90 deg (CCW)
    rgt = (np.sin(h), -np.cos(h))       # -90 deg
    cl = raycast(pxo, pyo, lft[0], lft[1], R)
    cr = raycast(pxo, pyo, rgt[0], rgt[1], R)
    cf = raycast(pxo, pyo, fwd[0], fwd[1], R)
    out["clearance_left"][ok] = cl
    out["clearance_right"][ok] = cr
    out["clearance_forward"][ok] = cf
    out["clearance_asymmetry"][ok] = cr - cl
    osx, ocx = nearest_bearing(pxo, pyo, h, R["obs_idx"], R)
    bsx, bcx = nearest_bearing(pxo, pyo, h, R["bnd_idx"], R)
    esx, ecx = entrance_bearing(pxo, pyo, h, R)
    out["obstacle_bearing_sin"][ok] = osx; out["obstacle_bearing_cos"][ok] = ocx
    out["boundary_bearing_sin"][ok] = bsx; out["boundary_bearing_cos"][ok] = bcx
    out["entrance_bearing_sin"][ok] = esx; out["entrance_bearing_cos"][ok] = ecx
    return out
