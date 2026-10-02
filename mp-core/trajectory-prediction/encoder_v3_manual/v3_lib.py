"""
Encoder V3 (manual architectural masks) -- core library.

Computes spatial features for each trajectory point relative to MANUALLY
ANNOTATED architectural masks (walkable / obstacle), instead of the old
trajectory-occupancy masks.

This module is ADDITIVE and read-only with respect to the old pipeline:
it never imports, modifies, or runs encode_spatial_auto_v2.py, the old encoded
CSVs, the master dataset, or any model.

Coordinate alignment (validated, see Encoder_V3_Manual_Report.md):
    calib.json method = "plan_scale_plus_correspondences".
    world <-> plan-pixel is an exact isotropic similarity. We fit it by least
    squares from the world_points <-> plan_points_px correspondences and verify
    the RMS residual (~0 px) and that the fitted scale == 1/meters_per_plan_pixel.
    Therefore plan-pixel distance * meters_per_plan_pixel = metres (constant scale).
"""

import os
import json
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# Large per-recording data (tracking CSVs, encoded outputs) live outside Git.
# Set MP_DATA_ROOT to that folder; see docs/DATA_AVAILABILITY.md.
DATASETS = os.environ.get("MP_DATA_ROOT", os.path.join(REPO_ROOT, "mp-data", "external"))
RECORDINGS_ROOT = os.path.join(REPO_ROOT, "mp-data", "recordings")   # calib.json + plans (in Git)
MASK_ROOT = os.path.join(REPO_ROOT, "mp-data", "annotations", "manual_masks_v3")
OUT_ROOT = os.path.join(DATASETS, "Barcelona_v3_manual_encoded")
OLD_ENCODED = os.path.join(DATASETS, "Barcelona_v1_encoded")

VALIDATED = [
    "stairs_montjuic_01",
    "red_bridge_combined_01",
    "esplanade_espanya_01",
    "placa_espanya_01",
    "placa_catalunya_01",
]

# ray-march parameters for directional clearance
CLEAR_STEP_M = 0.25
CLEAR_MAX_M = 25.0
EPS = 1e-6


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _calib_path(recording_id):
    """Prefer the calib.json committed under mp-data/recordings/, else MP_DATA_ROOT."""
    repo_calib = os.path.join(RECORDINGS_ROOT, recording_id, "calibration", "calib.json")
    if os.path.exists(repo_calib):
        return repo_calib
    return os.path.join(DATASETS, recording_id, "calibration", "calib.json")


def resolve_plan_image(calib, recording_id):
    """calib['plan_image'] may be absolute (original machine) or relative to the calib file."""
    p = calib.get("plan_image") or ""
    if os.path.isabs(p) and os.path.exists(p):
        return p
    cands = [os.path.join(os.path.dirname(_calib_path(recording_id)), p)] if p else []
    name = os.path.basename(p.replace("\\", "/"))
    cands += [os.path.join(RECORDINGS_ROOT, recording_id, "plan", name),
              os.path.join(DATASETS, recording_id, "plan", name)]
    for c in cands:
        if c and os.path.exists(c):
            return os.path.normpath(c)
    return p


def paths_for(recording_id):
    base = os.path.join(DATASETS, recording_id)
    mdir = os.path.join(MASK_ROOT, recording_id)
    return {
        "csv": os.path.join(base, "filtered_250m", "trajectories_world_filtered_250m.csv"),
        "calib": _calib_path(recording_id),
        "walkable": os.path.join(mdir, "walkable_mask_v3_manual.png"),
        "obstacle": os.path.join(mdir, "obstacle_mask_v3_manual.png"),
        "old_csv": os.path.join(OLD_ENCODED, recording_id, "spatial_v21C",
                                "trajectories_encoded.csv"),
        "outdir": os.path.join(OUT_ROOT, recording_id, "spatial_v3_manual"),
    }


def fit_transform(calib):
    """Fit world<->plan-pixel affine from correspondences; verify it is a clean
    similarity consistent with meters_per_plan_pixel.

    Returns dict with M (world->plan, 3x2), Minv (plan->world, 3x2), residuals,
    scale, and an 'ambiguous' flag."""
    W = np.array(calib["world_points"], float)
    P = np.array(calib["plan_points_px"], float)
    mpp = float(calib["meters_per_plan_pixel"])

    Aw = np.c_[W, np.ones(len(W))]
    M, *_ = np.linalg.lstsq(Aw, P, rcond=None)        # [wx,wy,1] @ M = [px,py]
    pred_p = Aw @ M
    rms_w2p = float(np.sqrt(((pred_p - P) ** 2).sum(1)).mean())

    Ap = np.c_[P, np.ones(len(P))]
    Minv, *_ = np.linalg.lstsq(Ap, W, rcond=None)     # [px,py,1] @ Minv = [wx,wy]
    pred_w = Ap @ Minv
    rms_p2w = float(np.sqrt(((pred_w - W) ** 2).sum(1)).mean())

    lin = M[:2, :].T                                  # 2x2 world->plan linear
    sv = np.linalg.svd(lin, compute_uv=False)
    scale_px_per_m = float(sv.mean())
    anisotropy = float(sv.max() / max(sv.min(), EPS))
    scale_consistency = float(abs(scale_px_per_m - 1.0 / mpp) / (1.0 / mpp))

    ambiguous = rms_w2p > 5.0 or anisotropy > 1.05 or scale_consistency > 0.05
    return {
        "M": M, "Minv": Minv,
        "rms_world_to_plan_px": round(rms_w2p, 4),
        "rms_plan_to_world_px": round(rms_p2w, 4),
        "scale_px_per_m": round(scale_px_per_m, 5),
        "meters_per_plan_pixel": mpp,
        "metres_per_pixel_from_fit": round(1.0 / scale_px_per_m, 6),
        "anisotropy": round(anisotropy, 5),
        "scale_consistency_rel_err": round(scale_consistency, 6),
        "n_correspondences": len(W),
        "ambiguous": bool(ambiguous),
    }


def world_to_plan(wx, wy, M):
    P = np.c_[wx, wy, np.ones(len(wx))] @ M
    return P[:, 0], P[:, 1]


def plan_to_world(px, py, Minv):
    Wp = np.c_[px, py, np.ones(len(px))] @ Minv
    return Wp[:, 0], Wp[:, 1]


# ---------------------------------------------------------------------------
# Heading
# ---------------------------------------------------------------------------
def compute_heading(df):
    """Per-track heading from consecutive world positions (radians). Forward/
    back-fill within track; stationary tracks flagged invalid (heading=0)."""
    s = df.sort_values(["track_id", "frame"])
    dx = s.groupby("track_id")["world_x"].diff()
    dy = s.groupby("track_id")["world_y"].diff()
    moved = (dx.abs() > EPS) | (dy.abs() > EPS)
    head = np.arctan2(dy, dx)
    head = head.where(moved)
    s = s.assign(_h=head)
    s["_h"] = s.groupby("track_id")["_h"].ffill()
    s["_h"] = s.groupby("track_id")["_h"].bfill()
    valid = s["_h"].notna()
    s["_h"] = s["_h"].fillna(0.0)
    heading = s["_h"].reindex(df.index).to_numpy()
    valid = valid.reindex(df.index).fillna(False).to_numpy()
    return heading, valid


# ---------------------------------------------------------------------------
# Field precompute (full-image EDT with nearest-pixel indices)
# ---------------------------------------------------------------------------
def precompute_fields(walk_bool, obst_bool):
    # distance (px) + nearest-pixel index to nearest OBSTACLE pixel
    obst_edt, obst_idx = ndimage.distance_transform_edt(~obst_bool, return_indices=True)
    # walkable boundary = morphological gradient of walkable
    k = np.ones((3, 3), np.uint8)
    dil = ndimage.binary_dilation(walk_bool, k)
    ero = ndimage.binary_erosion(walk_bool, k)
    boundary = dil & ~ero
    if not boundary.any():
        boundary = dil & ~walk_bool  # fallback
    bnd_edt, bnd_idx = ndimage.distance_transform_edt(~boundary, return_indices=True)
    return {
        "obst_edt": obst_edt, "obst_idx": obst_idx,
        "bnd_edt": bnd_edt, "bnd_idx": bnd_idx,
        "boundary": boundary,
    }


# ---------------------------------------------------------------------------
# Directional clearance (vectorized world-space ray march)
# ---------------------------------------------------------------------------
def clearance(wx, wy, ang, walk_bool, obst_bool, M, H, W,
              step=CLEAR_STEP_M, maxd=CLEAR_MAX_M):
    N = len(wx)
    nsteps = int(round(maxd / step))
    clr = np.full(N, maxd, float)
    active = np.ones(N, bool)
    cdx, cdy = np.cos(ang), np.sin(ang)
    for kk in range(1, nsteps + 1):
        d = kk * step
        sx = wx + cdx * d
        sy = wy + cdy * d
        ix, iy = world_to_plan(sx, sy, M)
        ix = np.round(ix).astype(int)
        iy = np.round(iy).astype(int)
        oob = (ix < 0) | (ix >= W) | (iy < 0) | (iy >= H)
        hit = oob.copy()
        ok = ~oob
        if ok.any():
            okx, oky = ix[ok], iy[ok]
            cell_hit = obst_bool[oky, okx] | (~walk_bool[oky, okx])
            tmp = np.zeros(N, bool)
            tmp[ok] = cell_hit
            hit |= tmp
        newly = active & hit
        clr[newly] = d
        active &= ~hit
        if not active.any():
            break
    return clr


# ---------------------------------------------------------------------------
# Main encode
# ---------------------------------------------------------------------------
def encode_recording(recording_id, verbose=True):
    if recording_id not in VALIDATED:
        raise ValueError(f"'{recording_id}' is not a validated recording. "
                         f"Allowed: {VALIDATED}")
    P = paths_for(recording_id)
    for key in ("csv", "calib", "walkable", "obstacle"):
        if not os.path.exists(P[key]):
            raise FileNotFoundError(f"[{recording_id}] missing {key}: {P[key]}")

    calib = json.load(open(P["calib"]))
    tf = fit_transform(calib)
    M, Minv = tf["M"], tf["Minv"]
    mpp = tf["meters_per_plan_pixel"]

    walk = cv2.imread(P["walkable"], cv2.IMREAD_GRAYSCALE) > 127
    obst = cv2.imread(P["obstacle"], cv2.IMREAD_GRAYSCALE) > 127
    H, W = walk.shape
    if obst.shape != walk.shape:
        raise ValueError(f"[{recording_id}] mask shape mismatch "
                         f"walk{walk.shape} obst{obst.shape}")

    df = pd.read_csv(P["csv"])
    n = len(df)
    wx = df["world_x"].to_numpy(float)
    wy = df["world_y"].to_numpy(float)

    if verbose:
        print(f"[{recording_id}] rows={n} tracks={df['track_id'].nunique()} "
              f"mask={W}x{H} rms={tf['rms_world_to_plan_px']}px mpp={mpp:.5f}")
    if tf["ambiguous"]:
        print(f"[{recording_id}] !! WARNING: transform flagged AMBIGUOUS "
              f"(rms={tf['rms_world_to_plan_px']} aniso={tf['anisotropy']} "
              f"scale_err={tf['scale_consistency_rel_err']})")

    # map to plan pixels
    px_f, py_f = world_to_plan(wx, wy, M)
    px = np.round(px_f).astype(int)
    py = np.round(py_f).astype(int)
    inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    pxc = np.clip(px, 0, W - 1)
    pyc = np.clip(py, 0, H - 1)

    fields = precompute_fields(walk, obst)

    # --- basic mask features ---
    inside_w = np.zeros(n, np.int8)
    inside_o = np.zeros(n, np.int8)
    inside_w[inb] = walk[pyc[inb], pxc[inb]].astype(np.int8)
    inside_o[inb] = obst[pyc[inb], pxc[inb]].astype(np.int8)

    dist_obst = np.full(n, np.nan)
    dist_bnd = np.full(n, np.nan)
    dist_obst[inb] = fields["obst_edt"][pyc[inb], pxc[inb]] * mpp
    dist_bnd[inb] = fields["bnd_edt"][pyc[inb], pxc[inb]] * mpp

    # nearest obstacle / boundary pixel (row=idx[0], col=idx[1])
    no_y = fields["obst_idx"][0][pyc, pxc]
    no_x = fields["obst_idx"][1][pyc, pxc]
    nb_y = fields["bnd_idx"][0][pyc, pxc]
    nb_x = fields["bnd_idx"][1][pyc, pxc]
    nearest_obst_px_x = np.where(inb, no_x, np.nan)
    nearest_obst_px_y = np.where(inb, no_y, np.nan)
    nearest_bnd_px_x = np.where(inb, nb_x, np.nan)
    nearest_bnd_px_y = np.where(inb, nb_y, np.nan)

    # --- heading ---
    heading, hvalid = compute_heading(df)

    # --- bearings (in world frame, relative to heading) ---
    no_wx, no_wy = plan_to_world(no_x.astype(float), no_y.astype(float), Minv)
    nb_wx, nb_wy = plan_to_world(nb_x.astype(float), nb_y.astype(float), Minv)
    obst_phi = np.arctan2(no_wy - wy, no_wx - wx)
    bnd_phi = np.arctan2(nb_wy - wy, nb_wx - wx)
    ob = obst_phi - heading
    bb = bnd_phi - heading
    valid_b = inb & hvalid
    obstacle_bearing_sin = np.where(valid_b, np.sin(ob), 0.0)
    obstacle_bearing_cos = np.where(valid_b, np.cos(ob), 0.0)
    boundary_bearing_sin = np.where(valid_b, np.sin(bb), 0.0)
    boundary_bearing_cos = np.where(valid_b, np.cos(bb), 0.0)

    # --- directional clearances ---
    clr_f = clearance(wx, wy, heading, walk, obst, M, H, W)
    clr_l = clearance(wx, wy, heading + np.pi / 2, walk, obst, M, H, W)
    clr_r = clearance(wx, wy, heading - np.pi / 2, walk, obst, M, H, W)
    # invalidate where heading undefined or start point OOB
    bad = (~hvalid) | (~inb)
    clr_f[bad] = np.nan
    clr_l[bad] = np.nan
    clr_r[bad] = np.nan
    asym = (clr_r - clr_l) / (clr_r + clr_l + EPS)

    # --- assemble (preserve existing columns) ---
    out = df.copy()
    out["dist_to_obstacle_v3_m"] = dist_obst
    out["dist_to_walkable_boundary_v3_m"] = dist_bnd
    out["inside_walkable_v3"] = inside_w
    out["inside_obstacle_v3"] = inside_o
    out["clearance_forward_v3_m"] = clr_f
    out["clearance_left_v3_m"] = clr_l
    out["clearance_right_v3_m"] = clr_r
    out["clearance_asymmetry_v3"] = asym
    out["obstacle_bearing_sin_v3"] = obstacle_bearing_sin
    out["obstacle_bearing_cos_v3"] = obstacle_bearing_cos
    out["boundary_bearing_sin_v3"] = boundary_bearing_sin
    out["boundary_bearing_cos_v3"] = boundary_bearing_cos
    out["nearest_obstacle_px_x"] = nearest_obst_px_x
    out["nearest_obstacle_px_y"] = nearest_obst_px_y
    out["nearest_boundary_px_x"] = nearest_bnd_px_x
    out["nearest_boundary_px_y"] = nearest_bnd_px_y
    out["plan_px_x_v3"] = px
    out["plan_px_y_v3"] = py
    out["inbounds_v3"] = inb.astype(np.int8)
    out["heading_v3_rad"] = heading
    out["heading_valid_v3"] = hvalid.astype(np.int8)

    # --- write outputs ---
    os.makedirs(P["outdir"], exist_ok=True)
    csv_out = os.path.join(P["outdir"], "trajectories_encoded_v3.csv")
    out.to_csv(csv_out, index=False)

    diag = build_diagnostics(recording_id, out, tf, inb, inside_w, inside_o,
                             dist_obst, dist_bnd, clr_f, clr_l, clr_r, hvalid, P)
    with open(os.path.join(P["outdir"], "encoding_v3_metadata.json"), "w") as f:
        json.dump(metadata(recording_id, calib, tf, walk, obst, df, P), f, indent=2)
    with open(os.path.join(P["outdir"], "feature_diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)
    with open(os.path.join(P["outdir"], "feature_diagnostics.md"), "w") as f:
        f.write(diagnostics_md(diag))
    make_overlay(recording_id, calib, walk, obst, px, py, inside_w, inb,
                 os.path.join(P["outdir"], "mask_alignment_overlay.png"))

    if verbose:
        print(f"[{recording_id}] STATUS={diag['status']}  "
              f"inWalk={diag['percent_inside_walkable']}%  "
              f"inObst={diag['percent_inside_obstacle']}%  "
              f"-> {csv_out}")
    return diag


# ---------------------------------------------------------------------------
def _stats(a):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"min": None, "mean": None, "max": None}
    return {"min": round(float(a.min()), 4),
            "mean": round(float(a.mean()), 4),
            "max": round(float(a.max()), 4)}


def build_diagnostics(rid, out, tf, inb, iw, io, dist_o, dist_b,
                      cf, cl, cr, hvalid, P):
    n = len(out)
    tracks = int(out["track_id"].nunique())
    inb_pct = round(100 * inb.mean(), 3)
    inb_n = max(int(inb.sum()), 1)
    pct_walk = round(100 * (iw == 1).sum() / n, 3)
    pct_obst = round(100 * (io == 1).sum() / n, 3)
    pct_walk_inb = round(100 * (iw == 1).sum() / inb_n, 3)
    nan_dist = round(100 * (~np.isfinite(dist_o)).mean(), 3)
    nan_clear = round(100 * (~np.isfinite(cf)).mean(), 3)

    warnings = []
    if tf["ambiguous"]:
        warnings.append("Coordinate transform flagged AMBIGUOUS - inspect calib.")
    if inb_pct < 90:
        warnings.append(f"{round(100-inb_pct,1)}% of points map OUTSIDE the mask "
                        f"image (plan crop does not cover full trajectory extent).")
    if pct_obst > 8:
        warnings.append(f"{pct_obst}% of points fall INSIDE obstacle mask "
                        f"(check mask edges near walkways / trajectory noise).")
    if pct_walk_inb < 85:
        warnings.append(f"Only {pct_walk_inb}% of in-bounds points are inside the "
                        f"walkable mask.")
    if not hvalid.all():
        warnings.append(f"{round(100*(~hvalid).mean(),2)}% rows had undefined "
                        f"heading (stationary); directional features NaN there.")

    # status
    if tf["ambiguous"] or pct_walk_inb < 50:
        status = "FAIL"
    elif inb_pct < 90 or pct_obst > 8 or pct_walk_inb < 85:
        status = "CHECK"
    else:
        status = "PASS"

    return {
        "recording_id": rid,
        "status": status,
        "row_count": n,
        "track_count": tracks,
        "percent_inbounds": inb_pct,
        "percent_inside_walkable": pct_walk,
        "percent_inside_walkable_of_inbounds": pct_walk_inb,
        "percent_inside_obstacle": pct_obst,
        "percent_invalid_dist_nan": nan_dist,
        "percent_invalid_clearance_nan": nan_clear,
        "dist_to_obstacle_v3_m": _stats(dist_o),
        "dist_to_walkable_boundary_v3_m": _stats(dist_b),
        "clearance_forward_v3_m": _stats(cf),
        "clearance_left_v3_m": _stats(cl),
        "clearance_right_v3_m": _stats(cr),
        "mean_metres_per_pixel": round(tf["meters_per_plan_pixel"], 6),
        "transform_rms_px": tf["rms_world_to_plan_px"],
        "transform_anisotropy": tf["anisotropy"],
        "warnings": warnings,
        "output_csv": os.path.join(P["outdir"], "trajectories_encoded_v3.csv"),
    }


def metadata(rid, calib, tf, walk, obst, df, P):
    return {
        "recording_id": rid,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "encoder": "v3_manual",
        "source_csv": P["csv"],
        "calib_json": P["calib"],
        "walkable_mask": P["walkable"],
        "obstacle_mask": P["obstacle"],
        "plan_image": calib.get("plan_image"),
        "mask_width": int(walk.shape[1]),
        "mask_height": int(walk.shape[0]),
        "invert_plan_y": calib.get("invert_plan_y"),
        "meters_per_plan_pixel": tf["meters_per_plan_pixel"],
        "scale_px_per_m_from_fit": tf["scale_px_per_m"],
        "metres_per_pixel_from_fit": tf["metres_per_pixel_from_fit"],
        "transform_world_to_plan_3x2": tf["M"].tolist(),
        "transform_plan_to_world_3x2": tf["Minv"].tolist(),
        "rms_world_to_plan_px": tf["rms_world_to_plan_px"],
        "rms_plan_to_world_px": tf["rms_plan_to_world_px"],
        "anisotropy": tf["anisotropy"],
        "scale_consistency_rel_err": tf["scale_consistency_rel_err"],
        "transform_ambiguous": tf["ambiguous"],
        "clearance_step_m": CLEAR_STEP_M,
        "clearance_max_m": CLEAR_MAX_M,
        "row_count": int(len(df)),
        "track_count": int(df["track_id"].nunique()),
        "left_right_convention": "left = heading + 90deg (CCW in world frame), "
                                 "right = heading - 90deg; clearance_asymmetry = "
                                 "(right-left)/(right+left+eps).",
    }


def diagnostics_md(d):
    L = [f"# Feature Diagnostics - {d['recording_id']} (Encoder V3 manual)", "",
         f"**Status: {d['status']}**", "",
         f"- Rows: {d['row_count']}",
         f"- Tracks: {d['track_count']}",
         f"- In-bounds (mapped inside mask image): {d['percent_inbounds']}%",
         f"- Inside walkable: {d['percent_inside_walkable']}% "
         f"({d['percent_inside_walkable_of_inbounds']}% of in-bounds)",
         f"- Inside obstacle: {d['percent_inside_obstacle']}%",
         f"- Invalid/NaN distance rows: {d['percent_invalid_dist_nan']}%",
         f"- Invalid/NaN clearance rows: {d['percent_invalid_clearance_nan']}%",
         f"- Mean metres/pixel: {d['mean_metres_per_pixel']}",
         f"- Transform RMS: {d['transform_rms_px']} px (anisotropy {d['transform_anisotropy']})",
         "", "## Feature ranges (finite values)", "",
         "| Feature | min | mean | max |", "|---|---|---|---|"]
    for key in ["dist_to_obstacle_v3_m", "dist_to_walkable_boundary_v3_m",
                "clearance_forward_v3_m", "clearance_left_v3_m", "clearance_right_v3_m"]:
        s = d[key]
        L.append(f"| {key} | {s['min']} | {s['mean']} | {s['max']} |")
    L += ["", "## Warnings", ""]
    L += [f"- {w}" for w in d["warnings"]] or ["- none"]
    return "\n".join(L) + "\n"


def make_overlay(rid, calib, walk, obst, px, py, inside_w, inb, out_path,
                 max_points=3000):
    img = cv2.imread(resolve_plan_image(calib, rid))
    if img is None or img.shape[:2] != walk.shape:
        img = cv2.cvtColor((walk * 0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    ov = img.copy()
    color = np.zeros_like(img)
    color[walk] = (0, 180, 0)
    color[obst] = (0, 0, 200)
    ov = cv2.addWeighted(img, 0.6, color, 0.4, 0)
    # sample trajectory points
    idx = np.where(inb)[0]
    if idx.size > max_points:
        idx = idx[np.linspace(0, idx.size - 1, max_points).astype(int)]
    for i in idx:
        c = (0, 255, 255) if inside_w[i] == 1 else (255, 0, 255)
        cv2.circle(ov, (int(px[i]), int(py[i])), 2, c, -1)
    cv2.rectangle(ov, (0, 0), (ov.shape[1], 28), (30, 30, 30), -1)
    cv2.putText(ov, f"{rid}  green=walkable red=obstacle  "
                    f"yellow=pt-in-walkable magenta=pt-not-in-walkable",
                (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, ov)
