"""
xc_common.py — shared MODEL_XC utilities. Builds on MODEL_XR's xr_common (magnitude/direction
metrics, rollout) and the Phase-5A macro curvature method. MODEL_X / MODEL_XR are imported
read-only; nothing in them is modified.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent                 # .../MODEL_XC
TP = HERE.parent
MODELXR = TP / "MODEL_XR"
sys.path.insert(0, str(MODELXR))
import xr_common as XC      # noqa  (also puts experiments/MODEL_X/model_x_lib on path)
L = XC.L

OBS = XC.OBS
BASE_H = XC.BASE_H
SEED = XC.SEED
RECS = XC.RECS
OPEN = ["placa_catalunya_01", "placa_espanya_01", "esplanade_espanya_01"]
CONSTR = ["stairs_montjuic_01", "red_bridge_combined_01"]
N_SEG = 10
TURN_DEG = 15.0
MIN_NET = 1.0
MIN_GT_HEAD = 15.0
OVERSHOOT_ND = 1.25
VPREV_K = 5    # smooth incoming heading over last K observed steps


def load_config(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def load_df(): return XC.load_df()
def load_split(): return XC.load_split()


def make_train_windows_xc(df, traj_ids):
    """Like model_x_lib.make_train_windows but also returns v_prev (M,2) = mean of last
    VPREV_K observed du,dv (raw) — the smoothed incoming displacement for the curvature loss."""
    X, Y = L.make_train_windows(df, traj_ids)
    du_i, dv_i = L.FEAT_COLS.index("du"), L.FEAT_COLS.index("dv")
    vprev = X[:, OBS - VPREV_K:OBS, [du_i, dv_i]].mean(axis=1)   # (M,2) raw
    return X, Y, vprev.astype(np.float32)


# ---- macro curvature (Phase 5A method) ----
def coarse_nodes(path, n_seg=N_SEG):
    P = path.shape[1]
    if P <= n_seg + 1:
        return path.copy()
    w = max(5, (P // n_seg) | 1)
    sm = XC.smooth_paths(path, w=w)
    idx = np.linspace(0, P - 1, n_seg + 1).astype(int)
    return sm[:, idx, :]


def curv_arrays(path):
    nodes = coarse_nodes(path)
    d = np.diff(nodes, axis=1)
    seg = np.linalg.norm(d, axis=2)
    h = np.arctan2(d[:, :, 1], d[:, :, 0])
    turns = np.arctan2(np.sin(np.diff(h, axis=1)), np.cos(np.diff(h, axis=1)))
    at = np.abs(np.degrees(turns))
    cum = at.sum(axis=1)
    plen = seg.sum(axis=1)
    net = np.linalg.norm(nodes[:, -1] - nodes[:, 0], axis=1)
    return {"cum_head": cum, "tort": plen / np.maximum(net, 1e-9),
            "n_turns": (at > TURN_DEG).sum(axis=1),
            "curv_density": cum / np.maximum(plen, 1e-9), "net": net}


def enrich_full(preds, gt, rec, tid, H):
    """Per-window df: magnitude+direction (xr_common.enrich) + macro curvature/shape."""
    d = XC.enrich(preds, gt, rec, tid, H)
    cp, cg = curv_arrays(preds), curv_arrays(gt)
    d["gt_cum_head"] = cg["cum_head"]; d["pred_cum_head"] = cp["cum_head"]
    d["gt_tort"] = cg["tort"]; d["pred_tort"] = cp["tort"]
    d["gt_nturns"] = cg["n_turns"]; d["pred_nturns"] = cp["n_turns"]
    d["curvy"] = (cg["net"] > MIN_NET) & (cg["cum_head"] > MIN_GT_HEAD)
    d["shape_ratio"] = np.where(d["curvy"], cp["cum_head"] / np.maximum(cg["cum_head"], 1e-9), np.nan)
    d["tort_ratio"] = np.where(d["curvy"], cp["tort"] / np.maximum(cg["tort"], 1e-9), np.nan)
    d["turn_count_ratio"] = np.where(d["curvy"], cp["n_turns"] / np.maximum(cg["n_turns"], 1e-9), np.nan)
    d["curv_density_ratio"] = np.where(d["curvy"], cp["curv_density"] / np.maximum(cg["curv_density"], 1e-9), np.nan)
    return d


def agg_full(d: pd.DataFrame) -> dict:
    base = XC.agg_block(d)
    mv = d[d.moving]; cv = d.dropna(subset=["shape_ratio"])
    def md(s): return float(np.nanmedian(s)) if len(s) else float("nan")
    base.update({
        "overshoot_rate_nd_gt125": round(100 * float(np.nanmean(mv.nd_ratio > OVERSHOOT_ND)), 2) if len(mv) else float("nan"),
        "shape_ratio_med": md(cv.shape_ratio), "tort_ratio_med": md(cv.tort_ratio),
        "turn_count_ratio_med": md(cv.turn_count_ratio), "curv_density_ratio_med": md(cv.curv_density_ratio),
        "pred_cum_head_med": md(cv.pred_cum_head), "gt_cum_head_med": md(cv.gt_cum_head),
        "pred_macro_turns_med": md(cv.pred_nturns), "n_curvy": int(len(cv)),
    })
    return base
