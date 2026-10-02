"""
xr_common.py — shared MODEL_XR utilities: config, data, scalers, rollout-based metric
enrichment (jitter-aware magnitude + direction), all built on the frozen MODEL_X library
(experiments/MODEL_X/model_x_lib.py). MODEL_X files are imported read-only, never modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent                    # .../MODEL_XR
TP = HERE.parent                                          # .../trajectory-prediction
MODELX = TP / "experiments" / "MODEL_X"
sys.path.insert(0, str(MODELX))
import model_x_lib as L  # noqa: E402

CSV = L.CONFIG["dataset"]["csv_path"]
SPLIT = MODELX / "splits" / "model_x_track_split.csv"
OBS = L.WINDOW_SIZE
BASE_H = L.HORIZON          # 10 — MODEL_X base horizon (control reproduction target)
SEED = L.SEED
RECS = ["placa_catalunya_01", "placa_espanya_01", "esplanade_espanya_01",
        "stairs_montjuic_01", "red_bridge_combined_01"]
SMOOTH_W = 5
NDISP_EPS = 0.2
STEP_EPS = 1e-3
PRED_DIR_EPS = 0.05

DATA_COLS = ["recording_id", "trajectory_id", "timestep", "world_x", "world_y"] + L.FEAT_COLS + L.TARGET_COLS


def load_config(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_df():
    return pd.read_csv(CSV, usecols=lambda c: c in set(DATA_COLS))


def load_split():
    return pd.read_csv(SPLIT)


def smooth_paths(a, w=SMOOTH_W):
    M, P, D = a.shape
    if P < 3:
        return a.copy()
    if w % 2 == 0:
        w += 1
    w = min(w, P if P % 2 == 1 else P - 1)
    pad = w // 2
    ap = np.concatenate([np.repeat(a[:, :1], pad, 1), a, np.repeat(a[:, -1:], pad, 1)], axis=1)
    cs = np.cumsum(ap, axis=1)
    cs = np.concatenate([np.zeros((M, 1, D)), cs], axis=1)
    return (cs[:, w:] - cs[:, :-w]) / w


def rollout_all(model, df, te_ids, wb, f_sc, t_sc, H, device, chunk=16384):
    seeds, start, gt, wba, meta = L.build_eval_windows(df, te_ids, wb, n_steps=H, stride=1)
    M = len(seeds)
    preds = np.empty((M, H + 1, 2))
    for i in range(0, M, chunk):
        sl = slice(i, i + chunk)
        preds[sl] = L.rollout_world_batch(model, seeds[sl], start[sl],
                                          {k: v[sl] for k, v in wba.items()}, f_sc, t_sc, H, device)
    rec = np.array([m["recording_id"] for m in meta])
    tid = np.array([m["trajectory_id"] for m in meta])
    return preds, gt, rec, tid


def enrich(preds, gt, rec, tid, H):
    """Per-window metric DataFrame (jitter-aware magnitude + direction)."""
    disp = np.linalg.norm(preds - gt, axis=2)
    ade = disp[:, 1:].mean(axis=1); fde = disp[:, -1]
    cum_gt = np.linalg.norm(np.diff(gt, axis=1), axis=2).sum(axis=1)
    cum_pred = np.linalg.norm(np.diff(preds, axis=1), axis=2).sum(axis=1)
    nd_gt = np.linalg.norm(gt[:, -1] - gt[:, 0], axis=1)
    nd_pred = np.linalg.norm(preds[:, -1] - preds[:, 0], axis=1)
    gt_sm, pr_sm = smooth_paths(gt), smooth_paths(preds)
    sm_gt = np.linalg.norm(np.diff(gt_sm, axis=1), axis=2).sum(axis=1)
    sm_pred = np.linalg.norm(np.diff(pr_sm, axis=1), axis=2).sum(axis=1)
    # t=1 step
    s1_gt = np.linalg.norm(gt[:, 1] - gt[:, 0], axis=1)
    s1_pred = np.linalg.norm(preds[:, 1] - preds[:, 0], axis=1)
    s1sm_gt = np.linalg.norm(gt_sm[:, 1] - gt_sm[:, 0], axis=1)
    s1sm_pred = np.linalg.norm(pr_sm[:, 1] - pr_sm[:, 0], axis=1)
    # direction (net-disp vectors)
    v_gt = gt[:, -1] - gt[:, 0]; v_pr = preds[:, -1] - preds[:, 0]
    cos = np.where((nd_gt > NDISP_EPS) & (nd_pred > PRED_DIR_EPS),
                   (v_gt * v_pr).sum(1) / np.maximum(nd_gt * nd_pred, 1e-9), np.nan)
    head = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    # t=1 cosine
    a1g = gt[:, 1] - gt[:, 0]; a1p = preds[:, 1] - preds[:, 0]
    n1g = np.linalg.norm(a1g, axis=1)
    n1p = np.linalg.norm(a1p, axis=1)
    cos1 = np.where((n1g > STEP_EPS) & (n1p > 1e-6),
                    (a1g * a1p).sum(1) / np.maximum(n1g * n1p, 1e-9), np.nan)
    return pd.DataFrame({
        "recording": rec, "trajectory_id": tid, "horizon": H,
        "ade": ade, "fde": fde,
        "cum_gt": cum_gt, "cum_pred": cum_pred,
        "cum_ratio": cum_pred / np.maximum(cum_gt, 1e-9),
        "nd_gt": nd_gt, "nd_pred": nd_pred,
        "nd_ratio": np.where(nd_gt > NDISP_EPS, nd_pred / np.maximum(nd_gt, 1e-9), np.nan),
        "sm_gt": sm_gt, "sm_pred": sm_pred,
        "sm_ratio": np.where(sm_gt > 1e-9, sm_pred / np.maximum(sm_gt, 1e-9), np.nan),
        "step1_ratio": np.where(s1_gt > STEP_EPS, s1_pred / np.maximum(s1_gt, 1e-9), np.nan),
        "step1_sm_ratio": np.where(s1sm_gt > STEP_EPS, s1sm_pred / np.maximum(s1sm_gt, 1e-9), np.nan),
        "cosine": cos, "cosine_t1": cos1, "heading_err_deg": head,
        "moving": nd_gt > NDISP_EPS,
    })


def agg_block(d: pd.DataFrame) -> dict:
    mv = d[d.moving]; cv = d.dropna(subset=["cosine"])
    def md(s): return float(np.nanmedian(s)) if len(s) else float("nan")
    def mn(s): return float(np.nanmean(s)) if len(s) else float("nan")
    return {
        "n_windows": int(len(d)),
        "ADE": mn(d.ade), "ADE_med": md(d.ade), "FDE": mn(d.fde), "FDE_med": md(d.fde),
        "step1_ratio_med": md(d.step1_ratio), "step1_sm_ratio_med": md(d.step1_sm_ratio),
        "cum_ratio_med": md(mv.cum_ratio), "nd_ratio_med": md(mv.nd_ratio),
        "sm_ratio_med": md(mv.sm_ratio),
        "pct_nd_lt05": round(100 * float(np.nanmean(mv.nd_ratio < 0.5)), 2) if len(mv) else float("nan"),
        "cosine_med": md(cv.cosine), "cosine_t1_med": md(d.cosine_t1),
        "heading_err_med": md(cv.heading_err_deg),
        "pct_cos_gt075": round(100 * float(np.nanmean(cv.cosine > 0.75)), 2) if len(cv) else float("nan"),
        "pct_cos_gt050": round(100 * float(np.nanmean(cv.cosine > 0.50)), 2) if len(cv) else float("nan"),
    }
