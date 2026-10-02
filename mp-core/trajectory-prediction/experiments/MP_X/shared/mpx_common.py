"""
MP_X shared core — turn taxonomy, labeled-window builder, turn-specific metrics,
and windowed (sliding) evaluation for Model C.

Everything here is REUSED across exp01..exp04. The frozen Model C architecture,
scalers, windowing primitives, and rollout are imported READ-ONLY from the frozen
recipe (run_bridge_ablation.py) and the Phase-5 helpers (phase5_gru_lib.py). No
frozen / Phase-4 / Phase-5 / Phase-6 / dataset file is modified.

Design choices (documented so every experiment is consistent):
  * A "window" is a 10-frame seed. Its GT future is the next `horizon` frames.
  * Turn statistics are computed from POSITIONS (reconstructed by cumulative
    du/dv for labeling, or the rollout's gt_pos/pred_pos for evaluation). The
    same `path_turn_stats` definition is used everywhere so GT and predicted
    quantities are measured identically.
  * Net heading change = |wrap(heading_last_moving - heading_first_moving)| over
    the future, where headings come from the moving (>eps) path segments.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── frozen recipe + phase-5 helpers (read-only) ──────────────────────────────
MP_ROOT = Path(r"C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels")
TP = MP_ROOT / "mp-core" / "trajectory-prediction"
FROZEN_DIR = TP / "experiments" / "schema_ablation_bridge"
PHASE5_DIR = TP / "experiments" / "phase5_v3_gru_comparison"
for p in (FROZEN_DIR, PHASE5_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import run_bridge_ablation as fz   # noqa: E402  frozen architecture + recipe
import phase5_gru_lib as P5        # noqa: E402  rollout + dataset helpers

# ── re-exported recipe constants ─────────────────────────────────────────────
FEAT_COLS = list(P5.FEAT_COLS)          # 10 frozen Model C features
TARGET_COLS = list(P5.TARGET_COLS)      # target_du, target_dv
SPATIAL_COLS = list(P5.SPATIAL_COLS)    # dist_to_obstacle_norm, dist_to_boundary_norm
ALL_FEAT_COLS = list(P5.ALL_FEAT_COLS)  # == FEAT_COLS for Model C
SPLIT_MAP = dict(P5.SPLIT_MAP)
WINDOW_SIZE = fz.WINDOW_SIZE            # 10
N_ROLLOUT = fz.N_ROLLOUT               # 20

# ── turn taxonomy thresholds (single source of truth) ────────────────────────
DISP_MIN_M = 2.0        # min GT net displacement for a genuine turn
HEAD_MIN_DEG = 30.0     # min GT net heading change for a genuine turn
MAX_STEP_M = 0.6        # artifact guard: drop windows with any larger single step
GT_TURN_DEG = 30.0      # a GT turn (for Turn Capture Rate)
PRED_TURN_DEG = 20.0    # a predicted turn (for Turn Capture Rate)
EPS = 1e-9

BIN_ORDER = ["straight", "mild_30_90", "sharp_90_150", "uturn_150_180"]
GENUINE_BINS = ["mild_30_90", "sharp_90_150", "uturn_150_180"]
BIN_WEIGHTS = {"straight": 1.0, "mild_30_90": 8.0, "sharp_90_150": 12.0, "uturn_150_180": 16.0}


# ─────────────────────────────────────────────────────────────────────────────
# Turn statistics (translation-invariant, position based)
# ─────────────────────────────────────────────────────────────────────────────
def path_turn_stats(pos: np.ndarray) -> dict:
    """Stats over a path given as (K+1, 2) positions (first row = start point).

    Returns net_disp (m), path_len (m), max_step (m), head_change_deg
    (net heading change magnitude in degrees, from first/last MOVING segment)."""
    out = {"net_disp": 0.0, "path_len": 0.0, "max_step": 0.0, "head_change_deg": 0.0}
    pos = np.asarray(pos, dtype=np.float64)
    if len(pos) < 2:
        return out
    steps = np.diff(pos, axis=0)
    seg_len = np.linalg.norm(steps, axis=1)
    out["path_len"] = float(seg_len.sum())
    out["net_disp"] = float(np.linalg.norm(pos[-1] - pos[0]))
    out["max_step"] = float(seg_len.max())
    moving = seg_len > EPS
    if moving.sum() >= 2:
        head = np.arctan2(steps[:, 1], steps[:, 0])[moving]
        out["head_change_deg"] = float(np.degrees(abs(fz.wrap_angle(head[-1] - head[0]))))
    return out


def classify_turn(net_disp: float, head_change_deg: float, max_step: float):
    """Return (bin_name, is_genuine_turn) from GT future stats."""
    genuine = (net_disp > DISP_MIN_M) and (head_change_deg > HEAD_MIN_DEG) and (max_step < MAX_STEP_M)
    if not genuine:
        return "straight", False
    if head_change_deg < 90.0:
        return "mild_30_90", True
    if head_change_deg < 150.0:
        return "sharp_90_150", True
    return "uturn_150_180", True


def positions_from_dudv(du: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """(K+1, 2) positions starting at origin from per-step du/dv (metres/step)."""
    pos = np.zeros((len(du) + 1, 2), dtype=np.float64)
    pos[1:, 0] = np.cumsum(du)
    pos[1:, 1] = np.cumsum(dv)
    return pos


# ─────────────────────────────────────────────────────────────────────────────
# Labeled window builder (same iteration order as fz.make_windows)
# ─────────────────────────────────────────────────────────────────────────────
def make_labeled_windows(df: pd.DataFrame, feat_cols, traj_ids, horizon: int,
                         target_cols=None):
    """Build (X, Y, meta) for the given trajectory ids.

    X : (N, WINDOW_SIZE, len(feat_cols))  raw (unscaled) features
    Y : (N, len(target_cols))             raw targets (default target_du/dv;
                                          exp04 passes 4 cols incl. heading)
    meta : DataFrame, one row per window, with the GT-future turn label
           (bin / is_genuine_turn / gt_net_disp_m / gt_head_change_deg / gt_max_step_m).

    Iteration order follows `traj_ids`; X aligns 1:1 with `meta` so sampler
    weights built from `meta` match the training windows.
    """
    tcols = list(TARGET_COLS if target_cols is None else target_cols)
    id_set = set(traj_ids)
    groups = {tid: g for tid, g in df[df["trajectory_id"].isin(id_set)].groupby("trajectory_id", sort=False)}
    Xs, Ys, meta = [], [], []
    for tid in traj_ids:
        g = groups[tid].sort_values("timestep")
        feat = g[feat_cols].to_numpy(np.float32)
        tgt = g[tcols].to_numpy(np.float32)
        du = g["du"].to_numpy(np.float64)
        dv = g["dv"].to_numpy(np.float64)
        rec = g["recording_id"].iloc[0]
        n = len(g)
        for i in range(n - WINDOW_SIZE):
            Xs.append(feat[i:i + WINDOW_SIZE])
            Ys.append(tgt[i + WINDOW_SIZE - 1])
            fa = i + WINDOW_SIZE
            fb = min(n, fa + horizon)
            fdu, fdv = du[fa:fb], dv[fa:fb]
            if len(fdu) >= 2:
                st = path_turn_stats(positions_from_dudv(fdu, fdv))
                b, gen = classify_turn(st["net_disp"], st["head_change_deg"], st["max_step"])
            else:
                st = {"net_disp": 0.0, "head_change_deg": 0.0, "max_step": 0.0}
                b, gen = "straight", False
            meta.append({
                "trajectory_id": tid, "recording_id": rec, "win_start": i,
                "future_len": int(len(fdu)), "gt_net_disp_m": st["net_disp"],
                "gt_head_change_deg": st["head_change_deg"], "gt_max_step_m": st["max_step"],
                "bin": b, "is_genuine_turn": gen,
            })
    X = np.asarray(Xs, dtype=np.float32)
    Y = np.asarray(Ys, dtype=np.float32)
    return X, Y, pd.DataFrame(meta)


def bin_counts(meta: pd.DataFrame) -> dict:
    c = meta["bin"].value_counts().to_dict()
    return {b: int(c.get(b, 0)) for b in BIN_ORDER}


# ─────────────────────────────────────────────────────────────────────────────
# Per-rollout turn metric row (works for any rollout result dict from P5)
# ─────────────────────────────────────────────────────────────────────────────
def rollout_metric_row(r: dict) -> dict:
    """Turn-specific metrics for one rollout result (P5.rollout_one_trajectory)."""
    gt = np.asarray(r["gt_pos"], dtype=np.float64)
    pr = np.asarray(r["pred_pos"], dtype=np.float64)
    k = min(len(gt), len(pr))
    gt, pr = gt[:k], pr[:k]
    gst = path_turn_stats(gt)
    pst = path_turn_stats(pr)
    gt_head = gst["head_change_deg"]
    pr_head = pst["head_change_deg"]
    ang_ratio = (pr_head / gt_head) if gt_head > 1e-6 else np.nan
    b, gen = classify_turn(gst["net_disp"], gt_head, gst["max_step"])
    return {
        "ade": float(r["ade"]), "fde": float(r["fde"]),
        "angular_err_deg": float(np.degrees(r["angular_error"])),
        "gt_head_change_deg": gt_head, "pred_head_change_deg": pr_head,
        "gt_path_len_m": gst["path_len"], "pred_path_len_m": pst["path_len"],
        "angularity_ratio": ang_ratio,
        "gt_net_disp_m": gst["net_disp"], "gt_max_step_m": gst["max_step"],
        "gt_turn": bool(gt_head > GT_TURN_DEG),
        "pred_turn": bool(pr_head > PRED_TURN_DEG),
        "bin": b, "is_genuine_turn": gen,
        "mean_turn_pred_rad": float(r.get("mean_turn_pred", np.nan)),
        "mean_turn_gt_rad": float(r.get("mean_turn_gt", np.nan)),
        "n_steps": int(r.get("n_steps", k - 1)),
    }


def turn_capture_rate(rows: pd.DataFrame) -> float:
    """captured GT turns / total GT turns. GT turn := gt_head>30°, captured := pred_head>20°."""
    gt_turns = rows[rows["gt_turn"]]
    if len(gt_turns) == 0:
        return float("nan")
    return float((gt_turns["pred_turn"]).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Windowed (sliding) evaluation
# ─────────────────────────────────────────────────────────────────────────────
def enumerate_full_horizon_windows(df: pd.DataFrame, splits, horizon: int) -> pd.DataFrame:
    """All windows in the given splits that have a FULL `horizon`-step GT future,
    each labeled with its GT turn bin. Cheap (no model)."""
    rows = []
    recs = [r for r in SPLIT_MAP if SPLIT_MAP[r] in splits]
    sub_df = df[df["recording_id"].isin(recs)]
    for (rec, tid), g in sub_df.groupby(["recording_id", "trajectory_id"], sort=False):
        g = g.sort_values("timestep")
        du = g["du"].to_numpy(np.float64)
        dv = g["dv"].to_numpy(np.float64)
        n = len(g)
        if True:
            last = n - WINDOW_SIZE - horizon  # inclusive max win_start with full future
            for i in range(last + 1):
                fa = i + WINDOW_SIZE
                st = path_turn_stats(positions_from_dudv(du[fa:fa + horizon], dv[fa:fa + horizon]))
                b, gen = classify_turn(st["net_disp"], st["head_change_deg"], st["max_step"])
                rows.append({"recording_id": rec, "split": SPLIT_MAP[rec], "trajectory_id": tid,
                             "win_start": i, "gt_net_disp_m": st["net_disp"],
                             "gt_head_change_deg": st["head_change_deg"],
                             "gt_max_step_m": st["max_step"], "bin": b, "is_genuine_turn": gen})
    return pd.DataFrame(rows)


def evaluate_windows(model, f_sc, t_sc, df, bounds, device, splits, horizon,
                     cap_all=4000, cap_genuine=2000, seed=42, keep_genuine_rolls=True,
                     rollout_fn=None):
    """Sliding-window rollout evaluation.

    `rollout_fn(model, traj_df, world_bnd, f_sc, t_sc, kdt, device, max_steps)` ->
    rollout result dict. Defaults to the frozen Model C rollout
    (P5.rollout_one_trajectory); exp04 passes its 4-output direction rollout.

    Rolls out:
      * a uniform random sample (<= cap_all) of ALL full-horizon windows -> the
        representative "all windows" set, and
      * genuine-turn windows (<= cap_genuine; sliding windows over the same turn
        can multiply into tens of thousands, so we subsample but always report
        the TRUE total from `cand`) -> the priority "genuine turns" set.

    Returns
    -------
    rolled : DataFrame   one row per rolled-out window (metric row + tags +
                         in_all_sample flag)
    cand   : DataFrame   every full-horizon window with its GT bin (for counts)
    genuine_rolls : list of (metric_row_dict, rollout_result_dict) for plotting
    """
    rollout_fn = P5.rollout_one_trajectory if rollout_fn is None else rollout_fn
    cand = enumerate_full_horizon_windows(df, splits, horizon)
    if len(cand) == 0:
        return pd.DataFrame(), cand, []

    rng = np.random.default_rng(seed)
    n = len(cand)
    all_sel = np.zeros(n, dtype=bool)
    sample_idx = rng.permutation(n)[:min(cap_all, n)]
    all_sel[sample_idx] = True
    gen_sel = cand["is_genuine_turn"].to_numpy().copy()
    gen_idx = np.where(gen_sel)[0]
    if cap_genuine is not None and len(gen_idx) > cap_genuine:
        drop = rng.permutation(gen_idx)[cap_genuine:]
        gen_sel[drop] = False  # subsample which genuine windows we actually roll out
    roll_sel = all_sel | gen_sel
    cand = cand.reset_index(drop=True)
    cand["in_all_sample"] = all_sel

    # KDTree per recording (spatial refresh during rollout) — built once.
    recs = [r for r in SPLIT_MAP if SPLIT_MAP[r] in splits]
    sub_df = df[df["recording_id"].isin(recs)]
    kdt = {rec: fz.build_kdt(sub_df[sub_df["recording_id"] == rec], SPATIAL_COLS) for rec in recs}
    # group each trajectory's rows ONCE (avoids re-filtering the full df per window)
    traj_groups = {tid: g.sort_values("timestep").reset_index(drop=True)
                   for tid, g in sub_df.groupby("trajectory_id", sort=False)}

    rolled, genuine_rolls = [], []
    sub = cand[roll_sel]
    for (rec, tid), grp in sub.groupby(["recording_id", "trajectory_id"], sort=False):
        wb = bounds[rec]
        tdf = traj_groups[tid]
        for _, c in grp.iterrows():
            i = int(c["win_start"])
            seg = tdf.iloc[i:].reset_index(drop=True)
            r = rollout_fn(model, seg, wb, f_sc, t_sc, kdt[rec], device, max_steps=horizon)
            if r is None:
                continue
            row = rollout_metric_row(r)
            row.update({"recording_id": rec, "trajectory_id": tid, "win_start": i,
                        "split": SPLIT_MAP[rec], "in_all_sample": bool(c["in_all_sample"]),
                        "is_genuine_turn": bool(c["is_genuine_turn"])})
            rolled.append(row)
            if keep_genuine_rolls and bool(c["is_genuine_turn"]):
                genuine_rolls.append((row, r))
    return pd.DataFrame(rolled), cand, genuine_rolls


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────
_METRIC_COLS = ["ade", "fde", "angular_err_deg", "gt_head_change_deg",
                "pred_head_change_deg", "gt_path_len_m", "pred_path_len_m",
                "angularity_ratio"]


def aggregate(rows: pd.DataFrame, label: str) -> dict:
    if rows is None or len(rows) == 0:
        d = {"subset": label, "n": 0}
        d.update({f"{c}_mean": float("nan") for c in _METRIC_COLS})
        d["turn_capture_rate"] = float("nan")
        return d
    d = {"subset": label, "n": int(len(rows))}
    for c in _METRIC_COLS:
        d[f"{c}_mean"] = float(np.nanmean(rows[c].to_numpy(dtype=np.float64)))
    d["turn_capture_rate"] = turn_capture_rate(rows)
    return d
