"""
model_x_lib.py — shared library for MODEL_X.

MODEL_X = general-purpose Horizon-10 retraining of Model C for best OVERALL
trajectory prediction on the available validated Barcelona dataset (5 recordings,
Barcelona_v3_manual_master). NOT a new architecture.

The TrajectoryLSTM class and ColumnScaler are faithful copies of the frozen Model C
implementation in experiments/schema_ablation_bridge/run_bridge_ablation.py — that
source file is NOT edited. Everything MODEL_X needs lives here so the phase is isolated.
"""
from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── load config ──────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config" / "model_x_config.json").read_text(encoding="utf-8"))

# Large datasets are not in Git (docs/DATA_AVAILABILITY.md). Relative dataset paths in the
# config resolve against MP_DATA_ROOT (default: <repo>/mp-data/external).
REPO_ROOT = HERE.parents[3]
DATA_ROOT = Path(os.environ.get("MP_DATA_ROOT", REPO_ROOT / "mp-data" / "external"))
for _k in ("csv_path", "model_c_csv_path"):
    _p = Path(CONFIG["dataset"].get(_k, ""))
    if CONFIG["dataset"].get(_k) and not _p.is_absolute():
        CONFIG["dataset"][_k] = str(DATA_ROOT / _p)

# Per-recording world bounds of the FULL master dataset, used to renormalise u,v during
# rollout. Persisted so that running on a subset/sample does not silently change predictions.
WORLD_BOUNDS_PATH = HERE / "config" / "recording_world_bounds.json"

FEAT_COLS:    List[str] = CONFIG["features"]["input_feature_order"]
TARGET_COLS:  List[str] = CONFIG["features"]["target_cols"]
SPATIAL_COLS: List[str] = CONFIG["features"]["spatial_cols"]
N_FEAT        = CONFIG["features"]["n_features"]

HIDDEN_SIZE   = CONFIG["architecture"]["hidden_size"]
NUM_LAYERS    = CONFIG["architecture"]["num_layers"]
DROPOUT       = CONFIG["architecture"]["dropout"]

WINDOW_SIZE   = CONFIG["windowing"]["obs_window"]
HORIZON       = CONFIG["windowing"]["horizon"]
MIN_SPEED     = 1e-6
SEED          = CONFIG["training"]["seed"]

# genuine-turn reporting thresholds (H10)
TURN_HEADING_DEG    = CONFIG["turn_labels_reporting_only"]["heading_change_deg"]
TURN_DISP_FLOOR_M   = CONFIG["turn_labels_reporting_only"]["h10_displacement_floor_m"]
TURN_MAX_STEP_M     = CONFIG["turn_labels_reporting_only"]["max_single_step_m"]


def set_seed(s: int = SEED) -> None:
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ── Model (verbatim Model C TrajectoryLSTM) ───────────────────────────────────
class TrajectoryLSTM(nn.Module):
    def __init__(self, n_feat: int = N_FEAT) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feat, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.head = nn.Linear(HIDDEN_SIZE, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ── Column-wise StandardScaler (numpy-backed, fast rollout) ────────────────────
class ColumnScaler:
    def __init__(self) -> None:
        self.means: Optional[np.ndarray] = None
        self.stds:  Optional[np.ndarray] = None
        self.cols:  List[str] = []

    def fit(self, arr: np.ndarray, cols: List[str]) -> "ColumnScaler":
        self.cols = list(cols)
        data = np.asarray(arr, dtype=np.float64)
        self.means = data.mean(axis=0)
        self.stds = data.std(axis=0)
        self.stds[self.stds < 1e-9] = 1.0
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.means) / self.stds

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.stds + self.means

    def scale_col(self, col: str, value: float) -> float:
        i = self.cols.index(col)
        return (value - self.means[i]) / self.stds[i]

    def to_dict(self) -> dict:
        return {"cols": self.cols, "means": self.means.tolist(), "stds": self.stds.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "ColumnScaler":
        s = cls(); s.cols = d["cols"]
        s.means = np.asarray(d["means"], dtype=np.float64)
        s.stds = np.asarray(d["stds"], dtype=np.float64)
        return s


# ── Training windows: single-step next-displacement (Model C recipe) ───────────
def make_train_windows(df: pd.DataFrame, traj_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding windows of WINDOW_SIZE observed steps -> target_du/dv at last obs frame."""
    Xs, Ys = [], []
    grp = {tid: g for tid, g in df.groupby("trajectory_id", sort=False)}
    for tid in traj_ids:
        g = grp.get(tid)
        if g is None:
            continue
        g = g.sort_values("timestep")
        feat = g[FEAT_COLS].to_numpy(np.float32)
        tgt = g[TARGET_COLS].to_numpy(np.float32)
        n = len(g)
        for i in range(n - WINDOW_SIZE):
            Xs.append(feat[i:i + WINDOW_SIZE])
            Ys.append(tgt[i + WINDOW_SIZE - 1])
    if not Xs:
        return np.empty((0, WINDOW_SIZE, N_FEAT), np.float32), np.empty((0, 2), np.float32)
    return np.asarray(Xs, np.float32), np.asarray(Ys, np.float32)


# ── World-space helpers ────────────────────────────────────────────────────────
def recording_world_bounds(df: pd.DataFrame) -> dict:
    """Per-recording world bounds for u,v normalisation during rollout.
    Uses the persisted full-dataset bounds (WORLD_BOUNDS_PATH) when available; on the full
    dataset they are identical to the computed ones. Warns if the loaded data disagrees."""
    computed = compute_world_bounds(df)
    if not WORLD_BOUNDS_PATH.exists():
        return computed
    saved = json.loads(WORLD_BOUNDS_PATH.read_text(encoding="utf-8"))["recordings"]
    out = {}
    for rec, b in computed.items():
        if rec not in saved:
            out[rec] = b
            continue
        s = {k: float(saved[rec][k]) for k in ("xmin", "xmax", "ymin", "ymax", "xrng", "yrng")}
        if any(abs(s[k] - b[k]) > 1e-9 for k in s):
            warnings.warn(f"{rec}: loaded data bounds differ from persisted full-dataset bounds "
                          f"(subset/sample?) - using persisted bounds.")
        out[rec] = s
    return out


def compute_world_bounds(df: pd.DataFrame) -> dict:
    """Per-recording min/max of world_x/world_y in the given dataframe."""
    out = {}
    for rec, g in df.groupby("recording_id"):
        xmin, xmax = float(g["world_x"].min()), float(g["world_x"].max())
        ymin, ymax = float(g["world_y"].min()), float(g["world_y"].max())
        out[rec] = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
                    "xrng": max(xmax - xmin, 1e-9), "yrng": max(ymax - ymin, 1e-9)}
    return out


def gt_world_path(g: pd.DataFrame, start_idx: int, n_steps: int) -> np.ndarray:
    """Ground-truth world positions for the horizon, anchored at last observed frame."""
    last = g.iloc[start_idx + WINDOW_SIZE - 1]
    wx0, wy0 = float(last["world_x"]), float(last["world_y"])
    fut = g.iloc[start_idx + WINDOW_SIZE: start_idx + WINDOW_SIZE + n_steps]
    return np.column_stack([
        np.concatenate([[wx0], fut["world_x"].to_numpy(float)]),
        np.concatenate([[wy0], fut["world_y"].to_numpy(float)]),
    ])


def build_eval_windows(df: pd.DataFrame, traj_ids: List[str], world_bounds: dict,
                       n_steps: int = HORIZON, stride: int = 1,
                       max_windows: Optional[int] = None, seed: int = SEED):
    """Collect rollout eval windows (10 obs + n_steps GT future) from given tracks.
    Returns seeds_raw (M,W,F), start_world (M,2), gt_paths (M,n_steps+1,2),
    wb_arr dict of (M,) arrays, and meta list of dicts."""
    grp = {tid: g.sort_values("timestep") for tid, g in df.groupby("trajectory_id", sort=False)
           if tid in set(traj_ids)}
    seeds, starts, gts, meta = [], [], [], []
    wb_keys = {"xmin": [], "xrng": [], "ymin": [], "yrng": []}
    need = WINDOW_SIZE + n_steps
    for tid in traj_ids:
        g = grp.get(tid)
        if g is None or len(g) < need:
            continue
        rec = g["recording_id"].iloc[0]
        wb = world_bounds[rec]
        feat = g[FEAT_COLS].to_numpy(np.float32)
        wxr = g["world_x"].to_numpy(np.float64); wyr = g["world_y"].to_numpy(np.float64)
        n = len(g)
        for i in range(0, n - need + 1, stride):
            seeds.append(feat[i:i + WINDOW_SIZE])
            anchor = i + WINDOW_SIZE - 1
            starts.append((wxr[anchor], wyr[anchor]))
            gx = wxr[anchor: anchor + n_steps + 1]
            gy = wyr[anchor: anchor + n_steps + 1]
            gts.append(np.column_stack([gx, gy]))
            for k in wb_keys:
                wb_keys[k].append(wb[k])
            meta.append({"trajectory_id": tid, "recording_id": rec, "start_idx": i})
    if not seeds:
        return (np.empty((0, WINDOW_SIZE, N_FEAT), np.float32), np.empty((0, 2)),
                np.empty((0, n_steps + 1, 2)), {k: np.empty(0) for k in wb_keys}, [])
    seeds = np.asarray(seeds, np.float32)
    starts = np.asarray(starts, np.float64)
    gts = np.asarray(gts, np.float64)
    wb_arr = {k: np.asarray(v, np.float64) for k, v in wb_keys.items()}
    if max_windows is not None and len(seeds) > max_windows:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(seeds), size=max_windows, replace=False)
        seeds, starts, gts = seeds[sel], starts[sel], gts[sel]
        wb_arr = {k: v[sel] for k, v in wb_arr.items()}
        meta = [meta[i] for i in sel]
    return seeds, starts, gts, wb_arr, meta


def rollout_world(model: TrajectoryLSTM, seed_feat_raw: np.ndarray,
                  start_world: Tuple[float, float], wb: dict,
                  f_sc: ColumnScaler, t_sc: ColumnScaler,
                  n_steps: int, device: torch.device) -> np.ndarray:
    """Autoregressive rollout in world metres. Spatial features FROZEN at last
    observed value (no KDTree re-query — documented MODEL_X simplification)."""
    model.eval()
    wx, wy = start_world
    hs0 = seed_feat_raw[-1, FEAT_COLS.index("heading_sin")]
    hc0 = seed_feat_raw[-1, FEAT_COLS.index("heading_cos")]
    prev_h = math.atan2(hs0, hc0)
    sp_frozen = {c: float(seed_feat_raw[-1, FEAT_COLS.index(c)]) for c in SPATIAL_COLS}

    window = torch.tensor(f_sc.transform(seed_feat_raw).astype(np.float32),
                          dtype=torch.float32, device=device)
    positions = [(wx, wy)]
    for _ in range(n_steps):
        with torch.no_grad():
            pred_sc = model(window.unsqueeze(0))[0].cpu().numpy()
        du_m, dv_m = t_sc.inverse(pred_sc.reshape(1, 2))[0]
        wx += du_m; wy += dv_m
        positions.append((wx, wy))

        u_new = float(np.clip((wx - wb["xmin"]) / wb["xrng"], 0.0, 1.0))
        v_new = float(np.clip((wy - wb["ymin"]) / wb["yrng"], 0.0, 1.0))
        speed = math.hypot(du_m, dv_m)
        heading = math.atan2(dv_m, du_m) if speed > MIN_SPEED else prev_h
        tr = wrap_angle(heading - prev_h)
        prev_h = heading
        raw = {"du": du_m, "dv": dv_m, "speed": speed,
               "heading_sin": math.sin(heading), "heading_cos": math.cos(heading),
               "turn_rate": tr, "u": u_new, "v": v_new, **sp_frozen}
        new_row = np.array([f_sc.scale_col(c, raw[c]) for c in FEAT_COLS], np.float32)
        window = torch.cat([window[1:], torch.tensor(new_row, device=device).unsqueeze(0)], dim=0)
    return np.asarray(positions)


def rollout_world_batch(model: TrajectoryLSTM, seeds_raw: np.ndarray,
                        start_world: np.ndarray, wb_arr: dict,
                        f_sc: ColumnScaler, t_sc: ColumnScaler,
                        n_steps: int, device: torch.device) -> np.ndarray:
    """Vectorised autoregressive rollout for B windows at once.
    seeds_raw: (B, W, F) unscaled.  start_world: (B, 2) world metres.
    wb_arr: dict of (B,) arrays xmin,xrng,ymin,yrng.  Spatial features frozen.
    Returns (B, n_steps+1, 2) world paths (index 0 = shared anchor)."""
    model.eval()
    B = seeds_raw.shape[0]
    idx = {c: FEAT_COLS.index(c) for c in FEAT_COLS}
    wx = start_world[:, 0].astype(np.float64).copy()
    wy = start_world[:, 1].astype(np.float64).copy()
    prev_h = np.arctan2(seeds_raw[:, -1, idx["heading_sin"]],
                        seeds_raw[:, -1, idx["heading_cos"]]).astype(np.float64)
    sp_frozen = {c: seeds_raw[:, -1, idx[c]].astype(np.float64) for c in SPATIAL_COLS}
    means, stds = f_sc.means, f_sc.stds

    window = torch.tensor(((seeds_raw - means) / stds).astype(np.float32), device=device)
    positions = np.empty((B, n_steps + 1, 2), np.float64)
    positions[:, 0, 0] = wx; positions[:, 0, 1] = wy

    for s in range(n_steps):
        with torch.no_grad():
            pred_sc = model(window).cpu().numpy()              # (B,2)
        dd = t_sc.inverse(pred_sc)                              # (B,2) metres
        du_m, dv_m = dd[:, 0], dd[:, 1]
        wx += du_m; wy += dv_m
        positions[:, s + 1, 0] = wx; positions[:, s + 1, 1] = wy

        u_new = np.clip((wx - wb_arr["xmin"]) / wb_arr["xrng"], 0.0, 1.0)
        v_new = np.clip((wy - wb_arr["ymin"]) / wb_arr["yrng"], 0.0, 1.0)
        speed = np.hypot(du_m, dv_m)
        moving = speed > MIN_SPEED
        heading = np.where(moving, np.arctan2(dv_m, du_m), prev_h)
        tr = np.arctan2(np.sin(heading - prev_h), np.cos(heading - prev_h))
        prev_h = heading

        new_raw = np.empty((B, N_FEAT), np.float64)
        new_raw[:, idx["du"]] = du_m
        new_raw[:, idx["dv"]] = dv_m
        new_raw[:, idx["speed"]] = speed
        new_raw[:, idx["heading_sin"]] = np.sin(heading)
        new_raw[:, idx["heading_cos"]] = np.cos(heading)
        new_raw[:, idx["turn_rate"]] = tr
        new_raw[:, idx["u"]] = u_new
        new_raw[:, idx["v"]] = v_new
        for c in SPATIAL_COLS:
            new_raw[:, idx[c]] = sp_frozen[c]
        new_sc = torch.tensor(((new_raw - means) / stds).astype(np.float32), device=device)
        window = torch.cat([window[:, 1:, :], new_sc.unsqueeze(1)], dim=1)
    return positions


# ── Metrics ────────────────────────────────────────────────────────────────────
def path_length(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def net_heading_change_deg(path: np.ndarray) -> float:
    """Heading change between first and last displacement segment (degrees)."""
    if len(path) < 3:
        return 0.0
    d = np.diff(path, axis=0)
    h0 = math.atan2(d[0, 1], d[0, 0])
    h1 = math.atan2(d[-1, 1], d[-1, 0])
    return abs(math.degrees(wrap_angle(h1 - h0)))


def window_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """pred/gt are (n_steps+1, 2) world paths incl. shared anchor at index 0."""
    n = min(len(pred), len(gt))
    p, g = pred[:n], gt[:n]
    disp = np.linalg.norm(p - g, axis=1)
    ade = float(disp[1:].mean()) if n > 1 else float("nan")   # exclude shared anchor
    fde = float(disp[-1])
    rmse = float(np.sqrt((disp[1:] ** 2).mean())) if n > 1 else float("nan")
    # final-heading angular error (deg)
    ang = float("nan")
    if n >= 2:
        dp = p[-1] - p[max(n - 2, 0)]
        dg = g[-1] - g[max(n - 2, 0)]
        if np.linalg.norm(dp) > 1e-9 and np.linalg.norm(dg) > 1e-9:
            ang = abs(math.degrees(wrap_angle(
                math.atan2(dp[1], dp[0]) - math.atan2(dg[1], dg[0]))))
    pl_p, pl_g = path_length(p), path_length(g)
    return {"ade": ade, "fde": fde, "rmse": rmse, "angular_err_deg": ang,
            "pred_path_len": pl_p, "gt_path_len": pl_g,
            "len_ratio": (pl_p / pl_g) if pl_g > 1e-9 else float("nan"),
            "gt_net_disp": float(np.linalg.norm(g[-1] - g[0])),
            "gt_net_heading_deg": net_heading_change_deg(g),
            "gt_max_step": float(np.max(np.linalg.norm(np.diff(g, axis=0), axis=1))) if n > 1 else 0.0}


def is_genuine_turn(m: dict) -> bool:
    """H10 genuine-turn label (reporting only)."""
    return (m["gt_net_heading_deg"] > TURN_HEADING_DEG
            and m["gt_net_disp"] >= TURN_DISP_FLOOR_M
            and m["gt_max_step"] < TURN_MAX_STEP_M)
