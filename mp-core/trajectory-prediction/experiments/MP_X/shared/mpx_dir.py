"""
MP_X shared — direction-output variant of Model C (exp04).

Same LSTM trunk as the frozen Model C (10 inputs, hidden 128, 2 layers, dropout
0.2) but the head is widened 2 -> 4:

    outputs = (target_du, target_dv, future_heading_sin, future_heading_cos)

where the heading target is the UNIT next-step displacement direction
(unit(target_du, target_dv), with a stationary fallback). Training loss is
    position_mse + 0.2 * heading_mse
both computed in the standardized target space.

At rollout the model's OWN predicted heading (normalized) drives the
heading_sin / heading_cos / turn_rate INPUT channels for the next step, so the
explicit heading output can actually steer the trajectory. Positions still
advance by the predicted du/dv, and all turn metrics are measured from the
realized positions — identical to exp01/exp02, so the comparison is apples-to-
apples. The returned dict matches P5.rollout_one_trajectory.
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import mpx_common as C
fz = C.fz

DIR_TARGET_COLS = ["target_du", "target_dv", "future_heading_sin", "future_heading_cos"]
MIN_SPEED = getattr(fz, "MIN_SPEED", 1e-6)
EPS = 1e-9


class TrajectoryLSTMDir(nn.Module):
    """Frozen Model C trunk, 4-wide output head."""

    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feat, hidden_size=fz.HIDDEN_SIZE,
            num_layers=fz.NUM_LAYERS, batch_first=True,
            dropout=fz.DROPOUT if fz.NUM_LAYERS > 1 else 0.0,
        )
        self.head = nn.Linear(fz.HIDDEN_SIZE, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def add_heading_targets(df: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """Add future_heading_sin/cos = unit(target_du, target_dv) in place.
    Falls back to the current-frame heading when the next step is ~stationary."""
    du = df["target_du"].to_numpy(np.float64)
    dv = df["target_dv"].to_numpy(np.float64)
    mag = np.hypot(du, dv)
    safe = np.where(mag > eps, mag, 1.0)
    df["future_heading_sin"] = np.where(mag > eps, dv / safe, df["heading_sin"].to_numpy())
    df["future_heading_cos"] = np.where(mag > eps, du / safe, df["heading_cos"].to_numpy())
    return df


def heading_loss_components(pred_s, y_s):
    """(position_mse, heading_mse) in standardized target space.
    pred_s/y_s are (B,4) scaled tensors: [du, dv, hsin, hcos]."""
    pos = nn.functional.mse_loss(pred_s[:, :2], y_s[:, :2])
    head = nn.functional.mse_loss(pred_s[:, 2:], y_s[:, 2:])
    return pos, head


# ── scaler IO (feat scaler + 4-col target scaler) ────────────────────────────
def save_dir_scalers(path: Path, f_sc, t_sc) -> None:
    blob = {"feat_means": np.asarray(f_sc.means).tolist(),
            "feat_stds": np.asarray(f_sc.stds).tolist(), "feat_cols": list(C.FEAT_COLS),
            "tgt_means": np.asarray(t_sc.means).tolist(),
            "tgt_stds": np.asarray(t_sc.stds).tolist(), "tgt_cols": list(DIR_TARGET_COLS)}
    Path(path).write_bytes(pickle.dumps(blob))


def load_dir_scalers(path: Path):
    blob = pickle.loads(Path(path).read_bytes())
    f = fz.ColumnScaler(); f.means = np.asarray(blob["feat_means"])
    f.stds = np.asarray(blob["feat_stds"]); f.cols = blob["feat_cols"]
    t = fz.ColumnScaler(); t.means = np.asarray(blob["tgt_means"])
    t.stds = np.asarray(blob["tgt_stds"]); t.cols = blob["tgt_cols"]
    return f, t


# ── rollout (4-output) ───────────────────────────────────────────────────────
def rollout_dir(model, seed_raw, feat_cols, all_feat_cols, spatial_cols,
                f_sc, t_sc, world_bnd, kdt, n_steps, device):
    """Autoregressive rollout for the 4-output model.

    Returns (positions (n+1,2) in world metres, headings (n,) = per-step
    displacement headings — the SAME quantity exp01/exp02 measure)."""
    model.eval()
    u0 = seed_raw[-1, all_feat_cols.index("u")]
    v0 = seed_raw[-1, all_feat_cols.index("v")]
    wx = u0 * world_bnd["xrng"] + world_bnd["xmin"]
    wy = v0 * world_bnd["yrng"] + world_bnd["ymin"]

    hs0 = seed_raw[-1, all_feat_cols.index("heading_sin")]
    hc0 = seed_raw[-1, all_feat_cols.index("heading_cos")]
    prev_h = math.atan2(hs0, hc0)   # tracks the EXPLICIT predicted heading channel

    seed_feat = seed_raw[:, [all_feat_cols.index(c) for c in feat_cols]]
    window = torch.tensor(f_sc.transform(seed_feat).astype(np.float32), device=device)

    positions = [(wx, wy)]
    headings = []
    for _ in range(n_steps):
        with torch.no_grad():
            pred_sc = model(window.unsqueeze(0))[0].cpu().numpy()   # (4,)
        du_m, dv_m, hs_p, hc_p = t_sc.inverse(pred_sc.reshape(1, 4))[0]

        # normalize the explicit predicted heading vector
        hn = math.hypot(hs_p, hc_p)
        if hn > EPS:
            hs_u, hc_u = hs_p / hn, hc_p / hn
        else:
            hs_u, hc_u = math.sin(prev_h), math.cos(prev_h)
        head_pred = math.atan2(hs_u, hc_u)

        wx += du_m; wy += dv_m
        positions.append((wx, wy))

        speed = math.hypot(du_m, dv_m)
        step_head = math.atan2(dv_m, du_m) if speed > MIN_SPEED else prev_h
        headings.append(step_head)   # realized-motion heading (for metrics vs GT)

        tr = fz.wrap_angle(head_pred - prev_h)   # turn_rate from EXPLICIT heading
        prev_h = head_pred
        u_new = float(np.clip((wx - world_bnd["xmin"]) / world_bnd["xrng"], 0.0, 1.0))
        v_new = float(np.clip((wy - world_bnd["ymin"]) / world_bnd["yrng"], 0.0, 1.0))

        raw = {"du": du_m, "dv": dv_m, "speed": speed,
               "heading_sin": hs_u, "heading_cos": hc_u, "turn_rate": tr,
               "u": u_new, "v": v_new}
        if kdt is not None:
            for col, val in zip(spatial_cols, fz.kdt_lookup(kdt, u_new, v_new)):
                raw[col] = val
        else:
            for col in spatial_cols:
                raw[col] = 0.0

        new_row = np.array([f_sc.scale_col(c, raw[c]) for c in feat_cols], dtype=np.float32)
        window = torch.cat([window[1:], torch.tensor(new_row, device=device).unsqueeze(0)], dim=0)

    return np.array(positions), np.array(headings)


def rollout_one_trajectory_dir(model, traj_df, world_bnd, f_sc, t_sc, kdt, device,
                               max_steps=None):
    """Mirror of P5.rollout_one_trajectory for the 4-output model; same dict shape."""
    max_steps = C.N_ROLLOUT if max_steps is None else max_steps
    traj_df = traj_df.sort_values("timestep").reset_index(drop=True)
    future = len(traj_df) - C.WINDOW_SIZE
    if future < 1:
        return None
    n_steps = int(min(max_steps, future))
    seed_raw = traj_df.iloc[:C.WINDOW_SIZE][C.ALL_FEAT_COLS].to_numpy(np.float32)
    pred_pos, pred_h = rollout_dir(model, seed_raw, C.FEAT_COLS, C.ALL_FEAT_COLS,
                                   C.SPATIAL_COLS, f_sc, t_sc, world_bnd, kdt, n_steps, device)
    gt_pos = fz.gt_positions_from_seed(traj_df, world_bnd, n_steps)
    gt_h = fz.gt_headings_from_df(traj_df, n_steps)
    su = traj_df.iloc[:C.WINDOW_SIZE]["u"].to_numpy()
    sv = traj_df.iloc[:C.WINDOW_SIZE]["v"].to_numpy()
    seed_pos = np.column_stack([su * world_bnd["xrng"] + world_bnd["xmin"],
                                sv * world_bnd["yrng"] + world_bnd["ymin"]])
    n = min(len(pred_pos), len(gt_pos))
    d = np.sqrt(((pred_pos[:n] - gt_pos[:n]) ** 2).sum(axis=1))[1:]
    if len(d) == 0:
        return None
    n_h = min(len(pred_h), len(gt_h))
    ang = float(np.mean([abs(fz.wrap_angle(pred_h[i] - gt_h[i])) for i in range(n_h)])) if n_h else float("nan")
    tr_pred = float(np.mean([abs(fz.wrap_angle(pred_h[i] - pred_h[i-1])) for i in range(1, len(pred_h))])) if len(pred_h) > 1 else float("nan")
    tr_gt = float(np.mean([abs(fz.wrap_angle(gt_h[i] - gt_h[i-1])) for i in range(1, len(gt_h))])) if len(gt_h) > 1 else float("nan")
    return {"trajectory_id": str(traj_df["trajectory_id"].iloc[0]),
            "recording_id": str(traj_df["recording_id"].iloc[0]),
            "split": str(traj_df["split"].iloc[0]), "length": int(len(traj_df)),
            "n_steps": n_steps, "seed_pos": seed_pos, "gt_pos": gt_pos,
            "pred_pos": pred_pos, "per_step_err": d, "ade": float(d.mean()),
            "fde": float(d[-1]), "angular_error": ang,
            "mean_turn_pred": tr_pred, "mean_turn_gt": tr_gt}
