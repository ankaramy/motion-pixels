"""
visualize_ablation_grid.py
--------------------------
Rollout + per-model + combined visualisations for the four ablation models.
Critically the rollout RECOMPUTES every derived feature at each predicted
step (no frozen features). Spatial features are re-queried via KDTree-IDW
over the v2.1C encoded CSV at the new predicted position.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _paths import (
    DATASET_CSV, ENCODED_CSV, MODELS_DIR, PLOTS_DIR, EXP,
    MODEL_DEFS, WINDOW_SIZE, HORIZON, IDW_K,
    STOP_THRESH_M, SHIFT_THRESH_RAD, ANGULAR_BIN_DEG,
    PRESET_TRACKS,
)
from train_ablation_models import TrajectoryLSTM


# --------------------------------------------------------------------------- #
# Spatial interpolator (3 features: obstacle/boundary/entrance)
# --------------------------------------------------------------------------- #
class SpatialInterpolator3:
    def __init__(self, df: pd.DataFrame, k: int = IDW_K):
        coords = df[["world_x", "world_y"]].to_numpy(dtype=np.float64)
        self.tree = cKDTree(coords)
        self.obs  = df["dist_to_obstacle"].to_numpy(dtype=np.float64)
        self.bnd  = df["dist_to_boundary"].to_numpy(dtype=np.float64)
        self.ent  = df["dist_to_entrance"].to_numpy(dtype=np.float64)
        self.k = k

    def query(self, x: float, y: float) -> Tuple[float, float, float]:
        d, idx = self.tree.query([x, y], k=self.k)
        if d[0] < 1e-9:
            return (float(self.obs[idx[0]]), float(self.bnd[idx[0]]),
                    float(self.ent[idx[0]]))
        w = 1.0 / d
        w = w / w.sum()
        return (float(np.dot(w, self.obs[idx])),
                float(np.dot(w, self.bnd[idx])),
                float(np.dot(w, self.ent[idx])))


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


# --------------------------------------------------------------------------- #
# Step-level full-feature builder used by the rollout
# --------------------------------------------------------------------------- #
def compute_full_step(wx, wy, du, dv, prev_heading,
                      interp: SpatialInterpolator3,
                      x_min, x_range, y_min, y_range) -> Dict[str, float]:
    speed   = float(np.hypot(du, dv))
    heading = float(np.arctan2(dv, du)) if speed > 1e-9 else prev_heading
    heading_delta = float(wrap(heading - prev_heading))
    obs, bnd, ent = interp.query(wx, wy)
    return {
        "world_x": wx, "world_y": wy,
        "u": (wx - x_min) / max(1e-9, x_range),
        "v": (wy - y_min) / max(1e-9, y_range),
        "delta_x": du, "delta_y": dv,
        "speed": speed,
        "heading_angle": heading,
        "is_stop":  1.0 if speed < STOP_THRESH_M else 0.0,
        "is_shift": 1.0 if abs(heading_delta) > SHIFT_THRESH_RAD else 0.0,
        "dist_to_obstacle": obs,
        "dist_to_boundary": bnd,
        "dist_to_entrance": ent,
        "heading_sin": float(np.sin(heading)),
        "heading_cos": float(np.cos(heading)),
        "turn_rate":   heading_delta,
    }


def load_model(model_def: Dict, device) -> Tuple[TrajectoryLSTM, object,
                                                  object, List[str]]:
    mdir = MODELS_DIR / model_def["name"]
    with open(mdir / "scaler.pkl", "rb") as fh:
        b = pickle.load(fh)
    fs = b["feature_scaler"]; ts = b["target_scaler"]
    features = b["feature_cols"]
    model = TrajectoryLSTM(len(features)).to(device)
    model.load_state_dict(torch.load(mdir / "model.pth",
                                      map_location=device))
    model.eval()
    return model, fs, ts, features


def rollout(model, fs, ts, features: List[str],
            track: pd.DataFrame, interp: SpatialInterpolator3,
            x_min, x_range, y_min, y_range,
            device, horizon: int) -> Tuple[np.ndarray, np.ndarray,
                                            np.ndarray]:
    """Autoregressive rollout. Returns (px, py, headings) with px/py of
    length horizon+1 (first entry = seed-end anchor) and headings of
    length horizon (per predicted step)."""
    # Initial WINDOW_SIZE-row window from real seed data.
    seed_rows = track.iloc[:WINDOW_SIZE][features].to_numpy(
        dtype=np.float32)
    window = fs.transform(seed_rows).copy()

    pos = track[["world_x", "world_y"]].to_numpy(dtype=np.float64)
    prev_x = float(pos[WINDOW_SIZE - 1, 0])
    prev_y = float(pos[WINDOW_SIZE - 1, 1])
    prev_heading = float(track.iloc[WINDOW_SIZE - 1]["heading_angle"])

    px = [prev_x]; py = [prev_y]
    headings: List[float] = []
    for _ in range(horizon):
        x_in = torch.tensor(window[np.newaxis], dtype=torch.float32
                            ).to(device)
        with torch.no_grad():
            p_s = model(x_in).cpu().numpy()[0]
        du, dv = ts.inverse_transform(p_s[np.newaxis])[0]
        du, dv = float(du), float(dv)

        wx, wy = prev_x + du, prev_y + dv
        px.append(wx); py.append(wy)

        step = compute_full_step(wx, wy, du, dv, prev_heading, interp,
                                  x_min, x_range, y_min, y_range)
        headings.append(step["heading_angle"])

        # Build new row in the EXACT feature order this model uses.
        row = np.array([step[c] for c in features], dtype=np.float32)
        scaled = fs.transform(row[np.newaxis])[0]
        window = np.vstack([window[1:], scaled])
        prev_x, prev_y, prev_heading = wx, wy, step["heading_angle"]

    return (np.asarray(px, dtype=np.float64),
            np.asarray(py, dtype=np.float64),
            np.asarray(headings, dtype=np.float64))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def per_track_metrics(pred_xy, pred_headings,
                      gt_future_xy, gt_future_headings,
                      seed_end) -> Dict[str, float]:
    """pred_xy length=horizon+1 (first = seed_end). Compare pred[1:] vs gt."""
    L = min(len(pred_xy) - 1, len(gt_future_xy))
    if L < 2:
        return {"ade": float("nan"), "fde": float("nan"),
                "heading_err_deg": float("nan"),
                "cum_heading_ratio": float("nan"),
                "path_len_ratio": float("nan"),
                "seed_continuity_err": float(np.hypot(
                    *(pred_xy[0] - seed_end))),
                "angularity_pred": 0,
                "angularity_gt": 0,
                "curvature_corr": float("nan")}

    p = pred_xy[1: 1 + L]
    g = gt_future_xy[:L]
    diffs = p - g
    ade = float(np.mean(np.hypot(diffs[:, 0], diffs[:, 1])))
    fde = float(np.hypot(*(p[-1] - g[-1])))

    ph = pred_headings[:L]
    gh = gt_future_headings[:L]
    heading_err = float(np.degrees(np.mean(np.abs(wrap(ph - gh)))))

    pred_cum = float(np.sum(np.abs(np.diff(np.unwrap(ph)))))
    gt_cum   = float(np.sum(np.abs(np.diff(np.unwrap(gh)))))
    cum_ratio = (pred_cum / gt_cum) if gt_cum > 1e-9 else float("nan")

    pred_path = float(np.sum(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))))
    gt_path   = float(np.sum(np.hypot(np.diff(g[:, 0]), np.diff(g[:, 1]))))
    path_ratio = (pred_path / gt_path) if gt_path > 1e-9 else float("nan")

    seed_cont = float(np.hypot(*(pred_xy[0] - seed_end)))

    # Angularity = number of per-step heading changes > ANGULAR_BIN_DEG.
    pred_dh = np.degrees(np.abs(np.diff(np.unwrap(ph))))
    gt_dh   = np.degrees(np.abs(np.diff(np.unwrap(gh))))
    ang_pred = int(np.sum(pred_dh > ANGULAR_BIN_DEG))
    ang_gt   = int(np.sum(gt_dh   > ANGULAR_BIN_DEG))

    # Curvature preservation = Pearson r between signed per-step heading deltas.
    pred_signed = np.diff(np.unwrap(ph))
    gt_signed   = np.diff(np.unwrap(gh))
    if (pred_signed.std() > 1e-9 and gt_signed.std() > 1e-9):
        curv_corr = float(np.corrcoef(pred_signed, gt_signed)[0, 1])
    else:
        curv_corr = float("nan")

    return {"ade": ade, "fde": fde,
            "heading_err_deg": heading_err,
            "cum_heading_ratio": cum_ratio,
            "path_len_ratio": path_ratio,
            "seed_continuity_err": seed_cont,
            "angularity_pred": ang_pred, "angularity_gt": ang_gt,
            "curvature_corr": curv_corr}


def gt_future_xy_headings(track: pd.DataFrame, horizon: int):
    xy = track[["world_x", "world_y"]].to_numpy(dtype=np.float64)[
        WINDOW_SIZE: WINDOW_SIZE + horizon]
    h  = track["heading_angle"].to_numpy(dtype=np.float64)[
        WINDOW_SIZE: WINDOW_SIZE + horizon]
    return xy, h


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def draw_world_panel(ax, full_gt, seed_xy, gt_future, pred_xy,
                     tid, ade, fde, color, label_extra=""):
    ax.plot(full_gt[:, 0], full_gt[:, 1], "-", color="#bdc3c7",
            lw=1.0, alpha=0.85, label="full GT")
    ax.plot(seed_xy[:, 0], seed_xy[:, 1], "-o", color="#1c2833",
            lw=1.8, ms=4, label="seed (10)")
    ax.plot(gt_future[:, 0], gt_future[:, 1], "--", color="#7f8c8d",
            lw=1.4, dashes=(4, 2), label="GT future")
    ax.plot(pred_xy[:, 0], pred_xy[:, 1], "-s", color=color,
            lw=1.7, ms=4, alpha=0.95, label=f"rollout {label_extra}")
    ax.plot(seed_xy[-1, 0], seed_xy[-1, 1], "o", color="#27ae60",
            ms=9, mfc="none", mew=1.6, label="seed end")
    ax.set_title(f"track {tid}   ADE={ade:.2f} m   FDE={fde:.2f} m",
                 fontsize=10)
    ax.set_xlabel("world_x (m)", fontsize=8)
    ax.set_ylabel("world_y (m)", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, lw=0.35, alpha=0.6)
    ax.legend(fontsize=7, loc="best")


def per_model_contact_sheet(model_def, tracks, interp, df, extents, device,
                            out_path: Path) -> List[Dict]:
    model, fs, ts, features = load_model(model_def, device)
    panels = []
    metrics_rows = []
    for tid in tracks:
        g = (df[df["track_id"] == tid].sort_values("frame")
             .iloc[: WINDOW_SIZE + HORIZON].reset_index(drop=True))
        if len(g) < WINDOW_SIZE + HORIZON:
            continue
        pos = g[["world_x", "world_y"]].to_numpy(dtype=np.float64)
        seed_xy = pos[:WINDOW_SIZE]
        gt_future, gt_headings = gt_future_xy_headings(g, HORIZON)

        px, py, headings = rollout(
            model, fs, ts, features, g, interp,
            extents["x_min"], extents["x_max"] - extents["x_min"],
            extents["y_min"], extents["y_max"] - extents["y_min"],
            device, HORIZON)
        pred_xy = np.column_stack([px, py])

        m = per_track_metrics(pred_xy, headings, gt_future, gt_headings,
                              seed_xy[-1])
        metrics_rows.append({"variant": model_def["name"], "track_id": int(tid),
                             **m})
        panels.append({"tid": tid, "full_gt": pos, "seed": seed_xy,
                       "gt_future": gt_future, "pred": pred_xy,
                       "ade": m["ade"], "fde": m["fde"]})

    # Render
    n = len(panels); cols = 3
    rows_n = max(1, int(np.ceil(n / cols)))
    fig, axes = plt.subplots(rows_n, cols,
                              figsize=(6 * cols, 5.5 * rows_n),
                              squeeze=False)
    for i, d in enumerate(panels):
        ax = axes[i // cols, i % cols]
        draw_world_panel(ax, d["full_gt"], d["seed"], d["gt_future"],
                          d["pred"], d["tid"], d["ade"], d["fde"],
                          model_def["color"])
    for j in range(n, rows_n * cols):
        axes[j // cols, j % cols].set_axis_off()
    fig.suptitle(
        f"{model_def['label']}  ·  features={len(features)}  ·  "
        f"horizon={HORIZON} steps  ·  rollout recomputes every derived "
        f"feature + KDTree-IDW spatial refresh",
        fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return metrics_rows


def combined_per_track_plot(track_id, all_rollouts, gt_full, seed_xy,
                            gt_future, extents, out_path: Path) -> None:
    """One panel per track. Show seed + GT future + all 4 model rollouts
    on normalised u/v axes."""
    fig, ax = plt.subplots(figsize=(8.5, 7))
    xr = extents["x_max"] - extents["x_min"]
    yr = extents["y_max"] - extents["y_min"]
    norm = lambda xy: np.column_stack([
        (xy[:, 0] - extents["x_min"]) / max(1e-9, xr),
        (xy[:, 1] - extents["y_min"]) / max(1e-9, yr)])
    full_n = norm(gt_full); seed_n = norm(seed_xy); gtf_n = norm(gt_future)
    ax.plot(full_n[:, 0], full_n[:, 1], "-", color="#cccccc",
            lw=1.0, alpha=0.85, label="full GT")
    ax.plot(seed_n[:, 0], seed_n[:, 1], "-o", color="#1c2833",
            lw=1.8, ms=4, label="seed")
    ax.plot(gtf_n[:, 0], gtf_n[:, 1], "--", color="#7f8c8d",
            lw=1.4, dashes=(4, 2), label="GT future")
    for mdef, pred_xy, ade in all_rollouts:
        pn = norm(pred_xy)
        ax.plot(pn[:, 0], pn[:, 1], "-", color=mdef["color"],
                lw=1.7, alpha=0.95,
                label=f"{mdef['label']}  (ADE={ade:.2f} m)")
    ax.plot(seed_n[-1, 0], seed_n[-1, 1], "o", color="#27ae60",
            ms=10, mfc="none", mew=1.8, label="seed end")
    ax.set_xlabel("u (normalised x)"); ax.set_ylabel("v (normalised y)")
    ax.set_title(f"All models vs GT — track {track_id}",
                 fontsize=11, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, lw=0.35, alpha=0.6)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device: {device}")

    df = pd.read_csv(DATASET_CSV)
    enc = pd.read_csv(ENCODED_CSV)
    interp = SpatialInterpolator3(enc, k=IDW_K)
    print(f"[INFO]  KDTree built on {len(enc):,} encoded rows")

    extents = pd.read_csv(EXP / "world_extents.csv", index_col=0
                          ).squeeze("columns").to_dict()
    extents = {k: float(v) for k, v in extents.items()}

    available = set(df["track_id"].unique().tolist())
    tracks = [t for t in PRESET_TRACKS if t in available]
    if len(tracks) < len(PRESET_TRACKS):
        miss = [t for t in PRESET_TRACKS if t not in available]
        print(f"[WARN]  missing tracks {miss}; using {len(tracks)}")
    print(f"[INFO]  tracks: {tracks}")

    all_metrics: List[Dict] = []
    # Per-model rollouts cached for the combined plots.
    rollouts_by_track: Dict[int, List[Tuple[Dict, np.ndarray, float]]] = {
        t: [] for t in tracks}
    gt_cache: Dict[int, Dict] = {}

    for mdef in MODEL_DEFS:
        print(f"\n========== {mdef['name']} ==========")
        sheet = PLOTS_DIR / f"contact_sheet_{mdef['name']}.png"
        rows = per_model_contact_sheet(mdef, tracks, interp, df, extents,
                                         device, sheet)
        all_metrics.extend(rows)
        print(f"[OK]    {sheet.name}")
        # Cache rollouts for combined plots.
        model, fs, ts, features = load_model(mdef, device)
        for tid in tracks:
            g = (df[df["track_id"] == tid].sort_values("frame")
                 .iloc[: WINDOW_SIZE + HORIZON].reset_index(drop=True))
            if len(g) < WINDOW_SIZE + HORIZON:
                continue
            px, py, _ = rollout(model, fs, ts, features, g, interp,
                                 extents["x_min"],
                                 extents["x_max"] - extents["x_min"],
                                 extents["y_min"],
                                 extents["y_max"] - extents["y_min"],
                                 device, HORIZON)
            pred_xy = np.column_stack([px, py])
            r = next(rr for rr in all_metrics
                      if rr["variant"] == mdef["name"]
                      and rr["track_id"] == tid)
            rollouts_by_track[tid].append((mdef, pred_xy, r["ade"]))
            if tid not in gt_cache:
                pos = g[["world_x", "world_y"]].to_numpy(dtype=np.float64)
                gt_cache[tid] = {
                    "full": pos,
                    "seed": pos[:WINDOW_SIZE],
                    "future": pos[WINDOW_SIZE:],
                }

    # Combined per-track plots
    combined_dir = PLOTS_DIR / "combined_all_models"
    combined_dir.mkdir(parents=True, exist_ok=True)
    for tid in tracks:
        if tid not in gt_cache:
            continue
        out = combined_dir / f"combined_track_{tid}.png"
        combined_per_track_plot(tid, rollouts_by_track[tid],
                                 gt_cache[tid]["full"],
                                 gt_cache[tid]["seed"],
                                 gt_cache[tid]["future"],
                                 extents, out)
    print(f"[OK]    combined plots → {combined_dir.name}/")

    pd.DataFrame(all_metrics).to_csv(EXP / "metrics_world.csv",
                                       index=False)
    print(f"[OK]    metrics_world.csv")


if __name__ == "__main__":
    main()
