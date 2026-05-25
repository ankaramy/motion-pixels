"""
eval_phase2b_final_multi.py
---------------------------
Phase 2B-Final extension: multi-trajectory robustness evaluation.

Reuses the LSTM and GRU models trained by train_phase2b_final.py and
runs N=50 rollouts (10-seed → 30-future) on randomly selected persons
with the *same* KDTree spatial recomputation used in training rollout.

No retraining. No change to rollout logic.

Outputs
-------
  mp-data/outputs/prediction/experiments/phase-2b-final/
      multi_eval_results.csv
      phase2b_final_multi_eval.png
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE    = Path(__file__).resolve().parent
MP_ROOT = HERE.parent.parent.parent.parent
MP_DATA = MP_ROOT / "mp-data"

sys.path.insert(0, str(HERE))
from train_phase2b_final import (   # noqa: E402  reuse training-time defs
    TrajectoryLSTM, TrajectoryGRU, SpatialInterpolator,
    add_deltas, FEATURE_COLS, WINDOW_SIZE, HIDDEN_SIZE, IDW_K,
)

SRC_CSV    = MP_DATA / "processed" / "encoded" / "trajectories_encoded.csv"
PHASE_DIR  = MP_DATA / "outputs" / "prediction" / "experiments" / "phase-2b-final"
LSTM_DIR   = PHASE_DIR / "lstm"
GRU_DIR    = PHASE_DIR / "gru"
CSV_OUT    = PHASE_DIR / "multi_eval_results.csv"
FIG_OUT    = PHASE_DIR / "phase2b_final_multi_eval.png"

N_TRAJ      = 50
N_ROLLOUT   = 30
SEED_STEPS  = WINDOW_SIZE         # 10
MIN_LEN     = SEED_STEPS + N_ROLLOUT
SEED        = 42


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_model(model_cls, model_dir: Path, n_feats: int, device):
    """Load a trained model + its scaler bundle from a phase-2b-final subdir."""
    pth = model_dir / "model.pth"
    sca = model_dir / "scaler.pkl"
    if not pth.exists() or not sca.exists():
        sys.exit(f"[ERROR] Missing artefacts in {model_dir}")
    model = model_cls(n_feats, HIDDEN_SIZE).to(device)
    model.load_state_dict(torch.load(pth, map_location=device))
    model.eval()
    with open(sca, "rb") as fh:
        bundle = pickle.load(fh)
    return model, bundle["feature_scaler"], bundle["target_scaler"]


# ---------------------------------------------------------------------------
# Rollout (mirrors train_phase2b_final.rollout — kept verbatim in spirit)
# ---------------------------------------------------------------------------

def rollout(model, fs, ts, track: pd.DataFrame,
            interp: SpatialInterpolator, device):
    feats     = track[FEATURE_COLS].to_numpy(dtype=np.float32)
    positions = track[["world_x", "world_y"]].to_numpy(dtype=np.float32)

    feats_scaled = fs.transform(feats)
    window = feats_scaled[:WINDOW_SIZE].copy()
    prev_x, prev_y = positions[WINDOW_SIZE - 1]
    pred_x, pred_y = [], []

    for _ in range(N_ROLLOUT):
        x_in = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            p_s = model(x_in).cpu().numpy()[0]
        dx, dy = ts.inverse_transform(p_s[np.newaxis])[0]
        wx = float(prev_x) + float(dx)
        wy = float(prev_y) + float(dy)
        pred_x.append(wx)
        pred_y.append(wy)

        obs, bnd = interp.query(wx, wy)
        new_raw = np.array([wx, wy, float(dx), float(dy), obs, bnd],
                           dtype=np.float32)
        window = np.vstack([window[1:], fs.transform(new_raw[np.newaxis])[0]])
        prev_x, prev_y = wx, wy

    return np.array(pred_x), np.array(pred_y), positions


def endpoint_drift(pred_x, pred_y, positions) -> float:
    gt_end = positions[WINDOW_SIZE + N_ROLLOUT - 1]
    return float(np.hypot(pred_x[-1] - gt_end[0], pred_y[-1] - gt_end[1]))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summary_stats(label: str, drifts: np.ndarray) -> dict:
    return {
        "label":  label,
        "mean":   float(np.mean(drifts)),
        "median": float(np.median(drifts)),
        "std":    float(np.std(drifts)),
        "min":    float(np.min(drifts)),
        "max":    float(np.max(drifts)),
    }


def print_summary(s: dict):
    print(f"  {s['label']:<6}  mean={s['mean']:6.3f}  median={s['median']:6.3f}  "
          f"std={s['std']:6.3f}  min={s['min']:6.3f}  max={s['max']:6.3f}  m")


def save_figure(lstm_drifts, gru_drifts, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    bp = ax1.boxplot([lstm_drifts, gru_drifts], labels=["LSTM", "GRU"],
                     widths=0.55, patch_artist=True, showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="white",
                                    markeredgecolor="black", markersize=6))
    for patch, color in zip(bp["boxes"], ["#c0392b", "#2980b9"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax1.set_ylabel("Endpoint drift (m)")
    ax1.set_title(f"Endpoint drift across {len(lstm_drifts)} trajectories")
    ax1.grid(True, lw=0.4, alpha=0.5, axis="y")

    bins = np.linspace(0, max(lstm_drifts.max(), gru_drifts.max()) * 1.05, 24)
    ax2.hist(lstm_drifts, bins=bins, alpha=0.55, color="#c0392b",
             label=f"LSTM (mean {lstm_drifts.mean():.2f} m)", edgecolor="white")
    ax2.hist(gru_drifts,  bins=bins, alpha=0.55, color="#2980b9",
             label=f"GRU  (mean {gru_drifts.mean():.2f} m)", edgecolor="white")
    ax2.set_xlabel("Endpoint drift (m)")
    ax2.set_ylabel("Count")
    ax2.set_title("Drift distributions")
    ax2.legend(fontsize=9)
    ax2.grid(True, lw=0.4, alpha=0.5, axis="y")

    fig.suptitle(
        f"Phase 2B-Final  —  multi-trajectory robustness  "
        f"({len(lstm_drifts)} tracks, seed={SEED})\n"
        "10-step seed → 30-step rollout  |  spatial features recomputed at every "
        "predicted position via KDTree",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]    Figure : {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device : {device}")

    if not SRC_CSV.exists():
        sys.exit(f"[ERROR] {SRC_CSV} not found.")

    df_full = pd.read_csv(SRC_CSV)
    if "track_id" in df_full.columns and "person_id" not in df_full.columns:
        df_full = df_full.rename(columns={"track_id": "person_id"})
    if "frame_idx" in df_full.columns and "frame_number" not in df_full.columns:
        df_full = df_full.rename(columns={"frame_idx": "frame_number"})
    print(f"[INFO]  Loaded {len(df_full):,} rows  |  "
          f"{df_full['person_id'].nunique()} persons")

    print("[INFO]  Building KDTree spatial interpolator ...")
    interp = SpatialInterpolator(df_full, k=IDW_K)

    df = add_deltas(df_full)

    track_lengths = df.groupby("person_id").size()
    eligible_pids = track_lengths[track_lengths >= MIN_LEN].index.to_numpy()
    print(f"[INFO]  {len(eligible_pids)} persons have ≥ {MIN_LEN} frames")
    if len(eligible_pids) < N_TRAJ:
        sys.exit(f"[ERROR] Only {len(eligible_pids)} eligible tracks — need {N_TRAJ}.")

    rng = np.random.default_rng(SEED)
    chosen_pids = rng.choice(eligible_pids, N_TRAJ, replace=False)
    print(f"[INFO]  Selected {N_TRAJ} trajectories (seed={SEED})")

    n_feats = len(FEATURE_COLS)
    lstm_model, lstm_fs, lstm_ts = load_model(TrajectoryLSTM, LSTM_DIR, n_feats, device)
    gru_model,  gru_fs,  gru_ts  = load_model(TrajectoryGRU,  GRU_DIR,  n_feats, device)
    print(f"[INFO]  LSTM + GRU loaded from {PHASE_DIR}")

    rows = []
    for i, pid in enumerate(chosen_pids, 1):
        track = (df[df["person_id"] == pid]
                 .reset_index(drop=True)
                 .iloc[:MIN_LEN]
                 .copy())

        lx, ly, pos = rollout(lstm_model, lstm_fs, lstm_ts, track, interp, device)
        gx, gy, _   = rollout(gru_model,  gru_fs,  gru_ts,  track, interp, device)

        d_lstm = endpoint_drift(lx, ly, pos)
        d_gru  = endpoint_drift(gx, gy, pos)
        rows.append({"trajectory_id": int(pid),
                     "lstm_drift":    d_lstm,
                     "gru_drift":     d_gru})

        if i % 10 == 0 or i == N_TRAJ:
            print(f"  [{i:>3d}/{N_TRAJ}]  pid={int(pid):<6}  "
                  f"LSTM={d_lstm:6.3f} m  GRU={d_gru:6.3f} m")

    out_df = pd.DataFrame(rows)
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(CSV_OUT, index=False)
    print(f"\n[OK]    CSV    : {CSV_OUT}")

    lstm_d = out_df["lstm_drift"].to_numpy()
    gru_d  = out_df["gru_drift"].to_numpy()
    s_lstm = summary_stats("LSTM", lstm_d)
    s_gru  = summary_stats("GRU",  gru_d)

    save_figure(lstm_d, gru_d, FIG_OUT)

    gru_better  = int((gru_d  < lstm_d).sum())
    lstm_better = int((lstm_d < gru_d ).sum())
    ties        = N_TRAJ - gru_better - lstm_better
    pct_gru     = 100.0 * gru_better  / N_TRAJ
    pct_lstm    = 100.0 * lstm_better / N_TRAJ

    print(f"\n{'='*64}")
    print(f"  PHASE 2B-FINAL  multi-trajectory evaluation  (N={N_TRAJ})")
    print(f"{'='*64}")
    print_summary(s_lstm)
    print_summary(s_gru)
    print(f"\n  average drift LSTM           : {s_lstm['mean']:.3f} m")
    print(f"  average drift GRU            : {s_gru ['mean']:.3f} m")
    print(f"  GRU outperforms LSTM         : {gru_better}/{N_TRAJ}  ({pct_gru:.1f} %)")
    print(f"  LSTM outperforms GRU         : {lstm_better}/{N_TRAJ}  ({pct_lstm:.1f} %)")
    if ties:
        print(f"  Ties                          : {ties}")

    delta = s_gru["mean"] - s_lstm["mean"]
    print(f"\n  Δ mean drift (GRU − LSTM)    : {delta:+.3f} m")

    print(f"\n  CONCLUSION:")
    if pct_gru >= 60 and s_gru["mean"] < s_lstm["mean"]:
        verdict = ("GRU is consistently better than LSTM under corrected spatial "
                   "rollout.")
    elif pct_lstm >= 60 and s_lstm["mean"] < s_gru["mean"]:
        verdict = ("LSTM is consistently better than GRU — the previous "
                   "GRU advantage was trajectory-specific.")
    else:
        verdict = ("LSTM and GRU perform comparably across trajectories — the "
                   "previous single-trajectory GRU win was not a robust trend.")
    print(f"    {verdict}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
