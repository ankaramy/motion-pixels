"""
predict_example.py — load the FINAL model (MODEL_XC_B_CURV_LIGHT) and roll out predictions.

Runs out of the box on the committed sample (25 test-split tracks, mp-data/sample/):
    python predict_example.py
    python predict_example.py --horizon 100 --plot out.png
    python predict_example.py --csv <MP_DATA_ROOT>/Barcelona_v3_manual_master_dataset/master_dataset.csv

Rollout uses the persisted full-dataset world bounds (experiments/MODEL_X/config/
recording_world_bounds.json), so predictions on the sample are identical to those on the
full dataset. Spatial features are frozen at the last observed step (documented convention).
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments" / "MODEL_X"))
import model_x_lib as L  # noqa: E402

CKPT_DIR = HERE / "checkpoints" / "MODEL_XC_B_CURV_LIGHT"
SAMPLE = L.REPO_ROOT / "mp-data" / "sample" / "barcelona_v3_test_sample.csv"


def load_final_model(device):
    sc = json.loads((CKPT_DIR / "scalers.json").read_text())
    f_sc = L.ColumnScaler.from_dict(sc["feature_scaler"])
    t_sc = L.ColumnScaler.from_dict(sc["target_scaler"])
    model = L.TrajectoryLSTM(L.N_FEAT).to(device)
    model.load_state_dict(torch.load(CKPT_DIR / "model_best.pth", map_location=device, weights_only=True))
    model.eval()
    return model, f_sc, t_sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(SAMPLE))
    ap.add_argument("--horizon", type=int, default=L.HORIZON, help="rollout steps (trained at H10)")
    ap.add_argument("--plot", default=None, help="optional PNG path")
    a = ap.parse_args()
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    model, f_sc, t_sc = load_final_model(device)

    df = pd.read_csv(a.csv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # subset data -> persisted bounds are used
        wb = L.recording_world_bounds(df)
    tids = df.trajectory_id.unique().tolist()
    seeds, start, gt, wba, meta = L.build_eval_windows(df, tids, wb, n_steps=a.horizon, stride=1)
    if len(seeds) == 0:
        sys.exit("no tracks long enough for this horizon")
    pred = L.rollout_world_batch(model, seeds, start, wba, f_sc, t_sc, a.horizon, device)
    err = np.linalg.norm(pred - gt, axis=2)
    print(f"MODEL_XC_B_CURV_LIGHT | {len(seeds)} windows from {len(tids)} tracks | H{a.horizon}")
    print(f"ADE {err[:, 1:].mean():.3f} m   FDE {err[:, -1].mean():.3f} m   (device={device})")

    if a.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        i = int(np.argmax(np.linalg.norm(gt[:, -1] - gt[:, 0], axis=1)))   # longest-moving window
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(gt[i, :, 0], gt[i, :, 1], "k-", label="ground truth")
        ax.plot(pred[i, :, 0], pred[i, :, 1], "m--", label="MODEL_XC prediction")
        ax.set_aspect("equal"); ax.legend(); ax.set_title(meta[i]["trajectory_id"])
        fig.savefig(a.plot, dpi=150, bbox_inches="tight")
        print("plot ->", a.plot)


if __name__ == "__main__":
    main()
