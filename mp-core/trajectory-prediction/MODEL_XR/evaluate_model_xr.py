"""
evaluate_model_xr.py --config configs/<variant>.json

Evaluates a trained MODEL_XR variant at the BASE horizon (H10, the MODEL_X control target):
standard (ADE/FDE), magnitude (step1 / smoothed-step1 / net-disp / smoothed-path ratios,
%nd<0.5) and direction (cosine, t1 cosine, heading err, %cos>0.75/0.50), overall + per recording.
Writes experiments/<variant>/eval_base.json. Inference only; no MODEL_X changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xr_common as XC  # noqa: E402
L = XC.L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = XC.load_config(ap.parse_args().config)
    variant = cfg["variant"]
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckdir = HERE / "checkpoints" / variant
    sc = json.loads((ckdir / "scalers.json").read_text())
    f_sc = L.ColumnScaler.from_dict(sc["feature_scaler"])
    t_sc = L.ColumnScaler.from_dict(sc["target_scaler"])
    model = L.TrajectoryLSTM(L.N_FEAT).to(device)
    model.load_state_dict(torch.load(ckdir / "model_best.pth", map_location=device)); model.eval()

    df = XC.load_df(); split = XC.load_split(); wb = L.recording_world_bounds(df)
    need = XC.OBS + XC.BASE_H
    lens = df.groupby("trajectory_id").size()
    te_ids = [t for t in split[split.split == "test"].trajectory_id if lens.get(t, 0) >= need]
    preds, gt, rec, tid = XC.rollout_all(model, df, te_ids, wb, f_sc, t_sc, XC.BASE_H, device)
    d = XC.enrich(preds, gt, rec, tid, XC.BASE_H)

    res = {"variant": variant, "horizon": XC.BASE_H, "lambda_mag": cfg["lambda_mag"],
           "lambda_dir": cfg["lambda_dir"], "ALL": XC.agg_block(d),
           "per_recording": {r: XC.agg_block(d[d.recording == r]) for r in XC.RECS
                             if (d.recording == r).any()}}
    expd = HERE / "experiments" / variant; expd.mkdir(parents=True, exist_ok=True)
    (expd / "eval_base.json").write_text(json.dumps(res, indent=2))
    a = res["ALL"]
    print(f"[{variant}] H{XC.BASE_H}: ADE={a['ADE']:.4f} FDE={a['FDE']:.4f} "
          f"step1={a['step1_ratio_med']:.3f} step1_sm={a['step1_sm_ratio_med']:.3f} "
          f"nd_ratio={a['nd_ratio_med']:.3f} cos={a['cosine_med']:.3f}")


if __name__ == "__main__":
    main()
