"""
run_horizon_xr.py --config configs/<variant>.json

Evaluates ONE trained MODEL_XR variant at rollout depths H20/H60/H100/H200/H400 (jitter-aware
magnitude + direction). NOTE (documented in README): unlike MODEL_X_HORIZON_SWEEP — which trained a
separate model per horizon — MODEL_XR trains ONE model per variant and evaluates it at several
rollout depths. This isolates the loss-function effect (the whole point of the controlled experiment)
and is much cheaper. Horizon DEFINITIONS are unchanged (track filter len>=obs+H, rollout depth=H).

Writes experiments/<variant>/horizon.json. Inference only.
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
HORIZONS = [20, 60, 100, 200, 400]
APPROX = {20: 1.0, 60: 3.0, 100: 5.0, 200: 10.0, 400: 20.0}


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
    lens = df.groupby("trajectory_id").size()
    out = {"variant": variant, "horizons": {}}
    for H in HORIZONS:
        need = XC.OBS + H
        te_ids = [t for t in split[split.split == "test"].trajectory_id if lens.get(t, 0) >= need]
        preds, gt, rec, tid = XC.rollout_all(model, df, te_ids, wb, f_sc, t_sc, H, device)
        d = XC.enrich(preds, gt, rec, tid, H)
        out["horizons"][str(H)] = {
            "approx_dist_m": APPROX[H], "ALL": XC.agg_block(d),
            "per_recording": {r: XC.agg_block(d[d.recording == r]) for r in XC.RECS
                             if (d.recording == r).any()}}
        a = out["horizons"][str(H)]["ALL"]
        print(f"[{variant} H{H}] ADE={a['ADE']:.3f} nd_ratio={a['nd_ratio_med']:.3f} "
              f"sm_ratio={a['sm_ratio_med']:.3f} cum_ratio={a['cum_ratio_med']:.3f} cos={a['cosine_med']:.3f}")
    expd = HERE / "experiments" / variant; expd.mkdir(parents=True, exist_ok=True)
    (expd / "horizon.json").write_text(json.dumps(out, indent=2))
    print(f"[{variant}] horizon sweep done")


if __name__ == "__main__":
    main()
