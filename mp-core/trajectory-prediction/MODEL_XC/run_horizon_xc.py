"""
run_horizon_xc.py --config configs/<variant>.json
Evaluate one trained variant at H20/H60/H100/H200/H400 (magnitude+direction+shape).
One trained model evaluated at multiple rollout depths (same convention as MODEL_XR — isolates
the loss effect; horizon definitions unchanged). Writes experiments/<variant>/horizon.json.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xc_common as XCC  # noqa
L = XCC.L
HORIZONS = [20, 60, 100, 200, 400]
APPROX = {20: 1.0, 60: 3.0, 100: 5.0, 200: 10.0, 400: 20.0}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = XCC.load_config(ap.parse_args().config); variant = cfg["variant"]
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = HERE / "checkpoints" / variant
    sc = json.loads((ck / "scalers.json").read_text())
    f_sc = L.ColumnScaler.from_dict(sc["feature_scaler"]); t_sc = L.ColumnScaler.from_dict(sc["target_scaler"])
    model = L.TrajectoryLSTM(L.N_FEAT).to(device)
    model.load_state_dict(torch.load(ck / "model_best.pth", map_location=device)); model.eval()
    df = XCC.load_df(); split = XCC.load_split(); wb = L.recording_world_bounds(df)
    lens = df.groupby("trajectory_id").size()
    out = {"variant": variant, "horizons": {}}
    for H in HORIZONS:
        need = XCC.OBS + H
        te_ids = [t for t in split[split.split == "test"].trajectory_id if lens.get(t, 0) >= need]
        preds, gt, rec, tid = XCC.XC.rollout_all(model, df, te_ids, wb, f_sc, t_sc, H, device)
        d = XCC.enrich_full(preds, gt, rec, tid, H)
        out["horizons"][str(H)] = {"approx_dist_m": APPROX[H], "ALL": XCC.agg_full(d),
                                   "per_recording": {r: XCC.agg_full(d[d.recording == r]) for r in XCC.RECS if (d.recording == r).any()}}
        a = out["horizons"][str(H)]["ALL"]
        print(f"[{variant} H{H}] ADE={a['ADE']:.3f} nd={a['nd_ratio_med']:.3f} cos={a['cosine_med']:.3f} shape={a['shape_ratio_med']:.3f}")
    expd = HERE / "experiments" / variant; expd.mkdir(parents=True, exist_ok=True)
    (expd / "horizon.json").write_text(json.dumps(out, indent=2))
    print(f"[{variant}] horizon sweep done")


if __name__ == "__main__":
    main()
