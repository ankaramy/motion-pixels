"""
evaluate_model_xc.py --config configs/<variant>.json
Base-H10 eval: magnitude + direction + curvature/shape, overall + per recording.
Writes experiments/<variant>/eval_base.json. Inference only.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xc_common as XCC  # noqa
L = XCC.L


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
    need = XCC.OBS + XCC.BASE_H; lens = df.groupby("trajectory_id").size()
    te_ids = [t for t in split[split.split == "test"].trajectory_id if lens.get(t, 0) >= need]
    preds, gt, rec, tid = XCC.XC.rollout_all(model, df, te_ids, wb, f_sc, t_sc, XCC.BASE_H, device)
    d = XCC.enrich_full(preds, gt, rec, tid, XCC.BASE_H)
    res = {"variant": variant, "horizon": XCC.BASE_H, **{k: cfg[k] for k in ("lambda_mag", "lambda_curv")},
           "lambda_dir": cfg.get("lambda_dir", 0.0), "ALL": XCC.agg_full(d),
           "per_recording": {r: XCC.agg_full(d[d.recording == r]) for r in XCC.RECS if (d.recording == r).any()},
           "topology": {"OPEN": XCC.agg_full(d[d.recording.isin(XCC.OPEN)]),
                        "CONSTRAINED": XCC.agg_full(d[d.recording.isin(XCC.CONSTR)])}}
    expd = HERE / "experiments" / variant; expd.mkdir(parents=True, exist_ok=True)
    (expd / "eval_base.json").write_text(json.dumps(res, indent=2))
    a = res["ALL"]
    print(f"[{variant}] H10 ADE={a['ADE']:.4f} step1_sm={a['step1_ratio_med']:.3f} "
          f"nd={a['nd_ratio_med']:.3f} cos={a['cosine_med']:.3f} shape={a['shape_ratio_med']:.3f}")


if __name__ == "__main__":
    main()
