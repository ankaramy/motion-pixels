"""
losses.py — MODEL_XR composite training loss.

Design (see MODEL_XR/README.md):
  loss = MSE(pred_scaled, gt_scaled)                         # base — IDENTICAL to MODEL_X
       + lambda_mag * MSE(||pred_m||/sigma, ||gt_m||/sigma)  # magnitude term (metric, isotropic-normed)
       + lambda_dir * mean(1 - cos(pred_m, gt_m))            # direction term (true metric cosine)

Why this split:
  - MODEL_X trains on per-axis StandardScaler-scaled targets; std_du/std_dv ~= 3.3, so a cosine
    computed in scaled space is angle-distorted. The base MSE is therefore kept in SCALED space
    (so lambda_mag=lambda_dir=0 reproduces MODEL_X exactly), while the magnitude and direction
    terms are computed in UNSCALED metric space (du,dv in metres).
  - The magnitude term is normalised by sigma = RMS metric step size of the train targets, so it is
    O(1) and the suggested lambdas (1.0 / 2.0) are comparable to the O(1) scaled-MSE base.
  - Isotropic /sigma scaling does NOT change the cosine, so the direction term is the true metric angle.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def composite_loss(pred_s: torch.Tensor, gt_s: torch.Tensor,
                   tgt_std: torch.Tensor, tgt_mean: torch.Tensor, sigma_iso: float,
                   lambda_mag: float = 0.0, lambda_dir: float = 0.0):
    """pred_s/gt_s: (B,2) scaled displacement. tgt_std/tgt_mean: (2,) target scaler params.
    Returns (loss, parts_dict)."""
    base = F.mse_loss(pred_s, gt_s)
    parts = {"mse": float(base.detach())}
    loss = base
    if lambda_mag != 0.0 or lambda_dir != 0.0:
        pred_m = pred_s * tgt_std + tgt_mean          # metric displacement (m)
        gt_m = gt_s * tgt_std + tgt_mean
        if lambda_mag != 0.0:
            pn = torch.linalg.norm(pred_m, dim=1) / sigma_iso
            gn = torch.linalg.norm(gt_m, dim=1) / sigma_iso
            mag = F.mse_loss(pn, gn)
            loss = loss + lambda_mag * mag
            parts["mag"] = float(mag.detach())
        if lambda_dir != 0.0:
            cos = F.cosine_similarity(pred_m, gt_m, dim=1, eps=1e-8)
            d = (1.0 - cos).mean()
            loss = loss + lambda_dir * d
            parts["dir"] = float(d.detach())
            parts["cos"] = float(cos.mean().detach())
    parts["total"] = float(loss.detach())
    return loss, parts
