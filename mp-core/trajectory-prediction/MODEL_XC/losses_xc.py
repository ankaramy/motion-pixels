"""
losses_xc.py — MODEL_XC composite loss = MODEL_XR magnitude loss + a curvature term.

loss = MSE(pred_scaled, gt_scaled)                          # base (= MODEL_X / control)
     + lambda_mag  * MSE(‖pred_m‖/σ, ‖gt_m‖/σ)             # magnitude (= MODEL_XR_B_MAG)
     + lambda_curv * MSE(Δθ_pred, Δθ_gt)   [moving mask]    # curvature (NEW)
     + lambda_dir  * mean(1 - cos(pred_m, gt_m))            # optional light direction

Curvature term (single-step, per Phase-5A denoising philosophy):
  v_prev = smoothed incoming displacement (mean of last 5 observed du,dv) — robust heading in.
  Δθ_pred = wrap(atan2(pred_m) - atan2(v_prev));  Δθ_gt = wrap(atan2(gt_m) - atan2(v_prev)).
  Only windows where ‖v_prev‖ and ‖gt_m‖ exceed STEP_EPS contribute (mask out near-stationary
  jitter). Wrapped angle = atan2(sin,cos) (differentiable). This penalises the model for NOT
  turning when GT turns — directly counters B_MAG's immediate straightening.

Limitation (documented): curvature is enforced at the single STEP level (instantaneous turn),
not as a macro multi-step rollout target, because training is single-step. Masking + smoothed
v_prev keep it stable; the GT one-step turn still carries some jitter, mitigated by the mask.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

STEP_EPS = 0.02  # m; min raw step magnitude for a defined turn


def _wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def composite_loss(pred_s, gt_s, v_prev, tgt_std, tgt_mean, sigma_iso,
                   lambda_mag=0.0, lambda_curv=0.0, lambda_dir=0.0):
    """pred_s/gt_s (B,2) scaled; v_prev (B,2) raw smoothed incoming displacement."""
    base = F.mse_loss(pred_s, gt_s)
    parts = {"mse": float(base.detach())}
    loss = base
    pred_m = pred_s * tgt_std + tgt_mean
    gt_m = gt_s * tgt_std + tgt_mean
    if lambda_mag != 0.0:
        pn = torch.linalg.norm(pred_m, dim=1) / sigma_iso
        gn = torch.linalg.norm(gt_m, dim=1) / sigma_iso
        mag = F.mse_loss(pn, gn); loss = loss + lambda_mag * mag
        parts["mag"] = float(mag.detach())
    if lambda_curv != 0.0:
        vp_n = torch.linalg.norm(v_prev, dim=1)
        gt_n = torch.linalg.norm(gt_m, dim=1)
        mask = (vp_n > STEP_EPS) & (gt_n > STEP_EPS)
        th_prev = torch.atan2(v_prev[:, 1], v_prev[:, 0])
        th_pred = torch.atan2(pred_m[:, 1], pred_m[:, 0])
        th_gt = torch.atan2(gt_m[:, 1], gt_m[:, 0])
        d_pred = _wrap(th_pred - th_prev)
        d_gt = _wrap(th_gt - th_prev)
        diff2 = _wrap(d_pred - d_gt) ** 2
        m = mask.float()
        curv = (diff2 * m).sum() / torch.clamp(m.sum(), min=1.0)
        loss = loss + lambda_curv * curv
        parts["curv"] = float(curv.detach())
    if lambda_dir != 0.0:
        cos = F.cosine_similarity(pred_m, gt_m, dim=1, eps=1e-8)
        d = (1.0 - cos).mean(); loss = loss + lambda_dir * d
        parts["dir"] = float(d.detach())
    parts["total"] = float(loss.detach())
    return loss, parts
