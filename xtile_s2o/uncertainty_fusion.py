"""Monte Carlo prediction and tile-recomposition utilities.

The functions operate on batches supplied by their caller. For CPA-enabled K=8
inference, generate each stochastic realization with ``blockgroup_infer.py``
using the ordered full-scene cover, then apply the fusion analysis to those
block-context outputs. A tile-wise caller does not activate CPA and should be
used only for the corresponding diagnostic comparison. The exact commands and
seed schedule are documented in ``docs/REPRODUCE.md``.

(a) mc_predict: run the (stochastic) generator K times on the same SAR input and
    return per-pixel mean mu and variance var. For the JiT flow-matching sampler
    the stochasticity is the initial noise, so K independent calls give K samples.

(b) fuse_scene: place per-tile (mu, var) on the scene canvas and fuse overlapping
    tiles. Precision weighting w_i = m_i / (sigma^2_i + eps) with a parameter-free
    MAD gate m_i that drops tiles whose tile-mean variance is a global outlier
    (high-variance tiles). Five fusion modes are available:
       hard     : no blending (each pixel from the nearest-center tile)
       average  : uniform mean over covering tiles (w_i = 1)
       feather  : cosine edge-taper position weights (classic seam blend)
       ad_hard  : MAD-reject anomalous tiles, then average survivors
       ucgated  : per-pixel precision weighting (soft)
    Returns the fused reflectance-space canvas and, for ucgated, the fused
    variance sigma^2(p) = 1 / sum_i w_i for uncertainty-aware post-translation review.

"""
import math

import torch


# ----------------------------------------------------------------------------- MC
def mc_predict(generate_fn, sar, K=8):
    """generate_fn(sar) -> (B,C,H,W) in [-1,1], stochastic across calls.
    Returns mu (B,C,H,W), var (B,C,H,W) over K samples (population variance)."""
    if K < 1:
        raise ValueError("K must be >= 1")
    samples = [generate_fn(sar) for _ in range(K)]
    stack = torch.stack(samples, dim=0)              # (K,B,C,H,W)
    mu = stack.mean(dim=0)
    var = stack.var(dim=0, unbiased=False) if K > 1 else torch.zeros_like(mu)
    return mu, var


# --------------------------------------------------------------------------- utils
def _edge_weight(h, w, device="cpu", dtype=torch.float32, eps=1e-3):
    """Separable cosine (Hann-like) taper: ~1 at center, ~eps at borders."""
    ys = torch.sin(torch.linspace(0.0, math.pi, h, device=device, dtype=dtype)).clamp_min(eps)
    xs = torch.sin(torch.linspace(0.0, math.pi, w, device=device, dtype=dtype)).clamp_min(eps)
    return ys[:, None] * xs[None, :]                  # (h,w)


def _mad_gate(tiles, k=3.0):
    """Tile-level robust outlier gate on tile-mean variance.
    Returns gate (N,) in {0,1}, the per-tile scalar uncertainty u (N,), threshold."""
    u = torch.stack([t["var"].mean() for t in tiles])           # (N,)
    med = u.median()
    mad = (u - med).abs().median()
    thr = med + k * 1.4826 * mad                                  # 1.4826: MAD->sigma
    gate = (u <= thr).float()
    return gate, u, thr


# ------------------------------------------------------------------------- fusion
def fuse_scene(tiles, canvas_hw, mode="ucgated", eps=1e-6, mad_k=3.0,
               feather_in_ucgated=False):
    """tiles: list of dict(mu=(C,th,tw), var=(C,th,tw)|None, x=int, y=int).
    Returns (fused (C,H,W), fused_var (1,H,W)|None, info dict).
    Pixels covered by no (surviving) tile are left 0 and flagged in info['uncovered'].
    """
    assert len(tiles) > 0
    H, W = canvas_hw
    C = tiles[0]["mu"].shape[0]
    dev = tiles[0]["mu"].device
    dt = tiles[0]["mu"].dtype

    if mode == "ad_hard":
        if any(t.get("var") is None for t in tiles):
            raise ValueError(f"mode '{mode}' needs per-tile variance")
        gate, u, thr = _mad_gate(tiles, mad_k)
        info = {"mode": mode, "gate": gate, "u": u, "thr": thr,
                "n_dropped": int((gate == 0).sum())}
    elif mode == "ucgated":
        # Soft precision weighting only: a high-variance tile is
        # down-weighted by its high variance, never hard-dropped -> no holes.
        if any(t.get("var") is None for t in tiles):
            raise ValueError(f"mode '{mode}' needs per-tile variance")
        gate = torch.ones(len(tiles), device=dev, dtype=dt)
        info = {"mode": mode, "n_dropped": 0}
    else:
        gate = torch.ones(len(tiles), device=dev, dtype=dt)
        info = {"mode": mode, "n_dropped": 0}

    # --- hard: nearest-center assignment, no blending
    if mode == "hard":
        out = torch.zeros(C, H, W, device=dev, dtype=dt)
        best = torch.full((1, H, W), float("inf"), device=dev, dtype=dt)
        ys = torch.arange(H, device=dev, dtype=dt)
        xs = torch.arange(W, device=dev, dtype=dt)
        for t in tiles:
            mu, x, y = t["mu"], int(t["x"]), int(t["y"])
            _, th, tw = mu.shape
            cy, cx = y + (th - 1) / 2.0, x + (tw - 1) / 2.0
            region = torch.full((1, H, W), float("inf"), device=dev, dtype=dt)
            d = (ys[:, None] - cy) ** 2 + (xs[None, :] - cx) ** 2
            region[0, y:y + th, x:x + tw] = d[y:y + th, x:x + tw]
            upd = region < best
            best = torch.where(upd, region, best)
            full = torch.zeros(C, H, W, device=dev, dtype=dt)
            full[:, y:y + th, x:x + tw] = mu
            out = torch.where(upd.expand(C, H, W), full, out)
        info["uncovered"] = int((best == float("inf")).sum())
        return out, None, info

    # --- soft modes: weighted accumulation
    num = torch.zeros(C, H, W, device=dev, dtype=dt)
    den = torch.zeros(1, H, W, device=dev, dtype=dt)
    prec = torch.zeros(1, H, W, device=dev, dtype=dt)            # sum of precisions (ucgated)
    for i, t in enumerate(tiles):
        mu, x, y = t["mu"], int(t["x"]), int(t["y"])
        _, th, tw = mu.shape
        g = gate[i]
        if mode == "average" or mode == "ad_hard":
            w = torch.full((1, th, tw), 1.0, device=dev, dtype=dt) * g
        elif mode == "feather":
            w = _edge_weight(th, tw, dev, dt)[None] * g
        elif mode == "ucgated":
            var_s = t["var"].mean(dim=0, keepdim=True)            # (1,th,tw) channel-mean sigma^2
            w = g / (var_s + eps)
            if feather_in_ucgated:
                w = w * _edge_weight(th, tw, dev, dt)[None]
        else:
            raise ValueError(f"unknown mode {mode}")
        num[:, y:y + th, x:x + tw] += w * mu
        den[:, y:y + th, x:x + tw] += w
        if mode == "ucgated":
            prec[:, y:y + th, x:x + tw] += w

    covered = den > eps
    fused = num / den.clamp_min(eps)
    fused_var = (1.0 / prec.clamp_min(eps)) if mode == "ucgated" else None
    info["uncovered"] = int((~covered).sum())
    return fused, fused_var, info
