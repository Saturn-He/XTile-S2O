"""
Differentiable geometry-matched losses for 2x2 tile blocks.

Base-agnostic pure-tensor functions. The training code calls these on the
network's clean-image estimate x_hat0 (the x-prediction the JiT denoiser already
computes) for a block batch of shape (G, 4, C, 256, 256), quadrant order
TL,TR,BL,BR.

Two terms:
  L_block   : assemble the 4 tiles into the 416x416 canvas (averaging overlaps,
              matching the naive inference fusion) and L1 against the GT mosaic
              assembled identically from the 4 optical tiles.
  L_overlap : GT-free L1 agreement on the 96-px shared strips of adjacent tiles
              (4 edge pairs + 2 diagonal corner pairs).

t-weighting (decision 3 = linear in t): x_hat0 is unreliable at small t, so the
block terms are weighted proportional to t. Pass a per-block t in [0,1]. A hard
threshold `block_t_min` is kept as a fallback gate.

Geometry: tile 256, stride 160 -> overlap 96, canvas 416. Offsets (x,y):
TL=(0,0) TR=(160,0) BL=(0,160) BR=(160,160).
"""
import torch
import torch.nn.functional as F

DEFAULT_OFFSETS = ((0, 0), (160, 0), (0, 160), (160, 160))  # (x, y) per quadrant
DEFAULT_CANVAS = (416, 416)
DEFAULT_OVERLAP = 96
TILE = 256
STRIDE = 160


def assemble_mosaic(tiles, offsets=DEFAULT_OFFSETS, canvas_hw=DEFAULT_CANVAS):
    """tiles (G,K,C,th,tw) -> mosaic (G,C,H,W), averaging overlapping regions.
    Differentiable (slice-add into a fresh canvas)."""
    G, K, C, th, tw = tiles.shape
    H, W = canvas_hw
    canvas = tiles.new_zeros(G, C, H, W)
    wsum = tiles.new_zeros(G, 1, H, W)
    for k in range(K):
        ox, oy = int(offsets[k][0]), int(offsets[k][1])
        canvas[:, :, oy:oy + th, ox:ox + tw] = canvas[:, :, oy:oy + th, ox:ox + tw] + tiles[:, k]
        wsum[:, :, oy:oy + th, ox:ox + tw] = wsum[:, :, oy:oy + th, ox:ox + tw] + 1.0
    return canvas / wsum.clamp_min(1.0)


def _per_block_l1(a, b):
    """L1 reduced over all dims except batch -> (G,)."""
    return (a - b).abs().flatten(1).mean(dim=1)


def block_mosaic_loss(pred_tiles, gt_tiles, offsets=DEFAULT_OFFSETS,
                      canvas_hw=DEFAULT_CANVAS):
    """Per-block L1 between the assembled prediction mosaic and GT mosaic -> (G,)."""
    pred_m = assemble_mosaic(pred_tiles, offsets, canvas_hw)
    gt_m = assemble_mosaic(gt_tiles, offsets, canvas_hw)
    return _per_block_l1(pred_m, gt_m)


def overlap_consistency_loss(pred_tiles, overlap=DEFAULT_OVERLAP):
    """GT-free per-block L1 over shared strips of adjacent tiles -> (G,).
    Quadrant order TL,TR,BL,BR. With tile=256, stride=160 the overlap is 96.
      edges: (TL,TR)&(BL,BR) share cols [160:256]~[0:96] (full rows);
             (TL,BL)&(TR,BR) share rows [160:256]~[0:96] (full cols);
      diag : (TL,BR) & (TR,BL) share the central 96x96 corner.
    """
    o = overlap
    s = TILE - o  # 160
    TL, TR, BL, BR = pred_tiles[:, 0], pred_tiles[:, 1], pred_tiles[:, 2], pred_tiles[:, 3]
    terms = [
        _per_block_l1(TL[..., :, s:TILE], TR[..., :, 0:o]),      # horizontal top
        _per_block_l1(BL[..., :, s:TILE], BR[..., :, 0:o]),      # horizontal bottom
        _per_block_l1(TL[..., s:TILE, :], BL[..., 0:o, :]),      # vertical left
        _per_block_l1(TR[..., s:TILE, :], BR[..., 0:o, :]),      # vertical right
        _per_block_l1(TL[..., s:TILE, s:TILE], BR[..., 0:o, 0:o]),  # diag TL-BR
        _per_block_l1(TR[..., s:TILE, 0:o], BL[..., 0:o, s:TILE]),  # diag TR-BL
    ]
    return torch.stack(terms, dim=0).mean(dim=0)  # (G,)


def _t_weights(t, n_blocks, device, block_t_min=0.0, linear=True):
    """Per-block weights from the flow-matching time t in [0,1]."""
    if t is None:
        return torch.ones(n_blocks, device=device)
    t = t.reshape(-1).to(device).float()
    if t.numel() != n_blocks:  # e.g. per-tile t (4G,) -> per-block mean
        if t.numel() % n_blocks == 0:
            t = t.view(n_blocks, -1).mean(dim=1)
        else:
            t = t[:n_blocks]
    if block_t_min > 0.0:                 # fallback: hard gate
        return (t >= block_t_min).float()
    return t if linear else torch.ones_like(t)


def compute_block_losses(xhat0, opt, t=None, lambda_block=1.0, lambda_overlap=1.0,
                         offsets=DEFAULT_OFFSETS, canvas_hw=DEFAULT_CANVAS,
                         overlap=DEFAULT_OVERLAP, block_t_min=0.0, t_linear=True):
    """Main entry.
    xhat0 : (G,4,C,256,256) clean-image estimate per tile (x-prediction)
    opt   : (G,4,C,256,256) GT optical tiles
    t     : per-block (G,) or per-tile (4G,) flow-matching time in [0,1], or None
    returns dict(L_block, L_overlap, L_total) as scalars (already t-weighted+mean).
    """
    assert xhat0.dim() == 5 and xhat0.shape[1] == 4, f"expected (G,4,C,H,W), got {tuple(xhat0.shape)}"
    G = xhat0.shape[0]
    w = _t_weights(t, G, xhat0.device, block_t_min=block_t_min, linear=t_linear)
    wn = w.sum().clamp_min(1e-6)

    lb = block_mosaic_loss(xhat0, opt, offsets, canvas_hw)        # (G,)
    lo = overlap_consistency_loss(xhat0, overlap)                 # (G,)
    L_block = (w * lb).sum() / wn
    L_overlap = (w * lo).sum() / wn
    L_total = lambda_block * L_block + lambda_overlap * L_overlap
    return {"L_block": L_block, "L_overlap": L_overlap, "L_total": L_total}
