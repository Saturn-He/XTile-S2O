# -*- coding: utf-8 -*-
"""Digital elevation model (DEM) conditioning with two fusion modes.

``--use_dem concat`` appends elevation and slope to the conditioning carrier,
increasing the model input from five to seven channels. ``--use_dem ca``
encodes elevation and slope as tokens and injects them through zero-gated
cross-attention in selected JiT blocks. These registered modules remain active
during inference, unlike the training-only DINOv3 projectors.

When DEM conditioning is enabled, the ``sar_img`` carrier follows the channel
convention ``[VV, VH, elevation, slope]``; the denoiser splits it as required.
Sampling-path signatures are unchanged. Elevation and slope use fixed
training-split 1st/99th-percentile bounds from ``dem_norm.json`` and are clipped
and mapped linearly to [-1, 1], never normalized per tile.
"""
import json
import os

import numpy as np
import tifffile
import torch
import torch.nn as nn


class DemNorm:
    """Loads train-split p1/p99 constants; maps raw (elev, slope) -> [-1,1]."""

    def __init__(self, json_path):
        with open(json_path) as f:
            c = json.load(f)
        self.elev_lo = float(c['elev_lo']); self.elev_hi = float(c['elev_hi'])
        self.slope_lo = float(c['slope_lo']); self.slope_hi = float(c['slope_hi'])

    def __call__(self, dem):        # dem: tensor/array (2,H,W) raw [elev_m, slope_deg]
        dem = torch.as_tensor(dem, dtype=torch.float32)
        e = (dem[0].clamp(self.elev_lo, self.elev_hi) - self.elev_lo) / max(self.elev_hi - self.elev_lo, 1e-6)
        s = (dem[1].clamp(self.slope_lo, self.slope_hi) - self.slope_lo) / max(self.slope_hi - self.slope_lo, 1e-6)
        return torch.stack([e, s]) * 2.0 - 1.0


def load_dem_pair(dem_split_dir, patch_id):
    """Reads <pid>_DEM.tif + <pid>_SLOPE.tif -> np.float32 (2,H,W) raw."""
    e = tifffile.imread(os.path.join(dem_split_dir, patch_id + "_DEM.tif")).astype(np.float32)
    s = tifffile.imread(os.path.join(dem_split_dir, patch_id + "_SLOPE.tif")).astype(np.float32)
    return np.stack([np.nan_to_num(e), np.nan_to_num(s)])


def _sincos_2d(dim, grid):
    """Fixed 2D sin-cos positional embedding, (grid*grid, dim)."""
    assert dim % 4 == 0
    d4 = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(d4, dtype=torch.float32) / d4))
    pos = torch.arange(grid, dtype=torch.float32)
    out = torch.einsum('p,d->pd', pos, omega)
    emb1d = torch.cat([out.sin(), out.cos()], dim=1)              # (grid, dim/2)
    embx = emb1d[None, :, :].expand(grid, grid, -1)               # rows vary in x
    emby = emb1d[:, None, :].expand(grid, grid, -1)               # cols vary in y
    return torch.cat([emby, embx], dim=-1).reshape(grid * grid, dim)


class DemEncoder(nn.Module):
    """Encode a two-channel DEM input as a 16x16 grid of hidden tokens."""

    def __init__(self, hidden=768, patch=16, in_ch=2, grid=16):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, hidden, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(hidden)
        self.register_buffer('pos', _sincos_2d(hidden, grid), persistent=False)

    def forward(self, dem):                                       # (B,2,H,W) in [-1,1]
        t = self.proj(dem).flatten(2).transpose(1, 2)             # (B,256,hidden)
        return self.norm(t) + self.pos
