"""Shared normalization utilities for training, inference, and evaluation.

All released data loaders and export scripts import these transforms so that
normalization, inverse normalization, and label filtering remain consistent.

Released corpus contract: band 0 = VV and band 1 = VH. External inputs must be
reordered to this convention before normalization. SAR backscatter is clipped
per channel and mapped linearly to [-1, 1]. Three-band optical reflectance is
clipped to [0, 4500] and mapped to the model target domain.
"""
import numpy as np

SAR_CLIP = [(-20.0, 2.0),    # band 0 = VV (p2 -19.5, p98 +1.8)
            (-28.0, -6.0)]   # band 1 = VH (p2 -27.8, p98 -6.2)
OPT_CLIP = (0.0, 4500.0)     # Shared by all three optical bands (p98 ~4500)
EXCLUDE_LABELS = {'0', '1', '5'}   # invalid / snow / cloud


def to_chw(arr):
    """Convert (H,W,C) or (C,H,W) to (C,H,W); add an axis to 2D input."""
    a = np.asarray(arr)
    if a.ndim == 2:
        return a[None, ...]
    if a.ndim == 3:
        c_axis = int(np.argmin(a.shape))   # The channel axis has length 2 or 3.
        return np.moveaxis(a, c_axis, 0)
    raise ValueError(f"unexpected array shape: {a.shape}")


def _fwd(channel, lo, hi):
    c = np.clip(channel, lo, hi)
    return (c - lo) / (hi - lo) * 2.0 - 1.0


def _inv(channel, lo, hi):
    c = np.clip(channel, -1.0, 1.0)
    return (c + 1.0) / 2.0 * (hi - lo) + lo


# ---------------- Forward transforms ----------------
def normalize_sar(arr):
    """Map raw SAR in either supported layout to (2,H,W) float32 in [-1,1]."""
    a = to_chw(arr).astype(np.float32)
    a = np.nan_to_num(a, nan=SAR_CLIP[0][0], posinf=SAR_CLIP[0][1], neginf=SAR_CLIP[0][0])
    return np.stack([_fwd(a[c], *SAR_CLIP[c]) for c in range(a.shape[0])], axis=0)


def normalize_opt(arr):
    """Map raw uint16 optical data to (3,H,W) float32 in [-1,1]."""
    a = to_chw(arr).astype(np.float32)
    a = np.nan_to_num(a, nan=OPT_CLIP[0], posinf=OPT_CLIP[1], neginf=OPT_CLIP[0])
    return np.stack([_fwd(a[c], *OPT_CLIP) for c in range(a.shape[0])], axis=0)


# ---------------- Inverse transforms ----------------
def denormalize_opt(t, layout='HWC'):
    """
    Convert model output in [-1,1] to uint16 optical reflectance.

    ``t`` may be a NumPy array or torch tensor with shape (3,H,W) or
    (1,3,H,W). Return (H,W,3) for ``layout='HWC'`` and (3,H,W) for
    ``layout='CHW'``.
    """
    a = np.asarray(t.detach().cpu()) if hasattr(t, 'detach') else np.asarray(t)
    if a.ndim == 4:
        a = a[0]
    refl = _inv(a, *OPT_CLIP)                       # (3,H,W) float
    refl = np.clip(np.round(refl), OPT_CLIP[0], OPT_CLIP[1]).astype(np.uint16)
    return np.transpose(refl, (1, 2, 0)) if layout == 'HWC' else refl


def clip_gt_opt(arr):
    """Clip reference optical data to the model's [0,4500] output range."""
    a = to_chw(arr).astype(np.float32)
    a = np.clip(a, OPT_CLIP[0], OPT_CLIP[1]).astype(np.uint16)
    return np.transpose(a, (1, 2, 0))


# ============================================================
#  sRGB target variant (--srgb_target).
#  Training: reflectance -> linear [0,1] -> sRGB transfer -> [-1,1].
#  Inference: model output -> inverse sRGB transfer -> uint16 reflectance.
#  The default linear pipeline is unaffected when this option is disabled.
# ============================================================
def _srgb_encode(lin):
    """Encode linear [0,1] values with the IEC 61966-2-1 sRGB transfer."""
    lin = np.clip(lin, 0.0, 1.0)
    a = 0.055
    return np.where(lin <= 0.0031308, 12.92 * lin, (1.0 + a) * np.power(lin, 1.0 / 2.4) - a)


def _srgb_decode(s):
    """Decode IEC 61966-2-1 sRGB [0,1] values to linear [0,1]."""
    s = np.clip(s, 0.0, 1.0)
    a = 0.055
    return np.where(s <= 0.04045, s / 12.92, np.power((s + a) / (1.0 + a), 2.4))


def denormalize_opt_srgb(t, layout='HWC'):
    """
    Convert an sRGB-target model output in [-1,1] to uint16 reflectance.

    The signature and layouts match :func:`denormalize_opt`; this variant adds
    the inverse sRGB transfer required by ``--srgb_target``.
    """
    a = np.asarray(t.detach().cpu()) if hasattr(t, 'detach') else np.asarray(t)
    if a.ndim == 4:
        a = a[0]
    s = np.clip((a + 1.0) / 2.0, 0.0, 1.0)          # (3,H,W) sRGB [0,1]
    lin = _srgb_decode(s)                            # linear [0,1]
    lo, hi = OPT_CLIP
    refl = np.clip(np.round(lin * (hi - lo) + lo), lo, hi).astype(np.uint16)
    return np.transpose(refl, (1, 2, 0)) if layout == 'HWC' else refl
