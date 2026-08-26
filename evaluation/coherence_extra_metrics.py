# -*- coding: utf-8 -*-
"""Coherence extra metrics on the disjoint test-block subset:
  OI  (overlap inconsistency, PRIMARY): within each 2x2 block, the 4 tiles overlap
      by 96 px; OI = mean |pred_i - pred_j| (reflectance) + mean (1-SSIM) over the
      4 adjacent tile pairs, on their shared 96-px band. Reported for pred and GT
      -> OI_delta = OI_pred - OI_gt (0 = as coherent as GT).
  DINO-seam (SUPPORTING): DINOv2 ViT-B/14 (facebook/dinov2-base, NOT the DINOv3
      teacher -> decircularized) patch-token cosine distance across the mosaic's
      two internal seams, pred vs GT -> seam_delta.
Reads a method's pred_tif (disjoint tiles, <pid>_fakeB.tif) + GT Optical_tif.
"""
import argparse
import csv
import os
import re

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

BLOCK_OFFSETS = {'top_left': (0, 0), 'top_right': (160, 0),
                 'bottom_left': (0, 160), 'bottom_right': (160, 160)}  # (x,y) in 416 canvas
QUAD = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
PID_RC = re.compile(r'_r(\d+)_c(\d+)_')
OPT_CLIP = 4500.0


def load_refl(path):
    a = tifffile.imread(path).astype(np.float32)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
        a = np.transpose(a, (1, 2, 0))
    return np.clip(a[..., :3], 0, OPT_CLIP)


def overlap_pairs(tiles):
    """tiles: dict quad->(256,256,3). Return list of (a_region, b_region) on shared 96px bands."""
    T = 256
    pairs = []
    # horizontal neighbours: TL|TR and BL|BR  (right 96 cols of left tile vs left 96 cols of right)
    for L, R in (('top_left', 'top_right'), ('bottom_left', 'bottom_right')):
        pairs.append((tiles[L][:, T - 96:T, :], tiles[R][:, 0:96, :]))
    # vertical neighbours: TL/BL and TR/BR  (bottom 96 rows of top vs top 96 rows of bottom)
    for U, Dn in (('top_left', 'bottom_left'), ('top_right', 'bottom_right')):
        pairs.append((tiles[U][T - 96:T, :, :], tiles[Dn][0:96, :, :]))
    return pairs


def oi_of(tiles):
    l1s, dssims = [], []
    for a, b in overlap_pairs(tiles):
        l1s.append(np.abs(a - b).mean())
        # 1-SSIM on the overlap band (grayscale mean of channels for stability)
        ga, gb = a.mean(-1), b.mean(-1)
        s = ssim(ga, gb, data_range=OPT_CLIP)
        dssims.append(1.0 - s)
    return float(np.mean(l1s)), float(np.mean(dssims))


def build_mosaic(tiles):
    canvas = np.zeros((416, 416, 3), 'float64')
    wt = np.zeros((416, 416, 1), 'float64')
    for q in QUAD:
        x0, y0 = BLOCK_OFFSETS[q]
        canvas[y0:y0 + 256, x0:x0 + 256] += tiles[q]
        wt[y0:y0 + 256, x0:x0 + 256] += 1.0
    return canvas / np.clip(wt, 1, None), wt[..., 0]      # (mosaic, count map)


def gds_of(mos, cnt, eps=1e-6):
    """GDS (Gradient Discontinuity at Seams), same def as metrics_s2oit.m_gds:
    mean |grad| on overlap pixels (cnt>=2) / mean |grad| on interior (cnt==1)."""
    g = mos.mean(-1)
    gy, gx = np.gradient(g)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ov, it = cnt >= 2, cnt == 1
    if ov.sum() == 0 or it.sum() == 0:
        return float('nan')
    return float(mag[ov].mean() / (mag[it].mean() + eps))


_IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DinoV2Seam:
    """DINOv2 ViT-B/14 dense features; seam = cosine distance between patch-token
    columns/rows straddling the mosaic's internal seams (x=208, y=208)."""
    def __init__(self, device):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained('facebook/dinov2-base').eval().requires_grad_(False).to(device)
        self.device = device
        self.grid = 16          # 224/14
        self.size = 224
        self.mean = _IMNET_MEAN.to(device); self.std = _IMNET_STD.to(device)

    @torch.no_grad()
    def tokens(self, mos_refl):
        x = torch.from_numpy((mos_refl / OPT_CLIP).astype('float32')).permute(2, 0, 1)[None].to(self.device)
        x = F.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        h = self.m(pixel_values=x).last_hidden_state[0, 1:, :]   # (256, D), drop CLS
        g = self.grid
        return h.reshape(g, g, -1)                               # (16,16,D)

    @torch.no_grad()
    def seam(self, mos_refl):
        t = F.normalize(self.tokens(mos_refl).float(), dim=-1)   # (16,16,D)
        g = self.grid
        # internal seam near canvas x=208,y=208 -> token index ~ 208/416*16 = 8
        c = g // 2
        # vertical seam: cosine distance between token columns c-1 and c
        v = (1 - (t[:, c - 1, :] * t[:, c, :]).sum(-1)).mean()
        # horizontal seam: rows c-1 and c
        hh = (1 - (t[c - 1, :, :] * t[c, :, :]).sum(-1)).mean()
        return float((v + hh) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--pred_dir', required=True)
    ap.add_argument('--optical_dir', required=True)
    ap.add_argument('--block_csv', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--seam', action='store_true', help='also compute DINOv2 seam (slower)')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    # index preds + gt
    pred = {}
    for f in os.listdir(args.pred_dir):
        if f.endswith('.tif'):
            pred[f.replace('_fakeB.tif', '').replace('.tif', '')] = os.path.join(args.pred_dir, f)
    gt = {}
    for f in os.listdir(args.optical_dir):
        if f.endswith('.tif'):
            gt[f.replace('_OPT.tif', '').replace('.tif', '')] = os.path.join(args.optical_dir, f)

    # read block rows, disjoint filter (TL even r/c), require all 4 pred+gt present
    rows = []
    with open(args.block_csv, newline='', encoding='utf-8-sig') as fh:
        rd = csv.DictReader(fh)
        cols = {q: next(c for c in rd.fieldnames if q in c.lower() and 'patch' in c.lower()) for q in QUAD}
        splitcol = next((c for c in rd.fieldnames if 'split' in c.lower()), None)
        for r in rd:
            if splitcol and (r.get(splitcol) or '').strip().lower() != args.split:
                continue
            pids = {q: (r.get(cols[q]) or '').strip() for q in QUAD}
            tl = pids['top_left']; m = PID_RC.search(tl + '_')
            if not m or int(m.group(1)) % 2 or int(m.group(2)) % 2:
                continue
            if any(p not in pred or p not in gt for p in pids.values()):
                continue
            rows.append(pids)
    if args.limit:
        rows = rows[:args.limit]
    print(f"[{args.name}] disjoint blocks with full pred+gt: {len(rows)}", flush=True)

    seamer = DinoV2Seam('cuda') if args.seam else None
    acc = {k: [] for k in ('OI_l1', 'OI_ssim', 'OI_l1_gt', 'OI_ssim_gt',
                           'GDS', 'GDS_gt', 'seam', 'seam_gt')}
    for i, pids in enumerate(rows):
        pt = {q: load_refl(pred[pids[q]]) for q in QUAD}
        gtt = {q: load_refl(gt[pids[q]]) for q in QUAD}
        l1, ds = oi_of(pt); l1g, dsg = oi_of(gtt)
        acc['OI_l1'].append(l1); acc['OI_ssim'].append(ds)
        acc['OI_l1_gt'].append(l1g); acc['OI_ssim_gt'].append(dsg)
        mp, cp = build_mosaic(pt); mg, cg = build_mosaic(gtt)
        acc['GDS'].append(gds_of(mp, cp)); acc['GDS_gt'].append(gds_of(mg, cg))
        if seamer is not None:
            acc['seam'].append(seamer.seam(mp))
            acc['seam_gt'].append(seamer.seam(mg))
        if i % 100 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    def mean(k):
        return float(np.mean(acc[k])) if acc[k] else float('nan')
    row = {
        'method': args.name, 'n_blocks': len(rows),
        'OI_l1': mean('OI_l1'), 'OI_l1_gt': mean('OI_l1_gt'),
        'OI_l1_delta': mean('OI_l1') - mean('OI_l1_gt'),
        'OI_dssim': mean('OI_ssim'), 'OI_dssim_gt': mean('OI_ssim_gt'),
        'OI_dssim_delta': mean('OI_ssim') - mean('OI_ssim_gt'),
        'GDS': mean('GDS'), 'GDS_gt': mean('GDS_gt'),
        'GDS_delta': mean('GDS') - mean('GDS_gt'),
        'DINOseam': mean('seam'), 'DINOseam_gt': mean('seam_gt'),
        'DINOseam_delta': (mean('seam') - mean('seam_gt')) if seamer else float('nan'),
    }
    new = not os.path.exists(args.out_csv)
    with open(args.out_csv, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[{args.name}] OI_l1={row['OI_l1']:.2f} (Δ{row['OI_l1_delta']:+.2f}) "
          f"OI_dssim={row['OI_dssim']:.4f} (Δ{row['OI_dssim_delta']:+.4f}) "
          f"GDS={row['GDS']:.3f} (Δ{row['GDS_delta']:+.3f}) "
          f"DINOseam={row['DINOseam']:.4f} (Δ{row['DINOseam_delta']:+.4f})", flush=True)
    print("COH_EXTRA_DONE", flush=True)


if __name__ == '__main__':
    main()
