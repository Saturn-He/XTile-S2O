# -*- coding: utf-8 -*-
"""Scene-level seam diagnostics on the saved hard-tiling mosaics (fs_h0 / fs_hb):
1) decompose GDS: absolute seam-line gradient (numerator) vs interior gradient
   (denominator) separately -> is Hb's higher GDS ratio a smooth-interior artifact?
2) DINOv2 seam distance at scene scale: 448-px windows centered on seam lines,
   token cosine distance across the seam; pred vs GT delta. Extractor = dinov2-base
   (NOT the DINOv3 teacher)."""
import argparse
import csv
import glob
import os
import re

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

PID_RE = re.compile(r'^(?P<city>.+?)_r\d+_c\d+_x(?P<x>\d+)_y(?P<y>\d+)_s(?P<s>\d+)_SAR$')
EXCLUDE = {0, 1, 5}
OPT_MAX = 4500.0
_IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def tile_layout(sar_dir, csv_path, city):
    lm = {}
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            try:
                lm[row['patch_id']] = (row['split'].strip().lower(), int(row['coarse_label']))
            except (KeyError, ValueError):
                pass
    origins = []
    size = 256
    for p in sorted(glob.glob(os.path.join(sar_dir, city + '_*_SAR.tif'))):
        stem = os.path.splitext(os.path.basename(p))[0]
        m = PID_RE.match(stem)
        if not m:
            continue
        pid = stem[:-4]
        meta = lm.get(pid)
        if meta is None or meta[0] != 'test' or meta[1] in EXCLUDE:
            continue
        origins.append((int(m.group('x')), int(m.group('y'))))
    if not origins:
        raise RuntimeError(f"{city}: no eligible test tiles found")
    W = max(x for x, _ in origins) + size
    H = max(y for _, y in origins) + size

    # Hard recomposition changes ownership at the nearest-center boundary between
    # available tiles. Deriving the boundary from the ownership map also handles
    # clamped final rows/columns whose displacement is not the nominal stride.
    owner = np.full((H, W), -1, dtype=np.int32)
    owner_dist = np.full((H, W), np.inf, dtype=np.float32)
    for idx, (x, y) in enumerate(origins):
        yy, xx = np.mgrid[y:y + size, x:x + size]
        d2 = (xx - (x + size / 2.0)) ** 2 + (yy - (y + size / 2.0)) ** 2
        sl = np.s_[y:y + size, x:x + size]
        closer = d2 < owner_dist[sl]
        owner_dist[sl][closer] = d2[closer]
        owner[sl][closer] = idx
    return H, W, owner


def grad_decompose(gray, owner):
    dx = np.abs(gray[:, 1:] - gray[:, :-1])
    dy = np.abs(gray[1:, :] - gray[:-1, :])
    valid = owner >= 0
    valid_v = valid[:, 1:] & valid[:, :-1]
    valid_h = valid[1:, :] & valid[:-1, :]
    seam_v = valid_v & (owner[:, 1:] != owner[:, :-1])
    seam_h = valid_h & (owner[1:, :] != owner[:-1, :])
    seam_vals = np.concatenate((dx[seam_v], dy[seam_h]))
    interior_vals = np.concatenate((dx[valid_v & ~seam_v], dy[valid_h & ~seam_h]))
    if not len(seam_vals) or not len(interior_vals):
        raise RuntimeError("empty ownership-seam or interior support")
    num = float(seam_vals.mean())
    den = float(interior_vals.mean())
    return num, den, num / (den + 1e-8)


class DinoSeam:
    def __init__(self, device='cuda'):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained('facebook/dinov2-base').eval().requires_grad_(False).to(device)
        self.device = device
        self.mean = _IMNET_MEAN.to(device); self.std = _IMNET_STD.to(device)

    @torch.no_grad()
    def window_seam(self, win_refl):
        """win_refl (448,448,3) reflectance, seam at the vertical center line.
        Returns mean cosine distance between the two token columns straddling it."""
        x = torch.from_numpy((win_refl / OPT_MAX).astype('float32')).permute(2, 0, 1)[None].to(self.device)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        h = self.m(pixel_values=x).last_hidden_state[0, 1:, :].reshape(16, 16, -1)
        t = F.normalize(h.float(), dim=-1)
        c = 8                                        # 224/14=16 tokens; seam at col 8
        return float((1 - (t[:, c - 1, :] * t[:, c, :]).sum(-1)).mean())


def seam_windows(img, owner, step=384, half=224):
    """Yield (448,448,3) windows centered on seams; vertical seams as-is,
    horizontal seams transposed so the seam is always vertical-center."""
    H, W = img.shape[:2]
    valid = owner >= 0
    seam_v = valid[:, 1:] & valid[:, :-1] & (owner[:, 1:] != owner[:, :-1])
    seam_h = valid[1:, :] & valid[:-1, :] & (owner[1:, :] != owner[:-1, :])
    seam_x = np.flatnonzero(seam_v.any(axis=0)) + 1
    seam_y = np.flatnonzero(seam_h.any(axis=1)) + 1
    for c in seam_x:
        if half <= c <= W - half:
            for y0 in range(0, H - 2 * half + 1, step):
                yield img[y0:y0 + 2 * half, c - half:c + half]
    for r in seam_y:
        if half <= r <= H - half:
            for x0 in range(0, W - 2 * half + 1, step):
                yield np.transpose(img[r - half:r + half, x0:x0 + 2 * half], (1, 0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fs_dirs', type=str, required=True, help='name:dir pairs comma-separated')
    ap.add_argument('--sar_dir', required=True)
    ap.add_argument('--csv_path', required=True)
    ap.add_argument('--cities', required=True)
    ap.add_argument('--fusion', default='hard')
    ap.add_argument('--out_csv', required=True)
    args = ap.parse_args()

    dino = DinoSeam()
    cities = [c.strip() for c in args.cities.split(',')]
    rows = []
    gt_cache = {}
    for spec in args.fs_dirs.split(','):
        name, d = spec.split(':', 1)
        for city in cities:
            H, W, owner = tile_layout(args.sar_dir, args.csv_path, city)
            pr = tifffile.imread(os.path.join(d, city, f'{args.fusion}_refl.tif')).astype('float32')
            if pr.shape[:2] != (H, W):
                raise RuntimeError(f"{name}/{city}: mosaic shape {pr.shape[:2]} != geometry {(H, W)}")
            num, den, ratio = grad_decompose(pr.mean(-1), owner)
            ds = [dino.window_seam(w) for w in seam_windows(pr, owner)]
            if city not in gt_cache:
                g = tifffile.imread(os.path.join(d, city, 'GT_refl.tif')).astype('float32')
                if g.shape[:2] != (H, W):
                    raise RuntimeError(f"GT/{city}: mosaic shape {g.shape[:2]} != geometry {(H, W)}")
                gn, gd, gr = grad_decompose(g.mean(-1), owner)
                gds_ = [dino.window_seam(w) for w in seam_windows(g, owner)]
                gt_cache[city] = (gn, gd, gr, float(np.mean(gds_)))
            gn, gd, gr, gseam = gt_cache[city]
            rows.append(dict(model=name, city=city,
                             seam_grad=num, interior_grad=den, GDS=ratio,
                             seam_grad_gt=gn, interior_grad_gt=gd, GDS_gt=gr,
                             dinoseam=float(np.mean(ds)), dinoseam_gt=gseam,
                             dinoseam_delta=float(np.mean(ds)) - gseam,
                             n_windows=len(ds)))
            print(f"[{name}/{city}] seam_grad={num:.1f} interior={den:.1f} GDS={ratio:.3f} "
                  f"(GT {gn:.1f}/{gd:.1f}/{gr:.3f}) dinoseam={np.mean(ds):.4f} (GT {gseam:.4f})", flush=True)

    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("SCENE_SEAM_DONE", flush=True)


if __name__ == '__main__':
    main()
