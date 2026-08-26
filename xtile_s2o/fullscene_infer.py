# -*- coding: utf-8 -*-
"""Diagnostic tile-wise Monte Carlo sampling and scene recomposition.

This utility samples tiles independently, so it does not activate the four-stream
cross-patch attention (CPA) path. Use ``blockgroup_infer.py`` for the primary
CPA-enabled block-context protocol. The explicit override below is reserved for
diagnostics in which bypassing CPA is intentional.

For each selected test city, the script samples every overlapping 256-pixel tile
``K`` times, estimates the per-pixel mean and variance, and recomposes the city
canvas using hard assignment, averaging, feathering, or uncertainty-based fusion.
It reports full-scene fidelity and seam diagnostics and can export qualitative
mosaics and uncertainty maps.

Example from the repository root::

    python xtile_s2o/fullscene_infer.py \
      --resume runs/xtile_s2o/checkpoint-40.pth \
      --sar_test_path data/corpus76/test/SAR_tif \
      --opt_test_path data/corpus76/test/Optical_tif \
       --csv_path manifests/patch_index.csv --K 8 \
       --cities Turkiye_Adana,Italy_Parma,Poland_Kielce \
       --fs_out out/fullscene --allow_tilewise_cpa_bypass
"""
import argparse, os, re, csv, copy, glob
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as Fnn
import tifffile

from main_jit import get_args_parser
from denoiser import Denoiser
from s2o_norm import SAR_CLIP, OPT_CLIP, _srgb_decode
from uncertainty_fusion import fuse_scene

_SRGB = False   # set in main() from args.srgb_target; gates sRGB encode/decode of the optical

from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

PID_RE = re.compile(r'^(?P<city>.+?)_r\d+_c\d+_x(?P<x>\d+)_y(?P<y>\d+)_s(?P<s>\d+)$')
EXCLUDE = {0, 1, 5}
FUSIONS = ["hard", "average", "feather", "ucgated"]  # ad_hard dropped for v1 (hole-unsafe)


# ----------------------------------------------------------------- data / model
def parse_pid(stem):
    if stem.endswith('_SAR') or stem.endswith('_OPT'):
        stem = stem[:-4]
    m = PID_RE.match(stem)
    if not m:
        return None
    return m.group('city'), int(m.group('x')), int(m.group('y'))


def load_label_map(csv_path):
    lm = {}
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            try:
                lm[row['patch_id']] = (row['split'].strip().lower(), int(row['coarse_label']))
            except (KeyError, ValueError):
                pass
    return lm


def load_model(args):
    ckpt_path = args.resume if args.resume.endswith('.pth') else os.path.join(args.resume, "checkpoint-last.pth")
    ck = torch.load(ckpt_path, map_location='cpu')
    ck_args = ck.get('args', None)
    if ck_args is not None:      # restore arch flags (DEM changes in_channels; dino_loss needed for strict load)
        for k in ('use_cpa', 'cpa_layers', 'srgb_target', 'model', 'img_size', 'class_num',
                  'use_dem', 'dem_ca_layers', 'use_dino_prior', 'lambda_dino', 'dino_layers',
                  'dino_model', 'allow_combined_priors'):
            if hasattr(ck_args, k):
                setattr(args, k, getattr(ck_args, k))
    print(f"[load] ckpt={ckpt_path} use_cpa={getattr(args,'use_cpa',False)} "
          f"use_dem={getattr(args,'use_dem','off')} dino={getattr(args,'use_dino_prior',False)}")
    model = Denoiser(args).to(args.device)
    model.load_state_dict(ck['model'])
    model.ema_params1 = [ck['model_ema1'][n].to(args.device) for n, _ in model.named_parameters()]
    sd = copy.deepcopy(model.state_dict())          # swap EMA weights into the net
    for i, (n, _) in enumerate(model.named_parameters()):
        sd[n] = model.ema_params1[i]
    model.load_state_dict(sd)
    model.eval()
    del ck
    return model


SAR_LO = torch.tensor([SAR_CLIP[0][0], SAR_CLIP[1][0]]).view(1, 2, 1, 1)
SAR_HI = torch.tensor([SAR_CLIP[0][1], SAR_CLIP[1][1]]).view(1, 2, 1, 1)


def load_sar_norm(path, size, device):
    a = tifffile.imread(path).astype(np.float32)
    if a.ndim == 2:
        a = a[None]
    elif a.shape[-1] in (1, 2, 3):
        a = np.transpose(a, (2, 0, 1))
    a = np.nan_to_num(a, nan=-28.0, posinf=2.0, neginf=-28.0)
    t = torch.from_numpy(a)[None].to(device)
    t = Fnn.interpolate(t, size=(size, size), mode='bilinear', align_corners=False)
    lo, hi = SAR_LO.to(device), SAR_HI.to(device)
    t = torch.max(torch.min(t, hi), lo)
    return ((t - lo) / (hi - lo) * 2 - 1).squeeze(0)        # (2,H,W) in [-1,1]


def load_opt_norm(path, size):
    a = tifffile.imread(path).astype(np.float32)
    if a.ndim == 2:
        a = a[None]
    elif a.shape[-1] in (1, 3):
        a = np.transpose(a, (2, 0, 1))
    a = np.nan_to_num(a, nan=0.0, posinf=OPT_CLIP[1], neginf=0.0)
    t = torch.from_numpy(a)[None]
    t = Fnn.interpolate(t, size=(size, size), mode='bilinear', align_corners=False).squeeze(0)
    lo, hi = OPT_CLIP
    lin = ((t.clamp(lo, hi) - lo) / (hi - lo)).clamp(0.0, 1.0)
    if _SRGB:
        b = 0.055
        srgb = torch.where(lin <= 0.0031308, 12.92 * lin,
                           (1.0 + b) * torch.clamp(lin, min=1e-12).pow(1.0 / 2.4) - b)
        return srgb * 2 - 1                                  # (3,H,W) in [-1,1], sRGB domain
    return lin * 2 - 1                                       # (3,H,W) in [-1,1], linear domain


@torch.no_grad()
def mc_tiles(model, sar_tiles, K, device, chunk):
    N = sar_tiles.size(0)
    mus, vars = [], []
    for s in range(0, N, chunk):
        batch = sar_tiles[s:s + chunk].to(device)
        n = batch.size(0)
        rep = batch.repeat_interleave(K, dim=0)
        lab = torch.zeros(n * K, dtype=torch.long, device=device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model.generate(rep, lab).float()          # (n*K,3,H,W) in [-1,1]
        out = out.view(n, K, 3, out.shape[-2], out.shape[-1])
        mus.append(out.mean(1).cpu())
        vars.append(out.var(1, unbiased=False).cpu())
        print(f"    mc {min(s+chunk,N)}/{N}", flush=True)
    return torch.cat(mus), torch.cat(vars)


# ----------------------------------------------------------------- metrics
def denorm(t_or_np):
    a = t_or_np.numpy() if hasattr(t_or_np, 'numpy') else np.asarray(t_or_np)  # (3,H,W) [-1,1]
    lo, hi = OPT_CLIP
    if _SRGB:
        s = np.clip((a + 1) / 2, 0.0, 1.0)                   # sRGB[0,1]
        lin = _srgb_decode(s)                                # -> linear [0,1]
        return np.clip(lin * (hi - lo) + lo, lo, hi)         # (3,H,W) reflectance
    return np.clip((a + 1) / 2 * (hi - lo) + lo, lo, hi)      # (3,H,W) reflectance


def sam_metric(p, g, cover):
    # p,g: (H,W,3) reflectance; cover (H,W) bool
    pv = p[cover].astype(np.float64); gv = g[cover].astype(np.float64)
    num = (pv * gv).sum(1)
    den = np.linalg.norm(pv, axis=1) * np.linalg.norm(gv, axis=1) + 1e-8
    cosv = np.clip(num / den, -1, 1)
    return float(np.degrees(np.arccos(cosv)).mean())


def seam_gradient_index(gray, cover, seam_x, seam_y):
    # gray (H,W) reflectance; ratio of mean |grad| on seam lines vs overall (lower=smoother)
    dx = np.zeros_like(gray); dy = np.zeros_like(gray)
    dx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    dy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    cx = cover[:, 1:] & cover[:, :-1]
    cy = cover[1:, :] & cover[:-1, :]
    overall = (dx[:, 1:][cx].mean() + dy[1:, :][cy].mean()) / 2 + 1e-8
    sg = []
    for c in seam_x:
        if 0 < c < gray.shape[1]:
            col_cov = cover[:, c] & cover[:, c - 1]
            if col_cov.any():
                sg.append(dx[:, c][col_cov].mean())
    for r in seam_y:
        if 0 < r < gray.shape[0]:
            row_cov = cover[r, :] & cover[r - 1, :]
            if row_cov.any():
                sg.append(dy[r, :][row_cov].mean())
    if not sg:
        return float('nan')
    return float(np.mean(sg) / overall)


# ----------------------------------------------------------------- per-city
def process_city(model, city, tiles, args):
    # tiles: list of (patch_id, sar_path, opt_path, x, y)
    size = args.img_size
    W = max(x for _, _, _, x, _ in tiles) + size
    H = max(y for _, _, _, _, y in tiles) + size
    print(f"[{city}] {len(tiles)} tiles, canvas {H}x{W}", flush=True)

    torch.manual_seed(args.seed)                 # per-city reset -> matched noise across models
    torch.cuda.manual_seed_all(args.seed)
    sar = torch.stack([load_sar_norm(p, size, args.device) for _, p, _, _, _ in tiles]).cpu()
    if str(getattr(args, 'use_dem', 'off')) != 'off':   # Append [elevation, slope] to [VV, VH].
        from dem_prior import DemNorm, load_dem_pair
        dn = DemNorm(os.path.join(args.dem_dir, 'dem_norm.json'))
        dsplit = os.path.join(args.dem_dir, args.gen_split)
        dems = torch.stack([dn(load_dem_pair(dsplit, pid)) for pid, _, _, _, _ in tiles])
        if dems.shape[-1] != size:
            dems = Fnn.interpolate(dems, size=(size, size), mode='bilinear', align_corners=False)
        sar = torch.cat([sar, dems], dim=1)
    mu, var = mc_tiles(model, sar, args.K, args.device, args.chunk)      # (N,3,256,256) [-1,1]

    pred_tiles, gt_tiles = [], []
    for i, (_, _, op, x, y) in enumerate(tiles):
        pred_tiles.append({"mu": mu[i], "var": var[i], "x": x, "y": y})
        gt_tiles.append({"mu": load_opt_norm(op, size), "x": x, "y": y})

    # per-tile uncertainty vs error (tile-level correlation + gate analysis)
    tunc = np.array([float(var[i].mean()) for i in range(len(tiles))])
    terr = np.array([float((mu[i] - gt_tiles[i]["mu"]).abs().mean()) for i in range(len(tiles))])

    gt_mosaic, _, _ = fuse_scene(gt_tiles, (H, W), mode="average")
    gt_ref = denorm(gt_mosaic).transpose(1, 2, 0)                        # (H,W,3)
    den = torch.zeros(1, H, W)
    for t in gt_tiles:
        den[:, t["y"]:t["y"]+size, t["x"]:t["x"]+size] += 1
    cover = (den[0].numpy() > 0)

    # per-pixel uncertainty map = mean tile sigma^2 over covering tiles (in [-1,1] space)
    umap = np.zeros((H, W)); ucnt = np.zeros((H, W))
    for i, (_, _, _, x, y) in enumerate(tiles):
        umap[y:y+size, x:x+size] += var[i].mean(0).numpy()
        ucnt[y:y+size, x:x+size] += 1
    umap = umap / np.clip(ucnt, 1, None)

    seam_x = sorted({x for _, _, _, x, _ in tiles if x > 0} | {x + size for _, _, _, x, _ in tiles if x + size < W})
    seam_y = sorted({y for _, _, _, _, y in tiles if y > 0} | {y + size for _, _, _, _, y in tiles if y + size < H})

    gds_gt = seam_gradient_index(gt_ref.mean(2), cover, seam_x, seam_y)  # GT's own seam statistic
    rows = []
    saved = {}
    fusions = [f.strip() for f in args.fusions.split(',') if f.strip()]
    for fusion in fusions:
        fused, fvar, info = fuse_scene(pred_tiles, (H, W), mode=fusion, mad_k=args.mad_k)
        pr = denorm(fused).transpose(1, 2, 0)                            # (H,W,3)
        m = cover
        psnr = psnr_fn(gt_ref[m], pr[m], data_range=OPT_CLIP[1])
        ssim = ssim_fn(gt_ref, pr, channel_axis=2, data_range=OPT_CLIP[1])
        sam = sam_metric(pr, gt_ref, m)
        gds = seam_gradient_index(pr.mean(2), cover, seam_x, seam_y)
        rows.append(dict(city=city, fusion=fusion, n_tiles=len(tiles),
                         PSNR=psnr, SSIM=ssim, SAM=sam, GDS=gds,
                         GDS_gt=gds_gt, GDS_delta=gds - gds_gt,
                         n_dropped=info.get("n_dropped", 0)))
        print(f"  [{city}/{fusion}] PSNR={psnr:.3f} SSIM={ssim:.4f} SAM={sam:.3f} "
              f"GDS={gds:.3f} (GT {gds_gt:.3f}, d{gds-gds_gt:+.3f}) dropped={info.get('n_dropped',0)}", flush=True)
        saved[fusion] = (pr, fvar)

    # uncertainty-vs-error correlation: does per-pixel sigma^2 predict error?
    # (error from the plain-average prediction so it is fusion-independent)
    ref_key = "average" if "average" in saved else list(saved)[0]
    err = np.abs(saved[ref_key][0] - gt_ref).mean(2)
    try:
        from scipy.stats import spearmanr
        corr = float(spearmanr(umap[cover], err[cover]).correlation)
    except Exception:
        corr = float(np.corrcoef(umap[cover], err[cover])[0, 1])
    for r in rows:
        r["unc_err_corr"] = corr
    print(f"  [{city}] PIXEL unc-error corr (Spearman) = {corr:.4f}", flush=True)

    try:
        from scipy.stats import spearmanr
        tcorr = float(spearmanr(tunc, terr).correlation)
    except Exception:
        tcorr = float(np.corrcoef(tunc, terr)[0, 1])
    for r in rows:
        r["tile_corr"] = tcorr
    print(f"  [{city}] TILE unc-error corr = {tcorr:.4f} | sigma^2 mean={tunc.mean():.5f} "
          f"p90={np.percentile(tunc, 90):.5f} max={tunc.max():.5f}", flush=True)

    # qualitative dump
    if args.save_vis:
        import cv2
        od = os.path.join(args.fs_out, city); os.makedirs(od, exist_ok=True)
        VIS = 3000.0
        def to8(ref):
            return np.clip(ref / VIS * 255, 0, 255).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(od, "GT.png"), to8(gt_ref))
        tifffile.imwrite(os.path.join(od, "GT_refl.tif"), gt_ref.astype('uint16'))
        for fusion, (pr, fvar) in saved.items():
            cv2.imwrite(os.path.join(od, f"{fusion}.png"), to8(pr))
            tifffile.imwrite(os.path.join(od, f"{fusion}_refl.tif"), pr.astype('uint16'))
        u8 = np.clip(umap / (umap[cover].max() + 1e-8) * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(od, "uncertainty.png"), cv2.applyColorMap(u8, cv2.COLORMAP_JET))
    return rows, tunc, terr


def main():
    parser = get_args_parser()
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--chunk', type=int, default=0, help='tiles per gen batch (0=auto from K)')
    parser.add_argument('--cities', type=str, default='Turkiye_Adana,Italy_Parma,Poland_Kielce')
    parser.add_argument('--max_cities', type=int, default=0, help='if >0 and --cities empty, top-N by tile count')
    parser.add_argument('--mad_k', type=float, default=3.0)
    parser.add_argument('--fusions', type=str, default='hard,average,feather',
                        help='comma list; ucgated needs K>1')
    parser.add_argument('--fs_out', type=str, default='out/fullscene')
    parser.add_argument('--save_vis', action='store_true', default=True)
    parser.add_argument('--allow_tilewise_cpa_bypass', action='store_true',
                        help='explicitly allow a CPA checkpoint to run tile-wise with CPA inactive')
    args = parser.parse_args()
    args.device = 'cuda'
    global _SRGB
    _SRGB = getattr(args, 'srgb_target', False)
    print("fullscene OPT domain:", "sRGB (gamma)" if _SRGB else "linear reflectance")
    if args.chunk <= 0:
        args.chunk = max(1, 64 // max(1, args.K))
    os.makedirs(args.fs_out, exist_ok=True)
    torch.manual_seed(args.seed)

    lm = load_label_map(args.csv_path)
    by_city = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(args.sar_test_path, '*_SAR.tif'))):
        stem = os.path.splitext(os.path.basename(p))[0]
        pid = stem[:-4]                                   # drop _SAR
        meta = lm.get(pid)
        if meta is None or meta[0] != 'test' or meta[1] in EXCLUDE:
            continue
        parsed = parse_pid(stem)
        if parsed is None:
            continue
        city, x, y = parsed
        op = os.path.join(args.opt_test_path, pid + '_OPT.tif')
        if not os.path.exists(op):
            continue
        by_city[city].append((pid, p, op, x, y))

    if args.cities.strip():
        sel = [c.strip() for c in args.cities.split(',') if c.strip() in by_city]
    else:
        sel = sorted(by_city, key=lambda c: -len(by_city[c]))
        if args.max_cities > 0:
            sel = sel[:args.max_cities]
    print("Selected cities:", sel, flush=True)

    model = load_model(args)
    _SRGB = getattr(args, 'srgb_target', False)   # re-sync after ckpt-flag restore (global decl at top of main)
    if getattr(args, 'use_cpa', False) and not args.allow_tilewise_cpa_bypass:
        raise RuntimeError(
            'fullscene_infer.py samples tiles independently, so CPA would be inactive. '
            'Use blockgroup_infer.py for XTile-S2O block-context inference, or pass '
            '--allow_tilewise_cpa_bypass only for an intentional tile-wise diagnostic.'
        )
    all_rows = []
    all_tunc, all_terr = [], []
    for city in sel:
        rows, tunc, terr = process_city(model, city, by_city[city], args)
        all_rows += rows
        all_tunc.append(tunc); all_terr.append(terr)

    # pooled tile-level correlation over ALL tiles (most statistically meaningful)
    AU = np.concatenate(all_tunc); AE = np.concatenate(all_terr)
    try:
        from scipy.stats import spearmanr
        pooled = float(spearmanr(AU, AE).correlation)
    except Exception:
        pooled = float(np.corrcoef(AU, AE)[0, 1])
    print(f"\n=== POOLED TILE unc-error Spearman over {len(AU)} tiles = {pooled:.4f} ===", flush=True)

    # write CSV + averaged summary
    out_csv = os.path.join(args.fs_out, "fullscene_metrics.csv")
    keys = ["city", "fusion", "n_tiles", "PSNR", "SSIM", "SAM", "GDS", "GDS_gt", "GDS_delta", "n_dropped", "unc_err_corr", "tile_corr"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print("\n==== MEAN over cities (by fusion) ====", flush=True)
    print(f"{'fusion':9} {'PSNR':>7} {'SSIM':>7} {'SAM':>7} {'GDS':>7}")
    for fusion in [f.strip() for f in args.fusions.split(',') if f.strip()]:
        rs = [r for r in all_rows if r["fusion"] == fusion]
        if rs:
            print(f"{fusion:9} {np.mean([r['PSNR'] for r in rs]):7.3f} "
                  f"{np.mean([r['SSIM'] for r in rs]):7.4f} "
                  f"{np.mean([r['SAM'] for r in rs]):7.3f} "
                  f"{np.mean([r['GDS'] for r in rs]):7.3f}")
    print("\nwrote", out_csv, flush=True)


if __name__ == '__main__':
    main()
