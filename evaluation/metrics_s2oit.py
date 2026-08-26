#!/usr/bin/env python3
"""Unified S2O image-translation evaluation at patch and block levels.

Inputs:
  Predictions: ``<pred_dir>/<patch_id>_fakeB.tif`` (uint16 reflectance)
  References: ``<optical_dir>/<patch_id>_OPT.tif`` (uint16 reflectance)

Patch-level evaluation computes paired metrics for each tile. Block-level
evaluation reads ``block_2x2_index.csv``, reassembles four overlapping
256-pixel predictions on a 416x416 canvas (stride 160; overlap 96), applies
average, feather, or Poisson fusion, and recomputes fidelity and seam metrics.

Reflectance-domain metrics are PSNR, SSIM, CW-SSIM, RMSE, SAM, and block-level
GDS. Display-domain metrics are FID, LPIPS, FSIM, and optional NIQE after the
same sRGB conversion for predictions and references.

Gradient discontinuity at seams (GDS) is the ratio of mean gradient magnitude
in overlap supports to that in single-contributor interiors. The same
construction is applied to predictions and references; ``GDS_delta`` reports
their difference. Lower values indicate less excess boundary contrast.

The output CSV contains one row per method, evaluation level, and fusion mode.
Required dependencies are numpy, tifffile, scikit-image, scipy, pandas,
opencv-python, and imagecodecs. Optional metrics use lpips, piq,
torch-fidelity, pyiqa, and dtcwt.
"""
import os, re, csv as csvmod, glob, argparse, warnings
import numpy as np
import tifffile
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from s2o_norm import clip_gt_opt, _srgb_encode, OPT_CLIP

DATA_RANGE = OPT_CLIP[1]
PATCH = 256   # Released tile size in pixels.


# ----------------- Image loading and radiometric mapping -----------------
def _to_hwc(a):
    a = np.asarray(a).astype(np.float32)
    if a.ndim == 2: a = a[..., None]
    if a.shape[0] in (1, 2, 3) and a.shape[0] < a.shape[-1]: a = np.moveaxis(a, 0, -1)
    return a

def load_pred(path):           return _to_hwc(tifffile.imread(path))
def load_gt(path):             return clip_gt_opt(tifffile.imread(path)).astype(np.float32)  # (H,W,3)

def unify_to_srgb8(refl_hwc):
    lin = np.clip((refl_hwc - OPT_CLIP[0]) / (OPT_CLIP[1] - OPT_CLIP[0]), 0, 1)
    return np.clip(np.round(_srgb_encode(lin) * 255.0), 0, 255).astype(np.uint8)


# ----------------- Reflectance-domain metrics -----------------
def m_psnr(p, g): return float(sk_psnr(g, p, data_range=DATA_RANGE))
def m_ssim(p, g): return float(sk_ssim(g, p, data_range=DATA_RANGE, channel_axis=-1))
def m_rmse(p, g): return float(np.sqrt(np.mean((p - g) ** 2)))

def m_sam(p, g, eps=1e-8):
    p = p.reshape(-1, p.shape[-1]); g = g.reshape(-1, g.shape[-1])
    npn = np.linalg.norm(p, axis=1); ngn = np.linalg.norm(g, axis=1)
    cos = np.clip(np.sum(p*g, axis=1) / (npn*ngn + eps), -1, 1)
    v = (npn > eps) & (ngn > eps)
    return float(np.mean(np.arccos(cos[v]))) if v.any() else float('nan')

_DTCWT = None
def _dtcwt():
    global _DTCWT
    if _DTCWT is None:
        try: import dtcwt; _DTCWT = dtcwt
        except Exception: _DTCWT = False
    return _DTCWT

def m_cwssim(p, g, levels=4, level=2, K=0.01):
    d = _dtcwt()
    if d is False: return float('nan')
    t = d.Transform2d()
    cp = t.forward(p.mean(-1), nlevels=levels); cg = t.forward(g.mean(-1), nlevels=levels)
    hp, hg = cp.highpasses[level], cg.highpasses[level]
    num = 2*np.abs(np.sum(hp*np.conj(hg), axis=(0,1))) + K
    den = np.sum(np.abs(hp)**2, (0,1)) + np.sum(np.abs(hg)**2, (0,1)) + K
    return float(np.mean(num/den))


# ----------------- Block reassembly and fusion -----------------
def parse_xy(pid):
    m = re.search(r'_x(\d+)_y(\d+)_s(\d+)', pid)
    return int(m.group(1)), int(m.group(2))

def _edge_weight(size=PATCH):
    """Return normalized feather weights that taper toward all tile edges."""
    r = np.arange(size)
    d = np.minimum(r, size-1-r).astype(np.float32) + 1.0
    w = np.minimum.outer(d, d)
    return w / w.max()

_FEATHER_W = _edge_weight()

def assemble(patches, offsets, fusion):
    """Reassemble HWC patches at relative offsets; return canvas and coverage count."""
    oxs = [o[0] for o in offsets]; oys = [o[1] for o in offsets]
    W = max(oxs) + PATCH; H = max(oys) + PATCH
    C = patches[0].shape[-1]
    acc = np.zeros((H, W, C), np.float32); wsum = np.zeros((H, W, 1), np.float32)
    cnt = np.zeros((H, W), np.float32)
    for arr, (ox, oy) in zip(patches, offsets):
        sl = (slice(oy, oy+PATCH), slice(ox, ox+PATCH))
        cnt[sl] += 1
        if fusion == 'feather':
            w = _FEATHER_W[..., None]
            acc[sl] += arr * w; wsum[sl] += w
        else:  # Poisson fusion starts from the same average canvas.
            acc[sl] += arr; wsum[sl] += 1.0
    canvas = acc / np.maximum(wsum, 1e-6)
    if fusion == 'poisson':
        canvas = _poisson_refine(canvas, patches, offsets)
    return canvas, cnt

def _poisson_refine(base, patches, offsets):
    """Refine an average canvas using patch gradients and sparse Poisson solves.

    Gradients are averaged where tiles overlap. Each channel is solved
    independently; the function returns ``base`` if sparse solvers are absent.
    """
    try:
        import scipy.sparse as sp
        from scipy.sparse.linalg import lsqr
    except Exception:
        return base
    H, W, C = base.shape
    # Average each patch's horizontal and vertical gradients on the canvas.
    gx = np.zeros((H, W, C), np.float32); gy = np.zeros((H, W, C), np.float32); gc = np.zeros((H, W, 1), np.float32)
    for arr, (ox, oy) in zip(patches, offsets):
        ax = np.zeros_like(arr); ay = np.zeros_like(arr)
        ax[:, 1:] = arr[:, 1:] - arr[:, :-1]
        ay[1:, :] = arr[1:, :] - arr[:-1, :]
        sl = (slice(oy, oy+PATCH), slice(ox, ox+PATCH))
        gx[sl] += ax; gy[sl] += ay; gc[sl] += 1.0
    gx /= np.maximum(gc, 1e-6); gy /= np.maximum(gc, 1e-6)
    # Solve argmin ||grad(u)-g||^2 + lambda||u-base||^2 for each channel.
    N = H * W; idx = np.arange(N).reshape(H, W)
    rows_, cols_, vals_, b_blocks = [], [], [], []
    eqn = 0
    # Horizontal-gradient constraints.
    r = idx[:, 1:].ravel(); l = idx[:, :-1].ravel()
    rows_ += list(range(eqn, eqn+len(r)))*2; cols_ += list(r)+list(l); vals_ += [1.0]*len(r)+[-1.0]*len(l)
    ex_n = len(r); eqn += ex_n
    r2 = idx[1:, :].ravel(); u2 = idx[:-1, :].ravel()
    rows_ += list(range(eqn, eqn+len(r2)))*2; cols_ += list(r2)+list(u2); vals_ += [1.0]*len(r2)+[-1.0]*len(r2)
    ey_n = len(r2); eqn += ey_n
    lam = 0.1
    rows_ += list(range(eqn, eqn+N)); cols_ += list(range(N)); vals_ += [lam]*N
    eqn += N
    A = sp.csr_matrix((vals_, (rows_, cols_)), shape=(eqn, N))
    out = np.empty((H, W, C), np.float32)
    for c in range(C):
        bx = gx[:, 1:, c].ravel(); by = gy[1:, :, c].ravel(); ba = lam*base[:, :, c].ravel()
        b = np.concatenate([bx, by, ba]).astype(np.float32)
        u = lsqr(A, b, atol=1e-5, btol=1e-5, iter_lim=300)[0]
        out[:, :, c] = u.reshape(H, W)
    return np.clip(out, OPT_CLIP[0], OPT_CLIP[1])


def m_gds(block_hwc, cnt, strip=2, eps=1e-6):
    """Return overlap-to-interior gradient ratio; lower values are smoother."""
    g = block_hwc.mean(-1)
    gy, gx = np.gradient(g)
    mag = np.sqrt(gx**2 + gy**2)
    overlap = cnt >= 2; interior = cnt == 1
    if overlap.sum() == 0 or interior.sum() == 0: return float('nan')
    return float(mag[overlap].mean() / (mag[interior].mean() + eps))


# ----------------- Optional display-domain metrics -----------------
class PngMetrics:
    def __init__(self, use_niqe=False, device=None):
        self.ok = {}; self.use_niqe = use_niqe; self.device = device
        self._t = self._lp = self._piq = self._niqe = None
        try:
            import torch; self._t = torch
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        except Exception:
            warnings.warn("PyTorch unavailable; skipping display-domain metrics"); return
        try:
            import lpips; self._lp = lpips.LPIPS(net='alex').to(self.device).eval(); self.ok['LPIPS']=True
        except Exception: warnings.warn("lpips unavailable; skipping LPIPS")
        try:
            import piq; self._piq = piq; self.ok['FSIM']=True
        except Exception: warnings.warn("piq unavailable; skipping FSIM")
        if use_niqe:
            try:
                import pyiqa; self._niqe = pyiqa.create_metric('niqe', device=self.device); self.ok['NIQE']=True
            except Exception: warnings.warn("pyiqa unavailable; skipping NIQE")

    def _tt(self, png):
        return self._t.from_numpy(png.astype(np.float32)/255.).permute(2,0,1).unsqueeze(0).to(self.device)

    def per_image(self, pred_png, gt_png):
        out = {}
        if self._t is None: return out
        tp, tg = self._tt(pred_png), self._tt(gt_png)
        with self._t.no_grad():
            if self.ok.get('LPIPS'): out['LPIPS']=float(self._lp(tp*2-1, tg*2-1).item())
            if self.ok.get('FSIM'):  out['FSIM']=float(self._piq.fsim(tp, tg, data_range=1.).item())
            if self.ok.get('NIQE'):  out['NIQE']=float(self._niqe(tp).item())
        return out

    def fid(self, d1, d2):
        try: import torch_fidelity
        except Exception: warnings.warn("torch-fidelity unavailable; skipping FID"); return float('nan')
        r = torch_fidelity.calculate_metrics(input1=d1, input2=d2, cuda=(self.device=='cuda'),
                                             fid=True, isc=False, kid=False, verbose=False)
        return float(r['frechet_inception_distance'])


# ----------------- Patch-level evaluation -----------------
def eval_patch(name, pred_dir, optical_dir, args, png):
    import cv2
    files = sorted(glob.glob(os.path.join(pred_dir, f"*{args.pred_suffix}")))
    pp = os.path.join(args.png_root, name, 'patch_pred'); gp = os.path.join(args.png_root, name, 'patch_gt')
    os.makedirs(pp, exist_ok=True); os.makedirs(gp, exist_ok=True)
    acc = {k: [] for k in ['PSNR','SSIM','CW-SSIM','RMSE','SAM','LPIPS','FSIM','NIQE']}; n = 0
    for pf in files:
        pid = os.path.basename(pf)[:-len(args.pred_suffix)]
        gtp = os.path.join(optical_dir, f"{pid}{args.gt_suffix}")
        if not os.path.isfile(gtp): continue
        pred = load_pred(pf); gt = load_gt(gtp)
        if pred.shape != gt.shape: continue
        n += 1
        acc['PSNR'].append(m_psnr(pred,gt)); acc['SSIM'].append(m_ssim(pred,gt))
        acc['RMSE'].append(m_rmse(pred,gt)); acc['SAM'].append(m_sam(pred,gt))
        if not args.no_cwssim: acc['CW-SSIM'].append(m_cwssim(pred,gt))
        pr8, gt8 = unify_to_srgb8(pred), unify_to_srgb8(gt)
        cv2.imwrite(os.path.join(pp, pid+'.png'), pr8[:,:,::-1]); cv2.imwrite(os.path.join(gp, pid+'.png'), gt8[:,:,::-1])
        for k,v in png.per_image(pr8, gt8).items(): acc[k].append(v)
    if n == 0: return None
    fid = png.fid(pp, gp)
    row = {'method':name,'level':'patch','fusion':'-','n':n}
    for k in acc: 
        vv=[x for x in acc[k] if not (isinstance(x,float) and np.isnan(x))]; row[k]=float(np.mean(vv)) if vv else float('nan')
    row['FID']=fid
    if not args.export_png:
        import shutil; shutil.rmtree(os.path.join(args.png_root, name), ignore_errors=True)
    print(f"  [{name}/patch] n={n} PSNR={row['PSNR']:.3f} SSIM={row['SSIM']:.4f} SAM={row['SAM']:.4f}")
    return row


# ----------------- block-level -----------------
def load_blocks(block_csv, split):
    rows = []
    for r in csv_iter(block_csv):
        if r['split'] != split: continue
        pass  # usable flag is placeholder; rely on pred-file existence to prune incomplete blocks
        rows.append(r)
    return rows

def csv_iter(path):
    with open(path, encoding='utf-8-sig') as f:
        for r in csvmod.DictReader(f): yield r

def eval_block(name, pred_dir, optical_dir, block_csv, split, fusion, args, png):
    import cv2
    blocks = load_blocks(block_csv, split)
    pp = os.path.join(args.png_root, name, f'block_{fusion}_pred'); gp = os.path.join(args.png_root, name, f'block_{fusion}_gt')
    os.makedirs(pp, exist_ok=True); os.makedirs(gp, exist_ok=True)
    keys = ['top_left_patch_id','top_right_patch_id','bottom_left_patch_id','bottom_right_patch_id']
    acc = {k: [] for k in ['PSNR','SSIM','CW-SSIM','RMSE','SAM','LPIPS','FSIM','NIQE','GDS','GDS_gt','GDS_delta']}; n = 0
    for b in blocks:
        pids = [b[k] for k in keys]
        pred_paths = [os.path.join(pred_dir, f"{p}{args.pred_suffix}") for p in pids]
        gt_paths   = [os.path.join(optical_dir, f"{p}{args.gt_suffix}") for p in pids]
        if not all(os.path.isfile(x) for x in pred_paths+gt_paths): continue
        offs = [parse_xy(p) for p in pids]; ox0 = min(o[0] for o in offs); oy0 = min(o[1] for o in offs)
        offs = [(x-ox0, y-oy0) for x,y in offs]
        pred_patches = [load_pred(x) for x in pred_paths]; gt_patches = [load_gt(x) for x in gt_paths]
        pblk, cnt = assemble(pred_patches, offs, fusion)
        gblk, _   = assemble(gt_patches,   offs, fusion)
        n += 1
        acc['PSNR'].append(m_psnr(pblk,gblk)); acc['SSIM'].append(m_ssim(pblk,gblk))
        acc['RMSE'].append(m_rmse(pblk,gblk)); acc['SAM'].append(m_sam(pblk,gblk))
        if not args.no_cwssim: acc['CW-SSIM'].append(m_cwssim(pblk,gblk))
        gpred = m_gds(pblk, cnt); ggt = m_gds(gblk, cnt)
        acc['GDS'].append(gpred); acc['GDS_gt'].append(ggt); acc['GDS_delta'].append(gpred-ggt)
        pr8, gt8 = unify_to_srgb8(pblk), unify_to_srgb8(gblk)
        bid = b['block_id']
        cv2.imwrite(os.path.join(pp, bid+'.png'), pr8[:,:,::-1]); cv2.imwrite(os.path.join(gp, bid+'.png'), gt8[:,:,::-1])
        for k,v in png.per_image(pr8, gt8).items(): acc[k].append(v)
    if n == 0:
        print(f"  [{name}/block/{fusion}] no usable blocks"); return None
    fid = png.fid(pp, gp)
    row = {'method':name,'level':'block','fusion':fusion,'n':n}
    for k in acc:
        vv=[x for x in acc[k] if not (isinstance(x,float) and np.isnan(x))]; row[k]=float(np.mean(vv)) if vv else float('nan')
    row['FID']=fid
    if not args.export_png:
        import shutil; shutil.rmtree(os.path.join(args.png_root, name), ignore_errors=True)
    print(f"  [{name}/block/{fusion}] n={n} PSNR={row['PSNR']:.3f} SSIM={row['SSIM']:.4f} "
          f"SAM={row['SAM']:.4f} GDS={row['GDS']:.3f} GDS_delta={row['GDS_delta']:.3f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', required=True, help='name:pred_dir ...')
    ap.add_argument('--optical_dir', required=True)
    ap.add_argument('--block_csv', default=None, help='block_2x2_index.csv (required for block-level evaluation)')
    ap.add_argument('--split', default='test')
    ap.add_argument('--level', default='both', choices=['patch','block','both'])
    ap.add_argument('--fusion', nargs='+', default=['average','feather','poisson'],
                    choices=['average','feather','poisson'])
    ap.add_argument('--out_csv', default='./metrics_s2oit.csv')
    ap.add_argument('--png_root', default='./_metrics_png')
    ap.add_argument('--pred_suffix', default='_fakeB.tif')
    ap.add_argument('--gt_suffix', default='_OPT.tif')
    ap.add_argument('--export_png', action='store_true')
    ap.add_argument('--no_cwssim', action='store_true')
    ap.add_argument('--niqe', action='store_true')
    args = ap.parse_args()

    methods = [(s.split(':',1)[0], s.split(':',1)[1]) for s in args.methods]
    png = PngMetrics(use_niqe=args.niqe)
    print("Available display-domain metrics:", [k for k,v in png.ok.items() if v] or "none (reflectance domain only)")

    rows = []
    for name, pred_dir in methods:
        print(f"Evaluating {name} ...")
        if args.level in ('patch','both'):
            r = eval_patch(name, pred_dir, args.optical_dir, args, png)
            if r: rows.append(r)
        if args.level in ('block','both'):
            if not args.block_csv:
                print("  Skipping block-level evaluation: --block_csv was not provided"); continue
            for fz in args.fusion:
                r = eval_block(name, pred_dir, args.optical_dir, args.block_csv, args.split, fz, args, png)
                if r: rows.append(r)

    cols = ['method','level','fusion','n','PSNR','SSIM','CW-SSIM','FSIM','RMSE','SAM',
            'LPIPS','FID','NIQE','GDS','GDS_gt','GDS_delta']
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csvmod.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({c: r.get(c,'') for c in cols})
    print(f"\nWrote: {args.out_csv}")


if __name__ == '__main__':
    main()
