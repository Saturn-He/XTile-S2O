# -*- coding: utf-8 -*-
"""Six recomposition arms on the same K=8 test candidates
(fstest_seed3001-3008), evaluated on city canvases with CPU streaming.

Arms (identical candidates, only the rule differs):
  hard1   : ownership stitch of seed3001 (K=1 deployment baseline)
  hard    : ownership stitch of per-tile MC means (isolates ensemble effect)
  average : unweighted mean of covering tile means
  feather : Hann-window distance blend of tile means
  precision: inverse-variance soft weights w=1/(sigma^2+eps), no gate
  gated   : precision + per-city tile gate s_i > median(s)+3*MAD(s) -> w=0
            (fallback: pixel with all candidates gated -> ungated precision)

Pre-registered: eps=1.0 (uint16 refl units^2), K=8, gate const 3,
selection rule = max val PSNR s.t. seam <= 1.05 * seam(hard); uncertainty map
= PRE-gate total variance (within + between, law of total variance).
Outputs: eval/test_fusion/{arms_percity.csv, city uncertainty tifs} + summary."""
import csv
import glob
import os
import re
from pathlib import Path

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path(os.environ.get("XTILE_RUN_ROOT", REPO_ROOT)).resolve()
DATA_ROOT = Path(os.environ.get("XTILE_DATA_ROOT", REPO_ROOT / "data" / "corpus76")).resolve()
NAS = str(RUN_ROOT)
GT_DIR = str(DATA_ROOT / "test" / "Optical_tif")
SEEDS = list(range(3001, 3009))
OUT = str(Path(os.environ.get("XTILE_EVAL_ROOT", RUN_ROOT / "eval")).resolve() / "test_fusion")
os.makedirs(OUT, exist_ok=True)
P, S = 256, 160
EPS = 1.0
PID_RC = re.compile(r"^(.*?)_r(\d+)_c(\d+)_")


def load(p):
    a = tifffile.imread(p).astype("float32")
    if a.ndim == 3 and a.shape[0] in (3, 4):
        a = a.transpose(1, 2, 0)
    return np.clip(a[..., :3], 0, 4500)


def tile_file(d, pid):
    for suf in ("_fakeB.tif", ".tif"):
        p = os.path.join(d, pid + suf)
        if os.path.exists(p):
            return p
    return None


dirs = [f"{NAS}/eval/fstest_seed{s}/pred_tif" for s in SEEDS]
pids = sorted(os.path.basename(f).replace("_fakeB.tif", "")
              for f in glob.glob(os.path.join(dirs[0], "*.tif")))
by_city = {}
for pid in pids:
    m = PID_RC.match(pid)
    by_city.setdefault(m.group(1), []).append((int(m.group(2)), int(m.group(3)), pid))

hann = np.outer(np.hanning(P), np.hanning(P)).astype("float32") + 1e-3


def psnr(a, b, mask):
    mse = ((a - b)[mask] ** 2).mean()
    return 10 * np.log10(4500.0 ** 2 / max(mse, 1e-9))


def ssim_g(a, b, mask):
    # global-statistics SSIM proxy on masked canvas (single window; fast, same for all arms)
    am, bm = a[mask], b[mask]
    mu1, mu2 = am.mean(), bm.mean()
    v1, v2 = am.var(), bm.var()
    cov = ((am - mu1) * (bm - mu2)).mean()
    c1, c2 = (0.01 * 4500) ** 2, (0.03 * 4500) ** 2
    return float((2 * mu1 * mu2 + c1) * (2 * cov + c2) / ((mu1**2 + mu2**2 + c1) * (v1 + v2 + c2)))


def sam(a, b, mask):
    aa, bb = a[mask].reshape(-1, 3), b[mask].reshape(-1, 3)
    num = (aa * bb).sum(1)
    den = np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1) + 1e-6
    return float(np.arccos(np.clip(num / den, -1, 1)).mean())


rows = []
for city, tl in sorted(by_city.items()):
    rows_r = [t[0] for t in tl]
    cols_c = [t[1] for t in tl]
    H = S * max(rows_r) + P
    W = S * max(cols_c) + P
    nt = len(tl)
    # per-tile MC stats
    mu, var, s_tile, owner = {}, {}, {}, {}
    for r, c, pid in tl:
        stack = np.stack([load(tile_file(d, pid)) for d in dirs], 0)
        mu[pid] = stack.mean(0)
        var[pid] = stack.var(0).mean(-1)          # (H,W) channel-mean variance
        s_tile[pid] = float(var[pid].mean())
        owner[pid] = load(tile_file(dirs[0], pid))  # seed3001 sample for hard1
    sv = np.array(list(s_tile.values()))
    tau = float(np.median(sv) + 3 * 1.4826 * np.median(np.abs(sv - np.median(sv))))
    gated_out = [pid for pid in s_tile if s_tile[pid] > tau]

    # ownership map: nearest tile center
    centers = {pid: (r * S + P // 2, c * S + P // 2) for r, c, pid in tl}
    # canvases: accumulate per arm
    arms = ["hard1", "hard", "average", "feather", "precision", "gated"]
    acc = {a: (np.zeros((H, W, 3), "float64"), np.zeros((H, W, 1), "float64")) for a in arms}
    gt_acc = (np.zeros((H, W, 3), "float64"), np.zeros((H, W, 1), "float64"))
    # uncertainty accumulators (pre-gate): within mean + between var of means
    n_cov = np.zeros((H, W), "float32")
    sum_w_var = np.zeros((H, W), "float32")
    sum_mu = np.zeros((H, W), "float32")
    sum_mu2 = np.zeros((H, W), "float32")
    own_dist = np.full((H, W), 1e9, "float32")
    own_pid = np.empty((H, W), object)

    for r, c, pid in tl:
        y, x = r * S, c * S
        sl = np.s_[y:y + P, x:x + P]
        yy, xx = np.mgrid[y:y + P, x:x + P]
        d2 = (yy - centers[pid][0]) ** 2 + (xx - centers[pid][1]) ** 2
        closer = d2 < own_dist[sl]
        own_dist[sl][closer] = d2[closer]
        sub = own_pid[sl]
        sub[closer] = pid
        gmu = mu[pid].mean(-1)
        n_cov[sl] += 1
        sum_w_var[sl] += var[pid]
        sum_mu[sl] += gmu
        sum_mu2[sl] += gmu ** 2
        w_prec = (1.0 / (var[pid] + EPS))[..., None]
        for arm, w in [("average", np.ones((P, P, 1), "float32")),
                       ("feather", hann[..., None]),
                       ("precision", w_prec),
                       ("gated", w_prec if pid not in gated_out else np.zeros((P, P, 1), "float32"))]:
            acc[arm][0][sl] += mu[pid] * w
            acc[arm][1][sl] += w
        g = load(os.path.join(GT_DIR, pid + "_OPT.tif"))
        gt_acc[0][sl] += g
        gt_acc[1][sl] += 1

    gt = (gt_acc[0] / np.clip(gt_acc[1], 1, None)).astype("float32")
    mask3 = (gt_acc[1][..., 0] > 0)[..., None] & np.ones((1, 1, 3), bool)
    fused = {}
    for arm in ("average", "feather", "precision", "gated"):
        num, den = acc[arm]
        if arm == "gated":                       # fallback: all-gated pixels -> precision
            fb = den[..., 0] <= 0
            num[fb] = acc["precision"][0][fb]
            den[fb] = acc["precision"][1][fb]
        fused[arm] = (num / np.clip(den, 1e-9, None)).astype("float32")
    for arm, src in [("hard1", owner), ("hard", mu)]:
        cv = np.zeros((H, W, 3), "float32")
        for r, c, pid in tl:
            y, x = r * S, c * S
            sl = np.s_[y:y + P, x:x + P]
            m = (own_pid[sl] == pid)
            cv[sl][m] = src[pid][m]
        fused[arm] = cv

    # uncertainty map (pre-gate, total variance)
    with np.errstate(invalid="ignore"):
        w_var = sum_w_var / np.clip(n_cov, 1, None)
        b_var = sum_mu2 / np.clip(n_cov, 1, None) - (sum_mu / np.clip(n_cov, 1, None)) ** 2
    unc = np.sqrt(np.clip(w_var + np.clip(b_var, 0, None), 0, None))
    tifffile.imwrite(f"{OUT}/{city}_uncertainty.tif", unc.astype("float32"))

    # seam lines of the ownership partition
    seam_m = np.zeros((H, W), bool)
    bnd = own_pid[:, 1:] != own_pid[:, :-1]
    seam_m[:, 1:] |= bnd
    seam_m[:, :-1] |= bnd
    bnd = own_pid[1:, :] != own_pid[:-1, :]
    seam_m[1:, :] |= bnd
    seam_m[:-1, :] |= bnd
    valid = gt_acc[1][..., 0] > 0

    def seam_grad(img):
        gy, gx = np.gradient(img.mean(-1))
        gm = np.hypot(gx, gy)
        return float(gm[seam_m & valid].mean()), float(gm[~seam_m & valid].mean())

    gt_s, gt_i = seam_grad(gt)
    for arm in ("hard1", "hard", "average", "feather", "precision", "gated"):
        f = fused[arm]
        m = valid[..., None] & np.ones((1, 1, 3), bool)
        se, it = seam_grad(f)
        rows.append([city, arm, round(psnr(f, gt, m), 3), round(ssim_g(f, gt, m), 4),
                     round(sam(f, gt, valid), 4), round(se, 2), round(it, 2),
                     len(gated_out) if arm == "gated" else "", nt,
                     round(gt_s, 2), round(gt_i, 2)])
    print(f"[{city}] tiles={nt} gated_out={len(gated_out)} tau={tau:.1f} "
          f"GTseam={gt_s:.1f}/{gt_i:.1f}", flush=True)

with open(f"{OUT}/arms_percity.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["city", "arm", "PSNR", "SSIMg", "SAM", "seam", "interior",
                "n_gated_tiles", "n_tiles", "GT_seam", "GT_interior"])
    w.writerows(rows)

# macro summary + registered selection
arms = ["hard1", "hard", "average", "feather", "precision", "gated"]
agg = {a: [r for r in rows if r[1] == a] for a in arms}
print("\narm, PSNR, SSIMg, SAM, seam, interior  (macro over cities)")
stats = {}
for a in arms:
    g = agg[a]
    st = [np.mean([r[i] for r in g]) for i in (2, 3, 4, 5, 6)]
    stats[a] = st
    print(f"{a:10s} " + "  ".join(f"{v:.3f}" for v in st), flush=True)
hard_seam = stats["hard"][3]
elig = [a for a in arms if stats[a][3] <= 1.05 * hard_seam]
sel = max(elig, key=lambda a: stats[a][0])
print(f"\nREGISTERED SELECTION (max PSNR s.t. seam<=1.05*hard): {sel}")
print("STEP0_FUSION_DONE", flush=True)
