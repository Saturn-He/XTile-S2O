# -*- coding: utf-8 -*-
"""Seam/interior gradient decomposition for all compared methods (shared
hard-stitching convention). Answers: is CycleGAN's apparent seamlessness low
BETWEEN-tile variance or just low overall contrast?"""
import csv
import glob
import os
import re
from pathlib import Path

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
E = str(Path(os.environ.get("XTILE_EVAL_ROOT", REPO_ROOT / "eval")).resolve())
B = str(Path(os.environ.get("XTILE_BASELINES_ROOT", REPO_ROOT / "baselines_out")).resolve())
D = str(Path(os.environ.get("XTILE_DATA_ROOT", REPO_ROOT / "data" / "corpus76")).resolve())
LABEL_CSV = str(Path(os.environ.get("XTILE_PATCH_MANIFEST",
                                    REPO_ROOT / "manifests" / "patch_index.csv")).resolve())
CITIES = ["Italy_Parma", "Poland_Kielce", "Turkiye_Adana",
          "Turkiye_Gaziantep", "Turkiye_Hatay", "Turkiye_Malatya"]
PID_RE = re.compile(r'^(?P<city>.+?)_r\d+_c\d+_x(?P<x>\d+)_y(?P<y>\d+)_s(?P<s>\d+)$')
EXCLUDE = {0, 1, 5}
SIZE = 256


def first(g):
    r = glob.glob(g); return r[0] if r else None


COLS = [
    ("GT",         D + "/test/Optical_tif",                                    "_OPT"),
    ("pix2pix",    B + "/pix2pix/results/pix2pix_p1/test_30/pred_tif",  "_fakeB"),
    ("CycleGAN",   B + "/CycleGAN/results/cyclegan_p1/test_50/pred_tif",      "_fakeB"),
    ("CUT",        B + "/CUT2026/results_bs8/cut_p1_bs8/test_110/pred_tif",            "_fakeB"),
    ("ControlNet", B + "/ControlNet/rerun_results/test_ep179_cfg3p0/pred_tif",    "_fakeB"),
    ("Bai",        B + "/Bai_CondDiff/results/test_150/pred_tif",               "_fakeB"),
    ("Prior-strengthened base", first(E + "/d2_both_test/*_pred_tif"),          "_fakeB"),
    ("OursFinal",  E + "/scene_final/pred_tif",                                 "_fakeB"),
    ("Block consistency (retrofit)", first(E + "/hb_test/*_pred_tif"),          "_fakeB"),
]


def usable_tiles(city, lm):
    out = []
    for p in sorted(glob.glob(os.path.join(D, 'test/SAR_tif', city + '_*_SAR.tif'))):
        stem = os.path.splitext(os.path.basename(p))[0][:-4]
        m = PID_RE.match(stem)
        meta = lm.get(stem)
        if m is None or meta is None or meta[0] != 'test' or meta[1] in EXCLUDE:
            continue
        out.append((stem, int(m.group('x')), int(m.group('y'))))
    return out


def load_tile(dirp, pid, suffix):
    for cand in (os.path.join(dirp, pid + suffix + '.tif'), os.path.join(dirp, pid + '.tif')):
        if os.path.exists(cand):
            a = tifffile.imread(cand).astype('float32')
            if a.ndim == 2:
                a = np.stack([a] * 3, -1)
            if a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
                a = np.transpose(a, (1, 2, 0))
            return np.clip(a[..., :3], 0, 4500)
    return None


lm = {}
with open(LABEL_CSV, newline='', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        try:
            lm[row['patch_id']] = (row['split'].strip().lower(), int(row['coarse_label']))
        except (KeyError, ValueError):
            pass

acc = {n: {'seam': [], 'interior': []} for n, _, _ in COLS}
for city in CITIES:
    tiles = usable_tiles(city, lm)
    W = max(x for _, x, _ in tiles) + SIZE
    H = max(y for _, _, y in tiles) + SIZE

    # The hard-stitch boundary is an ownership transition, not every tile edge.
    # Computing it from the geometry also handles clamped final rows/columns whose
    # displacement is not the nominal 160-px stride.
    owner = np.full((H, W), -1, dtype=np.int32)
    owner_dist = np.full((H, W), np.inf, dtype=np.float32)
    for idx, (_, x, y) in enumerate(tiles):
        yy, xx = np.mgrid[y:y + SIZE, x:x + SIZE]
        d2 = (xx - (x + SIZE / 2.0)) ** 2 + (yy - (y + SIZE / 2.0)) ** 2
        sl = np.s_[y:y + SIZE, x:x + SIZE]
        closer = d2 < owner_dist[sl]
        owner_dist[sl][closer] = d2[closer]
        owner[sl][closer] = idx
    geom_valid = owner >= 0
    seam_v = (owner[:, 1:] != owner[:, :-1]) & geom_valid[:, 1:] & geom_valid[:, :-1]
    seam_h = (owner[1:, :] != owner[:-1, :]) & geom_valid[1:, :] & geom_valid[:-1, :]

    for name, dirp, suf in COLS:
        canvas = np.zeros((H, W, 3), 'float32')
        cover = np.zeros((H, W), bool)
        loaded = {}
        for idx, (pid, x, y) in enumerate(tiles):
            t = load_tile(dirp, pid, suf)
            if t is None:
                continue
            loaded[idx] = t
        for idx, (pid, x, y) in enumerate(tiles):
            if idx not in loaded:
                continue
            sl = np.s_[y:y + SIZE, x:x + SIZE]
            take = owner[sl] == idx
            canvas[sl][take] = loaded[idx][take]
            cover[sl][take] = True
        g = canvas.mean(-1)
        dx = np.abs(g[:, 1:] - g[:, :-1])
        dy = np.abs(g[1:, :] - g[:-1, :])
        valid_v = cover[:, 1:] & cover[:, :-1]
        valid_h = cover[1:, :] & cover[:-1, :]
        sv = np.concatenate((dx[valid_v & seam_v], dy[valid_h & seam_h]))
        iv = np.concatenate((dx[valid_v & ~seam_v], dy[valid_h & ~seam_h]))
        if not len(sv) or not len(iv):
            raise RuntimeError(f"{name}/{city}: empty seam or interior support")
        seam = float(sv.mean())
        interior = float(iv.mean())
        acc[name]['seam'].append(seam)
        acc[name]['interior'].append(interior)
    print(city, 'done', flush=True)

print(f"\n{'method':12} {'seam':>7} {'interior':>9} {'ratio':>7}  (6-city mean)")
for name, _, _ in COLS:
    s = np.mean(acc[name]['seam']); i = np.mean(acc[name]['interior'])
    print(f"{name:12} {s:7.1f} {i:9.1f} {s / (i + 1e-8):7.3f}")
print("DECOMP_ALL_DONE", flush=True)
