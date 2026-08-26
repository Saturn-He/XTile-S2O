# -*- coding: utf-8 -*-
"""Overlap-support sensitivity for OI (Supplementary Note S9).

Computes, for one method's per-tile predictions, the block-level overlap
inconsistency under two support conventions:

  OI_fixed : the primary protocol — every edge-sharing pair of each disjoint
             2x2 anchor block, compared on the nominal 96-px band of the
             stride-160 grid (identical to coherence_extra_metrics.py).
  OI_nom   : the same statistic restricted to strictly nominal-stride pairs
             (relative tile displacement exactly 160 px), on which the
             compared bands represent identical ground support. Reference
             crops are bit-identical on these pairs, so the reference value
             is exactly 0.

Pairs involving a scene's clamped final row or column (displacement < 160 px)
are the difference between the two conventions: their true overlap exceeds
96 px and the nominal band compares offset pixels.

Aggregation matches the primary protocol: pair values are averaged within a
block, then averaged over blocks with equal weight. Blocks whose four pairs
are all clamped drop out of OI_nom (test split: 260 of 264 blocks remain).

Usage (release layout):
  python evaluation/oi_support_sensitivity.py --name GT \
    --pred_dir ./data/corpus76/test/Optical_tif --suffix _OPT
  python evaluation/oi_support_sensitivity.py --name mymethod \
    --pred_dir out/scene_final/pred_tif --suffix _fakeB
"""
import argparse
import csv
import os
import re

import numpy as np
import tifffile

OPT_CLIP = 4500.0
TILE = 256
OV = 96
PID_XY = re.compile(r"_x(\d+)_y(\d+)_s\d+$")
QUAD = ["top_left_patch_id", "top_right_patch_id",
        "bottom_left_patch_id", "bottom_right_patch_id"]
EXCLUDE = {"0", "1", "5"}


def anchor_blocks(label_csv, block_csv, split):
    usable = set()
    with open(label_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if (r.get("split") or "").strip().lower() == split and \
               (r.get("coarse_label") or "").strip() not in EXCLUDE:
                usable.add(r["patch_id"].strip())
    blocks = []
    with open(block_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if (r.get("split") or "").strip().lower() != split:
                continue
            pids = [r[q].strip() for q in QUAD]
            if all(p in usable for p in pids) and \
               int(r["top_left_row"]) % 2 == 0 and int(r["top_left_col"]) % 2 == 0:
                blocks.append(pids)
    return blocks


def xy(pid):
    m = PID_XY.search(pid)
    return int(m.group(1)), int(m.group(2))


def pair_spec(pids):
    tl, tr, bl, br = pids
    return [(tl, tr, "h", xy(tr)[0] - xy(tl)[0]),
            (bl, br, "h", xy(br)[0] - xy(bl)[0]),
            (tl, bl, "v", xy(bl)[1] - xy(tl)[1]),
            (tr, br, "v", xy(br)[1] - xy(tr)[1])]


def load_refl(d, suffix, pid):
    a = tifffile.imread(os.path.join(d, pid + suffix + ".tif")).astype("float64")
    return np.clip(a[..., :3], 0, OPT_CLIP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--suffix", default="_fakeB")
    ap.add_argument("--split", default="test")
    ap.add_argument("--label_csv", default="./manifests/patch_index.csv")
    ap.add_argument("--block_csv", default="./manifests/block_2x2_index.csv")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    blocks = anchor_blocks(args.label_csv, args.block_csv, args.split)
    print(f"[{args.split}] disjoint anchor blocks: {len(blocks)}")
    block_all, block_nom = [], []
    for pids in blocks:
        tiles = {p: load_refl(args.pred_dir, args.suffix, p) for p in set(pids)}
        la, ln = [], []
        for a, b, ax, gap in pair_spec(pids):
            A, B = tiles[a], tiles[b]
            if ax == "h":
                reg = (A[:, TILE - OV:], B[:, :OV])
            else:
                reg = (A[TILE - OV:], B[:OV])
            v = float(np.abs(reg[0] - reg[1]).mean())
            la.append(v)
            if gap == 160:
                ln.append(v)
        block_all.append(float(np.mean(la)))
        if ln:
            block_nom.append(float(np.mean(ln)))
    oi_fixed, oi_nom = float(np.mean(block_all)), float(np.mean(block_nom))
    print(f"{args.name}: OI_fixed {oi_fixed:.3f} ({len(block_all)} blocks) | "
          f"OI_nom {oi_nom:.3f} ({len(block_nom)} blocks)")
    if args.out_csv:
        new = not os.path.exists(args.out_csv)
        with open(args.out_csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["method", "OI_fixed", "OI_nom", "n_blocks_fixed", "n_blocks_nom"])
            w.writerow([args.name, oi_fixed, oi_nom, len(block_all), len(block_nom)])


if __name__ == "__main__":
    main()
