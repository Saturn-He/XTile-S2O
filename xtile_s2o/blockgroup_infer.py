# -*- coding: utf-8 -*-
"""Block-context inference for XTile-S2O and matched ablations.

CPA requires four block-major streams in each forward pass. Tile-wise inference
does not activate CPA. This script samples batches ordered as
``[TL, TR, BL, BR]`` for each block and passes ``n_per_block=4`` through the
ordinary differential equation sampling chain. The same path can evaluate
checkpoints without CPA, where ``n_per_block`` has no effect.

Coverage: by default, only disjoint blocks are retained. Their top-left tiles
have even row and column indices, so each exported tile is predicted once. With
``--all_blocks``, every valid row in ``--block_csv`` is processed in CSV order.
If a tile occurs in multiple rows, its flat ``pred_tif`` and ``pred_png`` files
are overwritten; the final file comes from the last covering block. Passing the
ordered ``manifests/scene_cover_test.csv`` reproduces the paper's 412-block
last-covering-block ownership rule. Passing the complete
``manifests/block_2x2_index.csv`` instead evaluates the full valid-block
population and is not equivalent to that scene cover. Per-block mosaics saved
with ``--save_mosaics`` are keyed by ``block_id`` and do not collide.

Run from the repository root:

  torchrun --standalone --nproc_per_node=1 xtile_s2o/blockgroup_infer.py \
    --model JiT-B/8 --img_size 256 \
    --resume runs/xtile_s2o/checkpoint-40.pth \
    --sar_test_path data/corpus76/test/SAR_tif \
    --opt_test_path data/corpus76/test/Optical_tif \
    --csv_path manifests/patch_index.csv \
    --block_csv manifests/scene_cover_test.csv --all_blocks \
    --gen_split test --gen_bsz 32 --save_mosaics \
    --output_dir out/test_block

Architecture and target-domain options are restored from checkpoint arguments
when present; command-line values are fallbacks for older checkpoints.
"""
import argparse
import copy
import os
import re

import numpy as np
import tifffile
import torch

from main_jit import get_args_parser, PairedTrainTransform
from denoiser import Denoiser
from block_dataset import BlockPairedDataset, block_collate, BLOCK_OFFSETS
from s2o_norm import denormalize_opt, denormalize_opt_srgb

PID_RC = re.compile(r'_r(\d+)_c(\d+)_')


def pct_png(refl):
    """(H,W,3) reflectance -> uint8 RGB, per-channel 2-98% stretch (paper display recipe)."""
    a = np.asarray(refl).astype('float32')
    out = np.zeros(a.shape[:2] + (3,), 'float32')
    for c in range(3):
        ch = a[..., c]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        out[..., c] = np.clip((ch - lo) / max(hi - lo, 1e-3), 0, 1)
    return (out * 255).astype('uint8')


def load_model(args):
    """Build Denoiser with the checkpoint's own arch flags, then load EMA weights."""
    ckpt_path = args.resume if args.resume.endswith('.pth') else os.path.join(args.resume, "checkpoint-last.pth")
    ck = torch.load(ckpt_path, map_location='cpu')
    ck_args = ck.get('args', None)
    if ck_args is not None:                      # restore arch-relevant flags from training
        for k in ('use_cpa', 'cpa_layers', 'srgb_target', 'model', 'img_size', 'class_num',
                  'use_dem', 'dem_ca_layers', 'use_dino_prior', 'lambda_dino', 'dino_layers',
                  'dino_model', 'allow_combined_priors'):  # DEM changes in_channels; dino_loss must exist for strict load
            if hasattr(ck_args, k):
                setattr(args, k, getattr(ck_args, k))
    print(f"[load] ckpt={ckpt_path} epoch={ck.get('epoch')} "
          f"use_cpa={getattr(args, 'use_cpa', False)} cpa_layers={getattr(args, 'cpa_layers', None)} "
          f"srgb_target={getattr(args, 'srgb_target', False)}")
    model = Denoiser(args).to(args.device)
    model.load_state_dict(ck['model'])
    model.ema_params1 = [ck['model_ema1'][n].to(args.device) for n, _ in model.named_parameters()]
    sd = copy.deepcopy(model.state_dict())       # swap EMA weights into the net
    for i, (n, _) in enumerate(model.named_parameters()):
        sd[n] = model.ema_params1[i]
    model.load_state_dict(sd)
    model.eval()
    del ck
    print(f"[load] CPA layers active at inference: {sorted(model.net.cpa_layers) if model.net.cpa_layers else 'NONE'}")
    return model


def main():
    parser = argparse.ArgumentParser('block-grouped inference', parents=[get_args_parser()])
    parser.add_argument('--num_blocks', type=int, default=0, help='limit #blocks (0 = all)')
    parser.add_argument('--all_blocks', action='store_true',
                        help=('process every valid block_csv row in CSV order; repeated flat tile '
                              'outputs are overwritten, so the last covering block wins '
                              '(default: disjoint blocks only, even r/c)'))
    parser.add_argument('--save_mosaics', action='store_true',
                        help='also save 416x416 stitched pred/GT mosaics (overlap-averaged) for seam inspection')
    args = parser.parse_args()
    args.device = 'cuda'
    if args.gen_bsz > 32:   # gen_bsz = BLOCKS per batch; net sees 4x that many tiles
        print(f"[warn] gen_bsz {args.gen_bsz} -> 32 (4 tiles/block; avoid OOM)")
        args.gen_bsz = 32
    torch.manual_seed(args.seed)

    model = load_model(args)
    use_srgb = bool(getattr(args, 'srgb_target', False))
    denorm = denormalize_opt_srgb if use_srgb else denormalize_opt

    transform = PairedTrainTransform(args.img_size, flip_prob=0.0, srgb=use_srgb)
    use_dem = str(getattr(args, 'use_dem', 'off')) != 'off'
    dem_dir = dem_norm = None
    if use_dem:                                  # Emit [VV, VH, elevation, slope] when DEM conditioning is enabled.
        from dem_prior import DemNorm
        dem_dir = os.path.join(args.dem_dir, args.gen_split)
        dem_norm = DemNorm(os.path.join(args.dem_dir, 'dem_norm.json'))
        print(f"[data] DEM carrier ON: {dem_dir}")
    dataset = BlockPairedDataset(
        args.sar_test_path, args.opt_test_path, args.block_csv, args.csv_path,
        split=args.gen_split, transform=transform, dem_dir=dem_dir, dem_norm=dem_norm)
    print(f"[data] split={args.gen_split} stats={dataset.stats}")

    if not args.all_blocks:                      # disjoint cover: TL tile at even (r, c)
        kept = []
        for s in dataset.samples:
            m = PID_RC.search(str(s[1][0].name))
            if m and int(m.group(1)) % 2 == 0 and int(m.group(2)) % 2 == 0:
                kept.append(s)
        print(f"[data] disjoint filter: {len(dataset.samples)} -> {len(kept)} blocks")
        dataset.samples = kept
    else:
        print("[data] ordered all-block mode: repeated flat tile outputs are overwritten "
              "in block_csv order (last covering block wins)")
    if args.num_blocks > 0:
        dataset.samples = dataset.samples[:args.num_blocks]
        print(f"[data] limited to {len(dataset.samples)} blocks")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.gen_bsz, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=block_collate)

    out_tif = os.path.join(args.output_dir, 'pred_tif')
    out_png = os.path.join(args.output_dir, 'pred_png')
    os.makedirs(out_tif, exist_ok=True)
    os.makedirs(out_png, exist_ok=True)
    if args.save_mosaics:
        out_mos = os.path.join(args.output_dir, 'mosaics_png')
        os.makedirs(out_mos, exist_ok=True)

    from PIL import Image
    n_tiles = 0
    for bi, batch in enumerate(loader):
        sar = batch['sar'].to(args.device).to(torch.float32)        # (G,4,2,H,W)
        G = sar.size(0)
        sar = sar.view(G * 4, *sar.shape[2:])                       # block-major [TL,TR,BL,BR]*G
        labels = torch.zeros(G * 4, device=args.device, dtype=torch.long)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model.generate(sar, labels, n_per_block=4)        # (4G,3,H,W) in [-1,1]
        out = out.detach().float().cpu()

        for g in range(G):
            bid = batch['block_id'][g]
            # tile patch_ids from the sample list (loader is shuffle=False, in order)
            sample = dataset.samples[bi * args.gen_bsz + g]
            tile_pids = [str(p.name).replace('_SAR.tif', '').replace('.tif', '') for p in sample[1]]

            if args.save_mosaics:
                canvas = np.zeros((416, 416, 3), 'float64')
                weight = np.zeros((416, 416, 1), 'float64')
                gt_canvas = np.zeros((416, 416, 3), 'float64')

            for k in range(4):
                pred_refl = denorm(out[g * 4 + k], layout='HWC')    # (H,W,3) uint16 reflectance
                pid = tile_pids[k]
                tifffile.imwrite(os.path.join(out_tif, f'{pid}_fakeB.tif'), pred_refl)
                Image.fromarray(pct_png(pred_refl)).save(os.path.join(out_png, f'{pid}_fakeB.png'))
                n_tiles += 1
                if args.save_mosaics:
                    x0, y0 = BLOCK_OFFSETS[k]
                    canvas[y0:y0 + 256, x0:x0 + 256] += pred_refl.astype('float64')
                    weight[y0:y0 + 256, x0:x0 + 256] += 1.0
                    gt_refl = denorm(batch['opt'][g, k], layout='HWC').astype('float64')
                    gt_canvas[y0:y0 + 256, x0:x0 + 256] += gt_refl

            if args.save_mosaics:
                mos = (canvas / np.clip(weight, 1, None))
                gt_mos = (gt_canvas / np.clip(weight, 1, None))
                Image.fromarray(pct_png(mos)).save(os.path.join(out_mos, f'{bid}_pred.png'))
                Image.fromarray(pct_png(gt_mos)).save(os.path.join(out_mos, f'{bid}_gt.png'))

        print(f"[gen] batch {bi + 1}/{len(loader)}  tiles={n_tiles}", flush=True)

    print(f"DONE. {n_tiles} tile predictions -> {out_tif}")
    if args.save_mosaics:
        print(f"      mosaics -> {out_mos}")


if __name__ == '__main__':
    main()
