import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path
import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import util.misc as misc
from s2o_norm import SAR_CLIP, OPT_CLIP

import copy
from engine_jit import train_one_epoch, evaluate

from denoiser import Denoiser
from util.datasets import FilteredPairedImageDirDataset


class PairedTrainTransform:
    """Apply paired geometric augmentation and the shared S2O normalization.

    SAR input is two-band dB backscatter in released-corpus order [VV, VH].
    Each band is clipped with ``SAR_CLIP`` and mapped to [-1, 1]. Optical
    input is three-band reflectance clipped with ``OPT_CLIP``. Resize and
    horizontal flip are applied to paired raw values before normalization.

    When ``srgb=True``, optical reflectance is converted to linear [0, 1],
    encoded with the standard sRGB transfer function, and mapped to [-1, 1].
    SAR preprocessing is unchanged; inference uses ``denormalize_opt_srgb``
    to return predictions to the clipped reflectance domain.
    """

    def __init__(self, img_size, flip_prob=0.5, srgb=False):
        self.img_size = img_size
        self.flip_prob = flip_prob
        self.srgb = srgb

    def __call__(self, sar, opt, aux=None):
        # Optional pre-normalized auxiliary raster, such as two DEM channels in
        # [-1, 1]. It receives the same resize and flip as the paired images.
        sar = torch.from_numpy(sar).float()  # (2,H,W) dB
        opt = torch.from_numpy(opt).float()  # (3,H,W) reflectance

        sar = F.resize(sar, [self.img_size, self.img_size], antialias=True)
        opt = F.resize(opt, [self.img_size, self.img_size], antialias=True)
        if aux is not None:
            aux = F.resize(torch.as_tensor(aux).float(), [self.img_size, self.img_size], antialias=True)
        if torch.rand(1).item() < self.flip_prob:
            sar = F.hflip(sar)
            opt = F.hflip(opt)
            if aux is not None:
                aux = F.hflip(aux)

        # Map each SAR dB channel independently to [-1, 1].
        for c, (lo, hi) in enumerate(SAR_CLIP):
            sar[c] = (sar[c].clamp(lo, hi) - lo) / (hi - lo) * 2.0 - 1.0

        lo, hi = OPT_CLIP
        if self.srgb:
            # Reflectance -> linear [0, 1] -> sRGB transfer -> [-1, 1].
            lin = ((opt.clamp(lo, hi) - lo) / (hi - lo)).clamp(0.0, 1.0)
            a = 0.055
            srgb = torch.where(lin <= 0.0031308,
                               12.92 * lin,
                               (1.0 + a) * torch.clamp(lin, min=1e-12).pow(1.0 / 2.4) - a)
            opt = srgb * 2.0 - 1.0
        else:
            # Default linear mapping from clipped reflectance to [-1, 1].
            opt = (opt.clamp(lo, hi) - lo) / (hi - lo) * 2.0 - 1.0

        if aux is not None:
            return sar, opt, aux
        return sar, opt
        

def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

    # training
    parser.add_argument('--epochs', default=650, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=1.6e-3, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--seed', default=77, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--gen_split', default='test', type=str,
                        help='split to export in evaluate_gen mode: test or val')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')

    # dataset
    parser.add_argument('--sar_train_path', default='./data/corpus76/train/SAR_tif', type=str,
                        help='Path to the SAR training dataset')
    parser.add_argument('--opt_train_path', default='./data/corpus76/train/Optical_tif', type=str,
                        help='Path to the optical training dataset (recommended: Optical_tif)')
    parser.add_argument('--sar_test_path', default='./data/corpus76/test/SAR_tif', type=str,
                        help='Path to the SAR testing dataset')
    parser.add_argument('--opt_test_path', default='./data/corpus76/test/Optical_tif', type=str,
                        help='Path to the optical testing dataset')
    parser.add_argument('--class_num', default=1, type=int)
    parser.add_argument('--csv_path', default='./manifests/patch_index.csv', type=str,
                        help='CSV file with patch_id, split and coarse_label')

    # checkpointing
    parser.add_argument('--output_dir', default='./runs/xtile_s2o',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Checkpoint file, or folder containing checkpoint-last.pth')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--keep_outputs', action='store_true',
                        help='Keep generated outputs after evaluation')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    # Perceptual-loss weight; zero disables the term.
    parser.add_argument('--lambda_perc', type=float, default=0.0,
                        help='weight of t-weighted LPIPS(x0_pred, target) perceptual loss')

    # Training-only DINOv3 optical feature alignment.
    parser.add_argument('--use_dino_prior', action='store_true',
                        help='align JiT hidden IMAGE tokens to frozen DINOv3 features of the optical GT (training only)')
    parser.add_argument('--dino_model', type=str, default='facebook/dinov3-vitb16-pretrain-lvd1689m',
                        help='HF model id of the frozen optical teacher (gated: needs accepted license + HF_TOKEN)')
    parser.add_argument('--dino_layers', type=str, default='6,7,8,9',
                        help='ONE-BASED block numbers used on BOTH networks, comma-separated')
    parser.add_argument('--lambda_dino', type=float, default=0.5,
                        help='weight of the multi-layer cosine feature-alignment loss')

    # DEM conditioning: disabled, channel concatenation, or gated cross-attention.
    parser.add_argument('--use_dem', type=str, default='off', choices=['off', 'concat', 'ca'],
                        help='DEM input fusion: concat = 2 extra input channels; ca = gated cross-attn tokens')
    parser.add_argument('--dem_dir', type=str, default='./data/dem_tif',
                        help='root with {train,val,test}/<pid>_DEM.tif + _SLOPE.tif and dem_norm.json')
    parser.add_argument('--dem_ca_layers', type=int, nargs='+', default=[4, 8],
                        help='zero-based JiT blocks for gated DEM cross-attention')
    parser.add_argument('--allow_combined_priors', action='store_true',
                        help='explicitly allow DEM conditioning together with DINOv3 feature alignment')

    # Optional initialization for coherence fine-tuning.
    parser.add_argument('--ft_init', type=str, default='',
                        help='checkpoint dir/file to initialize model weights from (fine-tune): '
                             'loads MODEL only (strict except new coherence modules), fresh optimizer, '
                             'EMA re-initialized from the loaded weights, start_epoch=0')
    parser.add_argument('--ft_max_steps', type=int, default=0,
                        help='hard stop after exactly this many optimizer steps (0 = off)')

    # target-domain switch (off by default => identical linear-reflectance baseline)
    parser.add_argument('--srgb_target', action='store_true',
                        help='train optical target in sRGB display domain (gamma) instead of linear reflectance; '
                             'export uses denormalize_opt_srgb. Off = original linear pipeline.')

    # Cross-tile coherence components; all are disabled by default.
    parser.add_argument('--use_cpa', action='store_true',
                        help='cross-patch attention (CPA) across each 2x2 block')
    parser.add_argument('--cpa_layers', type=int, nargs='+', default=[4, 8],
                        help='ZERO-BASED insertion indices for cross-patch attention; '
                             'default 4 8 means after one-based transformer blocks 5 and 9')
    parser.add_argument('--use_blocks', action='store_true',
                        help='train on 2x2 block batches with geometry-matched block and overlap losses')
    parser.add_argument('--block_csv', type=str, default='./manifests/block_2x2_index.csv',
                        help='path to the resolved block_2x2_index.csv manifest (required if --use_blocks)')
    parser.add_argument('--block_bsz', type=int, default=4,
                        help='blocks per batch G (effective tiles per step = 4*G)')
    parser.add_argument('--lambda_block', type=float, default=1.0,
                        help='weight of the 2x2 mosaic-vs-GT L1 term')
    parser.add_argument('--lambda_overlap', type=float, default=0.5,
                        help='weight of the GT-free overlap-consistency term')
    parser.add_argument('--block_t_min', type=float, default=0.0,
                        help='hard t-gate fallback for block terms (0 = linear-in-t weighting)')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # Data augmentation transforms (paired for SAR/OPT consistency)
    # Construct the training dataset only for training; generation mode needs
    # only the requested evaluation split.
    if not args.evaluate_gen:
        transform_train = PairedTrainTransform(args.img_size, srgb=args.srgb_target)

        dataset_train = FilteredPairedImageDirDataset(
            args.sar_train_path,
            args.opt_train_path,
            csv_path=args.csv_path,
            split="train",
            transform=transform_train,
            dem_dir=(os.path.join(args.dem_dir, 'train') if args.use_dem != 'off' else None),
            dem_norm=(__import__('dem_prior').DemNorm(os.path.join(args.dem_dir, 'dem_norm.json'))
                      if args.use_dem != 'off' else None),
        )
        print(dataset_train)

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train =", sampler_train)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True
        )

        if args.use_blocks:
            from block_dataset import BlockPairedDataset, block_collate
            block_transform = PairedTrainTransform(args.img_size, flip_prob=0.0, srgb=args.srgb_target)  # deterministic (block coherence)
            dataset_block = BlockPairedDataset(
                args.sar_train_path, args.opt_train_path, args.block_csv, args.csv_path,
                split="train", transform=block_transform,
                dem_dir=(os.path.join(args.dem_dir, 'train') if args.use_dem != 'off' else None),
                dem_norm=(__import__('dem_prior').DemNorm(os.path.join(args.dem_dir, 'dem_norm.json'))
                          if args.use_dem != 'off' else None))
            print("Block dataset stats:", dataset_block.stats)
            sampler_block = torch.utils.data.DistributedSampler(
                dataset_block, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            data_loader_block = torch.utils.data.DataLoader(
                dataset_block, sampler=sampler_block, batch_size=args.block_bsz,
                num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
                collate_fn=block_collate)

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser
    model = Denoiser(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Resume from checkpoint if provided
    checkpoint_path = (args.resume if args.resume.endswith('.pth') else
                       os.path.join(args.resume, "checkpoint-last.pth")) if args.resume else None
    if args.ft_init:
        # coherence fine-tune init: model weights only; NEW coherence modules (CPA)
        # are the only keys allowed to be missing; optimizer stays fresh; EMA <- loaded weights.
        ft_path = args.ft_init if args.ft_init.endswith('.pth') else os.path.join(args.ft_init, 'checkpoint-last.pth')
        ck = torch.load(ft_path, map_location='cpu')
        missing, unexpected = model_without_ddp.load_state_dict(ck['model'], strict=False)
        allowed_prefix = ('net.cpa',)
        bad_missing = [k for k in missing if not k.startswith(allowed_prefix)]
        assert not bad_missing and not unexpected, \
            f"[ft_init] key mismatch: bad_missing={bad_missing[:6]} unexpected={list(unexpected)[:6]}"
        print(f"[ft_init] loaded {ft_path}; fresh-init new-module keys: {len(missing)} "
              f"(all under {allowed_prefix}); optimizer fresh; EMA re-initialized")
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        del ck
    elif checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    # Evaluate generation
    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, args, 0, batch_size=args.gen_bsz, log_writer=log_writer)
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    loader = data_loader_block if args.use_blocks else data_loader_train
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            loader.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, loader, optimizer, device, epoch, log_writer=log_writer, args=args)

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )

        if epoch % 20 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
