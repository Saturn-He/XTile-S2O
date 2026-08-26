import math
import sys
import os
import shutil

import torch
import numpy as np
import cv2
import tifffile
import torchvision.transforms as transforms

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import copy

from util.datasets import FilteredImageDirDataset
from s2o_norm import SAR_CLIP, OPT_CLIP, denormalize_opt, denormalize_opt_srgb  # Canonical normalization definitions.


def _strip_sar_suffix(fname):
    """Return a patch identifier after removing the extension and ``_SAR`` suffix."""
    stem = os.path.splitext(fname)[0]
    return stem[:-4] if stem.endswith('_SAR') else stem


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    use_blocks = getattr(args, 'use_blocks', False)
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # coherence fine-tune: hard stop at an exact optimizer-step budget
        if getattr(args, 'ft_max_steps', 0) > 0:
            gstep = epoch * len(data_loader) + data_iter_step
            if gstep >= args.ft_max_steps:
                print(f"[ft] reached ft_max_steps={args.ft_max_steps} "
                      f"(epoch {epoch}, iter {data_iter_step}); stopping.")
                break

        if use_blocks:
            # block batch: (G,4,C,H,W) block-major -> flatten to (4G,C,H,W)
            sar = batch["sar"].to(device, non_blocking=True).to(torch.float32)
            opt = batch["opt"].to(device, non_blocking=True).to(torch.float32)
            G = sar.shape[0]
            sar = sar.view(G * 4, *sar.shape[2:])
            opt = opt.view(G * 4, *opt.shape[2:])
            labels = torch.zeros(G * 4, device=device, dtype=torch.long)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                base, L_blk, L_ov, loss_dict = model(opt, sar, labels, block=True,
                                                     lambda_block=args.lambda_block,
                                                     lambda_overlap=args.lambda_overlap,
                                                     block_t_min=args.block_t_min)
            loss = base + args.lambda_block * L_blk + args.lambda_overlap * L_ov
            loss_dict = dict(loss_dict)
            loss_dict['L_block'] = float(L_blk.detach())
            loss_dict['L_overlap'] = float(L_ov.detach())
        else:
            sar_img, opt_img = batch
            # SAR/OPT already normalized to [-1,1] by the dataset transform; just move to device.
            sar_img = sar_img.to(device, non_blocking=True).to(torch.float32)
            opt_img = opt_img.to(device, non_blocking=True).to(torch.float32)
            labels = torch.zeros(opt_img.size(0), device=device, dtype=torch.long)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, loss_dict = model(opt_img, sar_img, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        metric_logger.update(**loss_dict)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
                for k, v in loss_dict.items():
                    log_writer.add_scalar('train_' + k, v, epoch_1000x)


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):
    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()

    # Select the evaluation/export split; validation exports support checkpoint selection.
    gen_split = getattr(args, 'gen_split', 'test')

    # Select the inverse transform that matches the model's optical target domain.
    use_srgb = getattr(args, 'srgb_target', False)
    denorm_fn = denormalize_opt_srgb if use_srgb else denormalize_opt
    print("OPT target domain:", "sRGB (gamma)" if use_srgb else "linear reflectance")

    # SAR read as tif (2-band dB) by the dataset (transform=None); normalization done below.
    sar_dataset = FilteredImageDirDataset(
        args.sar_test_path,
        csv_path=args.csv_path,
        split=gen_split,
        transform=None,
        mode="SAR_TIF",   # signals tif reading in the dataset
    )
    sampler = torch.utils.data.DistributedSampler(
        sar_dataset, num_replicas=world_size, rank=local_rank, shuffle=False
    )
    data_loader = torch.utils.data.DataLoader(
        sar_dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    num_images = min(args.num_images, len(sar_dataset))
    num_steps = (num_images // (batch_size * world_size)) + 1

    # Construct the folder for visualization PNGs and the optional online FID calculation.
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    # Keep reflectance TIFFs in a separate directory from visualization PNGs.
    pred_tif_folder = save_folder + "_pred_tif"
    print("Save to:", save_folder)
    print("Pred TIF to:", pred_tif_folder)
    if misc.get_rank() == 0:
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(pred_tif_folder, exist_ok=True)

    # Evaluate using the canonical exponential-moving-average parameter set.
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    # Prebuild the per-channel SAR bounds from the canonical normalization module.
    sar_lo = torch.tensor([SAR_CLIP[0][0], SAR_CLIP[1][0]], dtype=torch.float32).view(1, 2, 1, 1)
    sar_hi = torch.tensor([SAR_CLIP[0][1], SAR_CLIP[1][1]], dtype=torch.float32).view(1, 2, 1, 1)

    # Reuse the training-time DEM normalization constants for each patch.
    use_dem = str(getattr(args, 'use_dem', 'off')) != 'off'
    if use_dem:
        from dem_prior import DemNorm, load_dem_pair
        dem_norm = DemNorm(os.path.join(args.dem_dir, 'dem_norm.json'))
        dem_split_dir = os.path.join(args.dem_dir, gen_split)

    img_count = 0
    for i, (sar_img, sar_names) in enumerate(data_loader):
        if img_count >= num_images:
            break
        print("Generation step {}/{}".format(i, num_steps))

        sar_img = sar_img.to(torch.device(args.device)).to(torch.float32)
        sar_img = torch.nn.functional.interpolate(
            sar_img, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False)

        # Map each clipped SAR dB channel to [-1, 1], as in the training transform.
        lo = sar_lo.to(sar_img.device)
        hi = sar_hi.to(sar_img.device)
        sar_img = torch.max(torch.min(sar_img, hi), lo)
        sar_img = (sar_img - lo) / (hi - lo) * 2.0 - 1.0

        labels_gen = torch.zeros(sar_img.size(0), device=sar_img.device, dtype=torch.long)

        if use_dem:
            dems = torch.stack([dem_norm(load_dem_pair(dem_split_dir, _strip_sar_suffix(n)))
                                for n in sar_names[:sar_img.size(0)]]).to(sar_img.device)
            if dems.shape[-1] != args.img_size:
                dems = torch.nn.functional.interpolate(
                    dems, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False)
            sar_img = torch.cat([sar_img, dems], dim=1)   # carrier [VV,VH,elev,slope]

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(sar_img, labels_gen)  # [-1,1], (B,3,H,W)

        torch.distributed.barrier()
        sampled_images = sampled_images.detach().float().cpu()

        for b_id in range(sampled_images.size(0)):
            img_id = img_count + b_id
            if img_id >= num_images:
                break
            patch_id = _strip_sar_suffix(sar_names[b_id])

            # Export quantitative predictions as uint16 reflectance TIFFs. The sRGB
            # target path applies the exact inverse transfer function first.
            pred_refl = denorm_fn(sampled_images[b_id], layout='HWC')         # (H,W,3) uint16 reflectance
            tifffile.imwrite(os.path.join(pred_tif_folder, f"{patch_id}_fakeB.tif"), pred_refl)

            # Export stretched 8-bit PNGs for visualization and optional online FID;
            # quantitative metrics use the reflectance TIFFs above.
            VIS_CLIP = 3000.0
            arr = pred_refl.astype(np.float32)                               # (H,W,3) reflectance
            gen_img = np.round(np.clip(arr / VIS_CLIP * 255.0, 0, 255)).astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, sar_names[b_id].replace('.tif', '.png')), gen_img)

        img_count += sampled_images.size(0)

    torch.distributed.barrier()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    # Optional online FID and Inception Score use visualization PNGs. Reported
    # reflectance-domain metrics should be computed from ``*_pred_tif`` outputs.
    if log_writer is not None:
        if args.img_size == 256:
            fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
        elif args.img_size == 512:
            fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        if not args.keep_outputs:
            shutil.rmtree(save_folder)

    torch.distributed.barrier()
