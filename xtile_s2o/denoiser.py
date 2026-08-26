import torch
import torch.nn as nn
from model_jit import JiT_models


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        # DEM conditioning: off, channel concatenation, or gated cross-attention.
        # Concatenation increases the model input from five to seven channels;
        # cross-attention encodes the two DEM channels as tokens.
        # Carrier convention: `sar_img` = [VV,VH] (off) or [VV,VH,elev,slope].
        self.use_dem = str(getattr(args, 'use_dem', 'off'))
        assert self.use_dem in ('off', 'concat', 'ca'), f"bad --use_dem {self.use_dem}"
        if self.use_dem != 'off' and bool(getattr(args, 'use_dino_prior', False)) \
                and not bool(getattr(args, 'allow_combined_priors', False)):
            raise ValueError("Single-prior DEM experiments require --use_dino_prior false "
                             "(pass --allow_combined_priors only for the explicit +both run)")
        in_ch = 5 + (2 if self.use_dem == 'concat' else 0)
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=in_ch,
            out_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            use_cpa=getattr(args, 'use_cpa', False),
            cpa_layers=tuple(getattr(args, 'cpa_layers', (4, 8))),
            use_dem_ca=(self.use_dem == 'ca'),
            dem_ca_layers=tuple(getattr(args, 'dem_ca_layers', (4, 8))),
        )
        if self.use_dem == 'ca':
            from dem_prior import DemEncoder
            # REGISTERED (runs at inference), unlike the DINO training-only projectors.
            self.dem_encoder = DemEncoder(hidden=self.net.hidden_size)
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

        # Perceptual loss; disabled by default when lambda_perc is zero.
        # lpips model is held in a plain list so nn.Module does NOT register it =>
        # excluded from parameters()/EMA/state_dict (clean checkpoints, no DDP sync).
        self.lambda_perc = float(getattr(args, 'lambda_perc', 0.0))
        self._perc = []

        # Training-only DOGAN-style DINOv3 optical feature alignment.
        # Frozen ~86M teacher lives in a plain list (like _perc) => NOT registered.
        # The small per-layer projectors ARE registered normally (standard
        # optimizer/EMA/checkpoint; never called at inference). Flag off => no new
        # registered parameters, so disabling the feature alignment leaves the
        # baseline execution path unchanged.
        self.use_dino_prior = bool(getattr(args, 'use_dino_prior', False))
        self.lambda_dino = float(getattr(args, 'lambda_dino', 0.5))
        self.dino_layers = [int(v) for v in str(getattr(args, 'dino_layers', '6,7,8,9')).split(',') if v.strip()]
        self._dino_model_id = str(getattr(args, 'dino_model', 'facebook/dinov3-vitb16-pretrain-lvd1689m'))
        self._dino = []
        if self.use_dino_prior:
            from dino_prior import MultiLayerDinoLoss
            self.dino_loss = MultiLayerDinoLoss(len(self.dino_layers))

    def _dino_teacher(self, device):
        if not self._dino:
            from dino_prior import DinoV3OpticalTeacher
            self._dino.append(DinoV3OpticalTeacher(self._dino_model_id, self.dino_layers, device))
        return self._dino[0]

    def _split_cond(self, cond):
        """cond = the sar_img carrier. Returns (channels_to_concat, dem_tokens_or_None).
        off:    (2ch, None) | concat: (4ch, None) | ca: (2ch, tokens from ch 2:4)."""
        if self.use_dem == 'ca':
            return cond[:, :2], self.dem_encoder(cond[:, 2:4])
        return cond, None

    def _perceptual(self, x_pred, opt_img, t):
        """t-weighted LPIPS(x_pred, opt_img); both in [-1,1] 3ch. Trust at high t (clean est)."""
        if not self._perc:
            import lpips
            m = lpips.LPIPS(net='vgg', verbose=False).to(opt_img.device).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            self._perc.append(m)
        m = self._perc[0]
        d = m(x_pred.float().clamp(-1, 1), opt_img.float()).flatten()   # (N,) fp32 for VGG stability
        w = t.flatten()
        return (w * d).sum() / w.sum().clamp_min(1e-6)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, opt_img, sar_img, labels=None, block=False,
                lambda_block=1.0, lambda_overlap=1.0, block_t_min=0.0):
        if block:
            # Channel-concatenated DEM conditioning follows the block path;
            # DEM cross-attention is not supported by block training.
            assert self.use_dem in ('off', 'concat'), \
                "DEM 'ca' + block training not supported (concat carrier only)"
            return self._forward_block(opt_img, sar_img, labels,
                                       lambda_block, lambda_overlap, block_t_min)
        if labels is None:
            labels = torch.zeros(opt_img.size(0), device=opt_img.device, dtype=torch.long)
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(opt_img.size(0), device=opt_img.device).view(-1, *([1] * (opt_img.ndim - 1)))
        e = torch.randn_like(opt_img) * self.noise_scale

        z = t * opt_img + (1 - t) * e
        v = (opt_img - z) / (1 - t).clamp_min(self.t_eps)

        cond_ch, dem_tokens = self._split_cond(sar_img)
        z_cond = torch.cat([z, cond_ch], dim=1)
        if self.use_dino_prior:
            x_pred, jit_feats = self.net(z_cond, t.flatten(), labels_dropped,
                                         return_layers=self.dino_layers, dem_tokens=dem_tokens)
        else:
            x_pred = self.net(z_cond, t.flatten(), labels_dropped, dem_tokens=dem_tokens)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 (flow-matching) loss
        l_flow = ((v - v_pred) ** 2).mean(dim=(1, 2, 3)).mean()
        loss = l_flow
        loss_dict = {'L_flow': float(l_flow.detach())}

        # Optional perceptual term on the x0 estimate (time-weighted); disabled when lambda_perc=0.
        if self.lambda_perc > 0:
            l_perc = self._perceptual(x_pred, opt_img, t)
            loss = loss + self.lambda_perc * l_perc
            loss_dict['L_perc'] = float(l_perc.detach())

        # DINOv3 optical feature alignment (teacher stop-gradient; student gradients
        # flow through pool+projector into the JiT backbone).
        if self.use_dino_prior:
            teacher = self._dino_teacher(opt_img.device)
            with torch.no_grad():
                dino_feats = teacher.extract(opt_img)
            l_dino, per_layer = self.dino_loss(jit_feats, dino_feats)
            loss = loss + self.lambda_dino * l_dino
            loss_dict['L_dino'] = float(l_dino.detach())
            for lid, lv in zip(self.dino_layers, per_layer):
                loss_dict['L_d%d' % lid] = float(lv.detach())

        return loss, loss_dict

    def _forward_block(self, opt_img, sar_img, labels=None,
                       lambda_block=1.0, lambda_overlap=1.0, block_t_min=0.0):
        # opt_img: (4G, 3, H, W); sar_img: (4G, 2 or 4, H, W) conditioning CARRIER
        # ([VV,VH] or [VV,VH,elev,slope]), block-major [TL,TR,BL,BR] per block.
        # Returns (base_loss, L_block, L_overlap, loss_dict): base_loss carries the
        # Full per-tile base objective (flow, perceptual, and DINOv3 alignment terms,
        # applied here); L_block/L_overlap are t-weighted means WITHOUT the lambdas
        # (the training loop applies lambda_block/lambda_overlap).
        from block_losses import compute_block_losses
        B = opt_img.size(0)
        G = B // 4
        if labels is None:
            labels = torch.zeros(B, device=opt_img.device, dtype=torch.long)
        labels = self.drop_labels(labels) if self.training else labels
        # one t per block, shared across its 4 tiles (x_hat0 at a single noise level)
        t_blk = self.sample_t(G, device=opt_img.device)
        t = t_blk.view(G, 1, 1, 1, 1).expand(G, 4, 1, 1, 1).reshape(B, 1, 1, 1)
        e = torch.randn_like(opt_img) * self.noise_scale
        z = t * opt_img + (1 - t) * e
        v = (opt_img - z) / (1 - t).clamp_min(self.t_eps)
        cond_ch, dem_tokens = self._split_cond(sar_img)
        z_cond = torch.cat([z, cond_ch], dim=1)
        if self.use_dino_prior:
            x_pred, jit_feats = self.net(z_cond, t.flatten(), labels, n_per_block=4,
                                         return_layers=self.dino_layers, dem_tokens=dem_tokens)
        else:
            x_pred = self.net(z_cond, t.flatten(), labels, n_per_block=4, dem_tokens=dem_tokens)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        L_fm = ((v - v_pred) ** 2).mean(dim=(1, 2, 3)).mean()
        base = L_fm
        loss_dict = {'L_flow': float(L_fm.detach())}
        if self.lambda_perc > 0:
            l_perc = self._perceptual(x_pred, opt_img, t)
            base = base + self.lambda_perc * l_perc
            loss_dict['L_perc'] = float(l_perc.detach())
        if self.use_dino_prior:
            teacher = self._dino_teacher(opt_img.device)
            with torch.no_grad():
                dino_feats = teacher.extract(opt_img)
            l_dino, per_layer = self.dino_loss(jit_feats, dino_feats)
            base = base + self.lambda_dino * l_dino
            loss_dict['L_dino'] = float(l_dino.detach())
            for lid, lv in zip(self.dino_layers, per_layer):
                loss_dict['L_d%d' % lid] = float(lv.detach())
        C, H, W = opt_img.shape[1], opt_img.shape[2], opt_img.shape[3]
        d = compute_block_losses(
            x_pred.view(G, 4, C, H, W), opt_img.view(G, 4, C, H, W),
            t=t_blk, lambda_block=lambda_block, lambda_overlap=lambda_overlap,
            block_t_min=block_t_min)
        return base, d["L_block"], d["L_overlap"], loss_dict

    @torch.no_grad()
    def generate(self, sar_img, labels=None, n_per_block=1):
        # n_per_block=4: batch is block-major [TL,TR,BL,BR]*G -> CPA mixes the 4
        # tiles of each block at every net call (matches _forward_block training).
        # Default 1 = per-patch sampling, identical to before (CPA inactive).
        if labels is None:
            labels = torch.zeros(sar_img.size(0), device=sar_img.device, dtype=torch.long)
        device = sar_img.device
        bsz = sar_img.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels, sar_img, n_per_block=n_per_block)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels, sar_img, n_per_block=n_per_block)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels, sar_img, n_per_block=1):
        cond_ch, dem_tokens = self._split_cond(sar_img)
        # conditional
        z_cond = torch.cat([z, cond_ch], dim=1)
        x_cond = self.net(z_cond, t.flatten(), labels, n_per_block=n_per_block, dem_tokens=dem_tokens)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net(z_cond, t.flatten(), torch.full_like(labels, self.num_classes),
                            n_per_block=n_per_block, dem_tokens=dem_tokens)
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels, sar_img, n_per_block=1):
        v_pred = self._forward_sample(z, t, labels, sar_img, n_per_block=n_per_block)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels, sar_img, n_per_block=1):
        v_pred_t = self._forward_sample(z, t, labels, sar_img, n_per_block=n_per_block)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels, sar_img, n_per_block=n_per_block)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
