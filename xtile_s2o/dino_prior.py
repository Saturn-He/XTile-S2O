# -*- coding: utf-8 -*-
"""Training-only DINOv3 optical feature alignment.

The implementation follows the DOGAN-style representation-alignment design.

Teacher: frozen DINOv3 ViT-B/16 fed the *real optical reference* (model target domain
[-1,1] sRGB -> [0,1] -> ImageNet normalization). Held OUTSIDE nn.Module
registration (plain object in a list, like the LPIPS `_perc` precedent) so it
never enters parameters()/DDP/EMA/state_dict.

Student side: JiT IMAGE tokens (in-context tokens already sliced away by
model_jit) from the matching blocks, 32x32 -> 2x2 avg-pool -> 16x16, then a
small per-layer LayerNorm+Linear projector (registered for standard
optimizer/EMA/checkpoint; never called at inference).

Loss: per-layer mean(1 - cosine), averaged over layers. Teacher stopgrad only;
student gradients flow into the JiT backbone (that is the mechanism).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_IMNET_MEAN = (0.485, 0.456, 0.406)
_IMNET_STD = (0.229, 0.224, 0.225)


class DinoV3OpticalTeacher:
    """Frozen DINOv3 feature extractor. Plain class on purpose (non-registered)."""

    def __init__(self, model_id, layers_one_based, device):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval().requires_grad_(False)
        self.model.to(device)
        self.layers = [int(b) for b in layers_one_based]   # one-based block numbers
        depth = int(getattr(self.model.config, 'num_hidden_layers', 12))
        assert all(1 <= b <= depth for b in self.layers), \
            f"dino_layers {self.layers} out of range for depth-{depth} teacher"
        # Teacher-agnostic geometry: input size chosen so the patch grid is 16x16
        # (DINOv3 ViT-B/16 -> 256px; DINOv2 ViT-B/14 -> 224px). Grid alignment with
        # the pooled JiT student (16x16) is therefore identical for either teacher.
        self.grid = 16
        patch = int(getattr(self.model.config, 'patch_size', 16))
        self.input_size = self.grid * patch
        # HF sequence layout: [CLS] + register tokens (v3: 4, v2: none) + patch tokens.
        # Read from config; the 256-token assert below is the backstop.
        self.num_prefix = 1 + int(getattr(self.model.config, 'num_register_tokens', 0))
        self._mean = torch.tensor(_IMNET_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMNET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def extract(self, opt_img_m1p1):
        """opt_img_m1p1: (B,3,H,W) in [-1,1] (the model's sRGB target domain).
        Returns list of (B, 256, D) patch-token tensors, one per selected block."""
        x = (opt_img_m1p1.float() + 1.0) * 0.5            # [-1,1] -> [0,1]
        if x.shape[-1] != self.input_size:                 # e.g. 256 -> 224 for a patch-14 teacher
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode='bilinear', align_corners=False, antialias=True)
        x = (x - self._mean) / self._std                   # official ImageNet norm
        out = self.model(pixel_values=x, output_hidden_states=True)
        hs = out.hidden_states                             # hs[b] = output of one-based block b
        want = self.grid * self.grid
        feats = []
        for b in self.layers:
            h = hs[b][:, self.num_prefix:, :]              # strip CLS (+ registers if any)
            assert h.shape[1] == want, \
                f"expected {want} patch tokens, got {h.shape[1]} (prefix={self.num_prefix}, input={self.input_size})"
            feats.append(h)
        return feats


class MultiLayerDinoLoss(nn.Module):
    """Registered module: per-layer pool+project student tokens, cosine-align to teacher."""

    def __init__(self, num_layers, dim=768):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
            for _ in range(num_layers)
        ])

    def forward(self, student_tokens, teacher_feats):
        """student_tokens: list of (B, 1024, D) IMAGE tokens (in-context sliced away).
        teacher_feats: list of (B, 256, D), already detached.
        Returns (mean_loss, per_layer_losses)."""
        assert len(student_tokens) == len(self.projs) == len(teacher_feats)
        per_layer = []
        for proj, s, tt in zip(self.projs, student_tokens, teacher_feats):
            B, N, D = s.shape
            g = int(N ** 0.5)
            assert g * g == N, f"student tokens not square: {N}"
            s2 = s.transpose(1, 2).reshape(B, D, g, g)
            s2 = F.avg_pool2d(s2, 2)                       # 32x32 -> 16x16
            s2 = s2.flatten(2).transpose(1, 2)             # (B, 256, D)
            s2 = proj(s2)
            s2 = F.normalize(s2.float(), dim=-1)
            t2 = F.normalize(tt.detach().float(), dim=-1)  # stopgrad: teacher only
            per_layer.append((1.0 - (s2 * t2).sum(dim=-1)).mean())
        return torch.stack(per_layer).mean(), per_layer
