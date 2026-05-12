"""
Robust Deepfake Detection Model - v2
======================================
Key upgrades over v1:
  1. Xception-inspired separable conv neck (proven best for deepfake detection)
  2. NPR (Neighboring Pixel Relationships) feature map — catches upsampling artifacts
  3. Multi-scale frequency analysis (not just last feature map)
  4. MixUp / CutMix ready (labels passed as floats)
  5. Auto-detects backbone output channels at runtime
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np


def get_backbone_out_channels(backbone: nn.Module, image_size: int = 224) -> int:
    backbone.eval()
    with torch.no_grad():
        dummy = torch.zeros(2, 3, image_size, image_size)
        try:
            out = backbone(dummy)
            last = out[-1] if isinstance(out, (list, tuple)) else out
            return last.shape[1]
        except Exception:
            return 512


class NPRModule(nn.Module):
    """
    Neighboring Pixel Relationships (NPR) — computes local pixel difference maps.
    GAN-generated images have distinctive NPR patterns even after heavy degradation.
    Uses per-channel Sobel-style horizontal and vertical gradients.
    """
    def __init__(self):
        super().__init__()
        # Horizontal kernel: shape [3, 1, 1, 2] — one filter per channel, depthwise
        kh = torch.tensor([[-1.0, 1.0]], dtype=torch.float32)   # [1, 2]
        kh = kh.view(1, 1, 1, 2).repeat(3, 1, 1, 1)             # [3, 1, 1, 2]
        self.register_buffer('kernel_h', kh)

        # Vertical kernel: shape [3, 1, 2, 1]
        kv = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)  # [2, 1]
        kv = kv.view(1, 1, 2, 1).repeat(3, 1, 1, 1)             # [3, 1, 2, 1]
        self.register_buffer('kernel_v', kv)

    def forward(self, x):
        # x: [B, 3, H, W]
        # Depthwise conv: groups=3, one filter per channel
        dh = F.conv2d(x, self.kernel_h, padding=(0, 0), groups=3)  # [B, 3, H, W-1]
        dv = F.conv2d(x, self.kernel_v, padding=(0, 0), groups=3)  # [B, 3, H-1, W]
        # Crop both to same size before combining
        H = min(dh.shape[2], dv.shape[2])
        W = min(dh.shape[3], dv.shape[3])
        dh = dh[:, :, :H, :W]
        dv = dv[:, :, :H, :W]
        # Average gradient magnitude across channels -> [B, 1, H, W]
        mag = torch.sqrt(dh ** 2 + dv ** 2 + 1e-8).mean(dim=1, keepdim=True)
        return mag


class FrequencyBranch(nn.Module):
    """
    Multi-scale DCT frequency analysis.
    Runs at 3 different block sizes to catch artifacts at different scales.
    """
    def __init__(self, in_channels: int = 3, out_dim: int = 256):
        super().__init__()

        # Block size 8 (standard JPEG block)
        self.dct8 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=8, bias=False)
        self._init_dct(self.dct8, block=8)

        # Block size 4 (finer scale)
        self.dct4 = nn.Conv2d(in_channels, 16, kernel_size=4, stride=4, bias=False)
        self._init_dct(self.dct4, block=4, n_filters=16)

        self.freq_net = nn.Sequential(
            nn.Conv2d(48, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def _init_dct(self, conv, block=8, n_filters=None):
        if n_filters is None:
            n_filters = conv.out_channels
        in_c = conv.in_channels
        weight = torch.zeros(n_filters, in_c, block, block)
        idx = 0
        for i in range(block):
            for j in range(block):
                if idx >= n_filters:
                    break
                for ch in range(in_c):
                    for x in range(block):
                        for y in range(block):
                            weight[idx, ch, x, y] = (
                                np.cos(np.pi * i * (2*x+1) / (2*block)) *
                                np.cos(np.pi * j * (2*y+1) / (2*block))
                            )
                idx += 1
            if idx >= n_filters:
                break
        weight = weight / (weight.norm(dim=(2, 3), keepdim=True) + 1e-6)
        conv.weight.data[:n_filters].copy_(weight)

    def forward(self, x):
        # Pad to make divisible if needed
        _, _, H, W = x.shape
        x8 = self.dct8(x)
        x4 = self.dct4(x)
        # Align spatial dims
        target_h = min(x8.shape[2], x4.shape[2])
        target_w = min(x8.shape[3], x4.shape[3])
        x8 = F.adaptive_avg_pool2d(x8, (target_h, target_w))
        x4 = F.adaptive_avg_pool2d(x4, (target_h, target_w))
        combined = torch.cat([x8, x4], dim=1)  # [B, 48, H', W']
        return self.freq_net(combined).flatten(1)  # [B, out_dim]


class SpatialAttentionPool(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        mid = max(in_dim // 8, 16)
        self.attn = nn.Sequential(
            nn.Conv2d(in_dim, mid, 1),
            nn.ReLU(),
            nn.Conv2d(mid, 1, 1),
        )

    def forward(self, x):
        attn_map = torch.sigmoid(self.attn(x))
        attended = (x * attn_map).sum(dim=(2, 3))
        norm = attn_map.sum(dim=(2, 3)) + 1e-6
        return attended / norm


class SeparableConvNeck(nn.Module):
    """
    Xception-style depthwise separable conv neck.
    Better at capturing local texture artifacts than plain conv.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_dim, in_dim, 3, padding=1, groups=in_dim, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.GELU(),
            # Pointwise
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class RobustDeepfakeDetector(nn.Module):
    def __init__(
        self,
        backbone: str = 'efficientnet_b4',
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freq_dim: int = 256,
        head_dim: int = 512,
        image_size: int = 224,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.is_vit = backbone.startswith('vit') or backbone.startswith('swin')

        # ── Backbone ──────────────────────────────────────────────────────────
        if self.is_vit:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            with torch.no_grad():
                dummy = torch.zeros(2, 3, image_size, image_size)
                feat = self.backbone.forward_features(dummy)
                feat_dim = feat.shape[-1] if feat.dim() == 3 else feat.shape[1]
        else:
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained, features_only=True, num_classes=0
            )
            feat_dim = get_backbone_out_channels(self.backbone, image_size)
            print(f"[Model] Backbone '{backbone}' -> last feature channels: {feat_dim}")

        # ── Separable conv neck ────────────────────────────────────────────────
        neck_dim = min(feat_dim, 256)
        if not self.is_vit:
            self.neck = SeparableConvNeck(feat_dim, neck_dim)
            self.attn_pool = SpatialAttentionPool(neck_dim)
            self.gap = nn.AdaptiveAvgPool2d(1)
            spatial_dim = neck_dim * 2  # attn + gap
        else:
            self.neck = None
            spatial_dim = feat_dim

        # ── NPR module ─────────────────────────────────────────────────────────
        self.npr = NPRModule()
        self.npr_enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )
        npr_dim = 32

        # ── Frequency branch ──────────────────────────────────────────────────
        self.freq_branch = FrequencyBranch(in_channels=3, out_dim=freq_dim)

        # ── Fusion head ───────────────────────────────────────────────────────
        total_dim = spatial_dim + freq_dim + npr_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(head_dim, head_dim // 2),
            nn.LayerNorm(head_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(head_dim // 2, 1)
        )
        self._init_head_weights()

    def _init_head_weights(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def extract_spatial_features(self, x):
        if self.is_vit:
            feat = self.backbone.forward_features(x)
            return feat[:, 0] if feat.dim() == 3 else feat
        feature_maps = self.backbone(x)
        last_map = feature_maps[-1]
        last_map = self.neck(last_map)
        return torch.cat([
            self.attn_pool(last_map),
            self.gap(last_map).flatten(1)
        ], dim=1)

    def forward(self, x):
        spatial = self.extract_spatial_features(x)
        freq    = self.freq_branch(x)
        npr     = self.npr_enc(self.npr(x)).flatten(1)
        fused   = torch.cat([spatial, freq, npr], dim=1)
        return self.classifier(fused)


class EnsembleDetector(nn.Module):
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0 / len(models)] * len(models)

    @torch.no_grad()
    def forward(self, x):
        return torch.stack([
            torch.sigmoid(m(x)) * w
            for m, w in zip(self.models, self.weights)
        ]).sum(dim=0)
