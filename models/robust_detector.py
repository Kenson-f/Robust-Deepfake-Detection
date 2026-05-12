"""
Anti-Overfitting Deepfake Detector
Uses: FFT frequency + Patch-based training + ConvNeXt
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np


class FFTBranch(nn.Module):
    """
    Fourier frequency magnitude analysis.
    GAN upsampling creates high-frequency patterns that survive degradation.
    """
    def __init__(self, channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, x):
        # x: [B, 3, H, W]
        # FFT -> magnitude spectrum
        fft = torch.fft.rfft2(x, norm='ortho')
        mag = torch.abs(fft)  # [B, 3, H, W//2+1]
        # Pad to square
        if mag.shape[-1] < mag.shape[-2]:
            pad_w = mag.shape[-2] - mag.shape[-1]
            mag = F.pad(mag, (0, pad_w))
        # Log scale (high frequencies are small)
        mag = torch.log(mag + 1e-8)
        return self.encoder(mag).flatten(1)


class PatchExtractor(nn.Module):
    """Extract random patches during training to prevent identity memorization."""
    def __init__(self, patch_size=128, n_patches=4):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = n_patches
    
    def forward(self, x):
        if not self.training:
            return x  # Use full image at inference
        
        B, C, H, W = x.shape
        patches = []
        for _ in range(self.n_patches):
            # Random crop
            top = torch.randint(0, H - self.patch_size + 1, (1,)).item()
            left = torch.randint(0, W - self.patch_size + 1, (1,)).item()
            patch = x[:, :, top:top+self.patch_size, left:left+self.patch_size]
            patches.append(patch)
        return torch.cat(patches, dim=0)  # [B*n_patches, C, patch_size, patch_size]


class RobustDetector(nn.Module):
    def __init__(self, backbone='convnext_tiny', pretrained=True, 
                 dropout=0.4, image_size=224, use_patches=True):
        super().__init__()
        self.use_patches = use_patches
        
        # Patch extractor
        if use_patches:
            patch_size = min(image_size // 2, 128)
            self.patch_extract = PatchExtractor(patch_size=patch_size, n_patches=4)
        
        # Spatial backbone
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, 
            features_only=True, num_classes=0
        )
        
        # Detect output channels
        with torch.no_grad():
            test_size = patch_size if use_patches else image_size
            dummy = torch.zeros(1, 3, test_size, test_size)
            feats = self.backbone(dummy)
            spatial_dim = feats[-1].shape[1]
        
        print(f"[Model] {backbone} -> {spatial_dim} channels")
        
        # Frequency branch
        self.freq_branch = FFTBranch(channels=3)
        freq_dim = 256
        
        # Fusion
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout/2),
            nn.Linear(256, 1)
        )
        
        self._init_head()
    
    def _init_head(self):
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        orig_batch = x.shape[0]
        
        # Extract patches during training
        if self.use_patches and self.training:
            x = self.patch_extract(x)
        
        # Spatial features
        spatial_feats = self.backbone(x)[-1]  # last feature map
        spatial = self.spatial_pool(spatial_feats).flatten(1)
        
        # Frequency features (on original-size patches)
        freq = self.freq_branch(x)
        
        # Fuse
        fused = torch.cat([spatial, freq], dim=1)
        logits = self.fusion(fused)
        
        # Average predictions over patches during training
        if self.use_patches and self.training:
            logits = logits.view(orig_batch, -1).mean(dim=1, keepdim=True)
        
        return logits
