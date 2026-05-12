"""
Augmentation Pipeline v2 - Stronger & More Targeted
=====================================================
Key upgrades:
  - Higher degradation probabilities (train sees harder cases)
  - Mixup/CutMix support
  - Curriculum: mild → strong degradation over epochs
  - Extra JPEG rounds at very low quality (matches adversarial test set)
  - Gaussian noise at higher variance
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch


MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224, severity: str = 'strong') -> A.Compose:
    """
    severity: 'mild' | 'medium' | 'strong'
    Use 'mild' for first few epochs, 'strong' thereafter (curriculum).
    """
    if severity == 'mild':
        noise_p, blur_p, jpeg_p, lowres_p = 0.3, 0.2, 0.3, 0.1
    elif severity == 'medium':
        noise_p, blur_p, jpeg_p, lowres_p = 0.5, 0.4, 0.5, 0.2
    else:  # strong
        noise_p, blur_p, jpeg_p, lowres_p = 0.7, 0.6, 0.7, 0.4

    return A.Compose([
        # ── Geometry ──────────────────────────────────────────────────────────
        A.Resize(int(image_size * 1.15), int(image_size * 1.15)),
        A.RandomCrop(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=10, border_mode=0, p=0.4),

        # ── Noise (boosted) ───────────────────────────────────────────────────
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 150.0), p=1.0),    # higher max var
            A.ISONoise(color_shift=(0.01, 0.08), intensity=(0.1, 0.7), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.7, 1.3), p=1.0),
            A.GaussNoise(var_limit=(50.0, 200.0), p=1.0),    # very heavy noise
        ], p=noise_p),

        # ── Blur ──────────────────────────────────────────────────────────────
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 11), p=1.0),       # larger max kernel
            A.MotionBlur(blur_limit=(3, 15), p=1.0),
            A.MedianBlur(blur_limit=9, p=1.0),
            A.Defocus(radius=(1, 6), p=1.0),
            A.ZoomBlur(max_factor=1.1, p=1.0),
        ], p=blur_p),

        # ── JPEG Compression (critical for this challenge) ────────────────────
        A.OneOf([
            A.ImageCompression(quality_lower=5,  quality_upper=40, p=1.0),  # very heavy
            A.ImageCompression(quality_lower=20, quality_upper=70, p=1.0),
            A.ImageCompression(quality_lower=50, quality_upper=90, p=1.0),
            # Triple compression — aggressive social media simulation
            A.Sequential([
                A.ImageCompression(quality_lower=20, quality_upper=50),
                A.ImageCompression(quality_lower=40, quality_upper=70),
                A.ImageCompression(quality_lower=60, quality_upper=90),
            ], p=1.0),
        ], p=jpeg_p),

        # ── Low Resolution ────────────────────────────────────────────────────
        A.OneOf([
            A.Downscale(scale_min=0.15, scale_max=0.5,
                        interpolation={'downscale': 0, 'upscale': 0}),   # nearest (blocky)
            A.Downscale(scale_min=0.25, scale_max=0.6,
                        interpolation={'downscale': 2, 'upscale': 2}),   # cubic
            A.Downscale(scale_min=0.4,  scale_max=0.8,
                        interpolation={'downscale': 1, 'upscale': 3}),   # bilinear→lanczos
        ], p=lowres_p),

        # ── Combined degradation (simulates "worst case" test images) ─────────
        # Blur THEN compress THEN noise — stacked degradation
        A.Sequential([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.ImageCompression(quality_lower=10, quality_upper=40, p=0.7),
            A.GaussNoise(var_limit=(10, 50), p=0.5),
        ], p=0.2),

        # ── Color / Photometric ───────────────────────────────────────────────
        A.OneOf([
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=25, p=1.0),
            A.RandomGamma(gamma_limit=(60, 140), p=1.0),
        ], p=0.6),

        # ── Texture / Sharpness ───────────────────────────────────────────────
        A.OneOf([
            A.Sharpen(alpha=(0.1, 0.6), lightness=(0.8, 1.3), p=1.0),
            A.UnsharpMask(blur_limit=(3, 7), sigma_limit=0.5, alpha=(0.2, 0.7), p=1.0),
            A.CLAHE(clip_limit=6.0, p=1.0),
        ], p=0.25),

        # ── Occlusion ─────────────────────────────────────────────────────────
        A.CoarseDropout(max_holes=6, max_height=40, max_width=40,
                        min_holes=1, min_height=8,  min_width=8,
                        fill_value=0, p=0.25),

        # ── Normalize ─────────────────────────────────────────────────────────
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])


def get_tta_transforms(image_size: int = 224) -> list:
    """
    8-view TTA: original + flip + 3 crops + sharpen + denoise + combined
    More views = better AUC at inference time.
    """
    return [
        # 1. Original
        A.Compose([A.Resize(image_size, image_size),
                   A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
        # 2. Horizontal flip
        A.Compose([A.Resize(image_size, image_size), A.HorizontalFlip(p=1.0),
                   A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
        # 3. Larger center crop
        A.Compose([A.Resize(int(image_size*1.12), int(image_size*1.12)),
                   A.CenterCrop(image_size, image_size),
                   A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
        # 4. Flip + crop
        A.Compose([A.Resize(int(image_size*1.12), int(image_size*1.12)),
                   A.CenterCrop(image_size, image_size), A.HorizontalFlip(p=1.0),
                   A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
        # 5. Sharpen (recover blurry images)
        A.Compose([A.Resize(image_size, image_size),
                   A.Sharpen(alpha=(0.4, 0.4), lightness=(1.0, 1.0), p=1.0),
                   A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
        # 6. Mild denoise via median blur
        A.Compose([A.Resize(image_size, image_size),
                   A.MedianBlur(blur_limit=3, p=1.0),
                   A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
        # 7. CLAHE (contrast enhancement — helps on dark/flat images)
        A.Compose([A.Resize(image_size, image_size),
                   A.CLAHE(clip_limit=3.0, p=1.0),
                   A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
        # 8. Sharpen + flip
        A.Compose([A.Resize(image_size, image_size), A.HorizontalFlip(p=1.0),
                   A.Sharpen(alpha=(0.4, 0.4), lightness=(1.0, 1.0), p=1.0),
                   A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
    ]


class AlbumentationsWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        return self.transform(image=img)['image']
