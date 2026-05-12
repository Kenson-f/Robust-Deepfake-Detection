"""
NTIRE 2026 Robust Deepfake Detection — Inference & Submission Generator
========================================================================
Usage:

  # Basic (no TTA):
  python infer.py --test_dir PATH/TO/validation_data_final --checkpoint PATH/TO/best_model.pth

  # With Test-Time Augmentation (better AUC, ~4x slower):
  python infer.py --test_dir PATH/TO/validation_data_final --checkpoint PATH/TO/best_model.pth --tta

  # Ensemble of multiple checkpoints:
  python infer.py --test_dir PATH/TO/validation_data_final \
      --checkpoint checkpoints/best_model.pth checkpoints/checkpoint_epoch20.pth --tta

Output:
  submissions/submission.txt   — one probability per line, alphabetical filename order
  submissions/submission.zip   — ready to upload to the challenge platform
"""

import os
import sys
import json
import zipfile
import argparse
import numpy as np
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from PIL import Image, ImageFile
import albumentations as A
from albumentations.pytorch import ToTensorV2

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Bring sibling modules into scope regardless of cwd ──────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from models.detector import RobustDeepfakeDetector

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif'}


# ── Image discovery ──────────────────────────────────────────────────────────

def find_images_sorted(directory: Path):
    """All images under directory, sorted alphabetically by filename."""
    imgs = []
    for ext in VALID_EXTENSIONS:
        imgs.extend(directory.rglob(f'*{ext}'))
        imgs.extend(directory.rglob(f'*{ext.upper()}'))
    return sorted(set(imgs), key=lambda p: p.name)


# ── Transforms ──────────────────────────────────────────────────────────────

def base_transform(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def tta_transforms(image_size=224):
    """4 deterministic TTA views."""
    return [
        # 1. Original
        A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 2. Horizontal flip
        A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 3. Slightly larger crop
        A.Compose([
            A.Resize(int(image_size * 1.1), int(image_size * 1.1)),
            A.CenterCrop(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 4. Mild sharpen (helps on blurry degraded images)
        A.Compose([
            A.Resize(image_size, image_size),
            A.Sharpen(alpha=(0.3, 0.3), lightness=(1.0, 1.0), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get('config', {})
    model = RobustDeepfakeDetector(
        backbone=cfg.get('backbone', 'efficientnet_b4'),
        pretrained=False,
        dropout_rate=cfg.get('dropout_rate', 0.3),
        image_size=cfg.get('image_size', 224),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    epoch = ckpt.get('epoch', '?')
    auc   = ckpt.get('val_auc', 'N/A')
    print(f"  Loaded: {Path(checkpoint_path).name}  |  epoch {epoch}  |  val AUC {auc}")
    return model, cfg.get('image_size', 224)


# ── Core inference ───────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, image_paths, transforms_list, device, batch_size):
    """
    Run inference with one or more transforms (TTA).
    Returns array of shape [N] with averaged probabilities.
    """
    n = len(image_paths)
    accumulated = np.zeros(n, dtype=np.float64)

    for t_idx, tfm in enumerate(transforms_list, 1):
        print(f"    Transform pass {t_idx}/{len(transforms_list)} ...", end=' ', flush=True)
        pass_probs = []

        for i in range(0, n, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            tensors = []
            for p in batch_paths:
                try:
                    img = np.array(Image.open(p).convert('RGB'))
                except Exception:
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
                tensors.append(tfm(image=img)['image'])

            batch = torch.stack(tensors).to(device)
            with autocast():
                logits = model(batch).squeeze(1)
            probs = torch.sigmoid(logits).cpu().float().numpy()
            pass_probs.extend(probs.tolist())

        accumulated += np.array(pass_probs)
        print(f"done  (mean prob so far: {(accumulated / t_idx).mean():.4f})")

    return accumulated / len(transforms_list)


# ── Submission packaging ─────────────────────────────────────────────────────

def save_submission(probs, output_dir: Path, image_paths):
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / 'submission.txt'
    with open(txt_path, 'w') as f:
        for p in probs:
            f.write(f"{p:.6f}\n")

    zip_path = output_dir / 'submission.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt_path, 'submission.txt')

    # Human-readable debug file (filename → probability)
    debug = {img.name: float(prob) for img, prob in zip(image_paths, probs)}
    with open(output_dir / 'predictions_debug.json', 'w') as f:
        json.dump(debug, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  submission.txt  →  {txt_path}")
    print(f"  submission.zip  →  {zip_path}  (upload this)")
    print(f"  Lines: {len(probs)}  |  Fake (>0.5): {(probs > 0.5).sum()}  |  Real (≤0.5): {(probs <= 0.5).sum()}")
    print(f"  Prob  min={probs.min():.4f}  max={probs.max():.4f}  mean={probs.mean():.4f}")
    print(f"{'='*55}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate NTIRE 2026 deepfake submission')
    parser.add_argument('--test_dir',   required=True,
                        help='Path to test/validation image directory')
    parser.add_argument('--checkpoint', nargs='+', required=True,
                        help='Path(s) to model checkpoint(s). Multiple = ensemble.')
    parser.add_argument('--output_dir', default='./submissions',
                        help='Where to save submission.txt and submission.zip')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--tta',        action='store_true',
                        help='Use 4-view Test-Time Augmentation (recommended)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    # ── Find images (alphabetical — must match submission order) ─────────────
    test_dir    = Path(args.test_dir)
    image_paths = find_images_sorted(test_dir)
    print(f"Images : {len(image_paths)} found in {test_dir}")
    if not image_paths:
        raise FileNotFoundError(f"No images found in {test_dir}")

    # ── Load model(s) ─────────────────────────────────────────────────────────
    print(f"\nLoading {len(args.checkpoint)} checkpoint(s):")
    models, img_sizes = [], []
    for ckpt_path in args.checkpoint:
        m, sz = load_model(ckpt_path, device)
        models.append(m)
        img_sizes.append(sz)

    image_size = img_sizes[0]  # use first model's size

    # ── Build transform list ───────────────────────────────────────────────────
    if args.tta:
        tfm_list = tta_transforms(image_size)
        print(f"\nTTA enabled: {len(tfm_list)} views per image")
    else:
        tfm_list = [base_transform(image_size)]
        print(f"\nTTA disabled: single-pass inference")

    # ── Run inference ─────────────────────────────────────────────────────────
    all_model_probs = []
    for i, model in enumerate(models):
        print(f"\nModel {i+1}/{len(models)}:")
        probs = predict(model, image_paths, tfm_list, device, args.batch_size)
        all_model_probs.append(probs)

    final_probs = np.mean(all_model_probs, axis=0)

    # ── Save ───────────────────────────────────────────────────────────────────
    save_submission(final_probs, Path(args.output_dir), image_paths)


if __name__ == '__main__':
    main()
