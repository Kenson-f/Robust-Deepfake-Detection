# Robust Deepfake Detection — NTIRE 2026

A full solution for the **Robust Deepfake Detection Challenge** at NTIRE @ CVPR 2026.  
Optimized for **ROC-AUC robustness under image degradations**.

---

## Repository Structure

```
deepfake_detection/
├── train.py              # Main training script
├── infer.py              # Inference + submission generation
├── requirements.txt      # Dependencies
├── models/
│   └── detector.py       # RobustDeepfakeDetector architecture
├── utils/
│   ├── dataset.py        # Dataset with auto-format detection
│   ├── augmentation.py   # Train/Val/TTA transforms
│   ├── losses.py         # Focal + BCE combined loss
│   └── scheduler.py      # LR scheduler utilities
└── configs/
    └── config.py         # Training configurations
```

---

## Architecture

```
Input Image
    │
    ├──► Backbone (EfficientNet-B4 / ConvNeXt-Base)
    │         │
    │    [Feature Maps]
    │         │
    │    ┌────┴──────────────────┐
    │    │                       │
    │  Attention Pool       Multi-Scale Pool
    │    │                       │
    │    └──────────┬────────────┘
    │           Spatial Features
    │
    └──► Frequency Branch (DCT-initialized)
              │
         Frequency Features
              │
         ┌────┴────────────────┐
         │                     │
     Spatial Features  +  Frequency Features
         │
     Fusion MLP Head
         │
     Fake Probability (0–1)
```

**Why this design is robust:**
- **Frequency branch**: Deepfake GAN grids/artifacts survive in the frequency domain even when spatial appearance is degraded
- **Attention pooling**: Focuses on forgery-relevant face regions, not global average
- **Multi-scale features**: Handles degradations at different spatial frequencies
- **Heavy augmentation**: Trains explicitly on noise, blur, JPEG, low-res, etc.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Data layout (auto-detected)

**Format A — class subfolders:**
```
training_data_final/
    real/   img_001.png, img_002.jpg, ...
    fake/   img_001.png, img_002.jpg, ...
```

**Format B — flat + labels.txt:**
```
training_data_final/
    img_001.png
    img_002.png
    ...
    labels.txt      # one label (0 or 1) per line, alphabetical order
```

**Format C — labels.csv:**
```
training_data_final/
    img_001.png
    ...
    labels.csv      # columns: filename, label
```

### 3. Train

```bash
# Default config (EfficientNet-B4, 30 epochs)
python train.py \
    --train_dir ~/Downloads/ntire2026/training_data_final \
    --val_dir   ~/Downloads/ntire2026/validation_data_final \
    --output_dir ./checkpoints

# Faster iteration
python train.py --config fast --train_dir ... --val_dir ...

# Best performance (slower)
python train.py --config strong --train_dir ... --val_dir ...
```

### 4. Generate submission

**Single model, no TTA (fast):**
```bash
python infer.py \
    --test_dir ~/Downloads/ntire2026/validation_data_final \
    --checkpoints ./checkpoints/best_model.pth \
    --output_dir ./submissions \
    --phase val
```

**Single model + TTA (better AUC):**
```bash
python infer.py \
    --test_dir ~/Downloads/ntire2026/validation_data_final \
    --checkpoints ./checkpoints/best_model.pth \
    --use_tta \
    --phase val
```

**Ensemble of multiple checkpoints:**
```bash
python infer.py \
    --test_dir ~/Downloads/ntire2026/validation_data_final \
    --checkpoints ./checkpoints/best_model.pth ./checkpoints/checkpoint_epoch20.pth \
    --use_tta \
    --phase val
```

Outputs:
- `submissions/submission.txt` — probabilities in alphabetical filename order
- `submissions/submission_val.zip` — ready to upload

---

## Training Configurations

| Config | Backbone | Epochs | Batch | Notes |
|--------|----------|--------|-------|-------|
| `default` | EfficientNet-B4 | 30 | 32 | Good balance |
| `fast` | EfficientNet-B3 | 15 | 64 | Quick iteration |
| `strong` | ConvNeXt-Base | 50 | 16 | Best performance |
| `vit` | ViT-Small/16 | 40 | 32 | Alternative arch |

---

## Robustness Strategy

The augmentation pipeline during training simulates:

| Degradation | Implementation |
|-------------|----------------|
| Gaussian noise | `A.GaussNoise` |
| Camera noise | `A.ISONoise` |
| Motion blur | `A.MotionBlur` |
| Defocus blur | `A.Defocus` |
| JPEG artifacts | `A.ImageCompression` |
| Double JPEG | Sequential compression |
| Low resolution | `A.Downscale` |
| Brightness / contrast | `A.RandomBrightnessContrast` |
| Saturation shift | `A.HueSaturationValue` |
| Grid distortion | `A.GridDistortion` |
| Occlusion | `A.CoarseDropout` |

---

## Tips for Higher AUC

1. **Use TTA at inference** — typically +0.5–1.5% AUC
2. **Train ensemble** — run 3-5 seeds, ensemble predictions
3. **Try `strong` config** — ConvNeXt-Base often outperforms EfficientNet on face tasks
4. **Increase image size** — set `--image_size 256` or `320` if VRAM allows
5. **Curriculum training** — start with mild augmentations, increase degradation severity progressively
6. **Pseudo-labeling** — use confident validation predictions to augment training (carefully — check rules)
