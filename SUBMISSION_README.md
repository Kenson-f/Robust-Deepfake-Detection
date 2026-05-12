# NTIRE 2026 Robust Deepfake Detection - Public Test Submission

## Team Information
- **Team Name:** [YOUR TEAM NAME]
- **Team Leader:** [YOUR NAME] ([YOUR EMAIL])
- **CodaBench Username:** [YOUR USERNAME]

## Files Included
1. `best_model.pth` - Trained model checkpoint
2. `submission.zip` - Predictions on public test set
3. `FACTSHEET.pdf` - Compiled factsheet
4. `FACTSHEET.tex` - LaTeX source
5. `train_robust.py` - Training script
6. `infer_robust.py` - Inference script
7. `models/robust_detector.py` - Model architecture
8. `requirements.txt` - Python dependencies

## Reproducing Results

### Environment Setup
```bash
pip install torch torchvision torchaudio timm albumentations opencv-python numpy scikit-learn Pillow
```

### Training (Optional - checkpoint provided)
```bash
python train_robust.py \
    --train_dir path/to/training_data_final \
    --backbone convnext_tiny \
    --image_size 224 \
    --batch_size 32 \
    --epochs 30 \
    --lr 5e-5 \
    --use_patches \
    --output_dir checkpoints_robust
```

### Inference on Public Test Set
```bash
python infer_robust.py \
    --test_dir path/to/publictest_data_final \
    --checkpoint checkpoints_robust/best_model.pth \
    --output_dir submissions \
    --batch_size 32
```

This generates `submissions/submission.zip` containing `submission.txt` with predictions.

## Model Architecture
- **Backbone:** ConvNeXt-Tiny (28.8M params)
- **FFT Branch:** Frequency analysis via 2D Fourier Transform
- **Fusion:** Spatial + Frequency features → MLP classifier
- **Training:** Patch-based (4 patches/image) with label smoothing
- **TTA:** 3-view test-time augmentation

## Performance
- **Validation AUC:** ~0.75-0.85 (on 15% held-out training data)
- **Public Test AUC:** [FILL IN YOUR ACTUAL SCORE]

## Hardware Requirements
- GPU: NVIDIA RTX 3060 or better (12GB+ VRAM)
- Training time: ~30 minutes
- Inference time: ~2 minutes for 100 images

## Contact
[YOUR EMAIL]
