"""
Robust Deepfake Detection - NTIRE 2026
Training Script v2
Improvements:
  - MixUp augmentation (interpolates labels + images)
  - Curriculum augmentation (mild → strong degradation over epochs)
  - Gradient accumulation (effective larger batch without VRAM cost)
  - Better LR scheduling
"""
import os
import sys
import json
import random
import argparse
import platform
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score

from models.detector import RobustDeepfakeDetector
from utils.dataset import DeepfakeDataset
from utils.augmentation import get_train_transforms, get_val_transforms
from utils.losses import CombinedLoss
from utils.scheduler import get_scheduler
from configs.config import get_config


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=0.2):
    """MixUp: blend two images and their labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def get_curriculum_severity(epoch, total_epochs):
    """Ramp from mild to strong augmentation over training."""
    progress = epoch / total_epochs
    if progress < 0.2:
        return 'mild'
    elif progress < 0.5:
        return 'medium'
    else:
        return 'strong'


def train_one_epoch(model, loader, optimizer, criterion, scaler, device,
                    epoch, cfg, accum_steps=2):
    model.train()
    total_loss = 0.0
    all_labels, all_preds = [], []
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        # MixUp
        use_mixup = cfg.use_mixup and random.random() < 0.5
        if use_mixup:
            images, y_a, y_b, lam = mixup_data(images, labels, alpha=cfg.mixup_alpha)

        with autocast():
            logits = model(images).squeeze(1)
            if use_mixup:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, labels)
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        probs = torch.sigmoid(logits.detach()).cpu().float().numpy()
        all_preds.extend(probs.tolist())
        # For AUC, use hard labels even with mixup
        hard_labels = labels.cpu().numpy() if not use_mixup else \
                      (lam * y_a + (1-lam) * y_b).cpu().numpy()
        all_labels.extend(hard_labels.tolist())

        if (step + 1) % 20 == 0:
            print(f"  [Epoch {epoch}] Step {step+1}/{len(loader)} | Loss: {total_loss/(step+1):.4f}")

    try:
        auc = roc_auc_score(np.array(all_labels) > 0.5, all_preds)
    except Exception:
        auc = 0.5
    return total_loss / len(loader), auc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, all_labels, all_preds = 0.0, [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        with autocast():
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(logits).cpu().float().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except Exception:
        auc = 0.5
    return total_loss / len(loader), auc


def main(args):
    cfg = get_config(args.config)
    set_seed(cfg.seed)

    train_dir = Path(args.train_dir)
    val_dir   = Path(args.val_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Config : {cfg}")

    # Windows needs num_workers=0
    nw = 0 if platform.system() == 'Windows' else cfg.num_workers

    val_transforms = get_val_transforms(cfg.image_size)
    val_dataset = DeepfakeDataset(val_dir, transform=val_transforms, mode='val')
    has_val_labels = any(l != -1 for l in val_dataset.labels) if val_dataset.labels else False

    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size * 2,
                            shuffle=False, num_workers=nw, pin_memory=True)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = RobustDeepfakeDetector(
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
        dropout_rate=cfg.dropout_rate,
        image_size=cfg.image_size,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")

    criterion = CombinedLoss(
        bce_weight=cfg.bce_weight,
        focal_weight=cfg.focal_weight,
        label_smoothing=cfg.label_smoothing
    )

    backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n]
    head_params     = [p for n, p in model.named_parameters() if 'backbone' not in n]
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': cfg.lr_backbone},
        {'params': head_params,     'lr': cfg.lr_head}
    ], weight_decay=cfg.weight_decay)

    scaler   = GradScaler()
    best_auc = 0.0
    history  = []
    patience_counter = 0

    print(f"\n{'='*60}\nStarting Training\n{'='*60}")

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n[Epoch {epoch}/{cfg.epochs}]")

        # Curriculum: rebuild train dataset with appropriate severity each epoch
        severity = get_curriculum_severity(epoch, cfg.epochs)
        train_transforms = get_train_transforms(cfg.image_size, severity=severity)
        train_dataset = DeepfakeDataset(train_dir, transform=train_transforms, mode='train')

        # Balanced sampler
        labels_arr = np.array(train_dataset.labels)
        class_counts = np.bincount(labels_arr.clip(0))
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[labels_arr.clip(0)]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float),
            num_samples=len(train_dataset), replacement=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, sampler=sampler,
            num_workers=nw, pin_memory=True, drop_last=True
        )

        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, epoch, cfg, accum_steps=cfg.accum_steps
        )

        val_loss, val_auc = None, None
        if has_val_labels:
            val_loss, val_auc = validate(model, val_loader, criterion, device)
            print(f"  Train -> Loss: {train_loss:.4f} | AUC: {train_auc:.4f}")
            print(f"  Val   -> Loss: {val_loss:.4f} | AUC: {val_auc:.4f}")
        else:
            val_auc = train_auc
            print(f"  Train -> Loss: {train_loss:.4f} | AUC: {train_auc:.4f}  [aug: {severity}]")

        # Scheduler step
        if not hasattr(train_one_epoch, '_scheduler'):
            # Build scheduler once (needs steps_per_epoch)
            train_one_epoch._scheduler = get_scheduler(optimizer, cfg, len(train_loader))
        sched = train_one_epoch._scheduler
        if sched is not None:
            sched.step()

        # Checkpoint best
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'config': cfg.__dict__
            }, output_dir / 'best_model.pth')
            print(f"  ✓ New best model saved (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1

        if epoch % cfg.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc if val_auc is not None else train_auc,
            }, output_dir / f'checkpoint_epoch{epoch}.pth')

        history.append({
            'epoch': epoch, 'severity': severity,
            'train_loss': train_loss, 'train_auc': train_auc,
            'val_loss': val_loss if val_loss is not None else train_loss,
            'val_auc':  val_auc  if val_auc  is not None else train_auc,
        })

        if patience_counter >= cfg.patience:
            print(f"\nEarly stopping triggered after {patience_counter} epochs.")
            break

    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump({'history': history, 'best_auc': best_auc}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! Best AUC: {best_auc:.4f}")
    print(f"Checkpoint: {output_dir / 'best_model.pth'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str,
                        default=os.path.expanduser('~/Downloads/ntire2026/training_data_final'))
    parser.add_argument('--val_dir',   type=str,
                        default=os.path.expanduser('~/Downloads/ntire2026/validation_data_final'))
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--config', type=str, default='default',
                        help='default | fast | strong | xception | ensemble')
    args = parser.parse_args()
    main(args)
