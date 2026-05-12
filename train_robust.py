"""
FIXED: Removed patch-based training - it was breaking gradient flow
"""
import os, sys, random, argparse, platform
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from models.robust_detector import RobustDetector
from utils.dataset import DeepfakeDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(size, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(int(size*1.15), int(size*1.15)),
            A.RandomCrop(size, size),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3,7), p=1),
                A.MedianBlur(blur_limit=5, p=1),
            ], p=0.4),
            A.GaussNoise(p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2()
        ])
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])

class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, logits, targets):
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(logits, targets_smooth)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    losses, preds, labels = [], [], []
    for step, (imgs, lbls) in enumerate(loader):
        imgs, lbls = imgs.to(device), lbls.to(device).float()
        optimizer.zero_grad()
        with autocast('cuda'):
            out = model(imgs).squeeze(1)
            loss = criterion(out, lbls)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        losses.append(loss.item())
        preds.extend(torch.sigmoid(out.detach()).cpu().numpy())
        labels.extend(lbls.cpu().numpy())
        if (step+1) % 10 == 0:
            print(f"  [{epoch}] {step+1}/{len(loader)} Loss={np.mean(losses):.4f}")
    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return np.mean(losses), auc

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses, preds, labels = [], [], []
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device).float()
        with autocast('cuda'):
            out = model(imgs).squeeze(1)
            loss = criterion(out, lbls)
        losses.append(loss.item())
        preds.extend(torch.sigmoid(out).cpu().numpy())
        labels.extend(lbls.cpu().numpy())
    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return np.mean(losses), auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--output_dir', default='./checkpoints_robust')
    parser.add_argument('--backbone', default='convnext_tiny')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    train_tf = get_transforms(args.image_size, True)
    val_tf = get_transforms(args.image_size, False)
    
    full_ds = DeepfakeDataset(args.train_dir, transform=train_tf, mode='train')
    train_size = int(0.85 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds_indices = random_split(full_ds, [train_size, val_size])
    
    # Apply val transforms to val split
    class ValDataset:
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self): return len(self.subset)
        def __getitem__(self, idx):
            from PIL import Image
            img_path = self.subset.dataset.image_paths[self.subset.indices[idx]]
            label = self.subset.dataset.labels[self.subset.indices[idx]]
            img = np.array(Image.open(img_path).convert('RGB'))
            img = self.transform(image=img)['image']
            return img, label
    
    val_ds = ValDataset(val_ds_indices, val_tf)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    
    nw = 0 if platform.system() == 'Windows' else 4
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False,
                            num_workers=nw, pin_memory=True)
    
    # NO PATCHES - use_patches=False
    model = RobustDetector(
        backbone=args.backbone, pretrained=True, dropout=0.4,
        image_size=args.image_size, use_patches=False
    ).to(device)
    
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = LabelSmoothingBCE(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-7)
    scaler = GradScaler('cuda')
    
    best_auc, patience = 0.0, 0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch)
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"  Train: L={train_loss:.4f} AUC={train_auc:.4f}")
        print(f"  Val:   L={val_loss:.4f} AUC={val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_auc': val_auc, 'backbone': args.backbone,
                'image_size': args.image_size, 'use_patches': False
            }, out_dir / 'best_model.pth')
            print(f"  ✓ Best: {best_auc:.4f}")
        else:
            patience += 1
            if patience >= 10:
                print(f"\nEarly stop")
                break
    
    print(f"\n{'='*60}\nDone! Best: {best_auc:.4f}\n{'='*60}")

if __name__ == '__main__':
    main()
