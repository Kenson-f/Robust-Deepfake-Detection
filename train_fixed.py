"""
FIXED Training Script - Simpler, More Stable
"""
import os, sys, json, random, argparse, platform
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score

from models.detector import RobustDeepfakeDetector
from utils.dataset import DeepfakeDataset
from utils.augmentation import get_train_transforms, get_val_transforms
from utils.losses import CombinedLoss
from configs.config import get_config

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    losses, labels, preds = [], [], []
    
    for step, (images, lbls) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True).float()
        
        optimizer.zero_grad()
        with autocast():
            logits = model(images).squeeze(1)
            loss = criterion(logits, lbls)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        losses.append(loss.item())
        preds.extend(torch.sigmoid(logits.detach()).cpu().numpy())
        labels.extend(lbls.cpu().numpy())
        
        if (step+1) % 10 == 0:
            print(f"  [{epoch}] {step+1}/{len(loader)} Loss={np.mean(losses):.4f}")
    
    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return np.mean(losses), auc

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses, labels, preds = [], [], []
    for images, lbls in loader:
        images, lbls = images.to(device), lbls.to(device).float()
        with autocast():
            logits = model(images).squeeze(1)
            loss = criterion(logits, lbls)
        losses.append(loss.item())
        preds.extend(torch.sigmoid(logits).cpu().numpy())
        labels.extend(lbls.cpu().numpy())
    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return np.mean(losses), auc

def main(args):
    cfg = get_config(args.config)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Config: {cfg}")
    
    # Fixed augmentation - no curriculum
    train_tf = get_train_transforms(cfg.image_size, severity='strong')
    val_tf = get_val_transforms(cfg.image_size)
    
    train_ds = DeepfakeDataset(args.train_dir, transform=train_tf, mode='train')
    val_ds = DeepfakeDataset(args.val_dir, transform=val_tf, mode='val')
    
    nw = 0 if platform.system() == 'Windows' else cfg.num_workers
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                              num_workers=nw, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size*2, shuffle=False,
                            num_workers=nw, pin_memory=True)
    
    model = RobustDeepfakeDetector(
        backbone=cfg.backbone, pretrained=cfg.pretrained,
        dropout_rate=cfg.dropout_rate, image_size=cfg.image_size
    ).to(device)
    
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = CombinedLoss(cfg.bce_weight, cfg.focal_weight, cfg.label_smoothing)
    
    # Single LR for all params
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs, eta_min=1e-7)
    scaler = GradScaler()
    
    best_auc, patience = 0.0, 0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60 + "\nStarting Training\n" + "="*60)
    
    for epoch in range(1, cfg.epochs + 1):
        print(f"\n[Epoch {epoch}/{cfg.epochs}]")
        
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, 
                                           scaler, device, epoch)
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"  Train: Loss={train_loss:.4f} AUC={train_auc:.4f}")
        print(f"  Val:   Loss={val_loss:.4f} AUC={val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_auc': val_auc, 'config': cfg.__dict__
            }, out_dir / 'best_model.pth')
            print(f"  ✓ Best: {best_auc:.4f}")
        else:
            patience += 1
            if patience >= cfg.patience:
                print(f"\nEarly stop at epoch {epoch}")
                break
    
    print(f"\n{'='*60}\nDone! Best AUC: {best_auc:.4f}\n{'='*60}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--val_dir', required=True)
    parser.add_argument('--output_dir', default='./checkpoints')
    parser.add_argument('--config', default='default')
    main(parser.parse_args())
