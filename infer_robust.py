"""
Inference for robust detector
"""
import sys, argparse, zipfile
import numpy as np
from pathlib import Path
import torch
from torch.amp import autocast
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).parent))
from models.robust_detector import RobustDetector

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

def find_images(d):
    imgs = []
    for ext in EXTS:
        imgs.extend(Path(d).rglob(f'*{ext}'))
    return sorted(set(imgs), key=lambda p: p.name)

def get_transforms(size):
    base = A.Compose([A.Resize(size, size),
                      A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                      ToTensorV2()])
    return [
        base,
        A.Compose([A.Resize(size, size), A.HorizontalFlip(p=1.0),
                   A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),
        A.Compose([A.Resize(int(size*1.1), int(size*1.1)), A.CenterCrop(size, size),
                   A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),
    ]

@torch.no_grad()
def predict(model, paths, transforms, device, bs=32):
    all_probs = np.zeros(len(paths))
    for tfm in transforms:
        print(f"  TTA pass...")
        for i in range(0, len(paths), bs):
            batch_paths = paths[i:i+bs]
            tensors = []
            for p in batch_paths:
                try:
                    img = np.array(Image.open(p).convert('RGB'))
                except:
                    img = np.zeros((224,224,3), dtype=np.uint8)
                tensors.append(tfm(image=img)['image'])
            batch = torch.stack(tensors).to(device)
            with autocast('cuda'):
                logits = model(batch).squeeze(1)
            all_probs[i:i+len(batch_paths)] += torch.sigmoid(logits).cpu().numpy()
    return all_probs / len(transforms)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', default='./submissions')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths = find_images(args.test_dir)
    print(f"Images: {len(paths)}")
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = RobustDetector(
        backbone=ckpt.get('backbone', 'convnext_tiny'),
        pretrained=False,
        image_size=ckpt.get('image_size', 224),
        use_patches=ckpt.get('use_patches', False)
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    
    tfms = get_transforms(ckpt.get('image_size', 224))
    probs = predict(model, paths, tfms, device, args.batch_size)
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    txt = out_dir / 'submission.txt'
    with open(txt, 'w') as f:
        for p in probs:
            f.write(f"{p:.6f}\n")
    
    zip_path = out_dir / 'submission.zip'
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(txt, 'submission.txt')
    
    print(f"\n{'='*60}")
    print(f"  submission.zip → {zip_path}")
    print(f"  Lines: {len(probs)} | Fake: {(probs>0.5).sum()} | Real: {(probs<=0.5).sum()}")
    print(f"  Range: {probs.min():.3f} - {probs.max():.3f}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
