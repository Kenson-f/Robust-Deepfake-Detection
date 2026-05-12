"""
Dataset for NTIRE 2026 Robust Deepfake Detection Challenge

Supported directory layouts (auto-detected):

Format A — class subfolders:
    root_dir/
        real/   -> label 0
        fake/   -> label 1

Format B — flat images, labels encoded in filename:
    root_dir/
        0001_real.png  -> label 0
        0971_fake.png  -> label 1

Format C — flat images + labels.txt:
    root_dir/
        img_0001.png
        labels.txt   (one label per line: 0 or 1, alphabetical order)

Format D — flat images + labels.csv:
    root_dir/
        img_0001.png
        labels.csv   (columns: filename, label)

Format E — inference/validation with no labels:
    root_dir/
        img_0001.png   (no labels — used for submission generation)
"""

import csv
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif'}


def find_images(directory: Path) -> List[Path]:
    """Recursively collect all image files, sorted alphabetically."""
    imgs = []
    for ext in VALID_EXTENSIONS:
        imgs.extend(directory.rglob(f'*{ext}'))
        imgs.extend(directory.rglob(f'*{ext.upper()}'))
    return sorted(set(imgs))


def load_image_as_numpy(path: Path) -> np.ndarray:
    """Load image from path and return as uint8 RGB numpy array."""
    try:
        img = Image.open(path).convert('RGB')
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"Warning: could not load {path}: {e}. Using blank image.")
        return np.zeros((224, 224, 3), dtype=np.uint8)


class DeepfakeDataset(Dataset):
    """
    Flexible deepfake dataset supporting multiple directory layouts.
    Works with albumentations transforms (expects numpy arrays).
    Images with label=-1 are excluded from train/val modes automatically.
    """

    def __init__(
        self,
        root_dir,
        transform=None,
        mode='train',
        label_file='labels.txt',
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        self.image_paths = []
        self.labels = []

        self._load_dataset(label_file)

        # Drop unlabeled entries for train/val (keep for test/infer)
        if mode in ('train', 'val'):
            paired = [(p, l) for p, l in zip(self.image_paths, self.labels) if l != -1]
            if paired:
                self.image_paths, self.labels = map(list, zip(*paired))
            else:
                self.image_paths, self.labels = [], []

        n_real = self.labels.count(0)
        n_fake = self.labels.count(1)
        n_unlabeled = self.labels.count(-1)
        msg = f"[{mode.upper()}] {len(self.image_paths)} images | Real: {n_real} | Fake: {n_fake}"
        if n_unlabeled:
            msg += f" | Unlabeled: {n_unlabeled}"
        print(msg)

    def _load_dataset(self, label_file):
        """Auto-detect format and populate self.image_paths + self.labels."""

        # Format A: real/ and fake/ subfolders
        real_dir = self.root_dir / 'real'
        fake_dir = self.root_dir / 'fake'
        if real_dir.exists() or fake_dir.exists():
            pairs = []
            if real_dir.exists():
                for p in find_images(real_dir):
                    pairs.append((p, 0))
            if fake_dir.exists():
                for p in find_images(fake_dir):
                    pairs.append((p, 1))
            pairs.sort(key=lambda x: x[0].name)
            if pairs:
                self.image_paths, self.labels = map(list, zip(*pairs))
            return

        # Format B: filename encodes label e.g. "0971_fake.png" / "0001_real.png"
        all_imgs = sorted(find_images(self.root_dir))
        if all_imgs:
            labeled, unlabeled = [], []
            for img in all_imgs:
                stem = img.stem.lower()
                if '_fake' in stem:
                    labeled.append((img, 1))
                elif '_real' in stem:
                    labeled.append((img, 0))
                else:
                    unlabeled.append(img)

            if labeled:
                for img, lbl in labeled:
                    self.image_paths.append(img)
                    self.labels.append(lbl)
                for img in unlabeled:
                    self.image_paths.append(img)
                    self.labels.append(-1)
                return

        # Format C: labels.txt file
        label_path = self.root_dir / label_file
        if label_path.exists():
            with open(label_path, 'r') as f:
                raw_labels = [line.strip() for line in f if line.strip()]
            all_imgs = sorted(find_images(self.root_dir))
            if len(all_imgs) != len(raw_labels):
                raise ValueError(
                    f"Mismatch: {len(all_imgs)} images but {len(raw_labels)} labels in {label_path}"
                )
            for img, lbl in zip(all_imgs, raw_labels):
                self.image_paths.append(img)
                self.labels.append(int(lbl))
            return

        # Format D: labels.csv file
        csv_path = self.root_dir / 'labels.csv'
        if csv_path.exists():
            img_label_map = {}
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row.get('filename', row.get('image', row.get('name', '')))
                    label = int(row.get('label', row.get('class', 0)))
                    img_label_map[fname] = label
            all_imgs = sorted(find_images(self.root_dir))
            for img in all_imgs:
                if img.name in img_label_map:
                    self.image_paths.append(img)
                    self.labels.append(img_label_map[img.name])
            return

        # Format E: no labels at all (pure inference / validation set)
        all_imgs = sorted(find_images(self.root_dir))
        if all_imgs:
            self.image_paths = all_imgs
            self.labels = [-1] * len(all_imgs)
            return

        raise FileNotFoundError(
            f"No images found in {self.root_dir}.\n"
            "Supported formats:\n"
            "  * Subfolders: real/ and fake/\n"
            "  * Filename labels: *_real.png / *_fake.png\n"
            "  * labels.txt (one 0/1 per line, alphabetical order)\n"
            "  * labels.csv (columns: filename, label)"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load as numpy array (required by albumentations)
        image = load_image_as_numpy(img_path)

        if self.transform is not None:
            # albumentations requires keyword argument: aug(image=arr)
            augmented = self.transform(image=image)
            image = augmented['image']  # returns tensor via ToTensorV2

        return image, label

    def get_image_paths(self):
        return self.image_paths


class InferenceDataset(Dataset):
    """Dataset for inference-only (no labels required)."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = sorted(find_images(self.root_dir))
        print(f"[INFERENCE] Found {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = load_image_as_numpy(img_path)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, str(img_path.name)
