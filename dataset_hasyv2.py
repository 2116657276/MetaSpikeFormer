"""
HASYv2 Data Pipeline for Spiking Neural Networks.

HASYv2: Handwritten Symbol dataset v2 — 369 classes of LaTeX math symbols.
  - 168,233 total images (~151K train, ~17K test in fold-1)
  - 32x32 grayscale PNG images (flat directory)
  - Labels in CSV files (path, symbol_id, latex, user_id)
  - Source: Martin Thoma, https://github.com/MartinThoma/hasy

Loads HASYv2 and unrolls static images along a time dimension (T=4)
to produce spike-compatible input tensors of shape [T, B, C, H, W].
"""

import csv
import os
import tarfile
import urllib.request
from pathlib import Path
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


HASYV2_NUM_CLASSES = 369
HASYV2_MEAN = (0.9372,)
HASYV2_STD  = (0.2310,)

HASYV2_URLS = [
    "https://zenodo.org/record/259444/files/HASYv2.tar.bz2",
    "https://github.com/MartinThoma/hasy/raw/master/HASYv2.tar.gz",
]


class HASYv2Dataset(Dataset):
    """Custom Dataset for HASYv2 flat-image + CSV label structure."""

    def __init__(self, image_dir: str, csv_path: str, transform=None):
        self.image_dir = Path(image_dir).resolve()
        self.transform = transform

        self.samples = []
        self.class_to_idx = {}
        self.classes = []

        # Resolve CSV paths relative to the CSV file's directory
        csv_dir = Path(csv_path).parent.resolve()

        # Build unique classes first (faster — skip file checks)
        symbol_ids = set()
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = int(row['symbol_id'])
                symbol_ids.add(sid)

        # Sort for deterministic class ordering
        for sid in sorted(symbol_ids):
            self.class_to_idx[sid] = len(self.classes)
            self.classes.append(str(sid))

        # Load samples (use prebuilt class map for speed)
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_rel = row['path']
                # Resolve path — images are flat in hasy-data/
                img_name = Path(img_rel).name  # e.g., "v2-00000.png"
                img_path = self.image_dir / img_name

                symbol_id = int(row['symbol_id'])
                label_idx = self.class_to_idx[symbol_id]
                self.samples.append((str(img_path), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')  # grayscale
        if self.transform:
            img = self.transform(img)
        return img, label


def _download_and_extract(root: str = "./data") -> tuple:
    """
    Download HASYv2 to root/hasyv2/ if not present.
    Returns (image_dir, csv_dir) paths.
    """
    data_dir = Path(root) / "hasyv2"
    extracted_marker = data_dir / "extracted.txt"

    if extracted_marker.exists():
        # Already extracted
        return str(data_dir / "hasy-data"), str(data_dir / "classification-task" / "fold-1")

    # Need to download
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "HASYv2.tar.bz2"

    # Try download
    downloaded = False
    for url in HASYV2_URLS:
        try:
            print(f"Downloading HASYv2 from {url}...")
            print("  (~200MB, may take a few minutes)")
            urllib.request.urlretrieve(url, str(archive_path))
            downloaded = True
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if not downloaded:
        raise RuntimeError(
            f"Could not download HASYv2. Please download manually from:\n"
            f"  https://zenodo.org/record/259444\n"
            f"  and place HASYv2.tar.bz2 in {data_dir}/")

    # Extract
    print(f"Extracting {archive_path.name}...")
    try:
        with tarfile.open(str(archive_path), "r:bz2") as tf:
            tf.extractall(path=str(data_dir))
    except Exception:
        with tarfile.open(str(archive_path), "r:*") as tf:
            tf.extractall(path=str(data_dir))

    # Clean up archive to save space
    archive_path.unlink(missing_ok=True)

    extracted_marker.touch()

    return str(data_dir / "hasy-data"), str(data_dir / "classification-task" / "fold-1")


def build_hasyv2(
    batch_size: int = 16,
    num_workers: int = 0,
    T: int = 4,
    root: str = "./data",
    max_train_samples: int = 0,
    max_val_samples: int = 0,
):
    """
    Build HASYv2 train/val dataloaders using fold-1 split.

    Args:
        batch_size: mini-batch size.
        num_workers: data-loading worker processes (0 for MPS).
        T: number of SNN time steps.
        root: data root directory.
        max_train_samples: limit training samples (0 = all ~151K).
        max_val_samples: limit validation samples (0 = all ~17K).

    Returns:
        train_loader, val_loader, num_classes (369).
    """
    image_dir, fold_dir = _download_and_extract(root)

    print(f"HASYv2 image dir: {image_dir}")
    print(f"HASYv2 fold dir:  {fold_dir}")

    # ---- Transforms ----
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=2),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(HASYV2_MEAN, HASYV2_STD),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(HASYV2_MEAN, HASYV2_STD),
    ])

    # ---- Datasets ----
    train_csv = os.path.join(fold_dir, "train.csv")
    test_csv  = os.path.join(fold_dir, "test.csv")

    train_full = HASYv2Dataset(image_dir, train_csv, transform=transform_train)
    val_full   = HASYv2Dataset(image_dir, test_csv, transform=transform_val)

    num_classes = len(train_full.classes)
    print(f"HASYv2: {len(train_full)} train, {len(val_full)} test, {num_classes} classes")

    # Apply sample limits
    from torch.utils.data import Subset
    if max_train_samples > 0 and max_train_samples < len(train_full):
        train_full = Subset(train_full, range(max_train_samples))
    if max_val_samples > 0 and max_val_samples < len(val_full):
        val_full = Subset(val_full, range(max_val_samples))

    print(f"Using: {len(train_full)} train, {len(val_full)} val")

    # ---- DataLoaders ----
    train_loader = DataLoader(
        train_full, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_full, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    return train_loader, val_loader, num_classes
