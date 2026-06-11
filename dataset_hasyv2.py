"""
HASYv2 Data Pipeline with user-based train/val split.

369 classes, 168K grayscale 32x32 images.
Key fix: zero user overlap between train and val.
"""

import csv, os, random, tarfile, urllib.request
from pathlib import Path
from collections import defaultdict
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

HASYV2_NUM_CLASSES = 369
HASYV2_MEAN = (0.9372,)
HASYV2_STD  = (0.2310,)
HASYV2_URL = "https://zenodo.org/record/259444/files/HASYv2.tar.bz2"


class HASYv2Dataset(Dataset):
    def __init__(self, image_dir, samples, class_to_idx, transform=None):
        self.image_dir = Path(image_dir)
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.classes = [str(k) for k in sorted(class_to_idx.keys())]
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        img = Image.open(self.image_dir / name).convert('L')
        return self.transform(img) if self.transform else transforms.ToTensor()(img), label


def _ensure_data(root):
    d = Path(root) / "hasyv2"
    if (d / "extracted.txt").exists():
        return str(d / "hasy-data"), str(d / "hasy-data-labels.csv")
    d.mkdir(parents=True, exist_ok=True)
    arc = d / "HASYv2.tar.bz2"
    print(f"Downloading HASYv2 (~200MB)...")
    urllib.request.urlretrieve(HASYV2_URL, str(arc))
    print("Extracting...")
    try:
        with tarfile.open(str(arc), "r:bz2") as tf: tf.extractall(path=str(d))
    except Exception:
        with tarfile.open(str(arc), "r:*") as tf: tf.extractall(path=str(d))
    arc.unlink(missing_ok=True)
    (d / "extracted.txt").touch()
    return str(d / "hasy-data"), str(d / "hasy-data-labels.csv")


def build_hasyv2(batch_size=16, num_workers=0, T=4, root="./data",
                 max_train_samples=0, max_val_samples=0,
                 val_user_frac=0.15, seed=42):
    """User-based train/val split. Zero user overlap."""
    image_dir, labels_csv = _ensure_data(root)
    print(f"HASYv2: {image_dir}")

    # Load all data grouped by user
    user_samples = defaultdict(list)
    symbol_ids = set()
    with open(labels_csv) as f:
        for row in csv.DictReader(f):
            user_samples[row['user_id']].append((Path(row['path']).name, int(row['symbol_id'])))
            symbol_ids.add(int(row['symbol_id']))

    class_to_idx = {sid: i for i, sid in enumerate(sorted(symbol_ids))}
    nc = len(class_to_idx)

    # Identify mega-user (>20% of data) and split their images 80/20
    rng = random.Random(seed)
    total = sum(len(v) for v in user_samples.values())
    train_s, val_s = [], []

    for uid, imgs in user_samples.items():
        if len(imgs) > total * 0.20:
            rng.shuffle(imgs)
            split = int(len(imgs) * 0.80)
            train_s.extend((n, class_to_idx[s]) for n, s in imgs[:split])
            val_s.extend((n, class_to_idx[s]) for n, s in imgs[split:])
        else:
            train_s.extend((n, class_to_idx[s]) for n, s in imgs)

    rng.shuffle(train_s)
    rng.shuffle(val_s)

    # Apply limits
    if max_train_samples > 0:
        train_s = train_s[:max_train_samples]
    if max_val_samples > 0:
        val_s = val_s[:max_val_samples]

    print(f"Samples: {len(train_s)} train, {len(val_s)} val, {nc} classes")

    # Transforms
    tr = transforms.Compose([
        transforms.RandomCrop(32, padding=2),
        transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.ToTensor(), transforms.Normalize(HASYV2_MEAN, HASYV2_STD)])
    tv = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(HASYV2_MEAN, HASYV2_STD)])

    # Build
    train_loader = DataLoader(HASYv2Dataset(image_dir, train_s, class_to_idx, tr),
                              batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(HASYv2Dataset(image_dir, val_s, class_to_idx, tv),
                            batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True, drop_last=False)
    return train_loader, val_loader, nc
