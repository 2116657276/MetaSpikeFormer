"""
CIFAR-100 Data Pipeline for Spiking Neural Networks.

Loads CIFAR-100 and unrolls static images along a time dimension (T=4)
to produce spike-compatible input tensors of shape [T, B, C, H, W].
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# CIFAR-100 mean and std (standard values)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def build_cifar100(batch_size: int = 128, num_workers: int = 4, T: int = 4):
    """
    Build CIFAR-100 train/val dataloaders.

    Args:
        batch_size: mini-batch size.
        num_workers: data-loading worker processes.
        T: number of SNN time steps (static images are repeated T times).

    Returns:
        train_loader, val_loader, num_classes (100).
    """
    # --- Training transforms ---
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    # --- Validation transforms ---
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    # --- Datasets ---
    train_set = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train,
    )
    val_set = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_val,
    )

    # --- DataLoaders ---
    # The time-unrolling transform is applied inside the model's forward pass
    # by repeating the first dimension.  We keep the dataloader producing
    # standard [B, C, H, W] tensors; the model handles the T-repeat.
    # This is cleaner and avoids modifying collate_fn.
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    return train_loader, val_loader, 100


class TimeRepeatTransform:
    """
    Wraps a batch so the first-time caller gets [T, B, C, H, W] directly
    from a standard [B, C, H, W] dataloader by repeating along a new dim 0.
    """

    def __init__(self, T: int = 4):
        self.T = T

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] → repeat T times → [T, B, C, H, W]
        return x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)


# Convenience: collate function that does time-repeat on the fly.
# Usage: DataLoader(..., collate_fn=time_repeat_collate(4))
def time_repeat_collate(T: int = 4):
    """Returns a collate_fn that adds a T dimension to images."""

    def _collate(batch):
        images = torch.stack([item[0] for item in batch], dim=0)  # [B, C, H, W]
        labels = torch.tensor([item[1] for item in batch])
        # Repeat T times along a new first dimension → [T, B, C, H, W]
        images = images.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        return images, labels

    return _collate
