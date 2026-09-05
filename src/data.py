"""CIFAR-10 datasets and dataloaders with standard normalization/augmentation."""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_tf, test_tf


def get_dataloaders(data_dir="./data", batch_size=128, num_workers=4, seed=42):
    train_tf, test_tf = build_transforms()
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True, generator=g,
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


def get_test_loader(data_dir="./data", batch_size=256, num_workers=4):
    _, test_tf = build_transforms()
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)
    return DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


def get_calibration_loader(data_dir="./data", batch_size=128, num_workers=2, seed=42):
    """A held-out-from-augmentation loader over the training set, used only to
    collect activation min/max statistics for PTQ calibration (no labels/grads used)."""
    _, test_tf = build_transforms()
    calib_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=test_tf)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        calib_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, generator=g,
    )
