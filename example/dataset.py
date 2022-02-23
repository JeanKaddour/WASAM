import math
from typing import Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from example.config import TrainConfig
from example.cutout import Cutout

DATASETS = {
    "svhn": datasets.SVHN,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "imagenet": datasets.ImageNet,
}


def get_statistics(config: TrainConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    train_set = DATASETS[config.dataset_name](
        root=config.dataset_path,
        train=True,
        download=True,
        transform=transforms.transforms.ToTensor(),
    )
    data = torch.cat([d[0] for d in DataLoader(train_set)])
    return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])


def get_dataset(
    config: TrainConfig,
) -> Tuple[datasets.VisionDataset, datasets.VisionDataset]:
    mean, std = get_statistics(config=config)
    train_transform = [
        transforms.transforms.RandomCrop(size=(32, 32), padding=4),
        transforms.transforms.RandomHorizontalFlip(),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(mean, std),
        Cutout(),
    ]

    test_transform = [
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(mean, std),
    ]

    train_transform = transforms.transforms.Compose(train_transform)
    test_transform = transforms.transforms.Compose(test_transform)

    train_dataset = DATASETS[config.dataset_name](
        root=config.dataset_path,
        train=True,
        transform=train_transform,
        download=True,
    )
    test_dataset = DATASETS[config.dataset_name](
        root=config.dataset_path,
        train=False,
        transform=test_transform,
        download=True,
    )
    return train_dataset, test_dataset


def get_train_valid_test_set(
    config: TrainConfig,
) -> tuple[
    datasets.VisionDataset, Optional[datasets.VisionDataset], datasets.VisionDataset
]:
    train_set, test_set = get_dataset(config=config)
    val_set = None
    if config.use_val_set:
        train_size, valid_size = math.floor(
            len(train_set) * (1 - config.val_size)
        ), math.ceil(len(train_set) * config.val_size)
        train_set, val_set = torch.utils.data.random_split(
            train_set, [train_size, valid_size]
        )
    return train_set, val_set, test_set


def get_train_loader(config: Union) -> torch.utils.data.DataLoader:
    train_set, _ = get_dataset(config=config)
    return torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=False,
    )


def get_train_test_loader(
    config: TrainConfig,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_set, test_set = get_dataset(config=config)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, test_loader


def get_train_valid_test_loader(
    config: TrainConfig,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    train_set, val_set, test_set = get_train_valid_test_set(config=config)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
