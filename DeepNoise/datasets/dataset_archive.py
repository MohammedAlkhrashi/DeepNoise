from copy import copy, deepcopy

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100

from DeepNoise.noise_injectors import NoiseInjector


class NoisyDataset(Dataset):
    def __init__(self, images, clean_labels, noisy_labels, transfroms=None) -> None:
        self.images = images
        self.clean_labels = clean_labels
        self.noisy_labels = noisy_labels
        self.transforms = transfroms

        self.is_paths = isinstance(
            self.images[0], str
        )  # check if self.images are list of paths, (unlike cifar10/100 where self.images is a list of raw images)

    def __getitem__(self, index):
        item = dict()

        if self.is_paths:
            item["image"] = Image.open(self.images[index]).convert("RGB")
        else:
            item["image"] = self.images[index]

        if self.transforms:
            item["image"] = self.transforms(item["image"])

        item["clean_label"] = self.clean_labels[index]
        item["noisy_label"] = self.noisy_labels[index]
        item["sample_index"] = index
        return item

    def __len__(self):
        return len(self.images)


def create_cifar10_dataset(root="./data/cifar"):
    batch_size = 128
    num_workers = 4

    processing_transforms = []
    aug_transforms = []
    aug_transforms.append(T.RandomHorizontalFlip())
    processing_transforms.append(T.ToTensor())
    processing_transforms.append(
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    )

    train_transforms = T.Compose(
        [
            Image.fromarray,
            T.RandomCrop(32, padding=4, padding_mode="reflect"),
            *aug_transforms,
            *processing_transforms,
        ]
    )
    test_transforms = T.Compose([Image.fromarray, *processing_transforms])

    loaders = []
    for split_bool, transform in zip(
        [True, False], [train_transforms, test_transforms]
    ):
        train_set = CIFAR10(root=root, train=split_bool, download=True)
        images = deepcopy(train_set.data)
        target = deepcopy(train_set.targets)
        noisy_set = NoisyDataset(
            images, clean_labels=target, noisy_labels=target, transfroms=transform
        )

        loader = DataLoader(
            noisy_set,
            batch_size=batch_size,
            shuffle=split_bool,
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders.append(loader)
    loaders.append(loaders[-1])
    return loaders
