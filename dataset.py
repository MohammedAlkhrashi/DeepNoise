import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset


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
        # TODO: convert item from dict to dataclass
        item = dict()

        if self.is_paths:
            item["image"] = Image.open(item["image"]).convert("RGB")
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
