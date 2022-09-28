from copy import deepcopy

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

from DeepNoise.noise_injectors import (
    IdentityNoiseInjector,
    NoiseInjector,
)


class NoisyCIFAR10(Dataset):
    def __init__(
        self, noise_injector: NoiseInjector = None, transforms=None, **kwargs
    ) -> None:
        """
        Args:
            noise_injector (NoiseInjector): A noise injector that will be applied to the clean labels
                                            to produce the synthetic noisy labels. If None self.noisy_labels
                                            will be equal to self.clean_labels.
            transforms: The transformation pipeline that will be applied to the images.
            **kwargs: torchvision.datasets.CIFAR10 arguments (keyword arguments)
        """
        if noise_injector is None:
            noise_injector = IdentityNoiseInjector()
        if transforms is None:
            transforms = lambda x: x

        dataset = CIFAR10(**kwargs)
        self.images = deepcopy(dataset.data)
        self.clean_labels = deepcopy(dataset.targets)
        self.noisy_labels = noise_injector.apply(self.clean_labels)
        self.transforms = transforms

    def __getitem__(self, index):
        item = dict()

        item["image"] = self.transforms(self.images[index])
        item["clean_label"] = self.clean_labels[index]
        item["noisy_label"] = self.noisy_labels[index]
        item["sample_index"] = index
        return item

    def __len__(self):
        return len(self.images)
