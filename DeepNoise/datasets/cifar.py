from copy import deepcopy
from venv import create

from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
import torchvision.datasets
from torchvision.transforms.transforms import Compose

from DeepNoise.builders import DATASETS
from DeepNoise.noise_injectors import IdentityNoiseInjector, NoiseInjector


@DATASETS.register()
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
            transforms = Compose(transforms=[])

        dataset = self.create_dataset(**kwargs)
        self.images = deepcopy(dataset.data)
        self.clean_labels = deepcopy(dataset.targets)
        self.noisy_labels = noise_injector.apply(self.clean_labels)
        print(
            confusion_matrix(self.clean_labels, self.noisy_labels) / len(self) * 10
        )  # TODO only works for cifar10
        self.transforms = transforms

    def create_dataset(self, **kwargs):
        return torchvision.datasets.CIFAR10(**kwargs)

    def __getitem__(self, index):
        item = dict()
        item["image"] = self.transforms(Image.fromarray(self.images[index]))
        item["clean_label"] = self.clean_labels[index]
        item["noisy_label"] = self.noisy_labels[index]
        item["sample_index"] = index
        return item

    def __len__(self):
        return len(self.images)


@DATASETS.register()
class NoisyCIFAR100(NoisyCIFAR10):
    def __init__(
        self, noise_injector: NoiseInjector = None, transforms=None, **kwargs
    ) -> None:
        super().__init__(noise_injector, transforms, **kwargs)
        """
        Args:
            noise_injector (NoiseInjector): A noise injector that will be applied to the clean labels
                                            to produce the synthetic noisy labels. If None self.noisy_labels
                                            will be equal to self.clean_labels.
            transforms: The transformation pipeline that will be applied to the images.
            **kwargs: torchvision.datasets.CIFAR100 arguments (keyword arguments)
        """

    def create_dataset(self, **kwargs):
        return torchvision.datasets.CIFAR100(**kwargs)
