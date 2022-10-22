import timm
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.transforms as T
from torch.optim import SGD
from torch.utils.data import DataLoader

import wandb
from DeepNoise.algorithms.symmetric_loss import SymmetericLossTrainer
from DeepNoise.callbacks.lr_scheduler import StepLRScheduler
from DeepNoise.callbacks.statistics import SimpleStatistics
from DeepNoise.datasets.cifar import NoisyCIFAR10
from DeepNoise.noise_injectors import SymmetricNoiseInjector


def main():
    train_transform = T.Compose(
        [
            T.RandomCrop(size=32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    test_transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    noise_injector = SymmetricNoiseInjector(noise_prob=0.4)
    train_set = NoisyCIFAR10(
        noise_injector=noise_injector,
        train=True,
        download=True,
        transforms=train_transform,
        root="data/cifar10",
    )
    test_set = NoisyCIFAR10(
        train=False, download=True, transforms=test_transforms, root="data/cifar10"
    )

    epochs = 120
    batch_size = 16
    num_workers = 4
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model: nn.Module = timm.create_model("resnet18", pretrained=False, num_classes=10)
    optimizer = SGD(model.parameters(), 0.02, momentum=0.9, weight_decay=5e-4)

    callbacks = [
        SimpleStatistics(),
        StepLRScheduler(optimizer, milestones=[80, 120], gamma=0.1),
    ]

    wandb.init(project="DeepNoise")
    trainer = SymmetericLossTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        callbacks=callbacks,
    )
    trainer.start()


if __name__ == "__main__":
    main()
