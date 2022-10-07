import numpy as np
import timm
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import SGD
from torch.utils.data import DataLoader

from DeepNoise.algorithms.bootstrap import BootstrappingLossTrainer
from DeepNoise.algorithms.erm import ERM
from DeepNoise.algorithms.pencil import Pencil
from DeepNoise.algorithms.symmetric_loss import SymmetericLossTrainer
from DeepNoise.callbacks.statistics import Callback, SimpleStatistics
from DeepNoise.datasets.cifar import NoisyCIFAR10
from DeepNoise.noise_injectors import SymmetricNoiseInjector


def test_all():
    model: nn.Module = timm.create_model(
        "efficientnet_b0", pretrained=False, num_classes=10
    )
    optim = SGD(model.parameters(), 0.02, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    train_set = NoisyCIFAR10(
        noise_injector=SymmetricNoiseInjector(0.2),
        root="data",
        train=True,
        download=True,
    )

    val_set = NoisyCIFAR10(
        root="data",
        train=False,
        download=True,
    )
    test_set = val_set
    train_loader = DataLoader()

    callbacks = [SimpleStatistics()]

    print("START ERM")
    trainer = ERM(
        model=model,
        optimizer=optim,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=1,
        callbacks=callbacks,
    )
    trainer.start()

    print("START Symmetric")

    trainer = SymmetericLossTrainer(
        model=model,
        optimizer=optim,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=1,
        callbacks=callbacks,
    )
    trainer.start()

    print("Start Pencil")
    trainer = Pencil(
        model=model,
        optimizer=optim,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=1,
        callbacks=callbacks,
        training_labels=train_loader.dataset.noisy_labels,
        alpha=0.01,
        b=0.1,
        labels_lr=500,
        stages=[70, 200, 320],
    )
    trainer.start()


if __name__ == "__main__":
    test_all()
