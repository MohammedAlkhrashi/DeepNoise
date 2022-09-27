import timm
from torch.optim import SGD
import torchvision.transforms as T
import torch.nn as nn
import numpy as np
from DeepNoise.datasets.dataset_archive import NoisyDataset
from DeepNoise.callbacks.statistics import Callback, SimpleStatistics
from DeepNoise.algorithms.erm import ERM
from DeepNoise.algorithms.pencil import Pencil

import numpy as np
from torch.utils.data import DataLoader

def test_all():
    model: nn.Module = timm.create_model("efficientnet_b0", pretrained=False, num_classes=10)

    optim = SGD(model.parameters(), 0)
    loss_fn = nn.CrossEntropyLoss()

    transforms_list = []
    transforms_list.append(T.ToTensor())
    transforms = T.Compose(transforms_list)

    train_images = np.random.rand(6, 12, 12, 3).astype(np.float32)
    train_labels = np.random.randint(0, 9, (6,))
    train_set = NoisyDataset(train_images, train_labels, train_labels, transforms)

    val_images = np.random.rand(6, 12, 12, 3).astype(np.float32)
    val_labels = np.random.randint(0, 9, (6,))
    val_set = NoisyDataset(val_images, val_labels, val_labels, transforms)

    test_images = np.random.rand(6, 12, 12, 3).astype(np.float32)
    test_labels = np.random.randint(0, 9, (6,))
    test_set = NoisyDataset(test_images, test_labels, test_labels, transforms)

    train_loader = DataLoader(
        train_set,
        batch_size=3,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=3,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=3,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
    )

    callbacks = [SimpleStatistics()]

    # trainer = ERM(
    #     model=model,
    #     optimizer=optim,
    #     loss_fn=loss_fn,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     test_loader=test_loader,
    #     epochs=2,
    #     callbacks=callbacks,
    # )
    # trainer.start()

    # trainer = SymmetericLossTrainer(
    #     model=model,
    #     optimizer=optim,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     test_loader=test_loader,
    #     epochs=2,
    #     callbacks=callbacks,
    # )
    # trainer.start()

    # trainer = BootstrappingLossTrainer(
    #     model=model,
    #     optimizer=optim,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     test_loader=test_loader,
    #     epochs=2,
    #     callbacks=callbacks,
    # )
    # trainer.start()

    trainer = Pencil(
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=15,
        callbacks=callbacks,
        training_labels=train_labels,
        labels_lr=2,
        alpha=0,
        beta=0.1,
        stages=[1,12,15],
        num_classes=10
    )
    trainer.start()

if __name__ == "__main__":
    test_all()
    