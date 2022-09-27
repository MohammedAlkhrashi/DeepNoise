import timm
from torch.optim import SGD
import torch
import torchvision.transforms as T
import torch.nn as nn
import numpy as np
from dataset import NoisyDataset
from callbacks import Callback, SimpleStats
import numpy as np

from erm import ERM

from torch.utils.data import DataLoader

from symmetric_loss import SymmetericLossTrainer
from bootstrap import BootstrappingLossTrainer
from loss_correction import LossCorrectionTrainer


def test_all():
    model: nn.Module = timm.create_model("efficientnet_b0", pretrained=False, num_classes=10)
    optim = SGD(model.parameters(), 0.02)
    loss_fn = nn.CrossEntropyLoss()

    transforms_list = []
    transforms_list.append(T.ToTensor())
    transforms = T.Compose(transforms_list)

    train_images = np.random.rand(20, 12, 12, 3).astype(np.float32)
    train_labels = np.random.randint(0, 9, (20,))
    train_set = NoisyDataset(train_images, train_labels, train_labels, transforms)

    val_images = np.random.rand(6, 12, 12, 3).astype(np.float32)
    val_labels = np.random.randint(0, 9, (6,))
    val_set = NoisyDataset(val_images, val_labels, val_labels, transforms)

    test_images = np.random.rand(6, 12, 12, 3).astype(np.float32)
    test_labels = np.random.randint(0, 9, (6,))
    test_set = NoisyDataset(test_images, test_labels, test_labels, transforms)

    train_loader = DataLoader(
        train_set, batch_size=2, shuffle=True, num_workers=1, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=2, shuffle=False, num_workers=1, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=2, shuffle=False, num_workers=1, pin_memory=True,
    )

    callbacks = [SimpleStats()]

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

    trainer = SymmetericLossTrainer(
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=2,
        callbacks=callbacks,
    )
    trainer.start()

    # trainer = LossCorrectionTrainer(
    #     model=model,
    #     optimizer=optim,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     test_loader=test_loader,
    #     epochs=2,
    #     callbacks=callbacks,
    #     T=torch.eye(10),
    #     correction='backward'

    # )
    # trainer.start()


if __name__ == "__main__":
    test_all()
