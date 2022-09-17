from typing import List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from callbacks import Callback, SimpleStats
from dataset import NoisyDataset

from trainer import Trainer
from torch.optim import Optimizer

import numpy as np


class TrainERM(Trainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        callbacks: List[Callback] = None,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            loss_fn,
            train_loader,
            val_loader,
            test_loader,
            epochs,
            callbacks,
        )

    def step(self, batch):
        pred = self.model(batch["image"])
        loss = self.loss_fn(pred, batch["noisy_label"])

        if self.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            metrics = dict()
            clean_loss = self.loss_fn(pred, batch["clean_label"])
            metrics["prediction"] = pred
            metrics["noisy_loss"] = loss
            metrics["clean_loss"] = clean_loss
            metrics["sample_index"] = batch["sample_index"]
            metrics["noisy_label"] = batch["noisy_label"]
            metrics["clean_label"] = batch["clean_label"]
            [callback.on_step_end(metrics) for callback in self.callbacks]


if __name__ == "__main__":
    import timm
    from torch.optim import SGD
    import torchvision.transforms as T

    model: nn.Module = timm.create_model("resnet18", pretrained=False, num_classes=10)
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
        train_set,
        batch_size=2,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=2,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=2,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    callbacks = [SimpleStats()]
    trainer = TrainERM(
        model=model,
        optimizer=optim,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=10,
        callbacks=callbacks,
    )
    trainer.start()
