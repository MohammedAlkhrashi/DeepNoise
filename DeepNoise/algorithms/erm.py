from typing import List

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from DeepNoise.callbacks.statistics import Callback
from DeepNoise.algorithms import Trainer


class ERM(Trainer):
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
        batch = {key: value.to(self.device) for key, value in batch.items()}
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
