from typing import List
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from tqdm import tqdm

from callbacks import Callback


class Trainer:
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
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs

        self.callbacks = callbacks

        self._mode = "train"

    def train(self):
        """_summary_
        Changes the trainer object and self.model to train model.
        """
        self.model.train()
        self._mode = "train"

    def eval(self, mode: str):
        """_summary_
        Changes the trainer object and self.model to eval model.

        Args:
            mode (str): Updates the "epoch mode" of the trainer.
        """
        self.model.eval()
        self._mode = mode

    @property
    def training(self) -> bool:
        return self._mode == "train"

    @property
    def epoch_mode(self) -> str:
        """_summary_

        Returns:
            str: the current epoch mode of the trainer, the epoch mode is used for logging.
        """
        return self._mode

    def step(self, batch, val_step=False):
        raise NotImplementedError

    def one_epoch(self, loader: DataLoader, epoch: int):
        for batch in tqdm(
            loader,
            desc=f"Running {self.epoch_mode.capitalize()}, Epoch: {epoch}",
            leave=True,
        ):
            self.step(batch)

        metrics = dict()
        metrics["epoch_mode"] = self.epoch_mode
        [callback.on_epoch_end(metrics) for callback in self.callbacks]

    def start(self):
        for epoch in tqdm(range(1, self.epochs + 1)):
            self.train()
            self.one_epoch(self.train_loader, epoch)

            with torch.no_grad():
                self.eval(mode="val")
                self.one_epoch(self.val_loader, epoch)
                self.eval(mode="test")
                self.one_epoch(self.test_loader, epoch)
