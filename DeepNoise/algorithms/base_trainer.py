from typing import List

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from DeepNoise.callbacks.statistics import Callback


def _empty_loader():
    class EmptyDataset(Dataset):
        def __len__(self):
            return 0

    return DataLoader(EmptyDataset())


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        test_loader: DataLoader = None,
        callbacks: List[Callback] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.train_loader = train_loader
        self.val_loader = val_loader if val_loader is not None else _empty_loader()
        self.test_loader = test_loader if test_loader is not None else _empty_loader()
        self.epochs = epochs

        self.callbacks = callbacks

        self.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self._mode = "train"

    def train(self):
        """
        Changes the trainer object and self.model to train mode.
        """
        self.model.train()
        self._mode = "train"

    def eval(self, mode: str):
        """
        Changes the trainer object and self.model to eval mode.

        Args:
            mode (str): Updates the "epoch mode" of the trainer (val/test).
        """
        self.model.eval()
        self._mode = mode

    @property
    def training(self) -> bool:
        return self._mode == "train"

    @property
    def epoch_mode(self) -> str:
        """
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
            disable=len(loader) == 0,
        ):
            self.step(batch)

        metrics = dict()
        metrics["epoch"] = epoch
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
