from typing import List

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from DeepNoise.algorithms.erm import ERM
from DeepNoise.callbacks import Callback


class BackwardCorrectedLoss(Module):
    # TODO: add docstring
    def __init__(self, T: List[List[float]]):
        super(BackwardCorrectedLoss, self).__init__()
        self.T = T

    def forward(self, pred, target):
        T_inv = torch.linalg.inv(self.T).float()
        num_classes = pred.size(-1)
        one_hot_target = F.one_hot(target, num_classes=num_classes).float()
        pred_prop = F.softmax(pred, dim=1)
        return -torch.sum(torch.matmul(one_hot_target, T_inv) * torch.log(pred_prop))


class ForwardCorrectedLoss(Module):
    def __init__(self, T: List[List[float]]):
        super(ForwardCorrectedLoss, self).__init__()
        # TODO: check validitiy of T
        self.T = T

    def forward(self, pred, target):
        T = torch.Tensor(self.T).float()
        num_classes = pred.size(-1)
        one_hot_target = F.one_hot(target, num_classes=num_classes).float()
        pred_prop = F.softmax(pred, dim=1)
        return -torch.sum(one_hot_target * torch.log(torch.matmul(pred_prop, T)))


class LossCorrectionTrainer(ERM):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        T,
        correction: str = "backward",
        callbacks: List[Callback] = None,
    ) -> None:

        if correction == "backward":
            loss_fn = BackwardCorrectedLoss(T)
        elif correction == "forward":
            loss_fn = ForwardCorrectedLoss(T)
        else:
            raise ValueError(
                f'Correction type {correction} unrecognized. Pass either "forward" or "backward"'
            )

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