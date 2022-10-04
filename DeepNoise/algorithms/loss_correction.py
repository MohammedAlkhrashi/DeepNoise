from typing import List

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from DeepNoise.algorithms.erm import ERM
from DeepNoise.builders import TRAINERS
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


@TRAINERS.register("LossCorrection")
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
        ignore_passed_loss_fn=False,
        **kwargs,
    ) -> None:

        passed_loss_fn = kwargs.pop("loss_fn", None)
        if passed_loss_fn is not None and not ignore_passed_loss_fn:
            raise ValueError(
                "LossCorrection trainer does not accept"
                " a loss_fn arguement that is not None when"
                " 'ignore_passed_loss_fn' is False"
            )

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
