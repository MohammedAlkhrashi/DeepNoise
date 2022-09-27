import torch
from typing import List
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from erm import ERM
from callbacks import Callback


class BackwardCorrectedLoss(Module):
    # TODO: add docstring
    def __init__(self, T: List[List[float]]):
        super(BackwardCorrectedLoss, self).__init__()
        self.T = T

    def forward(self, pred, target):
        T_inv = torch.linalg.inv(self.T).float()
        num_classes = pred.size(-1)
        one_hot_target = F.one_hot(target, num_classes=num_classes).float()
        return -torch.sum(torch.matmul(one_hot_target, T_inv) * torch.log_softmax(pred, dim=1))


class ForwardCorrectedLoss(Module):
    def __init__(self, T: List[List[float]]):
        super(ForwardCorrectedLoss, self).__init__()
        # TODO: check validitiy of T
        self.T = T

    def forward(self, pred, target):
        T = torch.Tensor(self.T).float()
        num_classes = pred.size(-1)
        one_hot_target = F.one_hot(target, num_classes=num_classes).float()

        pred_prob = F.softmax(pred, dim=1)
        eps = 1e-10
        pred_prob = torch.clip(pred_prob, eps, 1.0 - eps)

        return -torch.sum(one_hot_target * torch.log(torch.matmul(pred_prob, T)))


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
