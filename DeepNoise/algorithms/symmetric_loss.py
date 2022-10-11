import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from DeepNoise.algorithms.erm import ERM
from DeepNoise.builders import TRAINERS
from DeepNoise.callbacks.statistics import Callback


class SCELoss(nn.Module):
    """
    ``Loss(t, p) = alpha*CE + beta*RCE`` where log(0) is defined to equal A.
    The default parameters  for alpha,betea and A are taken from the orignal paper.
    https://arxiv.org/pdf/1908.06112.pdf

    Args:
        alpha (float) Defaults to 0.01.
        beta (int): Defaults to 1.
        A (int): Defaults to -4.
    """

    def __init__(self, alpha=0.01, beta: int = 1, A: int = -4, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.val_before_log_0 = math.exp(A)  # log(val_before_log_0) = A

        self.reduction = reduction

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction=self.reduction)

        num_classes = pred.size(-1)
        one_hot_target = F.one_hot(target, num_classes).float()
        one_hot_target[one_hot_target == 0] = self.val_before_log_0
        pred_prop = F.softmax(pred, dim=1)
        rce_loss = -torch.sum(pred_prop * torch.log(one_hot_target), dim=1)
        if self.reduction == "mean":
            rce_loss = torch.mean(rce_loss)
        elif self.reduction == "sum":
            rce_loss = torch.sum(rce_loss)

        loss = self.alpha * ce_loss + self.beta * rce_loss
        return loss


@TRAINERS.register("SymmetericLoss")
class SymmetericLossTrainer(ERM):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        test_loader: DataLoader = None,
        callbacks: List[Callback] = None,
        alpha=0.01,
        beta=1,
        A=-4,
        reduction: str = "mean",
        ignore_passed_loss_fn=False,
        **kwargs
    ) -> None:

        passed_loss_fn = kwargs.pop("loss_fn", None)
        if passed_loss_fn is not None and not ignore_passed_loss_fn:
            raise ValueError(
                "SymmetricLoss trainer does not accept"
                " a loss_fn arguement that is not None when"
                " 'ignored_passed_loss_fn' is False"
            )

        loss_fn = SCELoss(alpha, beta, A, reduction)

        super().__init__(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            callbacks=callbacks,
        )
