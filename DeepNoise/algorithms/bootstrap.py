from typing import List

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from DeepNoise.algorithms.erm import ERM
from DeepNoise.builders import TRAINERS
from DeepNoise.callbacks import Callback


class SoftBootstrappingLoss(Module):
    """
    ``Loss(t, p) = - (beta * t + (1 - beta) * p) * log(p)``

    Args:
            beta (float): bootstrap parameter. Default, 0.95
            reduce (bool): computes mean of the loss. Default, True.
            as_pseudo_label (bool): Stop gradient propagation for the term ``(1 - beta) * p``.
                    Can be interpreted as pseudo-label.
    """

    def __init__(self, beta=0.95, reduction: str = "mean", as_pseudo_label=True):
        super(SoftBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.as_pseudo_label = as_pseudo_label

    def forward(self, y_pred, y):
        # cross_entropy = - t * log(p)
        beta_xentropy = self.beta * F.cross_entropy(y_pred, y, reduction="none")

        y_pred_a = y_pred.detach() if self.as_pseudo_label else y_pred
        # second term = - (1 - beta) * p * log(p)
        bootstrap = -(1.0 - self.beta) * torch.sum(
            F.softmax(y_pred_a, dim=1) * F.log_softmax(y_pred, dim=1), dim=1
        )

        if self.reduction == "mean":
            return torch.mean(beta_xentropy + bootstrap)
        elif self.reduction == "sum":
            return torch.sum(beta_xentropy + bootstrap)

        return beta_xentropy + bootstrap


class HardBootstrappingLoss(Module):
    """
    ``Loss(t, p) = - (beta * t + (1 - beta) * z) * log(p)``
    where ``z = argmax(p)``
    Args:
            beta (float): bootstrap parameter. Default, 0.95
            reduce (bool): computes mean of the loss. Default, True.
    """

    def __init__(self, beta=0.8, reduction: str = "mean"):
        super(HardBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, y_pred, y):
        # cross_entropy = - t * log(p)
        beta_xentropy = self.beta * F.cross_entropy(y_pred, y, reduction="none")

        # z = argmax(p)
        z = F.softmax(y_pred.detach(), dim=1).argmax(dim=1)
        z = z.view(-1, 1)
        bootstrap = F.log_softmax(y_pred, dim=1).gather(1, z).view(-1)
        # second term = (1 - beta) * z * log(p)
        bootstrap = -(1.0 - self.beta) * bootstrap

        if self.reduction == "mean":
            return torch.mean(beta_xentropy + bootstrap)
        elif self.reduction == "sum":
            return torch.sum(beta_xentropy + bootstrap)

        return beta_xentropy + bootstrap


@TRAINERS.register("Bootstrapping")
class BootstrappingLossTrainer(ERM):
    """
    Implementation of the the paper:
    Training Deep Neural Networks on Noisy Labels with Bootstrapping
    https://arxiv.org/abs/1412.6596
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        test_loader: DataLoader = None,
        bootstrapping: str = "soft",
        beta: float = 0.95,
        reduction: str = "mean",
        as_pseudo_label: bool = True,
        ignore_passed_loss_fn=False,
        callbacks: List[Callback] = None,
        **kwargs,
    ) -> None:
        """Create a BootsrapingLossTrainer.

        Args:
            bootstrapping (str, optional): The bootstrapingloss mode, (options: 'soft', 'hard').
            beta (int, optional): The beta hyperparameters as described in the paper.
            reduction (str, optional): Loss reduction mode.
        """

        passed_loss_fn = kwargs.pop("loss_fn", None)
        if passed_loss_fn is not None and not ignore_passed_loss_fn:
            raise ValueError(
                "BootstrappingLoss trainer does not accept"
                " a loss_fn arguement that is not None when"
                " 'ignore_passed_loss_fn' is False"
            )

        if bootstrapping == "soft":
            loss_fn = SoftBootstrappingLoss(beta, reduction, as_pseudo_label)
        elif bootstrapping == "hard":
            loss_fn = HardBootstrappingLoss(beta, reduction)
        else:
            raise ValueError(
                f'bootstrapping type {bootstrapping} unrecognized. Pass either "soft" or "hard"'
            )

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
