import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import List
from dataclasses import dataclass
from DeepNoise.algorithms.base_trainer import Trainer
from DeepNoise.callbacks import Callback
from DeepNoise.builders import TRAINERS, MODELS, OPTIMIZERS, build_optimizer


class CoTeachingLoss(nn.Module):
    def forward(self, y_1, y_2, y, forget_rate: float):
        loss_1 = F.cross_entropy(y_1, y, reduce=False)
        ind_1_sorted = np.argsort(loss_1.data).cuda()

        loss_2 = F.cross_entropy(y_2, y, reduce=False)
        ind_2_sorted = np.argsort(loss_2.data).cuda()

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sorted))
        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]

        # exchange
        loss_1_update = F.cross_entropy(y_1[ind_2_update], y[ind_2_update])
        loss_2_update = F.cross_entropy(y_2[ind_1_update], y[ind_1_update])

        return (
            torch.sum(loss_1_update) / num_remember,
            torch.sum(loss_2_update) / num_remember,
        )


@MODELS.register()
@dataclass
class CoTeachingModel(nn.Module):
    model_1: nn.Module
    model_2: nn.Module

    def forward(self, x):
        return self.model_1(x), self.model_2(x)

    def train(self):
        self.model_1.train()
        self.model_2.train()

    def eval(self):
        self.model_1.eval()
        self.model_2.eval()


@OPTIMIZERS.register()
class CoTeachingOptimizers(Optimizer):
    def __init__(self, optim_1_cfg, optim_2_cfg, params):
        params_1, params_2 = itertools.tee(params)
        self.optimizer_1 = build_optimizer(optim_1_cfg, params_1)
        self.optimizer_2 = build_optimizer(optim_2_cfg, params_2)


@TRAINERS.register("Co-Teaching")
class CoTeachingTrainer(Trainer):
    def __init__(
        self,
        model: CoTeachingModel,
        optimizer: CoTeachingOptimizers,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        loss_fn: nn.Module = None,
        callbacks: List[Callback] = None,
        forget_rate: float = 0.2,
        num_gradual: int = 10,
        exponent: float = 1,
    ) -> None:
        """
        Args:
            forget_rate (float, default = 0.2)
            num_gradual (int, default = 10): How many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in the Co-teaching paper.
            exponent (float, default = 1.0) exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in the Co-teaching paper.
        """
        loss_fn = CoTeachingLoss()
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

        # Initialise drop rate scheduler
        self.rate_schedule = np.ones(self.epochs) * forget_rate
        self.rate_schedule[:num_gradual] = np.linspace(
            0, forget_rate ** exponent, num_gradual
        )

    def step(self, batch):
        batch = {key: value.to(self.device) for key, value in batch.items()}
        logits_1, logits_2 = self.model(batch["image"])
        loss_1, loss_2 = self.loss_fn(
            logits_1, logits_2, self.cur_epoch, batch["noisy_label"]
        )

        if self.training:
            self.optimizer.optimizer_1.zero_grad()
            loss_1.backward()
            self.optimizer.optimizer_1.step()

            self.optimizer.optimizer_2.zero_grad()
            loss_2.backward()
            self.optimizer.optimizer_2.step()

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
