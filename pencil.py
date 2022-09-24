from typing import Iterable, List

import torch
import torch.nn as nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.nn import KLDivLoss, CrossEntropyLoss

from callbacks import Callback
from trainer import Trainer

import torch.nn.functional as F


def entropy(logits):
    return (
        (-torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1))
        .sum(dim=1)
        .mean()
    )


class SoftCrossEntropyLoss(nn.Module):
    def forward(self, x, y):
        if len(y.shape) == 1:
            assert y.size(0) == x.size(0)
            num_classes = x.size(-1)
            y = F.one_hot(y, num_classes=num_classes).float()

        loss = -torch.sum(y * torch.log_softmax(x, dim=1), dim=1)
        return loss.mean()


class Pencil(Trainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        callbacks: List[Callback],
        training_labels: Iterable,
        labels_lr: float,
        alpha: float,
        beta: float,
        stages: List[int],
        num_classes: int,
    ) -> None:
        """
        Args:
            training_labels (Iterable): The the training labels, which will be directely learned during the second stage
            labels_lr (float): learning rate for the labels' optimizer.
            alpha (float): weight of the compatibility loss
            beta (float): weight of the entropy loss
            stages (List[int]): a list of three integers indicating when each stage.
            num_classes (int): the number possible classes for the labels.
        """
        if len(stages) != 3:
            raise ValueError("Pencil only has 3 stages.")
        if epochs != stages[-1]:
            raise ValueError("Number of epochs should equal the final stage")

        loss_fn = SoftCrossEntropyLoss()
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
        self.learnable_label_dist = self.init_label_dist(training_labels, num_classes)
        self.labels_optim = SGD([self.learnable_label_dist], lr=labels_lr)
        self.stages = stages
        self.cur_stage = 1
        self.alpha = alpha
        self.beta = beta

        self.comp_loss_fn = SoftCrossEntropyLoss()

    def init_label_dist(self, hard_labels, num_classes):
        num_labels = len(hard_labels)
        label_dist = torch.zeros((num_labels, num_classes))
        for i in range(len(hard_labels)):
            label_dist[i][hard_labels[i]] = 1
        return label_dist

    def calc_loss(self, pred, label_dist, orignal_noisy_label):
        entropy_loss = 0
        compatibility_loss = 0
        if self.cur_stage in [2, 3]:
            if self.cur_stage == 2:
                compatibility_loss = self.comp_loss_fn(label_dist, orignal_noisy_label)
                entropy_loss = entropy(pred)

            assert isinstance(self.loss_fn, KLDivLoss)
            pred = torch.log_softmax(pred, dim=1)
            label_dist = torch.softmax(label_dist, dim=1)

        pred_loss = self.loss_fn(pred, label_dist)
        loss = pred_loss + self.alpha * compatibility_loss + self.beta * entropy_loss
        return loss

    def val_step(self, batch):
        label = batch["noisy_label"]
        pred = self.model(batch["image"])
        loss = F.cross_entropy(pred, label)

        metrics = dict()
        metrics["prediction"] = pred
        metrics["noisy_loss"] = loss
        metrics["noisy_label"] = batch["noisy_label"]
        return metrics

    def train_step(self, batch):
        orignal_noisy_label = batch["noisy_label"]
        label_dist = self.learnable_label_dist[batch["sample_index"]]
        pred = self.model(batch["image"])
        loss = self.calc_loss(pred, label_dist, orignal_noisy_label)

        self.optimizer.zero_grad()
        self.labels_optim.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.labels_optim.step()

        metrics = dict()
        metrics["prediction"] = pred
        metrics["noisy_loss"] = loss
        metrics["noisy_label"] = label_dist.argmax(dim=1)
        return metrics

    def step(self, batch):
        metrics = dict()
        if self.training:
            m = self.train_step(batch)
            metrics.update(m)
        else:
            m = self.val_step(batch)
            metrics.update(m)
        with torch.no_grad():
            clean_loss = F.cross_entropy(metrics["prediction"], batch["clean_label"])
            metrics["clean_loss"] = clean_loss
            metrics["sample_index"] = batch["sample_index"]
            metrics["noisy_label"] = batch["noisy_label"]
            metrics["clean_label"] = batch["clean_label"]
            [callback.on_step_end(metrics) for callback in self.callbacks]

    def one_epoch(self, loader: DataLoader, epoch: int):
        if epoch <= self.stages[0]:
            self.cur_stage = 1
            self.loss_fn = SoftCrossEntropyLoss()
            self.learnable_label_dist.requires_grad = False
        elif epoch <= self.stages[1]:
            self.cur_stage = 2
            self.loss_fn = KLDivLoss()
            self.learnable_label_dist.requires_grad = True
        elif epoch <= self.stages[2]:
            self.cur_stage = 3
            self.loss_fn = KLDivLoss()
            self.learnable_label_dist.requires_grad = True

        return super().one_epoch(loader, epoch)
