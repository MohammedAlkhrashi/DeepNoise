from argparse import Namespace
from doctest import testsource
from gc import callbacks
from typing import List

import torch
import torch.nn as nn
from matplotlib import transforms
from torch.utils.data import DataLoader
import wandb

import DeepNoise.builders as builders
from DeepNoise.algorithms.base_trainer import Trainer
from DeepNoise.builders.builders import build_cfg
from DeepNoise.callbacks.statistics import Callback


def main():

    args = Namespace()
    args.path = "configs/algorithms/default_erm.py"
    cfg = build_cfg(args.path)

    wandb.init(project="DeepNoise", entity="elytsn", config=cfg)

    trainset = builders.build_dataset(cfg["data"]["trainset"])
    valset = builders.build_dataset(cfg["data"]["valset"])
    testset = builders.build_dataset(cfg["data"]["testset"])

    train_loader = DataLoader(
        trainset,
        shuffle=True,
        pin_memory=True,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    val_loader = DataLoader(
        valset,
        shuffle=False,
        pin_memory=True,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    test_loader = DataLoader(
        testset,
        shuffle=False,
        pin_memory=True,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    model: nn.Module = builders.build_model(cfg["model"], cfg["num_classes"])
    optimizer: torch.optim.Optimizer = builders.build_optimizer(
        cfg["optimizer"], model=model
    )
    loss_fn: nn.Module = builders.build_loss(cfg["loss_fn"])
    callbacks: List[Callback] = [
        builders.build_callbacks(callback_cfg) for callback_cfg in cfg["callbacks"]
    ]
    callbacks.extend(
        [
            builders.build_callbacks(callback_cfg, optimizer=optimizer)
            for callback_cfg in cfg["optimizer_callbacks"]
        ]
    )

    trainer: Trainer = builders.build_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=cfg["epochs"],
        callbacks=callbacks,
        cfg=cfg["trainer"],
    )
    trainer.start()


if __name__ == "__main__":
    main()
