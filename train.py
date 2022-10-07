from argparse import Namespace
import argparse
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str)
    parser.add_argument("--noise_type", type=str, default="SymmetricNoise")
    parser.add_argument("--noise_prob", type=float, default=0)
    parser.add_argument("--allow_equal_flips", type=str2bool, default=True)

    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    cfg = build_cfg(args.cfg_path)
    cfg["data"]["trainset"]["noise_injector"] = dict(
        type=args.noise_type,
        noise_prob=args.noise_prob,
        allow_equal_flips=args.allow_equal_flips,
    )  # TODO: Raise warning if cfg file alrady contains a noise injector.

    wandb.init(project="DeepNoise", config=cfg)

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
    model: nn.Module = builders.build_model(
        cfg["model"], num_classes=cfg["num_classes"]
    )
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
