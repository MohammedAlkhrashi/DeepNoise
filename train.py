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
from DeepNoise.callbacks.statistics import Callback


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_config():
    # This should read from a config file and return a dict
    cfg = dict()

    cfg["batch_size"] = 2
    cfg["num_classes"] = 10
    cfg["num_workers"] = 2
    cfg["epochs"] = 5
    # Data Start
    data = dict()

    data["trainset"] = dict()
    data["trainset"]["type"] = "NoisyCIFAR10"
    data["trainset"]["train"] = True
    data["trainset"]["root"] = "data"
    data["trainset"]["download"] = True
    data["trainset"]["transforms"] = [
        dict(type="RandomCrop", size=32, padding=4, padding_mode="reflect"),
        dict(type="RandomHorizontalFlip"),
        dict(type="ToTensor"),
        dict(
            type="Normalize",
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ]
    data["trainset"]["noise_injector"] = dict(type="SymmetricNoise", noise_prob=0.4)

    data["valset"] = dict()
    data["valset"]["type"] = "NoisyCIFAR10"
    data["valset"]["train"] = False
    data["valset"]["root"] = "data"
    data["valset"]["download"] = True
    data["valset"]["transforms"] = [
        dict(type="ToTensor"),
        dict(
            type="Normalize",
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ]

    data["testset"] = dict()
    data["testset"]["type"] = "NoisyCIFAR10"
    data["testset"]["train"] = False
    data["testset"]["root"] = "data"
    data["testset"]["download"] = True
    data["testset"]["transforms"] = [
        dict(type="ToTensor"),
        dict(
            type="Normalize",
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ]

    cfg["data"] = data
    # Data End

    cfg["model"] = dict()
    cfg["model"]["type"] = "resnet34"
    cfg["model"]["pretrained"] = False

    cfg["optimizer"] = dict()
    cfg["optimizer"]["type"] = "SGD"
    cfg["optimizer"]["lr"] = 0.02
    cfg["optimizer"]["weight_decay"] = 0.0005
    cfg["optimizer"]["momentum"] = 0.9

    cfg["loss_fn"] = dict()
    cfg["loss_fn"]["type"] = "CrossEntropyLoss"

    cfg["callbacks"] = [dict(type="SimpleStatistics")]
    cfg["optimizer_callbacks"] = [
        dict(type="StepLR", milestones=[80, 100], gamma=0.1, last_epoch=-1)
    ]

    cfg["trainer"] = dict()
    cfg["trainer"]["type"] = "ERM"
    cfg["trainer"]

    cfg = dotdict(cfg)

    return cfg


def main():
    cfg = get_config()
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
    model: nn.Module = builders.build_model(cfg.model, cfg["num_classes"])
    optimizer: torch.optim.Optimizer = builders.build_optimizer(
        cfg.optimizer, model=model
    )
    loss_fn: nn.Module = builders.build_loss(cfg.loss_fn)
    callbacks: List[Callback] = [
        builders.build_callbacks(callback_cfg) for callback_cfg in cfg.callbacks
    ]
    callbacks.extend(
        [
            builders.build_callbacks(callback_cfg, optimizer=optimizer)
            for callback_cfg in cfg.optimizer_callbacks
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
        cfg=cfg.trainer,
    )
    trainer.start()


if __name__ == "__main__":
    main()
