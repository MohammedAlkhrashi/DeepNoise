from copy import deepcopy

import torch.nn as nn
from cv2 import transform

from DeepNoise.builders.registry import Registry

TRAINERS = Registry()
DATASETS = Registry()
MODELS = Registry()
CALLBACKS = Registry()
LOSSES = Registry()
OPTIMIZERS = Registry()
TRANSFORMS = Registry()

import torchvision.transforms.transforms


def _from_registry(registry, cfg):
    cfg = deepcopy(cfg)
    cls_key = cfg.pop("type")
    return registry.build(cls_key, **cfg)


def _from_module(module, cfg):
    cfg = deepcopy(cfg)
    cls_key = cfg.pop("type")
    cls = getattr(module, cls_key)
    return cls(**cfg)


def build_transforms(cfg):
    if cfg["type"] not in TRANSFORMS:
        return _from_module(torchvision.transforms.transforms, cfg)

    return _from_registry(TRANSFORMS, cfg)


def build_dataset(cfg):
    if cfg["transforms"] is not None:
        cfg["transforms"] = torchvision.transforms.transforms.Compose(
            [build_transforms(transforms_cfg) for transforms_cfg in cfg["transforms"]]
        )
    return _from_registry(DATASETS, cfg)


def build_trainer(cfg, **kwargs):
    combined_cfg: dict = {**cfg, **kwargs}
    return _from_registry(TRAINERS, combined_cfg)


def build_callbacks(cfg):
    return _from_registry(CALLBACKS, cfg)


def build_optimizer(cfg, model: nn.Module):
    cfg["params"] = list(model.parameters())
    if cfg["type"] not in OPTIMIZERS:
        import torch.optim

        return _from_module(torch.optim, cfg)

    return _from_registry(OPTIMIZERS, cfg)


def build_loss(cfg):
    if cfg["type"] not in LOSSES:
        import torch.nn

        return _from_module(torch.nn, cfg)

    return _from_registry(LOSSES, cfg)


def build_model(cfg, num_classes):
    if cfg["type"] not in LOSSES:
        import timm

        model_name = cfg.pop("type")
        model = timm.create_model(model_name, num_classes=num_classes, **cfg)
        return model

    return _from_registry(MODELS, cfg)
