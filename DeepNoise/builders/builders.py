from copy import deepcopy
import runpy

import torch.nn as nn
import torchvision.transforms.transforms
from cv2 import transform

from DeepNoise.builders.registry import Registry

TRAINERS = Registry()
DATASETS = Registry()
MODELS = Registry()
CALLBACKS = Registry()
LOSSES = Registry()
OPTIMIZERS = Registry()
TRANSFORMS = Registry()
NOISE_INJECTORS = Registry()


def _from_registry(registry, cfg):
    if cfg is None or cfg["type"] is None:
        return None
    cls_key = cfg.pop("type")
    return registry.build(cls_key, **cfg)


def _from_module(module, cfg):
    if cfg is None or cfg["type"] is None:
        return None
    cls_key = cfg.pop("type")
    cls = getattr(module, cls_key)
    return cls(**cfg)


def build_cfg(path: str):
    config_file = runpy.run_path(path)
    cfg = config_file["cfg"]
    return cfg


def build_transforms(cfg):
    if cfg["type"] not in TRANSFORMS:
        return _from_module(torchvision.transforms.transforms, cfg)

    return _from_registry(TRANSFORMS, cfg)


def build_noise_injector(cfg):
    return _from_registry(NOISE_INJECTORS, cfg)


def build_dataset(cfg):
    if "transforms" in cfg:
        cfg["transforms"] = torchvision.transforms.transforms.Compose(
            [build_transforms(transforms_cfg) for transforms_cfg in cfg["transforms"]]
        )
    if "noise_injector" in cfg:
        cfg["noise_injector"] = build_noise_injector(cfg["noise_injector"])
    return _from_registry(DATASETS, cfg)


def build_trainer(cfg, **kwargs):
    combined_cfg = {**cfg, **kwargs}
    return _from_registry(TRAINERS, combined_cfg)


def build_callbacks(cfg, **kwargs):
    combined_cfg = {**cfg, **kwargs}
    return _from_registry(CALLBACKS, combined_cfg)


def build_optimizer(cfg, params):
    cfg["params"] = params
    if cfg["type"] not in OPTIMIZERS:
        import torch.optim

        return _from_module(torch.optim, cfg)

    return _from_registry(OPTIMIZERS, cfg)


def build_loss(cfg):
    if cfg["type"] not in LOSSES:
        import torch.nn

        return _from_module(torch.nn, cfg)

    return _from_registry(LOSSES, cfg)


def build_model(cfg, **kwargs):
    combined_cfg = {**cfg, **kwargs}
    if cfg["type"] not in MODELS:
        import timm

        model_name = cfg.pop("type")
        model = timm.create_model(model_name, **combined_cfg)
        return model

    return _from_registry(MODELS, combined_cfg)
