#!/usr/bin/env python3

# Flashlight.optim

import torch      # noqa: F401
from .radam import RAdam
from .ranger import Ranger
from importlib import import_module


def get_optimizer(cfg):
    name = cfg.name
    optimizer_classes = {
        'radam': RAdam,
        'ranger': Ranger,
    }
    if name.lower() in optimizer_classes:
        opt = optimizer_classes[name.lower()]
    else:
        module = cfg.get('module', 'torch.optim')
        mod = import_module(module)
        optimizer_class = getattr(mod, name)
        opt = optimizer_class
    return opt


def get_scheduler(cfg):
    if 'name' in cfg:
        name = cfg.name
        module = cfg.get('module', 'torch.optim.lr_scheduler')
        mod = import_module(module)
        scheduler_class = getattr(mod, name)
        return scheduler_class


__all__ = ['get_optimizer', 'get_scheduler', 'RAdam', 'Ranger']
