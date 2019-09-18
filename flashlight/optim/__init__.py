#!/usr/bin/env python3

# Flashlight.optim

import torch
from .radam import RAdam
from .ranger import Ranger


def get_optimizer(params, **kwargs):
    name = kwargs['name'].lower()
    kwargs = kwargs.copy()
    kwargs.pop('name')
    optimizer_class = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'radam': RAdam,
        'ranger': Ranger,
    }[name]

    return optimizer_class(params, **kwargs)


__all__ = ['get_optimizer', 'RAdam', 'Ranger']
