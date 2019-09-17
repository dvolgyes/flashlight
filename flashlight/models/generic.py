#!/usr/bin/env python3
import torch
import torchvision  # noqa: F401
from importlib import import_module
import sys
from loguru import logger

# local models
from .unet import UNet
from .unetcoord import UNetCoordreg


def get_model(name='unet', **kwargs):
    if 'src' in kwargs:
        source = kwargs['src']
    else:
        source = kwargs.get('source', 'local')

    # sources: local, torchvision, modelzoo, torchhub
    if source == 'local':
        name = name.lower()
        modelclass = {'unet': UNet, 'unetcoordreg': UNetCoordreg}[name]
        model = modelclass(**kwargs)
    elif source == 'torchhub':
        model = torch.hub.load(kwargs['url'], name, **kwargs['kwargs'])
    elif source == 'module':
        mod = import_module(kwargs['module'])
        model = getattr(mod, name)(**kwargs['kwargs'])
    else:
        logger.error('Invalid model source.')
        sys.exit(1)

    device = kwargs.get('device')
    model.to(device)
    if device.startswith('cuda'):
        return torch.nn.DataParallel(model)
    return model
