#!/usr/bin/env python3
import torch
import torchvision  # noqa: F401
from importlib import import_module
import sys
from loguru import logger

# local models
from .unet import UNet
from .unetcoord import UNetCoordreg
from highresnet import HighRes3DNet


def get_model(cfg):
    if 'src' in cfg:
        source = cfg['src']
    else:
        source = cfg.get('source', 'local')
    # sources: local, torchvision, modelzoo, torchhub
    if source == 'local':
        name = cfg.name.lower()
        modelclass = {'unet': UNet,
                      'unetcoordreg': UNetCoordreg,
                      'highres3Dnet': HighRes3DNet,
                      }[name]
        model = modelclass(**cfg.kwargs)
    elif source == 'torchhub':
        model = torch.hub.load(cfg['url'], name, **cfg['kwargs'])
    elif source == 'module':
        mod = import_module(cfg['module'])
        model = getattr(mod, name)(**cfg['kwargs'])
    else:
        logger.error('Invalid model source.')
        sys.exit(1)

    device = cfg.get('device', 'cpu')
    model.to(device)
    if device.startswith('cuda'):
        return torch.nn.DataParallel(model)
    return model
