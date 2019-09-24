#!/usr/bin/env python3
import torch
from dsntnn import kl_reg_losses, js_reg_losses  # noqa: F401
from ..util import dcn
from .moments import moments  # noqa: F401
import dsntnn  # noqa: F401
from importlib import import_module
from box import SBox
import numpy as np
from loguru import logger
from functools import partial


def import_function(name):
    relatives = 0
    base = name.strip('.')
    relatives = len(name) - len(base)
    module, *parts = base.split('.')
    module = ('.' * relatives) + module
    result = import_module(module)
    for part in parts:
        if hasattr(result, part):
            result = getattr(result, part)
        else:
            module = f'{module}.{part}'
            result = import_module(module)
    return result


def parametrized_loss(params):
    if params in globals():
        func = globals()[params]
    else:
        func = import_function(params)
    return func, func.__class__.__name__ != 'function'


def l2(prediction, target):
    return torch.norm(target - prediction, dim=-1)


class LossEvaluator:

    def __init__(self, cfg):
        self.cfg = cfg
        self.loss_fn = {}
        for name, loss in self.cfg.items():
            fn, cls = parametrized_loss(loss.function)
            if cls:
                fn = fn(**loss.kwargs)
            else:
                if len(loss.kwargs) > 0:
                    fn = partial(fn, **loss.kwargs)
            self.loss_fn[name] = fn

    def __call__(self, prediction, data):
        lossDict = SBox()
        for name, loss in self.cfg.items():

            namespace = SBox({'prediction': prediction, **data}, default_box=True)
            args = [namespace[x] for x in loss.signature]
            fn = self.loss_fn[name]

            w = loss.get('weight', 1.0)
            L = torch.squeeze(w * fn(*args))
            N = len(L.shape)
            if N > 1:
                dims = tuple(range(1, N))
                lossDict[name] = L.sum(dim=dims)
            else:
                lossDict[name] = L
        total = 0
        for key in lossDict:
            total = lossDict[key] + total
        lossDict['total'] = total

        return total.mean(), lossDict


def loss_function_cls(prediction, label, cfg):

    namespace = {
        'prediction': torch.squeeze(prediction),
        'label': torch.squeeze(label),
    }

    lossDict = SBox()
    total = 0
    np_losses = SBox()
    for name, loss in cfg.loss.items():
        params = {**namespace, **loss.kwargs}
        args = [params[x] for x in loss.signature]

        fn, cls = parametrized_loss(loss.function)
        if cls:
            fn = fn(**loss.kwargs)
        w = loss.get('weight', 1.0)
        L = torch.squeeze(w * fn(*args))
        N = len(L.shape)
        if N > 1:
            dims = tuple(range(1, N))
            lossDict[name] = L.sum(dim=dims)
        else:
            lossDict[name] = L
        v = dcn(lossDict[name]).sum()
        total += v
        np_losses[name] = v

    total = 0
    for key in lossDict:
        total = lossDict[key] + total
        lossDict[key] = dcn(lossDict[key])
    lossDict['total'] = dcn(total)

    return total.mean(), lossDict


def loss_function_location(prediction, heatmap, target, cfg):
    target2d = target[:, :, 0:2]
    prediction2d = prediction[:, :, 0:2]

    namespace = {
        'prediction': prediction,
        'heatmap': heatmap,
        'target': target,
        'prediction2d': prediction2d,
        'target2d': target2d,
    }

    lossDict = SBox()
    relative_calc = False
    total = 0
    np_losses = SBox()
    relative_calc = False
    for name, loss in cfg.loss.items():
        params = {**namespace, **loss.kwargs}
        if 'relative_weight' in loss:
            relative_calc = True
        args = [params[x] for x in loss.signature]

        fn, cls = parametrized_loss(loss.function)
        if cls:
            fn = fn(**loss.kwargs)
        w = loss.get('weight', 1.0)
        L = torch.squeeze(w * fn(*args))
        N = len(L.shape)
        if N > 1:
            dims = tuple(range(1, N))
            lossDict[name] = L.sum(dim=dims)
        else:
            lossDict[name] = L
        v = dcn(lossDict[name]).sum()
        total += v
        np_losses[name] = v

    if relative_calc:
        for name, loss in cfg.loss.items():
            if 'relative_weight' in loss:
                low, high = loss.relative_weight
                ratio_old = np_losses[name] / (total + 1e-30)
                logger.warning(
                    f'Relative weighting enabled {low}<{ratio_old}<{high} = {low<ratio_old<high}'
                )
                if not (low < ratio_old < high):
                    logger.warning(
                        f'Loss ({name}) relative weight is adjusted.')
                    ratio_new = np.clip(ratio_old, low, high)
                    lossDict[name] = lossDict[name] * (ratio_new / ratio_old)
    total = 0
    for key in lossDict:
        total = lossDict[key] + total
        lossDict[key] = dcn(lossDict[key])
    lossDict['total'] = dcn(total)

    return total.mean(), lossDict


# ~ def partial_cross_entropy(prediction, target, mask):
