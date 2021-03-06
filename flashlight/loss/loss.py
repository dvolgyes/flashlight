#!/usr/bin/env python3
import torch
import torch.nn.functional as F
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

# https://github.com/meng-tang/rloss/blob/master/pytorch/pytorch-deeplab_v3_plus/utils/loss.py


def partial_loss(input_, target, mask, loss_fn, *args, pixel_weights=None, reduction='mean', **kwargs):
    n, c, *dims = input_.size()
    input_ = input_.view(n, c, -1)[mask.view(n, 1, -1)]
    target = target.view(n, -1)[mask.view(n, -1)]
    if pixel_weights is None:
        return loss_fn(input_, target, *args, reduction='none', **kwargs)
    loss = loss_fn(input_, target, *args, reduction='none', **kwargs) * pixel_weights[mask.view(n, -1)]

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return torch.mean(loss)
    else:
        return torch.sum(loss)


def focal_loss(input_, target, class_weights='auto', gamma=1, ignore_index=-100, reduction='mean'):
    n, c, *dims = input_.size()
    if isinstance(class_weights, str) and class_weights == 'auto':
        class_weights = torch.zeros(c).double()
        for i in range(c):
            if i != ignore_index:
                class_weights[i] = torch.sum(target == i)

        class_weights = (torch.sign(class_weights) * n / (class_weights + 1e-3)).float().to(input_.device)
        #~ logger.error(class_weights)

    logpt = - torch.nn.functional.cross_entropy(input_, target, class_weights, ignore_index, reduction=reduction)
    pt = torch.exp(logpt)
    loss = -((1 - pt) ** gamma) * logpt

    return loss


def laplace_regularization(input_, softmax=True):
    if softmax:
        input_ = F.softmax(input_, dim=1)

    if len(input_.shape) == 4:
        batch, C, H, W = input_.shape
        weight = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0.]], requires_grad=False).float()
        weight = weight.view(1, 1, 3, 3).repeat(C, 1, 1, 1).to(input_.device)
        return torch.abs(torch.nn.functional.conv2d(input_, weight, groups=C).mean())

    if len(input_.shape) == 5:
        batch, C, D, H, W = input_.shape
        weight = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0.]], requires_grad=False).float()
        weight = weight.view(1, 1, 1, 3, 3).repeat(C, 1, D, 1, 1).to(input_.device)
        return torch.abs(torch.nn.functional.conv3d(input_, weight, groups=C).mean())


def TV_regularization(input_, softmax=True):
    if softmax:
        input_ = F.softmax(input_, dim=1)
    if len(input_.shape) == 4:
        batch, C, H, W = input_.shape
        weight = torch.tensor([[-0.5, 0, 0.5]], requires_grad=False).float()
        weightT = weight.transpose(0, 1)

        weight = weight.view(1, 1, 1, 3).repeat(C, 1, 1, 1).to(input_.device)
        weightT = weightT.view(1, 1, 3, 1).repeat(C, 1, 1, 1).to(input_.device)

        TV1 = torch.abs(torch.nn.functional.conv2d(input_, weight, groups=C).mean())
        TV2 = torch.abs(torch.nn.functional.conv2d(input_, weightT, groups=C).mean())
        return (TV1 + TV2) / 2.

    if len(input_.shape) == 5:
        batch, C, D, H, W = input_.shape
        weight = torch.tensor([[-0.5, 0, 0.5]], requires_grad=False).float()

        weight = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0.]], requires_grad=False).float()
        weightT = weight.transpose(0, 1)

        weight = weight.view(1, 1, 1, 1, 3).repeat(C, 1, D, 1, 1).to(input_.device)
        weightT = weightT.view(1, 1, 1, 3, 1).repeat(C, 1, 1, 1).to(input_.device)

        TV1 = torch.abs(torch.nn.functional.conv2d(input_, weight, groups=C).mean())
        TV2 = torch.abs(torch.nn.functional.conv2d(input_, weightT, groups=C).mean())
        return (TV1 + TV2) / 2.


class DiceLoss(torch.nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - input_: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = kornia.losses.DiceLoss()
        >>> input_ = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input_, target)
        >>> output.backward()
    """

    def __init__(self, softmax=True, one_hot=True, reduction=None):
        super().__init__()
        self.softmax = softmax
        self.one_hot = one_hot
        self.reduction = reduction

    def forward(self, input_, target):

        if not torch.is_tensor(input_):
            raise TypeError(f'input_ type is not a torch.Tensor. Got {type(input_)}')
        if not len(input_.shape) == 4:
            raise ValueError(f'Invalid input_ shape, we expect BxNxHxW. Got: {input_.shape}')
        if not input_.shape[-2:] == target.shape[-2:]:
            raise ValueError(f'input_ and target shapes must be the same. Got: {input_.shape} , {target.shape}')
        if not input_.device == target.device:
            raise ValueError(f'input_ and target must be in the same device. Got: {input_.device}, {target.device}')

        # compute softmax over the classes axis
        if self.softmax:
            input_ = F.softmax(input_, dim=1)

        # create the labels one hot tensor
        if self.one_hot:
            target = F.one_hot(target, num_classes=input_.shape[1])

        # converting to same type, if necessary
        if target.dtype != input_.dtype:
            target = torch.as_tensor(target, dtype=input_.dtype)

        # compute the actual dice score
        dims = tuple(range(1, len(input_.shape)))
        intersection = torch.sum(input_ * target, dims)
        union = torch.sum(input_ + target, dims) # both input_ and target must be >=0 and =<1
        dice_score = 2. * intersection / (union + self.eps)
        loss = torch.tensor(1.) - dice_score
        if self.reduction is not None:
            if self.reduction == 'mean':
                return torch.mean(loss)
            return self.reduction(loss)
        return loss


class Focalize(torch.nn.Module):
    """ A module to decrease the learning rate of already well classified instances.
    """

    def __init__(self, *args, fn=None):
        super().__init__()
        if fn is None:
            def fn(x, k): return (1. - torch.pow(1 - x, k))
            if len(args) == 0:
                args = (3,)
        self.fn = fn
        self.args = args

    def forward(self, input_):
        if not torch.is_tensor(input_):
            raise TypeError(f'input_ type is not a torch.Tensor. Got {type(input_)}')
        return self.fn(input_, *self.args)


def gamma_loss(x, gamma):
    return torch.pow(x, gamma)


class GammaLoss(torch.nn.Module):
    """ A module to decrease the learning rate of already well classified instances.
    """

    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input_):
        return gamma_loss(input_, self.gamma)


def focal_dice_loss(input_, target, alpha=1.0, gamma=1.0):
    x = Focalize(alpha)(input_)
    score = DiceLoss()(x, target)
    out = GammaLoss(gamma)(score)
    return out


def partial_focal_loss(input_, target, mask, *args, **kwargs):
    return partial_loss(input_, target, mask, focal_loss, *args, **kwargs)


def partial_cross_entropy(input_, target, mask, *args, **kwargs):
    return partial_loss(input_, target, mask, torch.nn.functional.cross_entropy, *args, **kwargs)


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
                logger.warning(f'Relative weighting enabled {low}<{ratio_old}<{high} = {low<ratio_old<high}')
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


# # to be implemented: def partial_cross_entropy(prediction, target, mask):
