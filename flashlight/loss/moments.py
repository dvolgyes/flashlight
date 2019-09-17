#!/usr/bin/env python3
import torch
from dsntnn import normalized_linspace


def moments(heatmaps, center, p=2):
    """Differentiable higher order momentum of heatmaps.
       Spread out heatmaps will have high moments, while
       concentrated heatmaps will have low.
       It is effectively a regularization against spreading out.
       The moment is calculated as mr^p where r is the eucledian
       distance from the center of gravity.

    Args:
        heatmaps (torch.Tensor): Spatial representation of locations
        p (float): The order of moment

    Returns:
        Numerical coordinates corresponding to the locations in the heatmaps.
    """
    device = heatmaps.device
    N, M = heatmaps.shape[-2], heatmaps.shape[-1]
    heatmaps = heatmaps / heatmaps.sum(dim=(-1, -2), keepdim=True)

    result = []
    for b in range(0, heatmaps.shape[0]):
        offY, offX = center[b, 0, 0], center[b, 0, 1]
        xx = (
            (normalized_linspace(N).to(device) - offX / N * 2)
            .pow(2)
            .expand(M, N)
            .permute(1, 0)
        )
        yy = (normalized_linspace(M).to(device) - offY / M * 2).pow(2).expand(N, M)
        distance = (xx + yy).pow(0.5 * p)
        loss = (heatmaps[b, ...] * distance).sum(dim=(-1, -2))
        result.append(loss)
    return torch.cat(result)
