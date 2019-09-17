import torch
from torch import nn
import dsntnn
import numpy as np
from .unet import UNet
from duecredit import due, BibTeX


@due.dcite(
    BibTeX(
        """
        @article{nibali2018numerical,
          title={Numerical Coordinate Regression with Convolutional Neural Networks},
          author={Nibali, Aiden and He, Zhen and Morgan, Stuart and Prendergast, Luke},
          journal={arXiv preprint arXiv:1801.07372},
          year={2018}
        }
        """
    ),
    description='Machine learning - models',
    path='DSNT coordinate regression',
)
class UNetCoordreg(nn.Module):

    def __init__(
        self,
        in_channels=1,
        n_classes=1,
        depth=3,
        wf=3,
        padding=False,
        batch_norm=False,
        up_mode='nearest',
        no_of_filters=lambda x, y: 2 ** (x + y),
        **kwargs,
    ):
        super().__init__()
        self.unet = UNet(
            in_channels=in_channels,
            n_classes=n_classes,
            depth=depth,
            wf=wf,
            padding=padding,
            batch_norm=batch_norm,
            up_mode=up_mode,
            no_of_filters=no_of_filters,
        )
        self.padding = torch.nn.ConstantPad1d((0, 1), 1)

    def forward(self, data):
        heatmaps = dsntnn.flat_softmax(self.unet(data) * 1000)
        x, y = heatmaps.shape[-2], heatmaps.shape[-1]
        coords = (dsntnn.dsnt(heatmaps) + 1) / 2.0
        size = torch.from_numpy(np.asarray([x, y], dtype=np.float32)).to(coords.device)
        coords = coords * size
        coords = self.padding(coords)
        return heatmaps, coords
