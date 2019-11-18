#!/usr/bin/env python3
import highresnet
import warm
import warm.functional as W

class HDRN(nn.Module):
    """ HybridDilatedResNet
    """
    def __init__(
            self,
            out_channels,
            initial_out_channels_power=4,
            layers_per_residual_block=2,
            residual_blocks_per_dilation=3,
            dilations=3,
            ):

        super().__init__()
        self.out_channels = out_channels
        self.layers_per_residual_block = layers_per_residual_block
        self.residual_blocks_per_dilation = residual_blocks_per_dilation

        self.dilations = dilations
        warm.up(selfm)

    def forward(self, x):
        dilated_convolutions = []
        for dilation in range(1,self.dilations+1):
            dilated_convolutions.append(W.conv(x, out_channels, dilation = dilation))
        x = self.activation(W.torch.cat(dilated_convolutions, 1))

        return self.block(x)
