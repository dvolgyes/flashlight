class HDRN(nn.Module):
    """ HybridDilatedResNet
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            mid_channels = 16,
            kernel_size = 3,
            groups = 1,
            bias = True,
            padding_mode = 'zeros',
            # ~ initial_out_channels_power=4,
            # ~ layers_per_residual_block=2,
            # ~ residual_blocks_per_dilation=3,
            dilations=(1,2,4,8,12,16,24,32),
            activation = nn.ReLU,
            last_activation = None,
            dim = 1,
            iterations = 5,
            initial_state = None
            ):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilations = dilations
        stride = 1
        self.dimension = dim
        self.hybrid_conv = nn.ModuleList()
        self.iterations  = iterations
        self.initial_state = initial_state

        for d in dilations:
            pad = nn.ReflectionPad2d(padding=d)
            layer = nn.modules.conv.Conv2d(in_channels +out_channels,
                                      mid_channels,
                                      kernel_size,
                                      stride,
                                      padding=0,
                                      dilation=d,
                                      groups=groups,
                                      bias=bias)

            self.hybrid_conv.append(nn.Sequential(pad,layer))

        self.activation = activation()
        self.conv_1x1 = nn.modules.conv.Conv2d(
                                          mid_channels * len(dilations),
                                          out_channels,
                                          kernel_size,
                                          stride=1,
                                          padding=0,
                                          groups=1,
                                          bias=bias)
        if last_activation:
            self.last_activation = last_activation()
        else:
            self.last_activation = lambda x:x


    def forward(self, x):
        if not self.initial_state:
            shape = x.shape
            shape = (shape[0],) +(self.out_channels,) +shape[2:]
            hidden_state = (torch.ones(shape, dtype = torch.float32) / self.out_channels)
        else:
            hidden_state = initial_state
        if hidden_state.device != x.device:
            hidden_state = hidden_state.to(x.device)

        outcomes = []

        for i in range(self.iterations):
            input = torch.cat((x,hidden_state), dim=self.dimension)
            for conv in self.hybrid_conv:
                outcomes.append(conv(input))
            out = torch.cat(outcomes, dim=self.dimension)
            scores = self.conv_1x1(out)
            segmentation = F.softmax(scores, dim=1)
            segmentation = (segmentation + hidden_state) / 2.0
            hidden = segmentation
        return segmentation
