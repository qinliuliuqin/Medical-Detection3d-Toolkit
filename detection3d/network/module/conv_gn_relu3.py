import torch.nn as nn


class ConvGnRelu3(nn.Module):
    """ classic combination: conv + batch normalization [+ relu]
        post-activation mode """

    def __init__(self, in_channels, out_channels, ksize, stride, padding, do_act=True, bias=True):
        super(ConvGnRelu3, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, ksize, stride=stride, padding=padding, groups=1, bias=bias)
        self.gn = nn.GroupNorm(1, out_channels)
        self.do_act = do_act
        if do_act:
            self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.gn(self.conv(input))
        if self.do_act:
            out = self.act(out)
        return out


class BottConvGnRelu3(nn.Module):
    """Bottle neck structure"""

    def __init__(self, in_channels, out_channels, ksize, stride, padding, ratio, do_act=True, bias=True):
        super(BottConvGnRelu3, self).__init__()
        self.conv1 = ConvGnRelu3(in_channels, in_channels//ratio, ksize, stride, padding, do_act=True, bias=bias)
        self.conv2 = ConvGnRelu3(in_channels//ratio, in_channels//ratio, ksize, stride, padding, do_act=True, bias=bias)
        self.conv3 = ConvGnRelu3(in_channels//ratio, out_channels, ksize, stride, padding, do_act=do_act, bias=bias)

    def forward(self, input):
        out = self.conv3(self.conv2(self.conv1(input)))
        return out