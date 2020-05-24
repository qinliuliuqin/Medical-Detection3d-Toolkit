import torch.nn as nn
from segmentation3d.network.module.conv_gn_relu3 import ConvGnRelu3, BottConvGnRelu3


class ResidualBlock3(nn.Module):
    """ residual block with variable number of convolutions """

    def __init__(self, channels, ksize, stride, padding, num_convs):
        super(ResidualBlock3, self).__init__()

        layers = []
        for i in range(num_convs):
            if i != num_convs - 1:
                layers.append(ConvGnRelu3(channels, channels, ksize, stride, padding, do_act=True))
            else:
                layers.append(ConvGnRelu3(channels, channels, ksize, stride, padding, do_act=False))

        self.ops = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):

        output = self.ops(input)
        output = self.act(input + output)

        return output


class BottResidualBlock3(nn.Module):
    """ block with bottle neck conv"""

    def __init__(self, channels, ksize, stride, padding, ratio, num_convs):
        super(BottResidualBlock3, self).__init__()
        layers = []
        for i in range(num_convs):
            if i != num_convs - 1:
                layers.append(BottConvGnRelu3(channels, channels, ksize, stride, padding, ratio, do_act=True))
            else:
                layers.append(BottConvGnRelu3(channels, channels, ksize, stride, padding, ratio, do_act=False))

        self.ops = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.ops(input)
        return self.act(input + output)