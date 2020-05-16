import torch
import torch.nn as nn

from segmentation3d.network.module.weight_init import kaiming_weight_init, gaussian_weight_init
from segmentation3d.network.module.vnet_inblock import InputBlock
from segmentation3d.network.module.vnet_outblock import OutputBlock
from segmentation3d.network.module.vnet_upblock import UpBlock
from segmentation3d.network.module.vnet_downblock import DownBlock


def parameters_kaiming_init(net):
    """ model parameters initialization """
    net.apply(kaiming_weight_init)


def parameters_gaussian_init(net):
    """ model parameters initialization """
    net.apply(gaussian_weight_init)


class Net(nn.Module):
    """ volumetric segmentation network """

    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.in_block = InputBlock(in_channels, 16)
        self.down_32 = DownBlock(16, 1, compression=False)
        self.down_64 = DownBlock(32, 2, compression=False)
        self.up_64 = UpBlock(64, 64, 2, compression=False)
        self.up_32 = UpBlock(64, 32, 1, compression=False)
        self.out_block = OutputBlock(32, out_channels)


    def forward(self, input):
        assert isinstance(input, torch.Tensor)

        out16 = self.in_block(input)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out = self.up_64(out64, out32)
        out = self.up_32(out, out16)
        out = self.out_block(out)
        return out

    def max_stride(self):
        return 4
