import torch.nn as nn
from segmentation3d.network.module.residual_block3 import ResidualBlock3, BottResidualBlock3


class DownBlock(nn.Module):
  """ downsample block of v-net """

  def __init__(self, in_channels, num_convs, compression=False, ratio=4):
    super(DownBlock, self).__init__()
    out_channels = in_channels * 2
    self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, groups=1)
    self.down_gn = nn.GroupNorm(1, num_channels=out_channels)
    self.down_act = nn.ReLU(inplace=True)
    if compression:
      self.rblock = BottResidualBlock3(out_channels, 3, 1, 1, ratio, num_convs)
    else:
      self.rblock = ResidualBlock3(out_channels, 3, 1, 1, num_convs)

  def forward(self, input):
    out = self.down_act(self.down_gn(self.down_conv(input)))
    out = self.rblock(out)
    return out