import torch
import torch.nn as nn
from segmentation3d.network.module.residual_block3 import ResidualBlock3, BottResidualBlock3


class UpBlock(nn.Module):
  """ Upsample block of v-net """

  def __init__(self, in_channels, out_channels, num_convs, compression=False, ratio=4):
    super(UpBlock, self).__init__()
    self.up_conv = nn.ConvTranspose3d(in_channels, out_channels // 2, kernel_size=2, stride=2, groups=1)
    self.up_gn = nn.GroupNorm(1, out_channels // 2)
    self.up_act = nn.ReLU(inplace=True)
    if compression:
      self.rblock = BottResidualBlock3(out_channels, 3, 1, 1, ratio, num_convs)
    else:
      self.rblock = ResidualBlock3(out_channels, 3, 1, 1, num_convs)

  def forward(self, input, skip):
    out = self.up_act(self.up_gn(self.up_conv(input)))
    out = torch.cat((out, skip), 1)
    out = self.rblock(out)
    return out
