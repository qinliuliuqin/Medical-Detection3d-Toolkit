import torch.nn as nn


class InputBlock(nn.Module):
  """ input block of vb-net """

  def __init__(self, in_channels, out_channels):
    super(InputBlock, self).__init__()
    self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
    self.gn = nn.GroupNorm(1, num_channels=out_channels)
    self.act = nn.ReLU(inplace=True)

  def forward(self, input):
    out = self.act(self.gn(self.conv(input)))
    return out
