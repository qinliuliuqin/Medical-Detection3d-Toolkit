import torch.nn as nn


class OutputBlock(nn.Module):
  """ output block of v-net

      The output is a list of foreground-background probability vectors.
      The length of the list equals to the number of voxels in the volume
  """

  def __init__(self, in_channels, out_channels):
    super(OutputBlock, self).__init__()
    self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
    self.gn1 = nn.GroupNorm(1, out_channels)
    self.act1 = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=1)
    self.gn2 = nn.GroupNorm(1, out_channels)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, input):
    out = self.act1(self.gn1(self.conv1(input)))
    out = self.gn2(self.conv2(out))
    out = self.softmax(out)
    return out