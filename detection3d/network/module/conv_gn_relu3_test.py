import unittest
import torch
from segmentation3d.network.module.conv_gn_relu3 import ConvGnRelu3


class TestBnReluConv3Methods(unittest.TestCase):
  def setUp(self):
    self.in_channels = 1
    self.out_channels = 16
    self.model1 = ConvGnRelu3(self.in_channels, self.out_channels, ksize=3,
                              stride=1, padding=1, do_act=True)

    self.model2 = ConvGnRelu3(self.in_channels, self.out_channels, ksize=2,
                              stride=2, padding=0, do_act=True)

    print('Total params: %d' % sum(p.numel() for p in self.model1.parameters()))
    print('Total params: %d' % sum(p.numel() for p in self.model2.parameters()))

  def test_output_size(self):
    batch_size = 4
    dim_x, dim_y, dim_z = 16, 32, 48

    inputs = torch.rand([batch_size, self.in_channels, dim_z, dim_y, dim_x])
    inputs = torch.autograd.Variable(inputs)
    outputs1 = self.model1(inputs)
    outputs2 = self.model2(inputs)

    self.assertEqual(outputs1.size()[0], batch_size)
    self.assertEqual(outputs1.size()[1], self.out_channels)
    self.assertEqual(outputs1.size()[2], dim_z)
    self.assertEqual(outputs1.size()[3], dim_y)
    self.assertEqual(outputs1.size()[4], dim_x)

    self.assertEqual(outputs2.size()[0], batch_size)
    self.assertEqual(outputs2.size()[1], self.out_channels)
    self.assertEqual(outputs2.size()[2], dim_z // 2)
    self.assertEqual(outputs2.size()[3], dim_y // 2)
    self.assertEqual(outputs2.size()[4], dim_x // 2)


if __name__ == '__main__':
  unittest.main()