import torch

from detection.network.vdnet import Net


def test_vdnet():
    batch_size, in_channel, dim_z, dim_y, dim_x = 2, 1, 64, 32, 32
    input_tensor = torch.randn([batch_size, in_channel, dim_z, dim_y, dim_x])

    out_channel = 14
    network = Net(in_channel, out_channel)
    output_tensor = network(input_tensor)

    assert output_tensor.shape[0] == batch_size
    assert output_tensor.shape[1] == out_channel
    assert output_tensor.shape[2] == dim_z
    assert output_tensor.shape[3] == dim_y
    assert output_tensor.shape[4] == dim_x


if __name__ == '__main__':
    test_vdnet()