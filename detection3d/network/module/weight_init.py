import torch.nn as nn


def kaiming_weight_init(m, bn_std=0.02):

    classname = m.__class__.__name__
    if 'Conv3d' in classname or 'ConvTranspose3d' in classname:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif 'BatchNorm' in classname:
        m.weight.data.normal_(1.0, bn_std)
        m.bias.data.zero_()
    elif 'Linear' in classname:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


def gaussian_weight_init(m, conv_std=0.01, bn_std=0.01):

    classname = m.__class__.__name__
    if 'Conv3d' in classname or 'ConvTranspose3d' in classname:
        m.weight.data.normal_(0, conv_std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif 'BatchNorm' in classname:
        m.weight.data.normal_(1.0, bn_std)
        m.bias.data.zero_()
