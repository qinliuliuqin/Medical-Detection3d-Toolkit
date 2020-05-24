import torch.nn as nn
import torch


def kaiming_weight_init(m, bn_std=0.02):

    classname = m.__class__.__name__
    if 'Conv3d' in classname or 'ConvTranspose3d' in classname:
        version_tokens = torch.__version__.split('.')
        if int(version_tokens[0]) == 0 and int(version_tokens[1]) < 4:
            nn.init.kaiming_normal(m.weight)
        else:
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