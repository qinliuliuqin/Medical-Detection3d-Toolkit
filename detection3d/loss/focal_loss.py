import torch
from torch import nn


class FocalLoss(nn.Module):

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, use_gpu=True):

        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1) / class_num
        else:
            assert len(alpha) == class_num
            self.alpha = torch.FloatTensor(alpha)
            self.alpha = self.alpha.unsqueeze(1)
            self.alpha = self.alpha / self.alpha.sum()

        if use_gpu:
            self.alpha = self.alpha.cuda()
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.one_hot_codes = torch.eye(self.class_num)
        if use_gpu:
            self.one_hot_codes = self.one_hot_codes.cuda()

    def forward(self, input, target):
        # Assume that the input should has one of the following shapes:
        # 1. [sample, class_num]
        # 2. [batch, class_num, dim_y, dim_x]
        # 3. [batch, class_num, dim_z, dim_y, dim_x]
        assert input.dim() == 2 or input.dim() == 4 or input.dim() == 5
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input = input.view(input.numel() // self.class_num, self.class_num)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input = input.view(input.numel() // self.class_num, self.class_num)

        # Assume that the target should has one of the following shapes which
        # correspond to the shapes of the input:
        # 1. [sample, 1] or [sample, ]
        # 2. [batch, 1, dim_y, dim_x] or [batch, dim_y, dim_x]
        # 3. [batch, 1, dim_z, dim_y, dim_x], or [batch, dim_z, dim_y, dim_x]
        target = target.long().view(-1)
        mask = self.one_hot_codes[target.data]
        alpha = self.alpha[target.data]
        probs = (input * mask).sum(1).view(-1, 1) + 1e-10
        log_probs = probs.log()

        if self.gamma > 0:
            batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_probs
        else:
            batch_loss = -alpha * log_probs

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss