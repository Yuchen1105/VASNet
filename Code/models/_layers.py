"""Parameter-free stochastic depth with the original timm DropPath convention."""
import torch
from torch import nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1 - self.drop_prob
        mask = (keep + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1),
                                  dtype=x.dtype, device=x.device)).floor()
        return x / keep * mask
