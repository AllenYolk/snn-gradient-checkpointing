import torch
import torch.nn as nn
from .checkpointing import in_gc_1st_forward


class BatchNorm1d_(nn.BatchNorm1d):

    def forward(self, x):
        original_momentum = self.momentum
        self.momentum = 0.0 if in_gc_1st_forward() else original_momentum
        out = super().forward(x)
        self.momentum = original_momentum
        return out


class BatchNorm2d_(nn.BatchNorm2d):

    def forward(self, x):
        original_momentum = self.momentum
        self.momentum = 0.0 if in_gc_1st_forward() else original_momentum
        out = super().forward(x)
        self.momentum = original_momentum
        return out


class TEBNProjection(nn.Module):

    def __init__(self, T, input_ndim: int = 5):
        super().__init__()
        self.p = nn.Parameter(
            torch.ones(T, *[1 for _ in range(input_ndim - 1)])
        )

    def forward(self, x_seq):
        return x_seq * self.p
