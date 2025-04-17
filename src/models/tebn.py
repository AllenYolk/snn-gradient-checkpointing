import torch
import torch.nn as nn


class TEBNProjection(nn.Module):

    def __init__(self, T, input_ndim: int = 5):
        super().__init__()
        self.p = nn.Parameter(
            torch.ones(T, *[1 for _ in range(input_ndim - 1)])
        )

    def forward(self, x_seq):
        return x_seq * self.p
