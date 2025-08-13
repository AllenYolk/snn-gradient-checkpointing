import torch.nn as nn


class MergeTN(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x_seq):
        return x_seq.flatten(0, 1)


class SplitTN(nn.Module):

    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq):
        return x_seq.reshape(self.T, x_seq.shape[0] // self.T, *x_seq.shape[1:])


class RepeatT(nn.Module):

    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x):
        return x.repeat(self.T, *[1 for _ in range(x.ndim)])

    def extra_repr(self):
        return f"T={self.T}"


class AverageT(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=0)
