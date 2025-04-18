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


class MergeSplitTNWrapper(nn.Module):
    """Equal to spikingjelly.activation_based.layer.SeqToANNContainer
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x_seq):
        T, N = x_seq.shape[0], x_seq.shape[1]
        x_seq = x_seq.flatten(0, 1)
        x_seq = self.module(x_seq)
        x_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
        return x_seq


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
