import torch.nn as nn
import einops


class MergeTN(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x_seq):
        return einops.rearrange(x_seq, "T N ... -> (T N) ...")


class SplitTN(nn.Module):

    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq):
        return einops.rearrange(x_seq, "(T N) ... -> T N ...", T=self.T)
