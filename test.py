import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate
import einops


class VanillaLIF(nn.Module):

    def __init__(self, decay_lambda=0.5):
        super().__init__()
        self.decay_lambda = decay_lambda
        self.surrogate_function = surrogate.Sigmoid()

    def forward(self, x_seq):
        T = x_seq.shape[0]
        v = torch.zeros_like(x_seq[0])
        s_seq = []
        for t in range(T):
            x = x_seq[t]
            # single-step forward; a.k.a. "core"
            v = self.decay_lambda * v + x
            s = self.surrogate_function(v)
            v = v * (1.-s)

            s_seq.append(s)
        s_seq = torch.stack(s_seq, dim=0)
        return s_seq
