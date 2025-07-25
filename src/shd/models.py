import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate


def plif_update(x, v, spike, vth, _beta):
    beta = torch.sigmoid(_beta)
    v = v*beta + (1-beta) * x - vth*spike
    spike = surrogate.atan.apply(v - vth, 2.)
    return v, spike


def output_plif_update(x, v, _beta):
    beta = torch.sigmoid(_beta)
    v = v*beta + (1-beta) * x
    return v


class LinearPLIF(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        beta_initializer='uniform',
        beta_low=0,
        beta_high=4,
        vth=0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.vth = vth

        self.dense = nn.Linear(in_features, out_features)
        self._beta = nn.Parameter(torch.empty([self.out_features]))

        if beta_initializer == 'uniform':
            nn.init.uniform_(self._beta, beta_low, beta_high)
        elif beta_initializer == 'constant':
            nn.init.constant_(self._beta, beta_low)

    def forward(self, x_seq):
        # x_seq.shape = [T, N, in_features]
        T = x_seq.shape[0]

        x_seq = self.dense(x_seq)

        v = torch.rand_like(x_seq[0])  # rand: follow the practice of DH-SNN
        s = torch.rand_like(x_seq[0])
        s_seq = torch.empty_like(x_seq)
        for t in range(T):
            x = x_seq[t]
            v, s = plif_update(x, v, s, self.vth, self._beta)
            s_seq[t] = s
        return s_seq


class LinearOutputPLIF(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        beta_initializer='uniform',
        beta_low=0,
        beta_high=4,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.dense = nn.Linear(in_features, out_features)
        self._beta = nn.Parameter(torch.empty([self.out_features]))

        if beta_initializer == 'uniform':
            nn.init.uniform_(self._beta, beta_low, beta_high)
        elif beta_initializer == 'constant':
            nn.init.constant_(self._beta, beta_low)

    def forward(self, x_seq):
        # x_seq.shape = (T, N, in_features)
        T = x_seq.shape[0]

        x_seq = self.dense(x_seq)

        v = torch.rand_like(x_seq[0])
        v_seq = torch.rand_like(x_seq)
        for t in range(T):
            x = x_seq[t]
            v = output_plif_update(x, v, self._beta)
            v_seq[t] = v
        return v_seq


class PLIFSFNN(nn.Module):

    def __init__(self):
        super().__init__()
        H = 64
        self.dense_1 = LinearPLIF(700, H, vth=1.)
        self.dense_2 = LinearOutputPLIF(H, 20)
        nn.init.xavier_normal_(self.dense_2.dense.weight)
        nn.init.constant_(self.dense_2.dense.bias, 0)

    def forward(self, x_seq):
        x_seq = x_seq.transpose(0, 1)  # [T, N, C]
        x_seq = self.dense_1(x_seq)
        x_seq = self.dense_2(x_seq)  # [T, N, 20]

        logits = F.softmax(x_seq, dim=-1)  # [T, N, 20]
        return torch.sum(logits[10:], dim=0)  # [N, 20]; discard 1st 10 steps
