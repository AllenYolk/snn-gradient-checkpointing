import sys

sys.path.append("./src")

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate

from modules.blocks.checkpointing import BaseCheckpointingBlock
from modules.blocks.checkpointing import SNNCheckpointingBlockFunction
from modules.compress import get_spike_compressor


def plif_update(x, v, spike, vth, _beta):
    beta = torch.sigmoid(_beta)
    v = v*beta + (1-beta) * x - vth*spike
    spike = surrogate.atan.apply(v - vth, 2.)
    return v, spike


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


class LinearPLIFCheckpointing(BaseCheckpointingBlock):

    def __init__(
        self,
        in_features,
        out_features,
        beta_initializer='uniform',
        beta_low=0,
        beta_high=4,
        vth=0.5,
        spike_compressor="NullSpikeCompressor",
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

        self.spike_compressor = get_spike_compressor(spike_compressor)

    @staticmethod
    def conventional_forward(
        x_seq, weight, bias, _beta, vth, in_backward=False
    ):
        # x_seq.shape = [T, N, in_features]
        T = x_seq.shape[0]

        x_seq = F.linear(x_seq, weight, bias)

        v = torch.rand_like(x_seq[0])  # rand: follow the practice of DH-SNN
        s = torch.rand_like(x_seq[0])
        s_seq = torch.empty_like(x_seq)
        for t in range(T):
            x = x_seq[t]
            v, s = plif_update(x, v, s, vth, _beta)
            s_seq[t] = s
        return s_seq

    def forward(self, x_seq):
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.dense.weight,
            self.dense.bias,
            self._beta,
            self.vth,
        )


def output_plif_update(x, v, _beta):
    beta = torch.sigmoid(_beta)
    v = v*beta + (1-beta) * x
    return v


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


class LinearOutputPLIFCheckpointing(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        beta_initializer='uniform',
        beta_low=0,
        beta_high=4,
        spike_compressor="NullSpikeCompressor",
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

        self.spike_compressor = get_spike_compressor(spike_compressor)

    @staticmethod
    def conventional_forward(x_seq, weight, bias, _beta, in_backward=False):
        # x_seq.shape = (T, N, in_features)
        T = x_seq.shape[0]

        x_seq = F.linear(x_seq, weight, bias)

        v = torch.rand_like(x_seq[0])
        v_seq = torch.rand_like(x_seq)
        for t in range(T):
            x = x_seq[t]
            v = output_plif_update(x, v, _beta)
            v_seq[t] = v
        return v_seq

    def forward(self, x_seq):
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.dense.weight,
            self.dense.bias,
            self._beta,
        )


class PLIFSFNN(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        H = 512
        self.dense_1 = LinearPLIF(700, H, vth=1.)
        self.dense_2 = LinearPLIF(H, H, vth=1.)
        self.dense_3 = LinearPLIF(H, H, vth=1.)
        self.dense_out = LinearOutputPLIF(H, 20)
        nn.init.xavier_normal_(self.dense_out.dense.weight)
        nn.init.constant_(self.dense_out.dense.bias, 0)

    def forward(self, x_seq):
        x_seq = x_seq.transpose(0, 1)  # [T, N, C]
        x_seq = self.dense_1(x_seq)
        x_seq = self.dense_2(x_seq)
        x_seq = self.dense_3(x_seq)
        x_seq = self.dense_out(x_seq)  # [T, N, 20]

        logits = F.softmax(x_seq, dim=-1)  # [T, N, 20]
        return torch.sum(logits[10:], dim=0)  # [N, 20]; discard 1st 10 steps


class GCPLIFSFNN(nn.Module):

    def __init__(self, spike_compressor: str, *args, **kwargs):
        super().__init__()
        H = 512
        self.dense_1 = LinearPLIFCheckpointing(
            700, H, vth=1., spike_compressor="NullSpikeCompressor"
        )
        self.dense_2 = LinearPLIFCheckpointing(
            H, H, vth=1., spike_compressor=spike_compressor
        )
        self.dense_3 = LinearPLIFCheckpointing(
            H, H, vth=1., spike_compressor=spike_compressor
        )
        self.dense_out = LinearOutputPLIFCheckpointing(
            H, 20, spike_compressor=spike_compressor
        )
        nn.init.xavier_normal_(self.dense_out.dense.weight)
        nn.init.constant_(self.dense_out.dense.bias, 0)

    def forward(self, x_seq):
        x_seq = x_seq.transpose(0, 1)  # [T, N, C]
        x_seq = self.dense_1(x_seq)
        x_seq = self.dense_2(x_seq)
        x_seq = self.dense_3(x_seq)
        x_seq = self.dense_out(x_seq)  # [T, N, 20]

        logits = F.softmax(x_seq, dim=-1)  # [T, N, 20]
        return torch.sum(logits[10:], dim=0)  # [N, 20]; discard 1st 10 steps


class PGCPLIFSFNN(nn.Module):

    def __init__(self, spike_compressor: str, *args, **kwargs):
        super().__init__()
        H = 512
        self.dense_1 = LinearPLIFCheckpointing(
            700, H, vth=1., spike_compressor="NullSpikeCompressor"
        )
        self.dense_2 = LinearPLIFCheckpointing(
            H, H, vth=1., spike_compressor=spike_compressor
        )
        self.dense_3 = LinearPLIF(H, H, vth=1.)
        self.dense_out = LinearOutputPLIF(H, 20)
        nn.init.xavier_normal_(self.dense_out.dense.weight)
        nn.init.constant_(self.dense_out.dense.bias, 0)

    def forward(self, x_seq):
        x_seq = x_seq.transpose(0, 1)  # [T, N, C]
        x_seq = self.dense_1(x_seq)
        x_seq = self.dense_2(x_seq)
        x_seq = self.dense_3(x_seq)
        x_seq = self.dense_out(x_seq)  # [T, N, 20]

        logits = F.softmax(x_seq, dim=-1)  # [T, N, 20]
        return torch.sum(logits[10:], dim=0)  # [N, 20]; discard 1st 10 steps
