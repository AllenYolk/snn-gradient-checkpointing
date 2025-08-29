import sys

sys.path.append("./src")

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate

from modules.checkpointing import memory_optimization


def plif_update(x, v, _beta, vth):
    """(x, v) -> (s, v)"""
    beta = torch.sigmoid(_beta)
    v = v*beta + (1-beta) * x
    s = surrogate.atan.apply(v - vth, 2.)
    v = v - vth*s
    return s, v


class Linear(nn.Linear):
    """Linear layer that supports temporal chunking"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __tc_init_states__(self, *args, **kwargs):
        return []  # empty; no states

    def __tc_forward__(self, x_seq):
        return (super().forward(x_seq),)


class PLIF(nn.Module):

    def __init__(
        self,
        out_features,
        beta_initializer='uniform',
        beta_low=0,
        beta_high=4,
        vth=1.
    ):
        super().__init__()
        self.out_features = out_features
        self.vth = vth
        self._beta = nn.Parameter(torch.empty([self.out_features]))

        if beta_initializer == 'uniform':
            nn.init.uniform_(self._beta, beta_low, beta_high)
        elif beta_initializer == 'constant':
            nn.init.constant_(self._beta, beta_low)

    def forward(self, x_seq):
        # x_seq.shape = [T, N, in_features]
        T = x_seq.shape[0]
        v = torch.zeros_like(x_seq[0])
        s_seq = torch.empty_like(x_seq)
        for t in range(T):
            s_seq[t], v = plif_update(x_seq[t], v, self._beta, self.vth)
        return s_seq

    def __tc_init_states__(self, x_seq):
        return [torch.zeros([], device=x_seq.device, dtype=x_seq.dtype)]

    def __tc_forward__(self, xc, v):
        Tc = xc.shape[0]
        sc = torch.empty_like(xc)
        for t in range(Tc):
            sc[t], v = plif_update(xc[t], v, self._beta, self.vth)
        return sc, v


class LinearPLIF(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        beta_initializer='uniform',
        beta_low=0,
        beta_high=4,
        vth=1.,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dense = Linear(in_features, out_features)
        self.neuron = PLIF(
            out_features, beta_initializer, beta_low, beta_high, vth
        )

    def forward(self, x_seq):
        return self.neuron(self.dense(x_seq))

    def __spatial_split__(self):
        return self.dense, self.neuron

    def __tc_init_states__(self, x_seq):
        return [torch.zeros([], device=x_seq.device, dtype=x_seq.dtype)]

    def __tc_forward__(self, xc, v):
        xc = self.dense(xc)
        return self.neuron.__tc_forward__(xc, v)


def output_plif_update(x, v, _beta):
    beta = torch.sigmoid(_beta)
    v = v*beta + (1-beta) * x
    return v


class OutputPLIF(nn.Module):

    def __init__(
        self,
        out_features,
        beta_initializer='uniform',
        beta_low=0,
        beta_high=4
    ):
        super().__init__()
        self.out_features = out_features
        self._beta = nn.Parameter(torch.empty([self.out_features]))

        if beta_initializer == 'uniform':
            nn.init.uniform_(self._beta, beta_low, beta_high)
        elif beta_initializer == 'constant':
            nn.init.constant_(self._beta, beta_low)

    def forward(self, x_seq):
        # x_seq.shape = (T, N, in_features)
        T = x_seq.shape[0]
        v = torch.zeros_like(x_seq[0])
        v_seq = torch.empty_like(x_seq)
        for t in range(T):
            x = x_seq[t]
            v = output_plif_update(x, v, self._beta)
            v_seq[t] = v
        return v_seq


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
        self.dense = Linear(in_features, out_features)
        self.neuron = OutputPLIF(
            out_features, beta_initializer, beta_low, beta_high
        )

    def forward(self, x_seq):
        return self.neuron(self.dense(x_seq))

    def __spatial_split__(self):
        return self.dense, self.neuron


class MaskedLinear(nn.Module):

    def __init__(self, in_features, out_features, branch=4, mask_share=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.branch = branch
        self.mask_share = mask_share
        self.pad = (in_features//branch*branch + branch - in_features) % branch
        self.dense = nn.Linear(in_features + self.pad, out_features * branch)
        self.create_mask()

    def create_mask(self):
        input_size = self.in_features + self.pad  # the real input channels
        mask = torch.zeros(self.out_features * self.branch, input_size)
        for i in range(self.out_features // self.mask_share):
            seq = torch.randperm(input_size)
            for j in range(self.branch):
                for k in range(self.mask_share):
                    x = (i * self.mask_share + k) * self.branch + j
                    y_start = j * input_size // self.branch
                    y_end = (j+1) * input_size // self.branch
                    y = seq[y_start:y_end]
                    mask[x, y] = 1
        self.mask = nn.Parameter(mask, requires_grad=False)

    def apply_mask(self):
        self.dense.weight.data = self.dense.weight.data * self.mask

    def forward(self, x_seq):
        # x_seq.shape = [T, N, C]
        T, N = x_seq.shape[:2]
        padding = torch.zeros([T, N, self.pad], device=x_seq.device)
        x_seq = torch.cat((x_seq, padding), -1)  # [T, N, Cin]
        x_seq = F.linear(x_seq, self.dense.weight * self.mask, self.dense.bias)
        return x_seq

    def __tc_init_states__(self, x_seq):
        return []

    def __tc_forward__(self, x_seq):
        return (self.forward(x_seq),)


class DHLIF(nn.Module):

    def __init__(
        self,
        out_features,
        alpha_initializer='uniform',
        alpha_low=2,
        alpha_high=6,
        beta_initializer='uniform',
        beta_low=0,
        beta_high=4,
        vth=1.,
        branch=4,
    ):
        super().__init__()
        self.out_features = out_features
        self.vth = vth
        self.branch = branch

        self._alpha = nn.Parameter(torch.empty([self.out_features, branch]))
        self._beta = nn.Parameter(torch.empty([self.out_features]))

        if alpha_initializer == 'uniform':
            nn.init.uniform_(self._alpha, alpha_low, alpha_high)
        elif alpha_initializer == 'constant':
            nn.init.constant_(self._alpha, alpha_low)

        if beta_initializer == 'uniform':
            nn.init.uniform_(self._beta, beta_low, beta_high)
        elif beta_initializer == 'constant':
            nn.init.constant_(self._beta, beta_low)

    def forward(self, x_seq):
        # x_seq.shape = [T, N, C]
        T, N = x_seq.shape[:2]
        x_seq = x_seq.reshape(T, N, self.out_features, self.branch)

        alpha = torch.sigmoid(self._alpha)
        vd = torch.zeros_like(x_seq[0])
        v = torch.zeros([N, self.out_features], device=x_seq.device)
        s_seq = torch.empty([T, N, self.out_features], device=x_seq.device)
        for t in range(T):
            x = x_seq[t]
            vd = alpha*vd + (1-alpha) * x
            y = torch.sum(vd, dim=-1)  # [N, out_features]
            s, v = plif_update(y, v, self._beta, self.vth)
            s_seq[t] = s
        return s_seq

    def __tc_init_states__(self, x_seq):
        N = x_seq.shape[1]
        return [
            torch.zeros([], device=x_seq.device, dtype=x_seq.dtype),
            torch.zeros([], device=x_seq.device, dtype=x_seq.dtype)
        ]

    def __tc_forward__(self, xc, vd, v):
        T, N = xc.shape[:2]
        xc = xc.reshape(T, N, self.out_features, self.branch)

        alpha = torch.sigmoid(self._alpha)
        sc = torch.empty([T, N, self.out_features], device=xc.device)
        for t in range(T):
            x = xc[t]
            vd = alpha*vd + (1-alpha) * x
            y = torch.sum(vd, dim=-1)  # [N, out_features]
            s, v = plif_update(y, v, self._beta, self.vth)
            sc[t] = s
        return sc, vd, v


class LinearDHLIF(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        alpha_initializer='uniform',
        alpha_low=2,
        alpha_high=6,
        beta_initializer='uniform',
        beta_low=0,
        beta_high=4,
        vth=1.,
        branch=4,
        mask_share=1
    ):
        super().__init__()
        self.out_features = out_features
        self.branch = branch
        self.dense = MaskedLinear(in_features, out_features, branch, mask_share)
        self.neuron = DHLIF(
            out_features, alpha_initializer, alpha_low, alpha_high,
            beta_initializer, beta_low, beta_high, vth, branch
        )

    def forward(self, x_seq):
        return self.neuron(self.dense(x_seq))

    def __spatial_split__(self):
        return self.dense, self.neuron

    def __tc_init_states__(self, x_seq):
        N = x_seq.shape[1]
        return [
            torch.zeros([], device=x_seq.device, dtype=x_seq.dtype),
            torch.zeros([], device=x_seq.device, dtype=x_seq.dtype)
        ]

    def __tc_forward__(self, xc, vd, v):
        xc = self.dense(xc)
        return self.neuron.__tc_forward__(xc, vd, v)


class PLIFSFNN(nn.Module):

    def __init__(self):
        super().__init__()
        H = 1024
        self.dense_1 = LinearPLIF(700, H, vth=1.)
        self.dense_2 = LinearPLIF(H, H, vth=1.)
        self.dense_3 = LinearPLIF(H, H // 2, vth=1.)
        self.dense_out = LinearOutputPLIF(H // 2, 20)
        self.dense_1.disable_x_compressor = True
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


def GCPLIFSFNN(
    T: int, compress_x: bool, level: int = 1, temporal_split_factor: int = 5
):
    net = PLIFSFNN()
    return memory_optimization(
        net,
        instance=(LinearPLIF, LinearOutputPLIF),
        dummy_input=torch.zeros(128, T, 700),
        compress_x=compress_x,
        level=level,
        verbose=True,
        temporal_split_factor=temporal_split_factor,
    )


class DHLIFSFNN(nn.Module):

    def __init__(self):
        super().__init__()
        H = 1024
        self.dense_1 = LinearDHLIF(700, H, vth=1.)
        self.dense_2 = LinearDHLIF(H, H, vth=1.)
        self.dense_3 = LinearDHLIF(H, H // 2, vth=1.)
        self.dense_out = LinearOutputPLIF(H // 2, 20)
        self.dense_1.disable_x_compressor = True
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


def GCDHLIFSFNN(
    T: int, compress_x: bool, level: int = 1, temporal_split_factor: int = 5
):
    net = DHLIFSFNN()
    return memory_optimization(
        net,
        instance=(LinearDHLIF, LinearOutputPLIF),
        dummy_input=torch.zeros(128, T, 700),
        compress_x=compress_x,
        level=level,
        verbose=True,
        temporal_split_factor=temporal_split_factor,
    )
