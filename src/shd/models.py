import sys

sys.path.append("./src")

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate

from modules.checkpointing import BaseGCBlock, BaseTCGCBlock
from modules.checkpointing import InputCompressedGC
from modules.checkpointing import memory_optimization
from modules.compress import get_spike_compressor


def plif_update(x, v, _beta, vth):
    """(x, v) -> (s, v)"""
    beta = torch.sigmoid(_beta)
    v = v*beta + (1-beta) * x
    s = surrogate.atan.apply(v - vth, 2.)
    v = v - vth*s
    return s, v


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

        v = torch.zeros_like(x_seq[0])
        s_seq = torch.empty_like(x_seq)
        for t in range(T):
            x = x_seq[t]
            s, v = plif_update(x, v, self._beta, self.vth)
            s_seq[t] = s
        return s_seq

    def __tc_init_states__(self, x_seq):
        return [torch.zeros([], device=x_seq.device, dtype=x_seq.dtype)]

    def __tc_forward__(self, xc, v):
        Tc = xc.shape[0]
        xc = self.dense(xc)
        sc = torch.empty_like(xc)
        for t in range(Tc):
            sc[t], v = plif_update(xc[t], v, self._beta, self.vth)
        return sc, v


class LinearPLIFCheckpointing(BaseGCBlock):

    def __init__(
        self,
        in_features,
        out_features,
        beta_initializer='uniform',
        beta_low=0,
        beta_high=4,
        vth=1.,
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
        v = torch.zeros_like(x_seq[0])
        s_seq = torch.empty_like(x_seq)
        for t in range(T):
            x = x_seq[t]
            s, v = plif_update(x, v, _beta, vth)
            s_seq[t] = s
        return s_seq

    def forward(self, x_seq):
        return InputCompressedGC.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.dense.weight,
            self.dense.bias,
            self._beta,
            self.vth,
        )


class LinearPLIFTCCheckpointing(BaseTCGCBlock):

    def __init__(
        self,
        in_features,
        out_features,
        beta_initializer='uniform',
        beta_low=0,
        beta_high=4,
        vth=1.,
        spike_compressor="NullSpikeCompressor",
        n_chunk: int = 2,
    ):
        super().__init__(n_chunk)
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
        x_seq, weight, bias, v, _beta, vth, in_backward=False
    ):
        # x_seq.shape = [Tc, N, in_features]
        Tc = x_seq.shape[0]
        x_seq = F.linear(x_seq, weight, bias)
        s_seq = torch.empty_like(x_seq)
        for t in range(Tc):
            x = x_seq[t]
            s, v = plif_update(x, v, _beta, vth)
            s_seq[t] = s
        return s_seq, v

    def forward(self, x_seq):
        x_seqs = torch.chunk(x_seq, self.n_chunk, dim=0)
        v = torch.zeros([], device=x_seq.device, dtype=x_seq.dtype)
        out_seq = []
        for xc in x_seqs:
            sc, v = InputCompressedGC.apply(
                self.conventional_forward,
                self.spike_compressor,
                xc,
                self.dense.weight,
                self.dense.bias,
                v,
                self._beta,
                self.vth,
            )
            out_seq.append(sc)
        return torch.cat(out_seq, dim=0)


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

        v = torch.zeros_like(x_seq[0])
        v_seq = torch.empty_like(x_seq)
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

        v = torch.zeros_like(x_seq[0])
        v_seq = torch.empty_like(x_seq)
        for t in range(T):
            x = x_seq[t]
            v = output_plif_update(x, v, _beta)
            v_seq[t] = v
        return v_seq

    def forward(self, x_seq):
        return InputCompressedGC.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.dense.weight,
            self.dense.bias,
            self._beta,
        )


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
        self.in_features = in_features
        self.out_features = out_features
        self.vth = vth
        self.branch = branch

        self.mask_share = mask_share
        self.pad = (in_features//branch*branch + branch - in_features) % branch
        self.dense = nn.Linear(in_features + self.pad, out_features * branch)

        #sparsity
        self.sparsity = 1 / branch
        self.overlap = 1 / branch
        self._alpha = nn.Parameter(torch.empty([self.out_features, branch]))
        self._beta = nn.Parameter(torch.empty([self.out_features]))

        self.create_mask()

        if alpha_initializer == 'uniform':
            nn.init.uniform_(self._alpha, alpha_low, alpha_high)
        elif alpha_initializer == 'constant':
            nn.init.constant_(self._alpha, alpha_low)

        if beta_initializer == 'uniform':
            nn.init.uniform_(self._beta, beta_low, beta_high)
        elif beta_initializer == 'constant':
            nn.init.constant_(self._beta, beta_low)

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
        x_seq = x_seq.reshape(T, N, self.out_features, self.branch)

        alpha = torch.sigmoid(self._alpha)
        vd = torch.zeros_like(x_seq[0])
        v = torch.rand([N, self.out_features], device=x_seq.device)
        s_seq = torch.empty([T, N, self.out_features], device=s.device)
        for t in range(T):
            x = x_seq[t]
            vd = alpha*vd + (1-alpha) * x
            y = torch.sum(vd, dim=-1)  # [N, out_features]
            s, v = plif_update(y, v, self._beta, self.vth)
            s_seq[t] = s
        return s_seq


class LinearDHLIFCheckpointing(nn.Module):

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
        mask_share=1,
        spike_compressor="NullSpikeCompressor",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.vth = vth
        self.branch = branch

        self.mask_share = mask_share
        self.pad = (in_features//branch*branch + branch - in_features) % branch
        self.dense = nn.Linear(in_features + self.pad, out_features * branch)

        #sparsity
        self.sparsity = 1 / branch
        self.overlap = 1 / branch
        self._alpha = nn.Parameter(torch.empty([self.out_features, branch]))
        self._beta = nn.Parameter(torch.empty([self.out_features]))

        self.create_mask()

        if alpha_initializer == 'uniform':
            nn.init.uniform_(self._alpha, alpha_low, alpha_high)
        elif alpha_initializer == 'constant':
            nn.init.constant_(self._alpha, alpha_low)

        if beta_initializer == 'uniform':
            nn.init.uniform_(self._beta, beta_low, beta_high)
        elif beta_initializer == 'constant':
            nn.init.constant_(self._beta, beta_low)

        self.spike_compressor = get_spike_compressor(spike_compressor)

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

    @staticmethod
    def conventional_forward(
        x_seq, weight, bias, mask, _alpha, _beta, pad, out_features, branch,
        vth, in_backward
    ):
        # x_seq.shape = [T, N, C]
        T, N = x_seq.shape[:2]
        padding = torch.zeros([T, N, pad], device=x_seq.device)
        x_seq = torch.cat((x_seq, padding), -1)  # [T, N, Cin]
        x_seq = F.linear(x_seq, weight * mask, bias)
        x_seq = x_seq.reshape(T, N, out_features, branch)

        alpha = torch.sigmoid(_alpha)
        vd = torch.zeros_like(x_seq[0])
        v = torch.rand([N, out_features], device=x_seq.device)
        s_seq = torch.empty([T, N, out_features], device=s.device)
        for t in range(T):
            x = x_seq[t]
            vd = alpha*vd + (1-alpha) * x
            y = torch.sum(vd, dim=-1)  # [N, out_features]
            s, v = plif_update(y, v, _beta, vth)
            s_seq[t] = s
        return s_seq

    def forward(self, x_seq):
        return InputCompressedGC.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.dense.weight,
            self.dense.bias,
            self.mask,
            self._alpha,
            self._beta,
            self.pad,
            self.out_features,
            self.branch,
            self.vth,
        )


class PLIFSFNN(nn.Module):

    def __init__(self, *args, **kwargs):
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


class GCPLIFSFNN(nn.Module):

    def __init__(self, spike_compressor: str, *args, **kwargs):
        super().__init__()
        H = 1024
        self.dense_1 = LinearPLIFCheckpointing(
            700, H, vth=1., spike_compressor="NullSpikeCompressor"
        )
        self.dense_2 = LinearPLIFCheckpointing(
            H, H, vth=1., spike_compressor=spike_compressor
        )
        self.dense_3 = LinearPLIFCheckpointing(
            H, H // 2, vth=1., spike_compressor=spike_compressor
        )
        self.dense_out = LinearOutputPLIFCheckpointing(
            H // 2, 20, spike_compressor=spike_compressor
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


def AutoGCPLIFSFNN(spike_compressor: str, T: int, *args, **kwargs):
    net = PLIFSFNN(*args, **kwargs)
    return memory_optimization(
        net,
        instance=(LinearPLIF, LinearOutputPLIF),
        dummy_input=torch.zeros(128, T, 700),
        level=4,
        verbose=True,
        temporal_split_factor=5,
    )


class TCGCPLIFSFNN(nn.Module):

    def __init__(self, spike_compressor: str, *args, **kwargs):
        super().__init__()
        H = 1024
        self.dense_1 = LinearPLIFTCCheckpointing(
            700, H, vth=1., spike_compressor="NullSpikeCompressor", n_chunk=25
        )
        self.dense_2 = LinearPLIFTCCheckpointing(
            H, H, vth=1., spike_compressor=spike_compressor, n_chunk=10
        )
        self.dense_3 = LinearPLIFTCCheckpointing(
            H, H // 2, vth=1., spike_compressor=spike_compressor, n_chunk=25
        )
        self.dense_out = LinearOutputPLIFCheckpointing(
            H // 2, 20, spike_compressor=spike_compressor
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


class DHLIFSFNN(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        H = 512
        self.dense_1 = LinearDHLIF(700, H, vth=1.)
        self.dense_2 = LinearDHLIF(H, H, vth=1.)
        self.dense_3 = LinearDHLIF(H, H, vth=1.)
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


class GCDHLIFSFNN(nn.Module):

    def __init__(self, spike_compressor: str, *args, **kwargs):
        super().__init__()
        H = 512
        self.dense_1 = LinearDHLIFCheckpointing(
            700, H, vth=1., spike_compressor="NullSpikeCompressor"
        )
        self.dense_2 = LinearDHLIFCheckpointing(
            H, H, vth=1., spike_compressor=spike_compressor
        )
        self.dense_3 = LinearDHLIFCheckpointing(
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


class PGCDHLIFSFNN(nn.Module):

    def __init__(self, spike_compressor: str, *args, **kwargs):
        super().__init__()
        H = 512
        self.dense_1 = LinearDHLIFCheckpointing(
            700, H, vth=1., spike_compressor="NullSpikeCompressor"
        )
        self.dense_2 = LinearDHLIFCheckpointing(
            H, H, vth=1., spike_compressor=spike_compressor
        )
        self.dense_3 = LinearDHLIF(H, H, vth=1.)
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
