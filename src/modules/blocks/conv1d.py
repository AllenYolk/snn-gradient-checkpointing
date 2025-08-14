import torch
import torch.nn as nn
import torch.nn.functional as F

from ..compress import *
from ..neuron import SlidingPSN, PSN
from ..kernels import *
from .checkpointing import InputCompressedGCFunction, BaseGCBlock, BaseTCGCBlock


class Conv1d(BaseGCBlock):

    def __init__(
        self,
        proj: nn.Conv1d,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        in_backward=False
    ):
        T = x_seq.size(0)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv1d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.proj.weight,
            self.proj.bias,
            self.proj.stride,
            self.proj.padding,
            self.proj.dilation,
            self.proj.groups,
        )


class TCConv1d(BaseTCGCBlock):

    def __init__(
        self,
        proj: nn.Conv1d,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor(),
        n_chunk: int = 2,
    ):
        super().__init__(n_chunk)
        self.proj = proj
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        in_backward=False
    ):
        T = x_seq.size(0)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv1d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x_seq: torch.Tensor):
        x_seqs = torch.chunk(x_seq, self.n_chunk, dim=0)
        out_seq = []
        for xc in x_seqs:
            yc = InputCompressedGCFunction.apply(
                self.conventional_forward,
                self.spike_compressor,
                xc,
                self.proj.weight,
                self.proj.bias,
                self.proj.stride,
                self.proj.padding,
                self.proj.dilation,
                self.proj.groups,
            )
            out_seq.append(yc)
        return torch.cat(out_seq, dim=0)


class Conv1dLIF(BaseGCBlock):

    def __init__(
        self,
        proj: nn.Conv1d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        neuron,
        in_backward=False,  # will be used in checkpointing function
    ):
        T = x_seq.size(0)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv1d(x_seq, weight, bias, stride, padding, dilation, groups)
        y_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return neuron(y_seq)

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.proj.weight,
            self.proj.bias,
            self.proj.stride,
            self.proj.padding,
            self.proj.dilation,
            self.proj.groups,
            self.neuron,
        )


class TCConv1dLIF(BaseTCGCBlock):

    def __init__(
        self,
        proj: nn.Conv1d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor(),
        n_chunk: int = 2,
    ):
        super().__init__(n_chunk)
        self.proj = proj
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        neuron,
        v,
        in_backward=False,  # will be used in checkpointing function
    ):
        T = x_seq.size(0)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv1d(x_seq, weight, bias, stride, padding, dilation, groups)
        y_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return neuron.rnn_forward(y_seq, v)  # s_seq, v

    def forward(self, x_seq: torch.Tensor):
        x_seqs = torch.chunk(x_seq, self.n_chunk, dim=0)
        v = torch.zeros([], device=x_seq.device)
        out_seq = []
        for xc in x_seqs:
            sc, v = InputCompressedGCFunction.apply(
                self.conventional_forward,
                self.spike_compressor,
                xc,
                self.proj.weight,
                self.proj.bias,
                self.proj.stride,
                self.proj.padding,
                self.proj.dilation,
                self.proj.groups,
                self.neuron,
                v,
            )
            out_seq.append(sc)
        return torch.cat(out_seq, dim=0)


class Conv1dPSN(BaseGCBlock):

    def __init__(
        self,
        proj: nn.Conv1d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        neuron_weight,
        neuron_bias,
        in_backward=False,  # will be used in checkpointing function
    ):
        T = x_seq.size(0)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv1d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return PSN.forward_function(x_seq, neuron_weight, neuron_bias)

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.proj.weight,
            self.proj.bias,
            self.proj.stride,
            self.proj.padding,
            self.proj.dilation,
            self.proj.groups,
            self.neuron.weight,
            self.neuron.bias,
        )


class Conv1dSlidingPSN(BaseGCBlock):

    def __init__(
        self,
        proj: nn.Conv1d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        neuron_weight,
        neuron_bias,
        neuron_k,
        in_backward=False,  # will be used in checkpointing function
    ):
        T = x_seq.size(0)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv1d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.proj.weight,
            self.proj.bias,
            self.proj.stride,
            self.proj.padding,
            self.proj.dilation,
            self.proj.groups,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
        )


class Conv1dBNLIF(BaseGCBlock):

    def __init__(
        self,
        proj: nn.Conv1d,
        bn: nn.BatchNorm1d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        training,
        neuron,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv1d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = F.batch_norm(
            x_seq,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=training,
            momentum=0.1 if in_backward else 0.
        )
        x_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
        return neuron(x_seq)

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.proj.weight,
            self.proj.bias,
            self.proj.stride,
            self.proj.padding,
            self.proj.dilation,
            self.proj.groups,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
            self.neuron,
        )


class Conv1dBNPSN(BaseGCBlock):

    def __init__(
        self,
        proj: nn.Conv1d,
        bn: nn.BatchNorm1d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        training,
        neuron_weight,
        neuron_bias,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv1d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = F.batch_norm(
            x_seq,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=training,
            momentum=0.1 if in_backward else 0.
        )
        x_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
        return PSN.forward_function(x_seq, neuron_weight, neuron_bias)

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.proj.weight,
            self.proj.bias,
            self.proj.stride,
            self.proj.padding,
            self.proj.dilation,
            self.proj.groups,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
            self.neuron.weight,
            self.neuron.bias,
        )


class Conv1dBNSlidingPSN(BaseGCBlock):

    def __init__(
        self,
        proj: nn.Conv1d,
        bn: nn.BatchNorm1d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        training,
        neuron_weight,
        neuron_bias,
        neuron_k,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv1d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = F.batch_norm(
            x_seq,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=training,
            momentum=0.1 if in_backward else 0.
        )
        x_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
        return SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.proj.weight,
            self.proj.bias,
            self.proj.stride,
            self.proj.padding,
            self.proj.dilation,
            self.proj.groups,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
        )


class AvgPool1dConv1dBNLIF(BaseGCBlock):

    def __init__(
        self,
        pool: nn.AvgPool1d,
        proj: nn.Conv1d,
        bn: nn.BatchNorm1d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.pool = pool
        self.proj = proj
        self.bn = bn
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        training,
        neuron,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.avg_pool1d(x_seq, pool_kernel_size, pool_stride, pool_padding)
        x_seq = F.conv1d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = F.batch_norm(
            x_seq,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=training,
            momentum=0.1 if in_backward else 0.
        )
        y_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
        return neuron(y_seq)

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.pool.kernel_size[0],
            self.pool.stride[0],
            self.pool.padding[0],
            self.proj.weight,
            self.proj.bias,
            self.proj.stride,
            self.proj.padding,
            self.proj.dilation,
            self.proj.groups,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
            self.neuron,
        )


class AvgPool1dConv1dBNPSN(BaseGCBlock):

    def __init__(
        self,
        pool: nn.AvgPool1d,
        proj: nn.Conv1d,
        bn: nn.BatchNorm1d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.pool = pool
        self.proj = proj
        self.bn = bn
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        training,
        neuron_weight,
        neuron_bias,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.avg_pool1d(x_seq, pool_kernel_size, pool_stride, pool_padding)
        x_seq = F.conv1d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = F.batch_norm(
            x_seq,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=training,
            momentum=0.1 if in_backward else 0.
        )
        y_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
        return PSN.forward_function(y_seq, neuron_weight, neuron_bias)

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.pool.kernel_size[0],
            self.pool.stride[0],
            self.pool.padding[0],
            self.proj.weight,
            self.proj.bias,
            self.proj.stride,
            self.proj.padding,
            self.proj.dilation,
            self.proj.groups,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
            self.neuron.weight,
            self.neuron.bias,
        )


class AvgPool1dConv1dBNSlidingPSN(BaseGCBlock):

    def __init__(
        self,
        pool: nn.AvgPool1d,
        proj: nn.Conv1d,
        bn: nn.BatchNorm1d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.pool = pool
        self.proj = proj
        self.bn = bn
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        training,
        neuron_weight,
        neuron_bias,
        neuron_k,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.avg_pool1d(x_seq, pool_kernel_size, pool_stride, pool_padding)
        x_seq = F.conv1d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = F.batch_norm(
            x_seq,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=training,
            momentum=0.1 if in_backward else 0.
        )
        y_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
        return SlidingPSN.forward_function(
            y_seq, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.pool.kernel_size[0],
            self.pool.stride[0],
            self.pool.padding[0],
            self.proj.weight,
            self.proj.bias,
            self.proj.stride,
            self.proj.padding,
            self.proj.dilation,
            self.proj.groups,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
        )
