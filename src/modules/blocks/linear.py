import torch
import torch.nn as nn
import torch.nn.functional as F

from ..compress import *
from ..neuron import SlidingPSN, PSN
from ..kernels import *
from .checkpointing import InputCompressedGCFunction, BaseGCBlock


class LinearLIF(BaseGCBlock):

    def __init__(
        self,
        proj: nn.Linear,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor(),
    ):
        super().__init__()
        self.proj = proj
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(x_seq, weight, bias, neuron, in_backward=False):
        y_seq = F.linear(x_seq, weight, bias)
        return neuron(y_seq)

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.proj.weight,
            self.proj.bias,
            self.neuron,
        )


class LinearPSN(BaseGCBlock):

    def __init__(
        self,
        proj: nn.Linear,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor(),
    ):
        super().__init__()
        self.proj = proj
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq, weight, bias, neuron_weight, neuron_bias, in_backward=False
    ):
        y_seq = F.linear(x_seq, weight, bias)
        return PSN.forward_function(y_seq, neuron_weight, neuron_bias)

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward, self.spike_compressor, x_seq,
            self.proj.weight, self.proj.bias, self.neuron.weight,
            self.neuron.bias
        )


class LinearSlidingPSN(BaseGCBlock):

    def __init__(
        self,
        proj: nn.Linear,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor(),
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
        neuron_weight,
        neuron_bias,
        neuron_k,
        in_backward=False
    ):
        y_seq = F.linear(x_seq, weight, bias)
        return SlidingPSN.forward_function(
            y_seq, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.proj.weight,
            self.proj.bias,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
        )


class LinearBNLIF(BaseGCBlock):

    def __init__(
        self,
        proj: nn.Linear,
        bn: nn.BatchNorm1d,  # actually, not necessarily 1d
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor(),
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
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        training,
        neuron,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = F.linear(x_seq, weight, bias)
        x_seq = x_seq.flatten(0, 1)
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
            self.proj.weight,
            self.proj.bias,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
            self.neuron,
        )


class LinearBNPSN(BaseGCBlock):

    def __init__(
        self,
        proj: nn.Linear,
        bn: nn.BatchNorm1d,  # actually, not necessarily 1d
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor(),
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
        x_seq = F.linear(x_seq, weight, bias)
        x_seq = x_seq.flatten(0, 1)
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
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
            self.neuron.weight,
            self.neuron.bias,
        )


class LinearBNSlidingPSN(BaseGCBlock):

    def __init__(
        self,
        proj: nn.Linear,
        bn: nn.BatchNorm1d,  # actually, not necessarily 1d
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor(),
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
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        training,
        neuron_weight,
        neuron_bias,
        neuron_k,
        spike_compressor,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = F.linear(x_seq, weight, bias)
        x_seq = x_seq.flatten(0, 1)
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
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
            self.spike_compressor,
        )


class AvgPool1dFlattenLinearLIF(BaseGCBlock):

    def __init__(
        self,
        pool: nn.AvgPool1d,
        proj: nn.Linear,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor(),
    ):
        super().__init__()
        self.pool = pool
        self.proj = proj
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
        neuron,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        y_seq = x_seq.flatten(0, 1)
        y_seq = F.avg_pool1d(
            y_seq,
            kernel_size=pool_kernel_size,
            stride=pool_stride,
            padding=pool_padding
        )
        y_seq = y_seq.flatten(1)
        y_seq = F.linear(y_seq, weight, bias)
        y_seq = y_seq.reshape(T, N, -1)
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
            self.neuron,
        )


class AvgPool1dFlattenLinearPSN(BaseGCBlock):

    def __init__(
        self,
        pool: nn.AvgPool1d,
        proj: nn.Linear,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor(),
    ):
        super().__init__()
        self.pool = pool
        self.proj = proj
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
        neuron_weight,
        neuron_bias,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        y_seq = x_seq.flatten(0, 1)
        y_seq = F.avg_pool1d(
            y_seq,
            kernel_size=pool_kernel_size,
            stride=pool_stride,
            padding=pool_padding
        )
        y_seq = y_seq.flatten(1)
        y_seq = F.linear(y_seq, weight, bias)
        y_seq = y_seq.reshape(T, N, -1)
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
            self.neuron.weight,
            self.neuron.bias,
        )


class AvgPool1dFlattenLinearSlidingPSN(BaseGCBlock):

    def __init__(
        self,
        pool: nn.AvgPool1d,
        proj: nn.Linear,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor(),
    ):
        super().__init__()
        self.pool = pool
        self.proj = proj
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
        neuron_weight,
        neuron_bias,
        neuron_k,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        y_seq = x_seq.flatten(0, 1)
        y_seq = F.avg_pool1d(
            y_seq,
            kernel_size=pool_kernel_size,
            stride=pool_stride,
            padding=pool_padding
        )
        y_seq = y_seq.flatten(1)
        y_seq = F.linear(y_seq, weight, bias)
        y_seq = y_seq.reshape(T, N, -1)
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
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
        )
