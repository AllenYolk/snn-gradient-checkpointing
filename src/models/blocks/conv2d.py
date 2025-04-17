import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseCheckpointingBlock
from ..compress import *
from ..neuron import SlidingPSN, PSN
from ..kernels import *
from .checkpointing import SNNCheckpointingBlockFunction


class Conv2dLIF(BaseCheckpointingBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
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
        x_seq = conv2d_forward(
            x_seq, weight, bias, stride, padding, dilation, groups
        )
        return neuron(x_seq)

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class Conv2dPSN(BaseCheckpointingBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
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
        x_seq = conv2d_forward(
            x_seq, weight, bias, stride, padding, dilation, groups
        )
        return PSN.forward_function(x_seq, neuron_weight, neuron_bias)

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class Conv2dSlidingPSN(BaseCheckpointingBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
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
        x_seq = conv2d_forward(
            x_seq, weight, bias, stride, padding, dilation, groups
        )
        return SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class Conv2dBNLIF(BaseCheckpointingBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
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
        x_seq = conv2d_bn_forward(
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
            momentum=0.1 if in_backward else 0.
        )
        return neuron(x_seq)

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class Conv2dBNPSN(BaseCheckpointingBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
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
        x_seq = conv2d_bn_forward(
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
            momentum=0.1 if in_backward else 0.
        )
        return PSN.forward_function(x_seq, neuron_weight, neuron_bias)

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class Conv2dBNSlidingPSN(BaseCheckpointingBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
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
        x_seq = conv2d_bn_forward(
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
            momentum=0.1 if in_backward else 0.
        )
        return SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class Conv2dBNRepeatLIFMaxPool2d(BaseCheckpointingBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
        T: int,
        neuron: nn.Module,
        pool: nn.MaxPool2d,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
        self.T = T
        self.neuron = neuron
        self.pool = pool
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x,
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
        T,
        neuron,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        pool_dilation,
        in_backward=False
    ):
        x = conv2d_bn_ann_forward(
            x,
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
            momentum=0.1 if in_backward else 0.
        )
        x_seq = x.repeat(T, *[1 for _ in range(x.ndim)])
        x_seq = neuron(x_seq)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.max_pool2d(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )
        return x_seq.reshape(T, -1, *x_seq.shape[1:])

    def forward(self, x: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x,
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
            self.T,
            self.neuron,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
            self.pool.dilation,
        )


class Conv2dBNRepeatPSNMaxPool2d(BaseCheckpointingBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
        T: int,
        neuron: nn.Module,
        pool: nn.MaxPool2d,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
        self.T = T
        self.neuron = neuron
        self.pool = pool
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x,
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
        T,
        neuron_weight,
        neuron_bias,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        pool_dilation,
        in_backward=False
    ):
        x = conv2d_bn_ann_forward(
            x,
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
            momentum=0.1 if in_backward else 0.
        )
        x_seq = x.repeat(T, *[1 for _ in range(x.ndim)])
        x_seq = PSN.forward_function(x_seq, neuron_weight, neuron_bias)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.max_pool2d(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )
        return x_seq.reshape(T, -1, *x_seq.shape[1:])

    def forward(self, x: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x,
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
            self.T,
            self.neuron.weight,
            self.neuron.bias,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
            self.pool.dilation,
        )


class Conv2dBNSlidingRepeatPSNMaxPool2d(BaseCheckpointingBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
        T: int,
        neuron: nn.Module,
        pool: nn.MaxPool2d,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
        self.T = T
        self.neuron = neuron
        self.pool = pool
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x,
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
        T,
        neuron_weight,
        neuron_bias,
        neuron_k,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        pool_dilation,
        in_backward=False
    ):
        x = conv2d_bn_ann_forward(
            x,
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
            momentum=0.1 if in_backward else 0.
        )
        x_seq = x.repeat(T, *[1 for _ in range(x.ndim)])
        x_seq = SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.max_pool2d(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )
        return x_seq.reshape(T, -1, *x_seq.shape[1:])

    def forward(self, x: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x,
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
            self.T,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
            self.pool.dilation,
        )
