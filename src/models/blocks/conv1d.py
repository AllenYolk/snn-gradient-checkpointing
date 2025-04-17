import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from .base import BaseCheckpointingBlock
from ..compress import *
from ..neuron import SlidingPSN, PSN
from ..kernels import *
from .checkpointing import SNNCheckpointingBlockFunction


class Conv1dLIF(BaseCheckpointingBlock):

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
        y_seq = conv1d_forward(
            x_seq, weight, bias, stride, padding, dilation, groups
        )
        return neuron(y_seq)

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


class Conv1dPSN(BaseCheckpointingBlock):

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
        x_seq = conv1d_forward(
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


class Conv1dSlidingPSN(BaseCheckpointingBlock):

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
        x_seq = conv1d_forward(
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


class Conv1dBNLIF(BaseCheckpointingBlock):

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
        x_seq = conv1d_bn_forward(
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


class Conv1dBNPSN(BaseCheckpointingBlock):

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
        x_seq = conv1d_bn_forward(
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


class Conv1dBNSlidingPSN(BaseCheckpointingBlock):

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
        x_seq = conv1d_bn_forward(
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


class AvgPool1dConv1dBNLIF(BaseCheckpointingBlock):

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
        y_seq = avgpool1d_conv1d_bn_forward(
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
            momentum=0.1 if in_backward else 0.
        )
        return neuron(y_seq)

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class AvgPool1dConv1dBNPSN(BaseCheckpointingBlock):

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
        y_seq = avgpool1d_conv1d_bn_forward(
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
            momentum=0.1 if in_backward else 0.
        )
        return PSN.forward_function(y_seq, neuron_weight, neuron_bias)

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class AvgPool1dConv1dBNSlidingPSN(BaseCheckpointingBlock):

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

        y_seq = avgpool1d_conv1d_bn_forward(
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
            momentum=0.1 if in_backward else 0.
        )
        return SlidingPSN.forward_function(
            y_seq, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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
