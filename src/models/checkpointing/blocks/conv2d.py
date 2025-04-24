import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseCheckpointingBlock
from ...compress import *
from ...neuron import SlidingPSN, PSN
from ...kernels import *
from ..checkpointing import SNNCheckpointingBlockFunction
from ...tebn import TEBNProjection


class Conv2d(BaseCheckpointingBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
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
        return conv2d_forward(
            x_seq,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
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
        )


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


class Conv2dBN(BaseCheckpointingBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
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
        in_backward=False
    ):
        return conv2d_bn_forward(
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


class Conv2dTEBNLIF(BaseCheckpointingBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
        tebn_proj: TEBNProjection,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
        self.tebn_proj = tebn_proj
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
        tebn_proj_weight,
        in_backward=False
    ):
        x_seq = conv2d_tebn_forward(
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
            momentum=0.1 if in_backward else 0.,
            tebn_proj_weight=tebn_proj_weight
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
            self.tebn_proj.p,
        )


class Conv2dTEBNSlidingPSN(BaseCheckpointingBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
        tebn_proj: TEBNProjection,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
        self.tebn_proj = tebn_proj
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
        tebn_proj_weight,
        in_backward=False
    ):
        x_seq = conv2d_tebn_forward(
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
            momentum=0.1 if in_backward else 0.,
            tebn_proj_weight=tebn_proj_weight
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
            self.tebn_proj.p,
        )


class AvgPool2dConv2dBNLIF(BaseCheckpointingBlock):

    def __init__(
        self,
        pool: nn.AvgPool2d,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
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
        x_seq = avgpool2d_conv2d_bn_forward(
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
        return neuron(x_seq)

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
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


class AvgPool2dConv2dBNPSN(BaseCheckpointingBlock):

    def __init__(
        self,
        pool: nn.AvgPool2d,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
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
        x_seq = avgpool2d_conv2d_bn_forward(
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
        return PSN.forward_function(x_seq, neuron_weight, neuron_bias)

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
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


class AvgPool2dConv2dBNSlidingPSN(BaseCheckpointingBlock):

    def __init__(
        self,
        pool: nn.AvgPool2d,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
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
        x_seq = avgpool2d_conv2d_bn_forward(
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
            x_seq, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
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


class AvgPool2dConv2dTEBNLIF(BaseCheckpointingBlock):

    def __init__(
        self,
        pool: nn.AvgPool2d,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
        tebn_proj: TEBNProjection,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.pool = pool
        self.proj = proj
        self.bn = bn
        self.tebn_proj = tebn_proj
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
        tebn_proj_weight,
        in_backward=False
    ):
        x_seq = avgpool2d_conv2d_tebn_forward(
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
            momentum=0.1 if in_backward else 0.,
            tebn_proj_weight=tebn_proj_weight
        )
        return neuron(x_seq)

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
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
            self.tebn_proj.p,
        )


class AvgPool2dConv2dTEBNSlidingPSN(BaseCheckpointingBlock):

    def __init__(
        self,
        pool: nn.AvgPool2d,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
        tebn_proj: TEBNProjection,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.pool = pool
        self.proj = proj
        self.bn = bn
        self.tebn_proj = tebn_proj
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
        tebn_proj_weight,
        in_backward=False
    ):
        x_seq = avgpool2d_conv2d_tebn_forward(
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
            momentum=0.1 if in_backward else 0.,
            tebn_proj_weight=tebn_proj_weight
        )
        return SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
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
            self.tebn_proj.p,
        )


class Conv2dBNLIFMaxPool2d(BaseCheckpointingBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
        neuron: nn.Module,
        pool: nn.MaxPool2d,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
        self.neuron = neuron
        self.pool = pool
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
        pool_kernel_size,
        pool_stride,
        pool_padding,
        pool_dilation,
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
        x_seq = neuron(x_seq)
        return maxpool2d_forward(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )

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
            self.neuron,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
            self.pool.dilation,
        )


class Conv2dBNPSNMaxPool2d(BaseCheckpointingBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
        neuron: nn.Module,
        pool: nn.MaxPool2d,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
        self.neuron = neuron
        self.pool = pool
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
        pool_kernel_size,
        pool_stride,
        pool_padding,
        pool_dilation,
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
        x_seq = PSN.forward_function(x_seq, neuron_weight, neuron_bias)
        return maxpool2d_forward(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )

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
            self.neuron.weight,
            self.neuron.bias,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
            self.pool.dilation,
        )


class Conv2dBNSlidingPSNMaxPool2d(BaseCheckpointingBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
        neuron: nn.Module,
        pool: nn.MaxPool2d,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
        self.neuron = neuron
        self.pool = pool
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
        pool_kernel_size,
        pool_stride,
        pool_padding,
        pool_dilation,
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
        x_seq = SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )
        return maxpool2d_forward(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )

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
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
            self.pool.dilation,
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
        return maxpool2d_forward(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )

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
        return maxpool2d_forward(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )

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


class Conv2dBNRepeatSlidingPSNMaxPool2d(BaseCheckpointingBlock):

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
        return maxpool2d_forward(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )

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


class MaxPool2d(BaseCheckpointingBlock):

    def __init__(
        self,
        pool: nn.MaxPool2d,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.pool = pool
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        pool_dilation,
        in_backward=False
    ):
        return maxpool2d_forward(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )

    def forward(self, x: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            x,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
            self.pool.dilation,
        )
