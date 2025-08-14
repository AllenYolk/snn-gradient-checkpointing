import torch
import torch.nn as nn
import torch.nn.functional as F

from ..compress import *
from ..neuron import SlidingPSN, PSN
from ..kernels import *
from .checkpointing import InputCompressedGCFunction, BaseGCBlock, BaseTCGCBlock
from ..tebn import TEBNProjection


class Conv2d(BaseGCBlock):

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
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
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


class TCConv2d(BaseTCGCBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
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
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
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


class Conv2dLIF(BaseGCBlock):

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
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
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
            self.neuron,
        )


class TCConv2dLIF(BaseTCGCBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
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
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
        return neuron.rnn_forward(x_seq, v)  # s_seq, v

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


class Conv2dPSN(BaseGCBlock):

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
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
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
            self.neuron.weight,
            self.neuron.bias,
        )


class Conv2dSlidingPSN(BaseGCBlock):

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
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
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
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
        )


class Conv2dBN(BaseGCBlock):

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
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
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
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
        )


class Conv2dBNLIF(BaseGCBlock):

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
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
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


class Conv2dBNPSN(BaseGCBlock):

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
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
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


class Conv2dBNSlidingPSN(BaseGCBlock):

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
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
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


class Conv2dTEBN(BaseGCBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
        tebn_proj: TEBNProjection,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
        self.tebn_proj = tebn_proj
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
        tebn_proj_weight,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = F.batch_norm(
            x_seq,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=training,
            momentum=0.1 if in_backward else 0.,
        )
        y_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
        return y_seq * tebn_proj_weight

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
            self.tebn_proj.p,
        )


class Conv2dTEBNLIF(BaseGCBlock):

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
        tebn_proj_weight,
        neuron,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = F.batch_norm(
            x_seq,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=training,
            momentum=0.1 if in_backward else 0.,
        )
        x_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
        x_seq = x_seq * tebn_proj_weight
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
            self.tebn_proj.p,
            self.neuron,
        )


class Conv2dTEBNSlidingPSN(BaseGCBlock):

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
        tebn_proj_weight,
        neuron_weight,
        neuron_bias,
        neuron_k,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = F.batch_norm(
            x_seq,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=training,
            momentum=0.1 if in_backward else 0.,
        )
        x_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
        x_seq = x_seq * tebn_proj_weight
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
            self.tebn_proj.p,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
        )


class AvgPool2dConv2dBNLIF(BaseGCBlock):

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
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.avg_pool2d(x_seq, pool_kernel_size, pool_stride, pool_padding)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
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


class AvgPool2dConv2dBNPSN(BaseGCBlock):

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
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.avg_pool2d(x_seq, pool_kernel_size, pool_stride, pool_padding)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
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


class AvgPool2dConv2dBNSlidingPSN(BaseGCBlock):

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
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.avg_pool2d(x_seq, pool_kernel_size, pool_stride, pool_padding)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
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


class AvgPool2dConv2dTEBNLIF(BaseGCBlock):

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
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.avg_pool2d(x_seq, pool_kernel_size, pool_stride, pool_padding)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = F.batch_norm(
            x_seq,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=training,
            momentum=0.1 if in_backward else 0.,
        )
        x_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
        x_seq = x_seq * tebn_proj_weight
        return neuron(x_seq)

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
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


class AvgPool2dConv2dTEBNSlidingPSN(BaseGCBlock):

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
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.avg_pool2d(x_seq, pool_kernel_size, pool_stride, pool_padding)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = F.batch_norm(
            x_seq,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=training,
            momentum=0.1 if in_backward else 0.,
        )
        x_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
        x_seq = x_seq * tebn_proj_weight
        return SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
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


class Conv2dBNLIFMaxPool2d(BaseGCBlock):

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
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
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
        x_seq = neuron(x_seq)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.max_pool2d(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x: torch.Tensor):
        return InputCompressedGCFunction.apply(
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


class Conv2dBNPSNMaxPool2d(BaseGCBlock):

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
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
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
        x_seq = PSN.forward_function(x_seq, neuron_weight, neuron_bias)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.max_pool2d(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x: torch.Tensor):
        return InputCompressedGCFunction.apply(
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


class Conv2dBNSlidingPSNMaxPool2d(BaseGCBlock):

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
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
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
        x_seq = SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.max_pool2d(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x: torch.Tensor):
        return InputCompressedGCFunction.apply(
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


class Conv2dBNMaxPool2d(BaseGCBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
        pool: nn.MaxPool2d,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
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
        pool_kernel_size,
        pool_stride,
        pool_padding,
        pool_dilation,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = F.batch_norm(
            x_seq,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=training,
            momentum=0.1 if in_backward else 0.,
        )
        x_seq = F.max_pool2d(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )
        return x_seq.reshape(T, N, *x_seq.shape[1:])

    def forward(self, x: torch.Tensor):
        return InputCompressedGCFunction.apply(
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
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
            self.pool.dilation,
        )


class Conv2dBNMaxPool2dLIF(BaseGCBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
        pool: nn.MaxPool2d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
        self.pool = pool
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
        pool_kernel_size,
        pool_stride,
        pool_padding,
        pool_dilation,
        neuron,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = F.batch_norm(
            x_seq,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=training,
            momentum=0.1 if in_backward else 0.,
        )
        x_seq = F.max_pool2d(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )
        x_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
        return neuron(x_seq)

    def forward(self, x: torch.Tensor):
        return InputCompressedGCFunction.apply(
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
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
            self.pool.dilation,
            self.neuron,
        )


class Conv2dBNMaxPool2dPSN(BaseGCBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
        pool: nn.MaxPool2d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
        self.pool = pool
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
        pool_kernel_size,
        pool_stride,
        pool_padding,
        pool_dilation,
        neuron_weight,
        neuron_bias,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = F.batch_norm(
            x_seq,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=training,
            momentum=0.1 if in_backward else 0.,
        )
        x_seq = F.max_pool2d(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )
        x_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
        return PSN.forward_function(x_seq, neuron_weight, neuron_bias)

    def forward(self, x: torch.Tensor):
        return InputCompressedGCFunction.apply(
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
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
            self.pool.dilation,
            self.neuron.weight,
            self.neuron.bias,
        )


class Conv2dBNMaxPool2dSlidingPSN(BaseGCBlock):

    def __init__(
        self,
        proj: nn.Conv2d,
        bn: nn.BatchNorm2d,
        pool: nn.MaxPool2d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
        self.pool = pool
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
        pool_kernel_size,
        pool_stride,
        pool_padding,
        pool_dilation,
        neuron_weight,
        neuron_bias,
        neuron_k,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = F.batch_norm(
            x_seq,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=training,
            momentum=0.1 if in_backward else 0.,
        )
        x_seq = F.max_pool2d(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )
        x_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
        return SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, x: torch.Tensor):
        return InputCompressedGCFunction.apply(
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
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
            self.pool.dilation,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
        )


class Conv2dBNRepeatLIFMaxPool2d(BaseGCBlock):

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
        x = F.conv2d(x, weight, bias, stride, padding, dilation, groups)
        x = F.batch_norm(
            x,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=training,
            momentum=0.1 if in_backward else 0.
        )
        x_seq = x.repeat(T, *[1 for _ in range(x.ndim)])
        x_seq = neuron(x_seq)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.max_pool2d(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x: torch.Tensor):
        return InputCompressedGCFunction.apply(
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


class Conv2dBNRepeatPSNMaxPool2d(BaseGCBlock):

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
        x = F.conv2d(x, weight, bias, stride, padding, dilation, groups)
        x = F.batch_norm(
            x,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=training,
            momentum=0.1 if in_backward else 0.
        )
        x_seq = x.repeat(T, *[1 for _ in range(x.ndim)])
        x_seq = PSN.forward_function(x_seq, neuron_weight, neuron_bias)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.max_pool2d(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x: torch.Tensor):
        return InputCompressedGCFunction.apply(
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


class Conv2dBNRepeatSlidingPSNMaxPool2d(BaseGCBlock):

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
        x = F.conv2d(x, weight, bias, stride, padding, dilation, groups)
        x = F.batch_norm(
            x,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=training,
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
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x: torch.Tensor):
        return InputCompressedGCFunction.apply(
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
