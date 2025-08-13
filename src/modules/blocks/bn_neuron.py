import torch
import torch.nn as nn
import torch.nn.functional as F

from ..compress import *
from ..neuron import SlidingPSN, PSN
from ..kernels import *
from .checkpointing import InputCompressedGCFunction, BaseGCBlock
from ..tebn import TEBNProjection


class BNLIF(BaseGCBlock):

    def __init__(
        self,
        bn: nn.BatchNorm2d,
        neuron: nn.Module,
    ):
        super().__init__()
        self.bn = bn
        self.neuron = neuron

    @staticmethod
    def conventional_forward(
        x_seq,
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
        return neuron(x_seq)

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
            self.neuron,
        )


class BNLIFAvgPool2d(BaseGCBlock):

    def __init__(
        self,
        bn: nn.BatchNorm2d,
        neuron: nn.Module,
        pool: nn.AvgPool2d,
    ):
        super().__init__()
        self.bn = bn
        self.neuron = neuron
        self.pool = pool

    @staticmethod
    def conventional_forward(
        x_seq,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        training,
        neuron,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
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
        x_seq = neuron(x_seq)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.avg_pool2d(x_seq, pool_kernel_size, pool_stride, pool_padding)
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
            self.neuron,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
        )


class BNLIFMaxPool2d(BaseGCBlock):

    def __init__(
        self,
        bn: nn.BatchNorm2d,
        neuron: nn.Module,
        pool: nn.MaxPool2d,
    ):
        super().__init__()
        self.bn = bn
        self.neuron = neuron
        self.pool = pool

    @staticmethod
    def conventional_forward(
        x_seq,
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
        x_seq = neuron(x_seq)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.max_pool2d(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
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


class BNPSN(BaseGCBlock):

    def __init__(
        self,
        bn: nn.BatchNorm2d,
        neuron: nn.Module,
    ):
        super().__init__()
        self.bn = bn
        self.neuron = neuron

    @staticmethod
    def conventional_forward(
        x_seq,
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
        return PSN.forward_function(x_seq, neuron_weight, neuron_bias)

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
            self.neuron.weight,
            self.neuron.bias,
        )


class BNPSNAvgPool2d(BaseGCBlock):

    def __init__(
        self,
        bn: nn.BatchNorm2d,
        neuron: nn.Module,
        pool: nn.AvgPool2d,
    ):
        super().__init__()
        self.bn = bn
        self.neuron = neuron
        self.pool = pool

    @staticmethod
    def conventional_forward(
        x_seq,
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
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
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
        x_seq = PSN.forward_function(x_seq, neuron_weight, neuron_bias)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.avg_pool2d(x_seq, pool_kernel_size, pool_stride, pool_padding)
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
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
        )


class BNPSNMaxPool2d(BaseGCBlock):

    def __init__(
        self,
        bn: nn.BatchNorm2d,
        neuron: nn.Module,
        pool: nn.MaxPool2d,
    ):
        super().__init__()
        self.bn = bn
        self.neuron = neuron
        self.pool = pool

    @staticmethod
    def conventional_forward(
        x_seq,
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
        x_seq = PSN.forward_function(x_seq, neuron_weight, neuron_bias)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.max_pool2d(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
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


class BNSlidingPSN(BaseGCBlock):

    def __init__(
        self,
        bn: nn.BatchNorm2d,
        neuron: nn.Module,
    ):
        super().__init__()
        self.bn = bn
        self.neuron = neuron

    @staticmethod
    def conventional_forward(
        x_seq,
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
        return SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
        )


class BNSlidingPSNAvgPool2d(BaseGCBlock):

    def __init__(
        self,
        bn: nn.BatchNorm2d,
        neuron: nn.Module,
        pool: nn.AvgPool2d,
    ):
        super().__init__()
        self.bn = bn
        self.neuron = neuron
        self.pool = pool

    @staticmethod
    def conventional_forward(
        x_seq,
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
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
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
        x_seq = SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.avg_pool2d(x_seq, pool_kernel_size, pool_stride, pool_padding)
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
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
        )


class BNSlidingPSNMaxPool2d(BaseGCBlock):

    def __init__(
        self,
        bn: nn.BatchNorm2d,
        neuron: nn.Module,
        pool: nn.MaxPool2d,
    ):
        super().__init__()
        self.bn = bn
        self.neuron = neuron
        self.pool = pool

    @staticmethod
    def conventional_forward(
        x_seq,
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
        x_seq = SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.max_pool2d(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
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


class TEBNLIF(BaseGCBlock):

    def __init__(
        self,
        bn: nn.BatchNorm2d,
        tebn_proj: TEBNProjection,
        neuron: nn.Module,
    ):
        super().__init__()
        self.bn = bn
        self.tebn_proj = tebn_proj
        self.neuron = neuron

    @staticmethod
    def conventional_forward(
        x_seq,
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
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
            self.tebn_proj.p,
            self.neuron,
        )


class TEBNLIFAvgPool2d(BaseGCBlock):

    def __init__(
        self,
        bn: nn.BatchNorm2d,
        tebn_proj: TEBNProjection,
        neuron: nn.Module,
        pool: nn.AvgPool2d,
    ):
        super().__init__()
        self.bn = bn
        self.tebn_proj = tebn_proj
        self.neuron = neuron
        self.pool = pool

    @staticmethod
    def conventional_forward(
        x_seq,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        training,
        tebn_proj_weight,
        neuron,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
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
        x_seq = neuron(x_seq)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.avg_pool2d(x_seq, pool_kernel_size, pool_stride, pool_padding)
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
            self.tebn_proj.p,
            self.neuron,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
        )


class TEBNSlidingPSN(BaseGCBlock):

    def __init__(
        self,
        bn: nn.BatchNorm2d,
        tebn_proj: TEBNProjection,
        neuron: nn.Module,
    ):
        super().__init__()
        self.bn = bn
        self.tebn_proj = tebn_proj
        self.neuron = neuron

    @staticmethod
    def conventional_forward(
        x_seq,
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
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
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


class TEBNSlidingPSNAvgPool2d(BaseGCBlock):

    def __init__(
        self,
        bn: nn.BatchNorm2d,
        tebn_proj: TEBNProjection,
        neuron: nn.Module,
        pool: nn.AvgPool2d,
    ):
        super().__init__()
        self.bn = bn
        self.tebn_proj = tebn_proj
        self.neuron = neuron
        self.pool = pool

    @staticmethod
    def conventional_forward(
        x_seq,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        training,
        tebn_proj_weight,
        neuron_weight,
        neuron_bias,
        neuron_k,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        in_backward=False
    ):
        T, N = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.flatten(0, 1)
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
        x_seq = SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.avg_pool2d(x_seq, pool_kernel_size, pool_stride, pool_padding)
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
            self.tebn_proj.p,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
        )


class TEBNProjectionLIF(BaseGCBlock):

    def __init__(
        self,
        tebn_proj: TEBNProjection,
        neuron: nn.Module,
    ):
        super().__init__()
        self.tebn_proj = tebn_proj
        self.neuron = neuron

    @staticmethod
    def conventional_forward(
        x_seq, tebn_proj_weight, neuron, in_backward=False
    ):
        x_seq = x_seq * tebn_proj_weight
        return neuron(x_seq)

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
            self.tebn_proj.p,
            self.neuron,
        )


class TEBNProjectionLIFAvgPool2d(BaseGCBlock):

    def __init__(
        self,
        tebn_proj: TEBNProjection,
        neuron: nn.Module,
        pool: nn.AvgPool2d,
    ):
        super().__init__()
        self.tebn_proj = tebn_proj
        self.neuron = neuron
        self.pool = pool

    @staticmethod
    def conventional_forward(
        x_seq,
        tebn_proj_weight,
        neuron,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        in_backward=False
    ):
        x_seq = x_seq * tebn_proj_weight
        x_seq = neuron(x_seq)
        T = x_seq.size(0)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.avg_pool2d(x_seq, pool_kernel_size, pool_stride, pool_padding)
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
            self.tebn_proj.p,
            self.neuron,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
        )


class TEBNProjectionSlidingPSN(BaseGCBlock):

    def __init__(
        self,
        tebn_proj: TEBNProjection,
        neuron: nn.Module,
    ):
        super().__init__()
        self.tebn_proj = tebn_proj
        self.neuron = neuron

    @staticmethod
    def conventional_forward(
        x_seq,
        tebn_proj_weight,
        neuron_weight,
        neuron_bias,
        neuron_k,
        in_backward=False
    ):
        x_seq = x_seq * tebn_proj_weight
        return SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
            self.tebn_proj.p,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
        )


class TEBNProjectionSlidingPSNAvgPool2d(BaseGCBlock):

    def __init__(
        self,
        tebn_proj: TEBNProjection,
        neuron: nn.Module,
        pool: nn.AvgPool2d,
    ):
        super().__init__()
        self.tebn_proj = tebn_proj
        self.neuron = neuron
        self.pool = pool

    @staticmethod
    def conventional_forward(
        x_seq,
        tebn_proj_weight,
        neuron_weight,
        neuron_bias,
        neuron_k,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        in_backward=False
    ):
        x_seq = x_seq * tebn_proj_weight
        x_seq = SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )
        T = x_seq.size(0)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.avg_pool2d(x_seq, pool_kernel_size, pool_stride, pool_padding)
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
            self.tebn_proj.p,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
        )


class PSNOnly(BaseGCBlock):

    def __init__(
        self,
        neuron: nn.Module,
    ):
        super().__init__()
        self.neuron = neuron

    @staticmethod
    def conventional_forward(
        x_seq, neuron_weight, neuron_bias, in_backward=False
    ):
        return PSN.forward_function(x_seq, neuron_weight, neuron_bias)

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
            self.neuron.weight,
            self.neuron.bias,
        )


class PSNMaxPool2d(BaseGCBlock):

    def __init__(
        self,
        neuron: nn.Module,
        pool: nn.MaxPool2d,
    ):
        super().__init__()
        self.neuron = neuron
        self.pool = pool

    @staticmethod
    def conventional_forward(
        x_seq,
        neuron_weight,
        neuron_bias,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        pool_dilation,
        in_backward=False
    ):
        x_seq = PSN.forward_function(x_seq, neuron_weight, neuron_bias)
        T = x_seq.size(0)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.max_pool2d(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"), x_seq,
            self.neuron.weight, self.neuron.bias, self.pool.kernel_size,
            self.pool.stride, self.pool.padding, self.pool.dilation
        )


class PSNAvgPool2d(BaseGCBlock):

    def __init__(
        self,
        neuron: nn.Module,
        pool: nn.AvgPool2d,
    ):
        super().__init__()
        self.neuron = neuron
        self.pool = pool

    @staticmethod
    def conventional_forward(
        x_seq,
        neuron_weight,
        neuron_bias,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        in_backward=False
    ):
        x_seq = PSN.forward_function(x_seq, neuron_weight, neuron_bias)
        T = x_seq.size(0)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.avg_pool2d(x_seq, pool_kernel_size, pool_stride, pool_padding)
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
            self.neuron.weight,
            self.neuron.bias,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
        )


class SlidingPSNOnly(BaseGCBlock):

    def __init__(
        self,
        neuron: nn.Module,
    ):
        super().__init__()
        self.neuron = neuron

    @staticmethod
    def conventional_forward(
        x_seq, neuron_weight, neuron_bias, neuron_k, in_backward=False
    ):
        return SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
        )


class SlidingPSNMaxPool2d(BaseGCBlock):

    def __init__(
        self,
        neuron: nn.Module,
        pool: nn.MaxPool2d,
    ):
        super().__init__()
        self.neuron = neuron
        self.pool = pool

    @staticmethod
    def conventional_forward(
        x_seq,
        neuron_weight,
        neuron_bias,
        neuron_k,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        pool_dilation,
        in_backward=False
    ):
        x_seq = SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )
        T = x_seq.size(0)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.max_pool2d(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
            self.pool.dilation,
        )


class SlidingPSNAvgPool2d(BaseGCBlock):

    def __init__(
        self,
        neuron: nn.Module,
        pool: nn.AvgPool2d,
    ):
        super().__init__()
        self.neuron = neuron
        self.pool = pool

    @staticmethod
    def conventional_forward(
        x_seq,
        neuron_weight,
        neuron_bias,
        neuron_k,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        in_backward=False
    ):
        x_seq = SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )
        T = x_seq.size(0)
        x_seq = x_seq.flatten(0, 1)
        x_seq = F.avg_pool2d(x_seq, pool_kernel_size, pool_stride, pool_padding)
        x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
        return x_seq

    def forward(self, x_seq: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
        )
