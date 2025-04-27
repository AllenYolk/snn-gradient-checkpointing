import torch
import torch.nn as nn

from ..compress import *
from ..neuron import SlidingPSN, PSN
from ..kernels import *
from .checkpointing import SNNCheckpointingBlockFunction, BaseCheckpointingBlock
from ..tebn import TEBNProjection


class BNLIF(BaseCheckpointingBlock):

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
        x_seq = bn_forward(
            x_seq,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.1 if in_backward else 0.,
        )
        return neuron(x_seq)

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class BNLIFAvgPool2d(BaseCheckpointingBlock):

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
        x_seq = bn_forward(
            x_seq,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.1 if in_backward else 0.,
        )
        x_seq = neuron(x_seq)
        return avgpool2d_forward(
            x_seq, pool_kernel_size, pool_stride, pool_padding
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class BNLIFMaxPool2d(BaseCheckpointingBlock):

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
        x_seq = bn_forward(
            x_seq,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.1 if in_backward else 0.,
        )
        x_seq = neuron(x_seq)
        return maxpool2d_forward(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class BNPSN(BaseCheckpointingBlock):

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
        x_seq = bn_forward(
            x_seq,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.1 if in_backward else 0.,
        )
        return PSN.forward_function(x_seq, neuron_weight, neuron_bias)

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class BNPSNAvgPool2d(BaseCheckpointingBlock):

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
        x_seq = bn_forward(
            x_seq,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.1 if in_backward else 0.,
        )
        x_seq = PSN.forward_function(x_seq, neuron_weight, neuron_bias)
        return avgpool2d_forward(
            x_seq, pool_kernel_size, pool_stride, pool_padding
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class BNPSNMaxPool2d(BaseCheckpointingBlock):

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
        x_seq = bn_forward(
            x_seq,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.1 if in_backward else 0.,
        )
        x_seq = PSN.forward_function(x_seq, neuron_weight, neuron_bias)
        return maxpool2d_forward(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class BNSlidingPSN(BaseCheckpointingBlock):

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
        x_seq = bn_forward(
            x_seq,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.1 if in_backward else 0.,
        )
        return SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class BNSlidingPSNAvgPool2d(BaseCheckpointingBlock):

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
        x_seq = bn_forward(
            x_seq,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.1 if in_backward else 0.,
        )
        x_seq = SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )
        return avgpool2d_forward(
            x_seq, pool_kernel_size, pool_stride, pool_padding
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class BNSlidingPSNMaxPool2d(BaseCheckpointingBlock):

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
        x_seq = bn_forward(
            x_seq,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.1 if in_backward else 0.,
        )
        x_seq = SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )
        return maxpool2d_forward(
            x_seq, pool_kernel_size, pool_stride, pool_padding, pool_dilation
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class TEBNLIF(BaseCheckpointingBlock):

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
        x_seq = tebn_forward(
            x_seq,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.1 if in_backward else 0.,
            tebn_proj_weight=tebn_proj_weight,
        )
        return neuron(x_seq)

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class TEBNLIFAvgPool2d(BaseCheckpointingBlock):

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
        x_seq = tebn_forward(
            x_seq,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.1 if in_backward else 0.,
            tebn_proj_weight=tebn_proj_weight,
        )
        x_seq = neuron(x_seq)
        return avgpool2d_forward(
            x_seq, pool_kernel_size, pool_stride, pool_padding
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class TEBNSlidingPSN(BaseCheckpointingBlock):

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
        x_seq = tebn_forward(
            x_seq,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.1 if in_backward else 0.,
            tebn_proj_weight=tebn_proj_weight,
        )
        return SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class TEBNSlidingPSNAvgPool2d(BaseCheckpointingBlock):

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
        x_seq = tebn_forward(
            x_seq,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.1 if in_backward else 0.,
            tebn_proj_weight=tebn_proj_weight,
        )
        x_seq = SlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )
        return avgpool2d_forward(
            x_seq, pool_kernel_size, pool_stride, pool_padding
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class TEBNProjectionLIF(BaseCheckpointingBlock):

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
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
            self.tebn_proj.p,
            self.neuron,
        )


class TEBNProjectionLIFAvgPool2d(BaseCheckpointingBlock):

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
        return avgpool2d_forward(
            x_seq, pool_kernel_size, pool_stride, pool_padding
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
            self.tebn_proj.p,
            self.neuron,
            self.pool.kernel_size,
            self.pool.stride,
            self.pool.padding,
        )


class TEBNProjectionSlidingPSN(BaseCheckpointingBlock):

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
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            get_spike_compressor("NullSpikeCompressor"),
            x_seq,
            self.tebn_proj.p,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
        )


class TEBNProjectionSlidingPSNAvgPool2d(BaseCheckpointingBlock):

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
        return avgpool2d_forward(
            x_seq, pool_kernel_size, pool_stride, pool_padding
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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


class SlidingPSNAvgPool2d(BaseCheckpointingBlock):

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
        return avgpool2d_forward(
            x_seq, pool_kernel_size, pool_stride, pool_padding
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
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
