import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .base import BaseCheckpointingBlock
from ...compress import *
from ...neuron import SlidingPSN, PSN
from ...kernels import *
from ..checkpointing import SNNCheckpointingBlockFunction
from ...tebn import TEBNProjection


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
        x_seq = tebn_forward(
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
        x_seq = tebn_forward(
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
