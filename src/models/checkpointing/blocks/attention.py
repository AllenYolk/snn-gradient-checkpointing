import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseCheckpointingBlock
from ...compress import *
from ...neuron import SlidingPSN, PSN
from ...kernels import *
from ..checkpointing import SNNCheckpointingBlockFunction


class SSACoreLIF(BaseCheckpointingBlock):

    def __init__(
        self,
        scale: float,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.scale = scale
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(qkv, scale, neuron, in_backward=False):
        x = ssa_core_forward(qkv, scale)
        return neuron(x)

    def forward(self, qkv: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            qkv,
            self.scale,
            self.neuron,
        )


class SSACorePSN(BaseCheckpointingBlock):

    def __init__(
        self,
        scale: float,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.scale = scale
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        qkv, scale, neuron_weight, neuron_bias, in_backward=False
    ):
        x = ssa_core_forward(qkv, scale)
        return PSN.forward_function(x, neuron_weight, neuron_bias)

    def forward(self, qkv: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            qkv,
            self.scale,
            self.neuron.weight,
            self.neuron.bias,
        )


class SSACoreSlidingPSN(BaseCheckpointingBlock):

    def __init__(
        self,
        scale: float,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.scale = scale
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        qkv, scale, neuron_weight, neuron_bias, neuron_k, in_backward=False
    ):
        x = ssa_core_forward(qkv, scale)
        return SlidingPSN.forward_function(
            x, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, qkv: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            qkv,
            self.scale,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
        )
