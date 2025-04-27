import torch
import torch.nn as nn

from ..compress import *
from ..neuron import SlidingPSN, PSN
from ..kernels import *
from .checkpointing import SNNCheckpointingBlockFunction, BaseCheckpointingBlock


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


class QKACoreLIF(BaseCheckpointingBlock):

    def __init__(
        self,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(qk, neuron, in_backward=False):
        q, k = qk[0], qk[1]  # [T, B, num_heads, C//num_heads, num_patches]
        q = torch.sum(q, dim=3, keepdim=True)
        q = neuron(q)  # [T, B, num_heads, 1, num_patches]; token-wise
        k = torch.mul(q, k)  # [T, B, num_heads, C//num_heads, num_patches]
        return k.flatten(2, 3)  # [T, B, C, num_patches]

    def forward(self, qkv: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            qkv,
            self.neuron,
        )


class QKACorePSN(BaseCheckpointingBlock):

    def __init__(
        self,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(qk, neuron_weight, neuron_bias, in_backward=False):
        q, k = qk[0], qk[1]  # [T, B, num_heads, C//num_heads, num_patches]
        q = torch.sum(q, dim=3, keepdim=True)
        q = PSN.forward_function(q, neuron_weight, neuron_bias)
        k = torch.mul(q, k)  # [T, B, num_heads, C//num_heads, num_patches]
        return k.flatten(2, 3)  # [T, B, C, num_patches]

    def forward(self, qkv: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            qkv,
            self.neuron.weight,
            self.neuron.bias,
        )


class QKACoreSlidingPSN(BaseCheckpointingBlock):

    def __init__(
        self,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        qk, neuron_weight, neuron_bias, neuron_k, in_backward=False
    ):
        q, k = qk[0], qk[1]  # [T, B, num_heads, C//num_heads, num_patches]
        q = torch.sum(q, dim=3, keepdim=True)
        q = SlidingPSN.forward_function(q, neuron_weight, neuron_bias, neuron_k)
        k = torch.mul(q, k)  # [T, B, num_heads, C//num_heads, num_patches]
        return k.flatten(2, 3)  # [T, B, C, num_patches]

    def forward(self, qkv: torch.Tensor):
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            qkv,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
        )
