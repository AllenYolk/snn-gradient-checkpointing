import torch
import torch.nn as nn

from ..compress import *
from ..neuron import SlidingPSN, PSN
from ..kernels import *
from .checkpointing import InputCompressedGCFunction, BaseGCBlock, BaseTCGCBlock


class SSACoreLIF(BaseGCBlock):

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
        # qkv.shape = [3, T, B, num_heads, num_patches, C//num_heads]
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]  # [T, B, num_heads, num_patches, C//num_heads]

        x = k.transpose(-2, -1) @ v
        x = (q@x) * scale
        x = x.transpose(-1, -2)  # [T, B, num_heads, C//num_heads, num_patches]
        x = x.reshape(x.shape[0], x.shape[1], -1, x.shape[-1])
        return neuron(x)

    def forward(self, qkv: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            qkv,
            self.scale,
            self.neuron,
        )


class TCSSACoreLIF(BaseTCGCBlock):

    def __init__(
        self,
        scale: float,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor(),
        n_chunk: int = 2,
    ):
        super().__init__(n_chunk)
        self.scale = scale
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(qkv, scale, neuron, v_neuron, in_backward=False):
        # qkv.shape = [3, Tc, B, num_heads, num_patches, C//num_heads]
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]  # [Tc, B, num_heads, num_patches, C//num_heads]

        x = k.transpose(-2, -1) @ v
        x = (q@x) * scale
        x = x.transpose(-1, -2)  # [Tc, B, num_heads, C//num_heads, num_patches]
        x = x.reshape(x.shape[0], x.shape[1], -1, x.shape[-1])
        return neuron.rnn_forward(x, v_neuron)  # s_seq, v_neuron

    def forward(self, qkv: torch.Tensor):
        qkv_seqs = torch.chunk(qkv, self.n_chunk, dim=1)
        T, B, H, L, Cph = qkv.shape[1:]
        v_neuron = torch.zeros([B, H * Cph, L], device=qkv.device)
        out_seq = []
        for qkv_c in qkv_seqs:
            out_c, v_neuron = InputCompressedGCFunction.apply(
                self.conventional_forward,
                self.spike_compressor,
                qkv_c,
                self.scale,
                self.neuron,
                v_neuron,
            )
            out_seq.append(out_c)
        return torch.cat(out_seq, dim=0)


class SSACorePSN(BaseGCBlock):

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
        # qkv.shape = [3, T, B, num_heads, num_patches, C//num_heads]
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]  # [T, B, num_heads, num_patches, C//num_heads]

        x = k.transpose(-2, -1) @ v
        x = (q@x) * scale
        x = x.transpose(-1, -2)  # [T, B, num_heads, C//num_heads, num_patches]
        x = x.reshape(x.shape[0], x.shape[1], -1, x.shape[-1])
        return PSN.forward_function(x, neuron_weight, neuron_bias)

    def forward(self, qkv: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            qkv,
            self.scale,
            self.neuron.weight,
            self.neuron.bias,
        )


class SSACoreSlidingPSN(BaseGCBlock):

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
        # qkv.shape = [3, T, B, num_heads, num_patches, C//num_heads]
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]  # [T, B, num_heads, num_patches, C//num_heads]

        x = k.transpose(-2, -1) @ v
        x = (q@x) * scale
        x = x.transpose(-1, -2)  # [T, B, num_heads, C//num_heads, num_patches]
        x = x.reshape(x.shape[0], x.shape[1], -1, x.shape[-1])
        return SlidingPSN.forward_function(
            x, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, qkv: torch.Tensor):
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            qkv,
            self.scale,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
        )


class QKACoreLIF(BaseGCBlock):

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
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            qkv,
            self.neuron,
        )


class QKACorePSN(BaseGCBlock):

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
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            qkv,
            self.neuron.weight,
            self.neuron.bias,
        )


class QKACoreSlidingPSN(BaseGCBlock):

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
        return InputCompressedGCFunction.apply(
            self.conventional_forward,
            self.spike_compressor,
            qkv,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
        )
