"""Lossless spike compression tools.
"""
import abc
import torch

from ..amp import AUTOCAST_DTYPE, is_autocast_enabled
from ..kernels import *
from .nvcomp_compressor import *


def get_spike_compressor(spike_compressor: str):
    return globals()[spike_compressor]()


class BaseSpikeCompressor(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        pass

    def compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._compress(s_seq)

    def decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        with torch.no_grad():
            return self._decompress(s_seq, shape)


class NullSpikeCompressor(BaseSpikeCompressor):
    """Similar to IdentitySpikeCompressor, but the decompressed tensor must have
    the same dtype as the original one. 
    
    NullSpikeCompressor is used for dealing with non-binary tensors. It is the 
    only "spike compressor" module that can deal with non-binary tensors 
    losslessly (actually, we shouldn't call is a "spike" compressor). For 
    instance, the input layer should always use NullSpikeCompressor, as its 
    input is a float tensor rather than a spike tensor.
    """
    requires_strictly_binary = False

    def __init__(self):
        super().__init__()

    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        return s_seq

    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        return s_seq


class IdentitySpikeCompressor(BaseSpikeCompressor):
    """Similar to NullSpikeCompressor, but the decompressed tensor might have
    a dtype that is different from the original tensor. 
    
    IdentitySpikeCompressor is more memory-efficient than NullSpikeCompressor 
    if amp is enabled, as it decompresses the tensor to low-precision float even
    if the original tensor is with float32 dtype.
    """
    requires_strictly_binary = False

    def __init__(self):
        super().__init__()

    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        return s_seq

    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        ac = is_autocast_enabled()
        decompressed_type = AUTOCAST_DTYPE if ac else torch.float32
        return s_seq.to(dtype=decompressed_type)


class BooleanSpikeCompressor(BaseSpikeCompressor):

    requires_strictly_binary = True

    def __init__(self):
        super().__init__()

    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        return s_seq.to(dtype=torch.bool)

    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        ac = is_autocast_enabled()
        decompressed_type = AUTOCAST_DTYPE if ac else torch.float32
        return s_seq.to(dtype=decompressed_type).reshape(shape)


class Uint8SpikeCompressor(BaseSpikeCompressor):

    requires_strictly_binary = False

    def __init__(self):
        super().__init__()

    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        return s_seq.to(dtype=torch.uint8)

    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        ac = is_autocast_enabled()
        decompressed_type = AUTOCAST_DTYPE if ac else torch.float32
        return s_seq.to(dtype=decompressed_type).reshape(shape)


class BitSpikeCompressor(BaseSpikeCompressor):

    requires_strictly_binary = True

    def __init__(self):
        super().__init__()

    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        # s_seq: float32
        return bit_spike_compress(s_seq)

    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        s_seq_decompressed = bit_spike_decompress(s_seq, shape)

        ac = is_autocast_enabled()
        decompressed_type = AUTOCAST_DTYPE if ac else torch.float32
        return s_seq_decompressed.to(dtype=decompressed_type)


class NvcompSpikeCompressor(BaseSpikeCompressor):

    requires_strictly_binary = False

    def __init__(self):
        super().__init__()
        self.codec = NvcompCompressor(
            algorithm=DEFAULT_NVCOMP_CODEC_ALGORITHM,
            compressed_dtype=torch.uint8
        )

    def _compress(self, s_seq: torch.Tensor) -> nvcomp.Array:
        s_seq = s_seq.reshape(-1)
        self.target_dtype = s_seq.dtype
        s_seq_compressed = self.codec.compress(s_seq)
        return s_seq_compressed

    def _decompress(self, s_seq: nvcomp.Array, shape) -> torch.Tensor:
        s_seq = self.codec.decompress(
            s_seq, target_shape=(-1,), target_dtype=self.target_dtype
        )
        s_seq_compressed = s_seq.reshape(shape)

        ac = is_autocast_enabled()
        decompressed_type = AUTOCAST_DTYPE if ac else torch.float32
        return s_seq_compressed.to(dtype=decompressed_type)


class BitNvcompSpikeCompressor(BaseSpikeCompressor):

    requires_strictly_binary = True

    def __init__(self):
        super().__init__()
        self.codec = NvcompCompressor(
            algorithm=DEFAULT_NVCOMP_CODEC_ALGORITHM,
            compressed_dtype=torch.uint8
        )

    def _compress(self, s_seq: torch.Tensor) -> nvcomp.Array:
        s_seq_compressed = bit_spike_compress(s_seq)
        s_seq_compressed = self.codec.compress(s_seq_compressed)
        return s_seq_compressed

    def _decompress(self, s_seq: nvcomp.Array, shape) -> torch.Tensor:
        s_seq = self.codec.decompress(
            s_seq, target_shape=(-1,), target_dtype=torch.uint8
        ) + 0  #? An error occurs if s_seq is directly handled by triton. `+0` is a workaround.
        s_seq_decompressed = bit_spike_decompress(s_seq, shape)

        ac = is_autocast_enabled()
        decompressed_type = AUTOCAST_DTYPE if ac else torch.float32
        return s_seq_decompressed.to(dtype=decompressed_type)


class SparseSpikeCompressor(BaseSpikeCompressor):

    requires_strictly_binary = True

    def __init__(self, dtype=torch.int64):
        super().__init__()
        self.dtype = dtype

    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        indices = torch.nonzero(s_seq.reshape(-1))
        return indices.to(dtype=self.dtype)

    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        ac = is_autocast_enabled()
        decompressed_type = AUTOCAST_DTYPE if ac else torch.float32
        s_seq_decompressed = torch.zeros(
            shape.numel(), dtype=decompressed_type, device=s_seq.device
        )
        s_seq_decompressed = s_seq_decompressed.scatter_(
            dim=0,
            index=s_seq.to(dtype=torch.int64).reshape(-1),
            value=1,
        )
        return s_seq_decompressed.reshape(shape)
