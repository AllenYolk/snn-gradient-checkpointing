"""Lossless spike compression tools.
"""
import abc
import torch

from ..amp import AUTOCAST_DTYPE, is_autocast_enabled


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

    def __init__(self):
        super().__init__()

    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        return s_seq

    def _decompress(self, s_seq: torch.Tensor) -> torch.Tensor:
        return s_seq


class IdentitySpikeCompressor(BaseSpikeCompressor):
    """Similar to NullSpikeCompressor, but the decompressed tensor might have
    a dtype that is different from the original tensor. 
    
    IdentitySpikeCompressor is more memory-efficient than NullSpikeCompressor 
    if amp is enabled, as it decompresses the tensor to low-precision float even
    if the original tensor is with float32 dtype.
    """

    def __init__(self):
        super().__init__()

    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        return s_seq

    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        ac = is_autocast_enabled()
        decompressed_type = AUTOCAST_DTYPE if ac else torch.float32
        return s_seq.to(dtype=decompressed_type)


class BooleanSpikeCompressor(BaseSpikeCompressor):

    def __init__(self):
        super().__init__()

    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        return s_seq.to(dtype=torch.bool)

    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        ac = is_autocast_enabled()
        decompressed_type = AUTOCAST_DTYPE if ac else torch.float32
        return s_seq.to(dtype=decompressed_type).reshape(shape)


class Uint8SpikeCompressor(BaseSpikeCompressor):

    def __init__(self):
        super().__init__()

    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        return s_seq.to(dtype=torch.uint8)

    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        ac = is_autocast_enabled()
        decompressed_type = AUTOCAST_DTYPE if ac else torch.float32
        return s_seq.to(dtype=decompressed_type).reshape(shape)


class BitSpikeCompressor(BaseSpikeCompressor):

    def __init__(self):
        super().__init__()

    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        s_seq = s_seq.to(dtype=torch.bool).reshape(-1)
        compressed_shape = (s_seq.numel() + 7) // 8
        s_seq_compressed = torch.zeros(
            compressed_shape, dtype=torch.uint8, device=s_seq.device
        )
        for i in range(8):
            sliced = s_seq[i::8].to(dtype=torch.uint8)
            sliced_len = sliced.numel()
            if sliced_len > 0:
                s_seq_compressed[:sliced_len] |= (sliced << i)
        return s_seq_compressed

    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        decompressed_len = shape.numel()
        s_seq_decompressed = torch.zeros(
            decompressed_len, dtype=torch.bool, device=s_seq.device
        )
        for i in range(8):
            sliced_len = (decompressed_len-i+7) // 8
            sliced = ((s_seq >> i) & 1)[:sliced_len]
            s_seq_decompressed[i::8] = sliced
        s_seq_compressed = s_seq_decompressed.reshape(shape)

        ac = is_autocast_enabled()
        decompressed_type = AUTOCAST_DTYPE if ac else torch.float32
        return s_seq_compressed.to(dtype=decompressed_type)


class SparseSpikeCompressor(BaseSpikeCompressor):

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
