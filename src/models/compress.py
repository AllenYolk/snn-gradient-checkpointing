import abc
import torch


def get_spike_compressor(spike_compressor: str):
    return globals()[spike_compressor]()


class BaseSpikeCompressor(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        pass


class IdentitySpikeCompressor(BaseSpikeCompressor):

    def __init__(self):
        super().__init__()

    def compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        return s_seq

    def decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        return s_seq


class BooleanSpikeCompressor(BaseSpikeCompressor):

    def __init__(self):
        super().__init__()

    def compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        return s_seq.to(dtype=torch.bool)

    def decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        return s_seq.to(dtype=torch.float32).reshape(shape)


class Uint8SpikeCompressor(BaseSpikeCompressor):

    def __init__(self):
        super().__init__()

    def compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        return s_seq.to(dtype=torch.uint8)

    def decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        return s_seq.to(dtype=torch.float32).reshape(shape)


class BitSpikeCompressor(BaseSpikeCompressor):

    def __init__(self):
        super().__init__()

    def compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        s_seq = s_seq.to(dtype=torch.bool).reshape(-1)
        compressed_shape = (s_seq.numel() + 7) // 8
        s_seq_compresses = torch.zeros(
            compressed_shape, dtype=torch.uint8, device=s_seq.device
        )
        for i in range(8):
            s_seq_compresses |= (s_seq[i::8].to(dtype=torch.uint8) << i)
        return s_seq_compresses

    def decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        s_seq_decompressed = torch.zeros(
            shape.numel(), dtype=torch.bool, device=s_seq.device
        )
        for i in range(8):
            s_seq_decompressed[i::8] = (s_seq >> i) & 1
        s_seq_decompressed = s_seq_decompressed.reshape(shape).to(
            dtype=torch.float32
        )
        return s_seq_decompressed


class SparseSpikeCompressor(BaseSpikeCompressor):

    def __init__(self, dtype=torch.int16):
        super().__init__()
        self.dtype = dtype

    def compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        indices = torch.nonzero(s_seq.reshape(-1))
        return indices.to(dtype=self.dtype)

    def decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        s_seq_decompressed = torch.zeros(
            shape.numel(), dtype=torch.float32, device=s_seq.device
        )
        s_seq_decompressed = torch.scatter(
            s_seq_decompressed,
            dim=0,
            index=s_seq.to(dtype=torch.int64).reshape(-1),
            value=1,
        )
        return s_seq_decompressed.reshape(shape)
