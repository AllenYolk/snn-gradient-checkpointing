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
        s_seq_compressed = torch.zeros(
            compressed_shape, dtype=torch.uint8, device=s_seq.device
        )
        for i in range(8):
            sliced = s_seq[i::8].to(dtype=torch.uint8)
            sliced_len = sliced.numel()
            if sliced_len > 0:
                s_seq_compressed[:sliced_len] |= (sliced << i)
        return s_seq_compressed

    def decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        decompressed_len = shape.numel()
        s_seq_decompressed = torch.zeros(
            decompressed_len, dtype=torch.bool, device=s_seq.device
        )
        for i in range(8):
            sliced_len = (decompressed_len-i+7) // 8
            sliced = ((s_seq >> i) & 1)[:sliced_len]
            s_seq_decompressed[i::8] = sliced
        s_seq_decompressed = s_seq_decompressed.reshape(shape).to(
            dtype=torch.float32
        )
        return s_seq_decompressed


class SparseSpikeCompressor(BaseSpikeCompressor):

    def __init__(self, dtype=torch.int64):
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
