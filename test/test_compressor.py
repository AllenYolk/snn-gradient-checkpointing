import sys
import random

sys.path.insert(0, "./src")

import torch

from modules import *


def _test_spike_compressor(compressor, trials=10):
    for _ in range(trials):
        H = random.randint(1, 50)
        W = random.randint(1, 50)
        D = random.randint(1, 50)
        s = torch.rand([H, W, D])
        s = (s >= 0.8).to(dtype=torch.float32, device="cuda")

        compressed_s = compressor.compress(s)
        decompressed_s = compressor.decompress(compressed_s, s.shape)

        assert (s == decompressed_s).all()


def test_spike_compressor():
    _test_spike_compressor(IdentitySpikeCompressor())
    _test_spike_compressor(SparseSpikeCompressor(dtype=torch.int64))
    _test_spike_compressor(BooleanSpikeCompressor())
    _test_spike_compressor(Uint8SpikeCompressor())
    _test_spike_compressor(BitSpikeCompressor())


def test_spike_compressor_autocast():
    with torch.amp.autocast(
        device_type="cuda", enabled=True, dtype=torch.bfloat16
    ):
        _test_spike_compressor(IdentitySpikeCompressor())
        _test_spike_compressor(SparseSpikeCompressor(dtype=torch.int64))
        _test_spike_compressor(BooleanSpikeCompressor())
        _test_spike_compressor(Uint8SpikeCompressor())
        _test_spike_compressor(BitSpikeCompressor())


def test_bit_spike_compressor():
    compressor = BitSpikeCompressor()

    s = torch.rand([7])
    s = (s >= 0.8).to(dtype=torch.float32, device="cuda")

    compressed_s = compressor._compress(s)
    decompressed_s = compressor._decompress(compressed_s, s.shape)

    assert (s == decompressed_s).all(), torch.sum(torch.abs(s - decompressed_s))

    s = torch.rand([7, 9, 3, 11, 191])
    s = (s >= 0.8).to(dtype=torch.float32, device="cuda")

    compressed_s = compressor._compress(s)
    decompressed_s = compressor._decompress(compressed_s, s.shape)

    assert (s == decompressed_s).all(), torch.sum(torch.abs(s - decompressed_s))


if __name__ == "__main__":
    test_bit_spike_compressor()
    test_spike_compressor()
    test_spike_compressor_autocast()
