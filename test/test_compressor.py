import sys
import random

sys.path.insert(0, "./src")

from models import *


def _test_compressor(compressor, trials=10):
    for _ in range(trials):
        H = random.randint(1, 50)
        W = random.randint(1, 50)
        D = random.randint(1, 50)
        s = torch.rand([H, W, D])
        s = (s >= 0.8).to(dtype=torch.float32, device="cuda")

        compressed_s = compressor.compress(s)
        decompressed_s = compressor.decompress(compressed_s, s.shape)

        assert (s == decompressed_s).all()


def test_compressor():
    _test_compressor(IdentitySpikeCompressor())
    _test_compressor(SparseSpikeCompressor(dtype=torch.int64))
    _test_compressor(BooleanSpikeCompressor())
    _test_compressor(Uint8SpikeCompressor())
    _test_compressor(BitSpikeCompressor())


def test_bit_compressor():
    compressor = BitSpikeCompressor()

    s = torch.rand([7])
    s = (s >= 0.8).to(dtype=torch.float32, device="cuda")

    compressed_s = compressor.compress(s)
    decompressed_s = compressor.decompress(compressed_s, s.shape)

    assert (s == decompressed_s).all()

    s = torch.rand([7, 9, 3, 11, 191])
    s = (s >= 0.5).to(dtype=torch.float32, device="cuda")

    compressed_s = compressor.compress(s)
    decompressed_s = compressor.decompress(compressed_s, s.shape)

    assert (s == decompressed_s).all()


if __name__ == "__main__":
    test_bit_compressor()
    test_compressor()
