import sys
import time
import random

sys.path.insert(0, "./src")

import torch

from modules import *


def _test_spike_compressor(compressor, trials=100, N=int(1e7), rho=0.5):
    compressor_class = compressor.__class__.__name__

    for i in range(10):  # warm up
        s = torch.rand([1, N])
        s = (s <= rho).to(dtype=torch.float32, device="cuda")
        compressed_s = compressor.compress(s)
        decompressed_s = compressor.decompress(compressed_s, s.shape)

    s = torch.rand([trials, N])
    s = (s <= rho).to(dtype=torch.float32, device="cuda")

    t = time.time()

    mem_costs = []
    for i in range(trials):
        compressed_s = compressor.compress(s[i])
        decompressed_s = compressor.decompress(compressed_s, s[i].shape)
        mem = compressed_s.element_size() * compressed_s.numel()
        mem_costs.append(mem)

    time_cost = time.time() - t
    time_cost /= trials
    mem_cost = sum(mem_costs) / trials
    print(
        compressor.__class__.__name__, f"rho={rho},",
        "Time Cost: {:.4f} ms, Mem Cost: {:.4f} MB".format(
            time_cost * 1000, mem_cost / 1024 / 1024
        )
    )


def _test_nvcomp_spike_compressor(compressor, trials=1, N=int(1e6), rho=0.5):
    compressor_class = compressor.__class__.__name__

    for i in range(0):  # warm up
        s = torch.rand([1, N])
        s = (s <= rho).to(dtype=torch.float32, device="cuda")
        compressed_s = compressor.compress(s)
        decompressed_s = compressor.decompress(compressed_s, s.shape)

    s = torch.rand([trials, N])
    s = (s <= rho).to(dtype=torch.float32, device="cuda")

    t = time.time()

    mem_costs = []
    for i in range(trials):
        compressed_s = compressor.compress(s[i])
        decompressed_s = compressor.decompress(compressed_s, s[i].shape)
        # s = [chunk.buffer_size for chunk in compressed_s]
        mem = compressed_s.buffer_size
        mem_costs.append(mem)

    time_cost = time.time() - t
    time_cost /= trials
    mem_cost = sum(mem_costs) / trials
    print(
        compressor.__class__.__name__, f"rho={rho},",
        "Time Cost: {:.4f} ms, Mem Cost: {:.4f} MB".format(
            time_cost * 1000, mem_cost / 1024 / 1024
        )
    )


def test_spike_compressor():
    for rho in [0.01, 0.1, 0.2, 0.5, 0.8, 0.9]:
        #_test_spike_compressor(IdentitySpikeCompressor(), rho=rho)
        # _test_spike_compressor(BitSpikeCompressor(), rho=rho, N=int(1e6))
        #_test_spike_compressor(
        #SparseSpikeCompressor(dtype=torch.int64), rho=rho
        #)
        _test_nvcomp_spike_compressor(NvcompSpikeCompressor(), rho=rho)


if __name__ == "__main__":
    test_spike_compressor()
    test_spike_compressor()
    test_spike_compressor()
    test_spike_compressor()
