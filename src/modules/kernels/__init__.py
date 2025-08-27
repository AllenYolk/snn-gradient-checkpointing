"""Compiled / Triton computational kernels for accelerating training.

Three types of kernels are provided:
1. (pool) + weight + (BN)
2. spiking neurons
3. spike compressors
"""
import multiprocessing as mp

from .compiled_kernels import *
from .triton_kernels import *

# api dict
api = {
    "handwritten_lif_forward": handwritten_lif_forward_compiled,
    "handwritten_lif_backward": handwritten_lif_backward_compiled,
    "bit_spike_compress": bit_spike_compress_compiled,
    "bit_spike_decompress": bit_spike_decompress_compiled,
}

if TRITON_AVAILABLE:
    if mp.current_process().name == "MainProcess":
        print("Use Triton kernels for BitSpikeCompressor.")
        print("Using Triton kernels for HandWrittenLIF.")
    api["bit_spike_compress"] = bit_spike_compress_triton
    api["bit_spike_decompress"] = bit_spike_decompress_triton
    api["handwritten_lif_forward"] = handwritten_lif_forward_triton
    api["handwritten_lif_backward"] = handwritten_lif_backward_triton
else:
    if mp.current_process().name == "MainProcess":
        print("Using torch kernels for BitSpikeCompressor.")
        print("Using torch kernels for HandWrittenLIF.")

handwritten_lif_forward = api["handwritten_lif_forward"]
handwritten_lif_backward = api["handwritten_lif_backward"]
bit_spike_compress = api["bit_spike_compress"]
bit_spike_decompress = api["bit_spike_decompress"]

__all__ = [k for k in api.keys()]
