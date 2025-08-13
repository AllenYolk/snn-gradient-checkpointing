"""Compiled / Triton computational kernels for accelerating training.

Three types of kernels are provided:
1. (pool) + weight + (BN)
2. spiking neurons
3. spike compressors
"""
from .compiled_kernels import *
from .triton_kernels import *

# api dict
api = {
    "handwritten_lif_forward":
        handwritten_lif_forward_compiled,
    "handwritten_lif_backward_not_detached":
        handwritten_lif_backward_not_detached_compiled,
    "handwritten_lif_backward_detached":
        handwritten_lif_backward_detached_compiled,
    "bit_spike_compress":
        bit_spike_compress_compiled,
    "bit_spike_decompress":
        bit_spike_decompress_compiled,
}

if TRITON_AVAILABLE:
    print("Use Triton kernels for BitSpikeCompressor.")
    api["bit_spike_compress"] = bit_spike_compress_triton
    api["bit_spike_decompress"] = bit_spike_decompress_triton
    print("Using Triton kernels for HandWrittenLIF.")
    api["handwritten_lif_forward"] = handwritten_lif_forward_triton
    api["handwritten_lif_backward_not_detached"] = (
        handwritten_lif_backward_not_detached_triton
    )
    api["handwritten_lif_backward_detached"] = (
        handwritten_lif_backward_detached_triton
    )
else:
    print("Using torch kernels for BitSpikeCompressor.")
    print("Using torch kernels for HandWrittenLIF.")

handwritten_lif_forward = api["handwritten_lif_forward"]
handwritten_lif_backward_not_detached = (
    api["handwritten_lif_backward_not_detached"]
)
handwritten_lif_backward_detached = api["handwritten_lif_backward_detached"]
bit_spike_compress = api["bit_spike_compress"]
bit_spike_decompress = api["bit_spike_decompress"]

__all__ = [k for k in api.keys()]
