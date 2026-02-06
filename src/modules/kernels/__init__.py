"""Compiled / Triton computational kernels for accelerating training.

If Triton is available, use Triton kernels. Otherwise, use compiled PyTorch kernels.

Three types of kernels are provided:
2. spiking neurons
3. spike compressors
"""

import multiprocessing as mp

from .compiled_kernels import *
from .triton_kernels import *

# api dict
api = {
    "melif_forward": melif_forward_compiled,
    "melif_backward": melif_backward_compiled,
    "bit_spike_compress": bit_spike_compress_compiled,
    "bit_spike_decompress": bit_spike_decompress_compiled,
}

if TRITON_AVAILABLE:
    if mp.current_process().name == "MainProcess":
        print("Use Triton kernels for BitSpikeCompressor.")
        print("Using Triton kernels for MELIF.")
    api["bit_spike_compress"] = bit_spike_compress_triton
    api["bit_spike_decompress"] = bit_spike_decompress_triton
    api["melif_forward"] = melif_forward_triton
    api["melif_backward"] = melif_backward_triton
else:
    if mp.current_process().name == "MainProcess":
        print("Using torch kernels for BitSpikeCompressor.")
        print("Using torch kernels for MELIF.")

melif_forward = api["melif_forward"]
melif_backward = api["melif_backward"]
bit_spike_compress = api["bit_spike_compress"]
bit_spike_decompress = api["bit_spike_decompress"]

__all__ = [k for k in api.keys()]
