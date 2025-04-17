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
    "linear_forward":
        linear_forward_compiled,
    "linear_bn_forward":
        linear_bn_forward_compiled,
    "avgpool1d_flatten_linear_forward":
        avgpool1d_flatten_linear_forward_compiled,
    "conv1d_forward":
        conv1d_forward_compiled,
    "conv1d_bn_forward":
        conv1d_bn_forward_compiled,
    "avgpool1d_conv1d_bn_forward":
        avgpool1d_conv1d_bn_forward_compiled,
    "conv2d_forward":
        conv2d_forward_compiled,
    "conv2d_bn_forward":
        conv2d_bn_forward_compiled,
    "conv2d_tebn_forward":
        conv2d_tebn_forward_compiled,
    "avgpool2d_conv2d_bn_forward":
        avgpool2d_conv2d_bn_forward_compiled,
    "avgpool2d_conv2d_tebn_forward":
        avgpool2d_conv2d_tebn_forward_compiled,
    "conv2d_bn_ann_forward":
        conv2d_bn_ann_forward_compiled,
    "handwritten_lif_forward":
        handwritten_lif_forward_compiled,
    "handwritten_lif_backward_not_detached":
        handwritten_lif_backward_not_detached_compiled,
    "handwritten_lif_backward_detached":
        handwritten_lif_backward_detached_compiled,
    "handwritten_hqlif_forward":
        handwritten_hqlif_forward_compiled,
    "handwritten_hqlif_backward_not_detached":
        handwritten_hqlif_backward_not_detached_compiled,
    "handwritten_hqlif_backward_detached":
        handwritten_hqlif_backward_detached_compiled,
    "psn_forward":
        psn_forward_compiled,
    "sliding_psn_forward":
        sliding_psn_forward_compiled,
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

if TRITON_FLOAT8E4NV_AVAILABLE or (
    TRITON_AVAILABLE and (not TORCH_FLOAT8E4M3FN_AVAILABLE)
):
    print("Using Triton kernels for HQLIF.")
    api["handwritten_hqlif_forward"] = handwritten_hqlif_forward_triton
    api["handwritten_hqlif_backward_not_detached"] = (
        handwritten_hqlif_backward_not_detached_triton
    )
    api["handwritten_hqlif_backward_detached"] = (
        handwritten_hqlif_backward_detached_triton
    )
else:
    print("Using torch kernels for HQLIF.")

linear_forward = api["linear_forward"]
linear_bn_forward = api["linear_bn_forward"]
avgpool1d_flatten_linear_forward = api["avgpool1d_flatten_linear_forward"]
conv1d_forward = api["conv1d_forward"]
conv1d_bn_forward = api["conv1d_bn_forward"]
avgpool1d_conv1d_bn_forward = api["avgpool1d_conv1d_bn_forward"]
conv2d_forward = api["conv2d_forward"]
conv2d_bn_forward = api["conv2d_bn_forward"]
conv2d_tebn_forward = api["conv2d_tebn_forward"]
avgpool2d_conv2d_bn_forward = api["avgpool2d_conv2d_bn_forward"]
avgpool2d_conv2d_tebn_forward = api["avgpool2d_conv2d_tebn_forward"]
conv2d_bn_ann_forward = api["conv2d_bn_ann_forward"]
handwritten_lif_forward = api["handwritten_lif_forward"]
handwritten_lif_backward_not_detached = (
    api["handwritten_lif_backward_not_detached"]
)
handwritten_lif_backward_detached = api["handwritten_lif_backward_detached"]
handwritten_hqlif_forward = api["handwritten_hqlif_forward"]
handwritten_hqlif_backward_not_detached = (
    api["handwritten_hqlif_backward_not_detached"]
)
handwritten_hqlif_backward_detached = api["handwritten_hqlif_backward_detached"]
psn_forward = api["psn_forward"]
sliding_psn_forward = api["sliding_psn_forward"]
bit_spike_compress = api["bit_spike_compress"]
bit_spike_decompress = api["bit_spike_decompress"]

__all__ = [k for k in api.keys()]
