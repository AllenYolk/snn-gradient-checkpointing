import functools
import multiprocessing as mp

import torch
import torch.nn.functional as F

TORCH_VERSION = torch.__version__.split(".")[0]

DISABLE_COMPILE = int(TORCH_VERSION) < 2
DISABLE_COMPILE = True
DEFAULT_BACKEND = "inductor"

if mp.current_process().name == "MainProcess":
    print(
        f"TORCH_VERSION={torch.__version__}, "
        f"DISABLE_COMPILE should be {DISABLE_COMPILE}"
    )
    print(f"DISABLE_COMPILE is manually set to {DISABLE_COMPILE}. ")
    print(f"DEFAULT_BACKEND is manually set to {DEFAULT_BACKEND}. ")


def _conditional_compile(
    fullgraph=False,
    dynamic=True,
    backend=DEFAULT_BACKEND,
    mode="default",
    disable=DISABLE_COMPILE,
):
    """We must use conditional compilation rather than the `disable=False`
    argument, sine `torch.compile` is not available in PyTorch 1.x.x .
    """

    def compile_decorator(f):
        @functools.wraps(f)  # retain f's metadata
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        return (
            wrapper
            if disable
            else torch.compile(
                f, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode
            )
        )

    return compile_decorator


# ===============================================================================
#                           Spiking Neurons                                    =
# ===============================================================================
@_conditional_compile()
def melif_forward_compiled(x_seq, decay_lambda):
    T = x_seq.shape[0]
    v = torch.zeros_like(x_seq[0])  # hidden state
    s_seq = torch.empty_like(x_seq)
    h_seq = torch.empty_like(x_seq)
    for t in range(T):
        x = x_seq[t]
        # core
        v = decay_lambda * v + x
        h_seq[t] = v
        s_seq[t] = (v >= 1.0).to(v)
        v = v * (1.0 - s_seq[t])
    return s_seq, h_seq


@_conditional_compile()
def melif_backward_compiled(grad_s_seq, h_seq, decay_lambda, detach_reset):
    grad_x_seq = torch.empty_like(grad_s_seq)
    grad_v = 0.0
    T = grad_s_seq.shape[0]
    for t in range(T - 1, -1, -1):
        grad_s = grad_s_seq[t]
        h = h_seq[t]
        s = (h >= 1.0).to(h)
        sg_r = 1.0 + (torch.pi * (h - 1.0)).pow_(2)
        if detach_reset:
            grad_v = grad_s / sg_r + grad_v * (1.0 - s)
        else:
            grad_v = (grad_s - grad_v * h) / sg_r + grad_v * (1.0 - s)
        grad_x_seq[t] = grad_v
        grad_v *= decay_lambda
    return grad_x_seq


# ===============================================================================
#                           Spike Compressor                                   =
# ===============================================================================
@_conditional_compile()
def bit_spike_compress_compiled(s_seq: torch.Tensor) -> torch.Tensor:
    # s_seq: float32, ndim=1
    s_seq = s_seq.to(dtype=torch.bool).reshape(-1)
    compressed_shape = (s_seq.numel() + 7) // 8
    s_seq_compressed = torch.zeros(
        compressed_shape, dtype=torch.uint8, device=s_seq.device
    )
    for i in range(8):
        sliced = s_seq[i::8].to(dtype=torch.uint8)
        sliced_len = sliced.numel()
        if sliced_len > 0:
            s_seq_compressed[:sliced_len] |= sliced << i
    return s_seq_compressed


@_conditional_compile()
def bit_spike_decompress_compiled(
    s_seq_compressed: torch.Tensor, shape
) -> torch.Tensor:
    decompressed_len = shape.numel()
    s_seq_decompressed = torch.zeros(
        decompressed_len, dtype=torch.bool, device=s_seq_compressed.device
    )
    for i in range(8):
        sliced_len = (decompressed_len - i + 7) // 8
        sliced = ((s_seq_compressed >> i) & 1)[:sliced_len]
        s_seq_decompressed[i::8] = sliced
    return s_seq_decompressed.reshape(shape)
