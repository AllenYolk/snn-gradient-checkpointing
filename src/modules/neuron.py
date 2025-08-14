from typing import Tuple
import sys

sys.path.insert(0, "./src")

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate, neuron, functional

from utils import *
from .kernels import *

try:
    import cupy
    DEFAULT_SJ_BACKEND = "cupy"
except Exception:
    DEFAULT_SJ_BACKEND = "torch"
print(f"Using {DEFAULT_SJ_BACKEND} backend for spikingjelly by default.")


def get_neuron(neuron_type: str, **kwargs):
    return globals()[neuron_type](**kwargs)


@torch.jit.script
def atan_derivative(x: torch.Tensor, alpha: float = 2.):
    return alpha / 2 / (1 + (torch.pi / 2 * alpha * x).pow_(2))


# TODO: implement Triton kernels
def lif_rnn_function(
    x_seq: torch.Tensor, v: torch.Tensor, decay_lambda: float,
    detach_reset: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """T-step RNN-like LIF neuron: 
    s_seq[0...T-1], v[T-1] = lif_rnn(x_seq[0...T-1], v[-1])

    Args:
        x_seq[0...T-1] (torch.Tensor)
        v[-1] (torch.Tensor)
        decay_lambda (float)
        detach_reset (bool)

    Returns:
        s_seq[0...T-1], v[T-1]
    """
    T = x_seq.shape[0]
    s_seq = torch.empty_like(x_seq)
    for t in range(T):
        v = decay_lambda*v + x_seq[t]
        s = surrogate.atan.apply(v - 1., 2.)
        s_seq[t] = s
        if detach_reset:
            s = s.detach()
        v = v * (1.-s)
    return s_seq, v


# ================ Standard SpikingJelly Multi-step neurons ================
class SJLIF(neuron.LIFNode):
    """Multi-step spikingjelly LIF neuron with:
    * decay_input=False
    * v_threshold = 1.
    * hard_reset, v_reset = 0.
    * ATan surrogate function
    """

    def __init__(
        self,
        decay_lambda: float = 0.5,
        detach_reset: bool = True,
        backend: str = DEFAULT_SJ_BACKEND,
        *args,
        **kwargs
    ):
        if decay_lambda < 0. or decay_lambda > 1.:
            raise ValueError('`decay_lambda` should be in the range [0, 1).')
        tau = 1. / (1.-decay_lambda)
        self.decay_lambda = decay_lambda

        super().__init__(
            tau,
            decay_input=False,
            v_threshold=1.,
            v_reset=0.,
            surrogate_function=surrogate.ATan(),
            detach_reset=detach_reset,
            step_mode="m",
            backend=backend,
            store_v_seq=False,
        )

    def forward(self, x_seq):
        functional.reset_net(self)  #! reset internal states before forwarding
        return self.multi_step_forward(x_seq)

    def rnn_forward(self, x_seq, v):
        return lif_rnn_function(x_seq, v, self.decay_lambda, self.detach_reset)


# =========================== Simplified SJLIF ===========================
class AutogradLIF(nn.Module):
    """Multi-step LIF neuron with:
    * decay_input=False
    * v_threshold = 1.
    * hard_reset, v_reset = 0.
    * ATan surrogate function
    whose BP is implemented by pytorch autograd.
    """

    def __init__(
        self,
        decay_lambda: float = 0.5,
        detach_reset: bool = True,
        *args,
        **kwargs
    ):
        super().__init__()
        if decay_lambda < 0. or decay_lambda > 1.:
            raise ValueError('`decay_lambda` should be in the range [0, 1).')
        self.tau = 1. / (1.-decay_lambda)
        self.decay_lambda = decay_lambda
        self.detach_reset = detach_reset
        self.sg = surrogate.ATan()

    def forward(self, x_seq):
        T = x_seq.shape[0]
        v = torch.zeros_like(x_seq[0])  # hidden state
        s_seq = torch.empty_like(x_seq)
        for t in range(T):
            v = self.decay_lambda * v + x_seq[t]
            s = self.sg(v - 1.)
            s_seq[t] = s
            if self.detach_reset:
                s = s.detach()
            v = v * (1.-s)
        return s_seq

    def rnn_forward(self, x_seq, v):
        return lif_rnn_function(x_seq, v, self.decay_lambda, self.detach_reset)


# =========================== Multi-step PSN Family ===========================
class PSN(neuron.PSN):
    """Multi-step spikingjelly PSN with:
    * ATan surrogate function

    Also, we implement a forwarding function to facilitate the programming of
    PSN-based blocks.
    """

    def __init__(self, T: int, *args, **kwargs):
        super().__init__(T=T, surrogate_function=surrogate.ATan())

    @staticmethod
    def forward_function(x_seq, weight, bias):
        # x_seq.shape = [T, N, ...]; weight.shape = [T, T]; bias.shape = [T, 1]
        h_seq = torch.addmm(bias, weight, x_seq.flatten(1))
        s_seq = surrogate.atan.apply(h_seq, 2.)
        return s_seq.reshape(x_seq.shape)


class SlidingPSN(neuron.SlidingPSN):
    """Multi-step spikingjelly SlidingPSN with:
    * exponential weight initialization rule
    * ATan surrogate function
    * convolutional implementation

    Also, we implement a forwarding function to facilitate the programming of
    SlidingPSN-based blocks.
    """

    def __init__(self, k: int, *args, **kwargs):
        super().__init__(
            k=k,
            exp_init=True,
            surrogate_function=surrogate.ATan(),
            step_mode="m",
            backend="gemm"
        )

    def forward(self, x_seq):  # disable single-step forward!!!
        functional.reset_net(self)
        return self.multi_step_forward(x_seq)

    @staticmethod
    def forward_function(x_seq, weight, bias, k):
        T = x_seq.shape[0]
        gemm_weight = torch.zeros([T, T], device=weight.device)
        for i in range(T):
            end = i + 1
            start = max(0, i + 1 - k)
            length = min(end - start, k)
            gemm_weight[i][start:end] = weight[k - length:k]

        h_seq = torch.addmm(
            bias,
            gemm_weight,
            x_seq.flatten(1),
        ).reshape(x_seq.shape)
        return surrogate.atan.apply(h_seq, 2.)


# ================ Hand-written Multistep LIF neuron ================
class HandWrittenLIFAutogradFunction(autograd.Function):

    @staticmethod
    def forward(ctx, x_seq, decay_lambda: float, detach_reset: bool):
        s_seq, h_seq = handwritten_lif_forward(x_seq.contiguous(), decay_lambda)
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(h_seq)  # internal states
            ctx.decay_lambda = decay_lambda
            ctx.detach_reset = detach_reset
        return s_seq

    @staticmethod
    def backward(ctx, grad_s_seq):
        h_seq = ctx.saved_tensors[0]
        grad_x_seq = handwritten_lif_backward(
            grad_s_seq.contiguous(),
            h_seq.contiguous(),
            ctx.decay_lambda,
            ctx.detach_reset,
        )
        return grad_x_seq, None, None


class HandWrittenLIF(nn.Module):
    """Multi-step handwritten LIF neuron with:
    * decay_input=False
    * v_threshold = 1.
    * hard_reset, v_reset = 0.
    * ATan surrogate function
    Experiments show that HandWrittenLIFNode consumes much less memory than
    SJLIF, while its computational efficiency is nearly the same.

    Args:
        decay_lambda (float): the neuronal decay factor. Should be in
            the range [0, 1].
        detach_reset (bool): Whether to detach the reset operation from the
            computational graph.
    """

    def __init__(self, decay_lambda=0.5, detach_reset=True, *args, **kwargs):
        super().__init__()
        if decay_lambda < 0. or decay_lambda > 1.:
            raise ValueError('`decay_lambda` should be in the range [0, 1).')
        self.decay_lambda = decay_lambda
        self.detach_reset = detach_reset
        self.core = HandWrittenLIFAutogradFunction.apply

    def forward(self, x_seq):
        return self.core(x_seq, self.decay_lambda, self.detach_reset)

    def rnn_forward(self, x_seq, v) -> Tuple[torch.Tensor, torch.Tensor]:
        return lif_rnn_function(x_seq, v, self.decay_lambda, self.detach_reset)

    def extra_repr(self):
        return (
            f"decay_lambda={self.decay_lambda}, "
            f"detach_reset={self.detach_reset}, "
        )
