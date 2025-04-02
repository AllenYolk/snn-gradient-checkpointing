import sys

sys.path.insert(0, "./src")

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate, neuron, functional

from utils import *
from .kernels import *
from .compress.h_quantizer import *

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


class SJPSN(neuron.PSN):
    """Multi-step spikingjelly PSN with:
    * ATan surrogate function

    Also, we implement a forwarding function to facilitate the programming of
    PSN-based blocks.
    """

    def __init__(self, T: int, *args, **kwargs):
        super().__init__(T=T, surrogate_function=surrogate.ATan())

    @staticmethod
    def forward_function(x_seq, weight, bias):
        return psn_forward(x_seq, weight, bias)


class SJSlidingPSN(neuron.SlidingPSN):
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
            backend="conv"
        )

    def forward(self, x_seq):  # disable single-step forward!!!
        functional.reset_net(self)
        return self.multi_step_forward(x_seq)

    @staticmethod
    def forward_function(x_seq, weight, bias, k):
        return sliding_psn_forward(x_seq, weight, bias, k)


# ================ Hand-written Multistep LIF neuron ================
class _BaseHandWrittenLIFAutogradFunction(autograd.Function):

    @staticmethod
    def forward(ctx, x_seq, decay_lambda):
        s_seq, h_seq = handwritten_lif_forward(x_seq, decay_lambda)
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(h_seq)  # internal states
            ctx.decay_lambda = decay_lambda
            ctx.T = x_seq.shape[0]
        return s_seq

    @staticmethod
    def backward(ctx, grad_s_seq):
        raise NotImplementedError('`backward` method should be implemented.')


class _HandWrittenLIFAutogradFunctionNotDetached(
    _BaseHandWrittenLIFAutogradFunction
):

    @staticmethod
    def backward(ctx, grad_s_seq):
        h_seq = ctx.saved_tensors[0]
        grad_x_seq = handwritten_lif_backward_not_detached(
            grad_s_seq, h_seq, ctx.decay_lambda, ctx.T
        )
        return grad_x_seq, None


class _HandWrittenLIFAutogradFunctionDetached(
    _BaseHandWrittenLIFAutogradFunction
):

    @staticmethod
    def backward(ctx, grad_s_seq):
        h_seq = ctx.saved_tensors[0]
        grad_x_seq = handwritten_lif_backward_detached(
            grad_s_seq, h_seq, ctx.decay_lambda, ctx.T
        )
        return grad_x_seq, None


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
        h_quantizer (BaseHQuantizer): Quantizer for the hidden state. Set it to
            None to disable quantization. Default is None.
    """

    def __init__(self, decay_lambda=0.5, detach_reset=True, *args, **kwargs):
        super().__init__()
        if decay_lambda < 0. or decay_lambda > 1.:
            raise ValueError('`decay_lambda` should be in the range [0, 1).')
        self.decay_lambda = decay_lambda
        self.detach_reset = detach_reset

        if detach_reset:
            self.core = _HandWrittenLIFAutogradFunctionDetached.apply
        else:
            self.core = _HandWrittenLIFAutogradFunctionNotDetached.apply

    def forward(self, x_seq):
        return self.core(x_seq, self.decay_lambda)

    def extra_repr(self):
        return (
            f"decay_lambda={self.decay_lambda}, "
            f"detach_reset={self.detach_reset}, "
        )


# =========== Hand-written Multistep LIF neuron with H quantization ===========
class _BaseHandWrittenHQLIFAutogradFunction(autograd.Function):

    @staticmethod
    def forward(ctx, x_seq, decay_lambda, h_quantizer):
        s_seq, hq_seq = handwritten_hqlif_forward(
            x_seq, decay_lambda, h_quantizer
        )
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(hq_seq)  # internal states
            ctx.decay_lambda = decay_lambda
            ctx.h_quantizer = h_quantizer
            ctx.T = x_seq.shape[0]
        return s_seq

    @staticmethod
    def backward(ctx, grad_s_seq):
        raise NotImplementedError('`backward` method should be implemented.')


class _HandWrittenHQLIFAutogradFunctionNotDetached(
    _BaseHandWrittenHQLIFAutogradFunction
):

    @staticmethod
    def backward(ctx, grad_s_seq):
        hq_seq = ctx.saved_tensors[0]
        grad_x_seq = handwritten_hqlif_backward_not_detached(
            grad_s_seq, hq_seq, ctx.decay_lambda, ctx.T, ctx.h_quantizer
        )
        return grad_x_seq, None, None


class _HandWrittenHQLIFAutogradFunctionDetached(
    _BaseHandWrittenHQLIFAutogradFunction
):

    @staticmethod
    def backward(ctx, grad_s_seq):
        hq_seq = ctx.saved_tensors[0]
        grad_x_seq = handwritten_hqlif_backward_detached(
            grad_s_seq, hq_seq, ctx.decay_lambda, ctx.T, ctx.h_quantizer
        )
        return grad_x_seq, None, None


class HandWrittenHQLIF(nn.Module):
    """Multi-step handwritten LIF neuron with H quantization:
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
        h_quantizer (BaseHQuantizer): Quantizer for the hidden state. Set it to
            None to disable quantization. Default is ClampProjHQuantizer.
    """

    def __init__(
        self,
        decay_lambda=0.5,
        detach_reset=True,
        h_quantizer=ClampProjHQuantizer(),
        *args,
        **kwargs
    ):
        super().__init__()
        if decay_lambda < 0. or decay_lambda > 1.:
            raise ValueError('`decay_lambda` should be in the range [0, 1).')
        self.decay_lambda = decay_lambda
        self.detach_reset = detach_reset
        self.h_quantizer = h_quantizer

        if detach_reset:
            self.core = _HandWrittenHQLIFAutogradFunctionDetached.apply
        else:
            self.core = _HandWrittenHQLIFAutogradFunctionNotDetached.apply

    def forward(self, x_seq):
        return self.core(x_seq, self.decay_lambda, self.h_quantizer)

    def extra_repr(self):
        return (
            f"decay_lambda={self.decay_lambda}, "
            f"detach_reset={self.detach_reset}, "
            f"h_quantizer={str(self.h_quantizer)}, "
        )


# ================== LIF with O(1) internal state for BPTT ==================
class _BaseMELIFAutogradFunction(autograd.Function):

    @staticmethod
    def forward(ctx, x_seq, decay_lambda):
        T = x_seq.shape[0]
        v = torch.zeros_like(x_seq[0])  # hidden state
        s_seq = []
        for t in range(T):
            x = x_seq[t]

            # core
            h = decay_lambda*v + x
            s = surrogate.heaviside(h - 1.)
            v = h * (1.-s)

            s_seq.append(s)
        s_seq = torch.stack(s_seq, dim=0)
        #! Here, we store x_seq. However, for a weight-neuron block, we only need
        #! to store s_pre_seq (the input to the weight layer). s_pre_seq is required
        #! by the computation of grad_weight, which cannot be omitted. The computation
        #! of neuronal gradient can reuse s_pre_seq, and no additional memory is required.
        ctx.save_for_backward(x_seq)
        ctx.decay_lambda = decay_lambda
        ctx.T = T
        return s_seq

    @staticmethod
    def backward(ctx, grad_s_seq):
        raise NotImplementedError('`backward` method should be implemented.')


class _MELIFAutogradFunctionNotDetached(_BaseMELIFAutogradFunction):

    @staticmethod
    def backward(ctx, grad_s_seq):
        decay_lambda = ctx.decay_lambda
        x_seq = ctx.saved_tensors[0]

        grad_x_seq = torch.empty_like(grad_s_seq)
        grad_v = 0.
        for t in range(ctx.T - 1, -1, -1):
            grad_s = grad_s_seq[t]
            #? How to compute  h and s using x_seq? Forward pass!
            v = 0.
            for tt in range(t + 1):
                x = x_seq[tt]
                h = decay_lambda*v + x
                s = surrogate.heaviside(h - 1.)
                v = h * (1.-s)
            grad_v = ((grad_s - grad_v*h) * atan_derivative(h - 1) + grad_v *
                      (1-s))
            grad_x_seq[t] = grad_v
            grad_v *= decay_lambda

        return grad_x_seq, None


class _MELIFAutogradFunctionDetached(_BaseMELIFAutogradFunction):

    @staticmethod
    def backward(ctx, grad_s_seq):
        decay_lambda = ctx.decay_lambda
        x_seq = ctx.saved_tensors[0]

        grad_x_seq = torch.empty_like(grad_s_seq)
        grad_v = 0.
        for t in range(ctx.T - 1, -1, -1):
            grad_s = grad_s_seq[t]
            #? How to compute  h and s using x_seq? Forward pass!
            v = 0.
            for tt in range(t + 1):
                x = x_seq[tt]
                h = decay_lambda*v + x
                s = surrogate.heaviside(h - 1.)
                v = h * (1.-s)
            grad_v = grad_s * atan_derivative(h - 1) + grad_v * (1-s)
            grad_x_seq[t] = grad_v
            grad_v *= decay_lambda

        return grad_x_seq, None


class MELIF(nn.Module):
    """Multi-step memory-efficient LIF neuron with:
    * decay_input=False
    * v_threshold = 1.
    * hard_reset, v_reset = 0.
    * ATan surrogate function
    """

    def __init__(self, decay_lambda=0.5, detach_reset=True, *args, **kwargs):
        super().__init__()
        if decay_lambda < 0. or decay_lambda > 1.:
            raise ValueError('`decay_lambda` should be in the range [0, 1).')
        self.decay_lambda = decay_lambda
        self.detach_reset = detach_reset
        if detach_reset:
            self.core = _MELIFAutogradFunctionDetached.apply
        else:
            self.core = _MELIFAutogradFunctionNotDetached.apply

    def forward(self, x_seq):
        return self.core(x_seq, self.decay_lambda)
