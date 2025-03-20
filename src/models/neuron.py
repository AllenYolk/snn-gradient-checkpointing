import torch
import torch.nn as nn
import torch.autograd as autograd
from spikingjelly.activation_based import surrogate, neuron

try:
    import cupy
    DEFAULT_SJ_BACKEND = "cupy"
except Exception:
    DEFAULT_SJ_BACKEND = "torch"
print(f"Using {DEFAULT_SJ_BACKEND} backend by default.")


@torch.jit.script
def atan_derivative(x: torch.Tensor, alpha: float = 2.):
    return alpha / 2 / (1 + (torch.pi / 2 * alpha * x).pow_(2))


# ================ Standard SpikingJelly Multi-step LIF neuron ================
class SJLIFNode(neuron.LIFNode):
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
        return self.multi_step_forward(x_seq)


# ================ Hand-written Multistep LIF neuron ================
class _BaseHandWrittenLIFAutogradFunction(autograd.Function):

    @staticmethod
    def forward(ctx, x_seq, decay_lambda):
        T = x_seq.shape[0]
        v = torch.zeros_like(x_seq[0])  # hidden state
        s_seq = []
        h_seq = []
        for t in range(T):
            x = x_seq[t]

            # core
            h = decay_lambda*v + x
            s = surrogate.heaviside(h - 1.)
            v = h * (1.-s)

            s_seq.append(s)
            h_seq.append(h)
        s_seq = torch.stack(s_seq, dim=0)
        h_seq = torch.stack(h_seq, dim=0)
        ctx.save_for_backward(h_seq, s_seq)  # internal states
        ctx.decay_lambda = decay_lambda
        ctx.T = T
        return s_seq

    @staticmethod
    def backward(ctx, grad_s_seq):
        raise NotImplementedError('`backward` method should be implemented.')


class _HandWrittenLIFAutogradFunctionNotDetached(
    _BaseHandWrittenLIFAutogradFunction
):

    @staticmethod
    def backward(ctx, grad_s_seq):
        decay_lambda = ctx.decay_lambda
        h_seq, s_seq = ctx.saved_tensors

        grad_x_seq = []
        grad_v = 0.
        for t in range(ctx.T - 1, -1, -1):
            grad_s = grad_s_seq[t]
            h = h_seq[t]
            s = s_seq[t]
            grad_x = (grad_s - grad_v*h) * atan_derivative(h - 1)
            grad_x = grad_x + grad_v * (1-s)
            grad_v = decay_lambda * grad_x

            grad_x_seq.append(grad_x)

        grad_x_seq = torch.stack(grad_x_seq[::-1], dim=0)
        return grad_x_seq, None


class _HandWrittenLIFAutogradFunctionDetached(
    _BaseHandWrittenLIFAutogradFunction
):

    @staticmethod
    def backward(ctx, grad_s_seq):
        decay_lambda = ctx.decay_lambda
        h_seq, s_seq = ctx.saved_tensors

        grad_x_seq = []
        grad_v = 0.
        for t in range(ctx.T - 1, -1, -1):
            grad_s = grad_s_seq[t]
            h = h_seq[t]
            s = s_seq[t]
            grad_x = grad_s * atan_derivative(h - 1)  # detach_reset = True
            grad_x = grad_x + grad_v * (1-s)
            grad_v = decay_lambda * grad_x

            grad_x_seq.append(grad_x)

        grad_x_seq = torch.stack(grad_x_seq[::-1], dim=0)
        return grad_x_seq, None


class HandWrittenLIFNode(nn.Module):
    """Multi-step handwritten LIF neuron with:
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
            self.core = _HandWrittenLIFAutogradFunctionDetached.apply
        else:
            self.core = _HandWrittenLIFAutogradFunctionNotDetached.apply

    def forward(self, x_seq):
        return self.core(x_seq, self.decay_lambda)


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

        grad_x_seq = []
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
            grad_x = (grad_s - grad_v*h) * atan_derivative(h - 1)
            grad_x = grad_x + grad_v * (1-s)
            grad_v = decay_lambda * grad_x

            grad_x_seq.append(grad_x)

        grad_x_seq = torch.stack(grad_x_seq[::-1], dim=0)
        return grad_x_seq, None


class _MELIFAutogradFunctionDetached(_BaseMELIFAutogradFunction):

    @staticmethod
    def backward(ctx, grad_s_seq):
        decay_lambda = ctx.decay_lambda
        x_seq = ctx.saved_tensors[0]

        grad_x_seq = []
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
            grad_x = grad_s * atan_derivative(h - 1)  # detach_reset = True
            grad_x = grad_x + grad_v * (1-s)
            grad_v = decay_lambda * grad_x

            grad_x_seq.append(grad_x)

        grad_x_seq = torch.stack(grad_x_seq[::-1], dim=0)
        return grad_x_seq, None


class MELIFNode(nn.Module):
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
