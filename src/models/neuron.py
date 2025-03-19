import torch
import torch.nn as nn
import torch.autograd as autograd
from spikingjelly.activation_based import surrogate
import einops

surrogate_function = surrogate.Sigmoid()


def sigmoid_backward(x):
    return surrogate.sigmoid_backward(torch.ones_like(x), x, alpha=4.)[0]


# ================== Vanilla LIF ==================
class VanillaLIF(nn.Module):

    def __init__(self, decay_lambda=0.5):
        super().__init__()
        self.decay_lambda = decay_lambda

    def forward(self, x_seq):
        T = x_seq.shape[0]
        v = torch.zeros_like(x_seq[0])  # hidden state
        s_seq = []
        for t in range(T):
            x = x_seq[t]

            # single-step forward; a.k.a. "core"
            h = self.decay_lambda * v + x
            s = surrogate_function(h - 1.)
            v = h * (1.-s)

            s_seq.append(s)
        s_seq = torch.stack(s_seq, dim=0)
        return s_seq


# ================== LIF with Hand-Written BPTT ==================
class HandWrittenLIFAutogradFunction(autograd.Function):

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
        decay_lambda = ctx.decay_lambda
        h_seq, s_seq = ctx.saved_tensors

        grad_x_seq = []
        grad_v = 0.
        for t in range(ctx.T - 1, -1, -1):
            grad_s = grad_s_seq[t]
            h = h_seq[t]
            s = s_seq[t]
            grad_x = (grad_s - grad_v*h) * sigmoid_backward(h - 1)
            grad_x = grad_x + grad_v * (1-s)
            grad_v = decay_lambda * grad_x

            grad_x_seq.append(grad_x)

        grad_x_seq = torch.stack(grad_x_seq[::-1], dim=0)
        return grad_x_seq, None


class HandWrittenLIF(nn.Module):

    def __init__(self, decay_lambda=0.5):
        super().__init__()
        self.decay_lambda = decay_lambda

    def forward(self, x_seq):
        return HandWrittenLIFAutogradFunction.apply(x_seq, self.decay_lambda)


# ================== LIF with O(1)-memory BPTT ==================
class MELIFAutogradFunction(autograd.Function):

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
            grad_x = (grad_s - grad_v*h) * sigmoid_backward(h - 1)
            grad_x = grad_x + grad_v * (1-s)
            grad_v = decay_lambda * grad_x

            grad_x_seq.append(grad_x)

        grad_x_seq = torch.stack(grad_x_seq[::-1], dim=0)
        return grad_x_seq, None


class MELIF(nn.Module):

    def __init__(self, decay_lambda=0.5):
        super().__init__()
        self.decay_lambda = decay_lambda

    def forward(self, x_seq):
        return MELIFAutogradFunction.apply(x_seq, self.decay_lambda)
