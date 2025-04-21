import torch
import torch.nn as nn
from torch import autograd


class TEBNProjectionAutogradFunction(autograd.Function):
    """Avoid storing the broadcasted tebn weight!!!
    """

    @staticmethod
    def forward(ctx, x_seq, p):
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(x_seq, p)
        return x_seq * p

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output.shape = [T, B, ...]
        # p.shape = [T, 1, 1, 1, ...]
        x_seq, p = ctx.saved_tensors
        grad_x_seq = grad_output * p
        grad_p = (grad_output * x_seq).sum(
            dim=(i for i in range(1, len(x_seq.shape))), keepdim=True
        )
        return grad_x_seq, grad_p


class TEBNProjection(nn.Module):

    def __init__(self, T, input_ndim: int = 5):
        super().__init__()
        self.p = nn.Parameter(
            torch.ones(T, *[1 for _ in range(input_ndim - 1)])
        )

    def forward(self, x_seq):
        return TEBNProjectionAutogradFunction.apply(x_seq, self.p)
