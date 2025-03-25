import functools

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate

TORCH_VERSION = torch.__version__.split('.')[0]

DISABLE_COMPILE = int(TORCH_VERSION) < 2
print(
    f"TORCH_VERSION={torch.__version__}, "
    f"DISABLE_COMPILE should be {DISABLE_COMPILE}"
)
DISABLE_COMPILE = True
print(f"DISABLE_COMPILE is manually set to {DISABLE_COMPILE}. ")

DEFAULT_BACKEND = "inductor"
print(f"DEFAULT_BACKEND is manually set to {DEFAULT_BACKEND}. ")


def _conditional_compile(
    fullgraph=False,
    dynamic=False,
    backend=DEFAULT_BACKEND,
    mode="default",
    disable=DISABLE_COMPILE
):
    """We must use conditional compilation rather than the `disable=False`
    argument, sine `torch.compile` is not available in PyTorch 1.x.x .
    """

    def compile_decorator(f):

        @functools.wraps(f)  # retain f's metadata
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        return wrapper if disable else torch.compile(
            f, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode
        )

    return compile_decorator


@_conditional_compile()
def linear_forward_compiled(x_seq, weight, bias):
    return F.linear(x_seq, weight, bias)


@_conditional_compile()
def linear_bn_forward_compiled(
    x_seq, weight, bias, bn_weight, bn_bias, bn_running_mean, bn_running_var,
    training, momentum
):
    T, N = x_seq.size(0), x_seq.size(1)
    x_seq = F.linear(x_seq, weight, bias)
    x_seq = x_seq.flatten(0, 1)
    x_seq = F.batch_norm(
        x_seq,
        bn_running_mean,
        bn_running_var,
        bn_weight,
        bn_bias,
        training=training,
        momentum=momentum
    )
    y_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
    return y_seq


@_conditional_compile()
def conv1d_forward_compiled(
    x_seq, weight, bias, stride, padding, dilation, groups
):
    T = x_seq.size(0)
    x_seq = x_seq.flatten(0, 1)
    x_seq = F.conv1d(x_seq, weight, bias, stride, padding, dilation, groups)
    x_seq = x_seq.reshape(T, -1, *x_seq.shape[1:])
    return x_seq


@_conditional_compile()
def conv1d_bn_forward_compiled(
    x_seq, weight, bias, stride, padding, dilation, groups, bn_weight, bn_bias,
    bn_running_mean, bn_running_var, training, momentum
):
    T, N = x_seq.size(0), x_seq.size(1)
    x_seq = x_seq.flatten(0, 1)
    x_seq = F.conv1d(x_seq, weight, bias, stride, padding, dilation, groups)
    x_seq = F.batch_norm(
        x_seq,
        bn_running_mean,
        bn_running_var,
        bn_weight,
        bn_bias,
        training=training,
        momentum=momentum
    )
    y_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
    return y_seq


@_conditional_compile()
def conv2d_forward_compiled(
    x_seq, weight, bias, stride, padding, dilation, groups
):
    T, N = x_seq.size(0), x_seq.size(1)
    x_seq = x_seq.flatten(0, 1)
    x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
    x_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
    return x_seq


@_conditional_compile()
def conv2d_bn_forward_compiled(
    x_seq, weight, bias, stride, padding, dilation, groups, bn_weight, bn_bias,
    bn_running_mean, bn_running_var, training, momentum
):
    T, N = x_seq.size(0), x_seq.size(1)
    x_seq = x_seq.flatten(0, 1)
    x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
    x_seq = F.batch_norm(
        x_seq,
        bn_running_mean,
        bn_running_var,
        bn_weight,
        bn_bias,
        training=training,
        momentum=momentum
    )
    y_seq = x_seq.reshape(T, N, *x_seq.shape[1:])
    return y_seq


#===============================================================================
#                           Spiking Neurons                                    =
#===============================================================================
@_conditional_compile()
def handwritten_lif_forward_compiled(x_seq, decay_lambda):
    T = x_seq.shape[0]
    v = torch.zeros_like(x_seq[0])  # hidden state
    s_seq = torch.empty_like(x_seq)
    h_seq = torch.empty_like(x_seq)
    for t in range(T):
        x = x_seq[t]

        # core
        h = decay_lambda*v + x
        s = (h >= 1.).to(h)
        v = h * (1.-s)

        s_seq[t] = s
        h_seq[t] = h
    return s_seq, h_seq


@_conditional_compile()
def handwritten_lif_backward_not_detached_compiled(
    grad_s_seq, h_seq, decay_lambda, T
):
    grad_x_seq = torch.empty_like(grad_s_seq)
    grad_v = 0.
    for t in range(T - 1, -1, -1):
        grad_s = grad_s_seq[t]
        h = h_seq[t]
        sg = 1. / (1. + (torch.pi * (h-1.)).pow_(2))
        grad_v = (grad_s - grad_v*h) * sg + grad_v * (1 - (h >= 1.).to(h))
        grad_x_seq[t] = grad_v
        grad_v *= decay_lambda
    return grad_x_seq


@_conditional_compile()
def handwritten_lif_backward_detached_compiled(
    grad_s_seq, h_seq, decay_lambda, T
):
    grad_x_seq = torch.empty_like(grad_s_seq)
    grad_v = 0.
    for t in range(T - 1, -1, -1):
        grad_s = grad_s_seq[t]
        h = h_seq[t]
        sg = 1. / (1. + (torch.pi * (h-1.)).pow_(2))
        grad_v = grad_s*sg + grad_v * (1 - (h >= 1.).to(h))
        grad_x_seq[t] = grad_v
        grad_v *= decay_lambda
    return grad_x_seq


@_conditional_compile()
def psn_forward_compiled(x_seq, weight, bias):
    # x_seq.shape = [T, N, ...]; weight.shape = [T, T]; bias.shape = [T, 1]
    h_seq = torch.addmm(bias, weight, x_seq.flatten(1))
    s_seq = surrogate.atan.apply(h_seq, 2.)
    return s_seq.reshape(x_seq.shape)


@_conditional_compile()
def sliding_psn_forward_compiled(x_seq, weight, bias, k):
    # x_seq.shape = [T, N, ...]; weight.shape = [k], bias.shape = []
    x_seq_shape = x_seq.shape
    x_seq = x_seq.flatten(1).t().unsqueeze(1)  # [T, N, ...] -> [(N*...), 1, T]
    x_seq = F.pad(x_seq, pad=(k - 1, 0), mode="constant", value=0.)
    x_seq = F.conv1d(x_seq, weight.reshape(1, 1, -1), stride=1)
    x_seq = x_seq = x_seq.squeeze(1).t().view(
        x_seq_shape
    )  # [(N*...), 1, T] -> [T, N, ...]
    return surrogate.atan.apply(x_seq + bias, 2.)
