import functools

import torch
import torch.nn.functional as F
import einops

TORCH_VERSION = torch.__version__.split('.')[0]
DISABLE_COMPILE = int(TORCH_VERSION) < 2
print(f"TORCH_VERSION={torch.__version__}, DISABLE_COMPILE={DISABLE_COMPILE}")

DEFAULT_BACKEND = "inductor"


def _conditional_compile(
    fullgraph=False, dynamic=False, backend=DEFAULT_BACKEND
):
    """We must use conditional compilation rather than the `disable=False`
    argument, sine `torch.compile` is not available in PyTorch 1.x.x .
    """

    def compile_decorator(f):

        @functools.wraps(f)  # retain f's metadata
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        return wrapper if DISABLE_COMPILE else torch.compile(
            f, fullgraph=fullgraph, dynamic=dynamic, backend=backend
        )

    return compile_decorator


@_conditional_compile()
def linear_forward_compiled(x_seq, weight, bias):
    return F.linear(x_seq, weight, bias)


@_conditional_compile()
def linear_bn_forward_compiled(
    x_seq, weight, bias, bn_weight, bn_bias, bn_running_mean, bn_running_var,
    training
):
    T = x_seq.size(0)
    x_seq = F.linear(x_seq, weight, bias)
    x_seq = einops.rearrange(x_seq, "T N ... -> (T N) ...")
    x_seq = F.batch_norm(
        x_seq,
        bn_running_mean,
        bn_running_var,
        bn_weight,
        bn_bias,
        training=training
    )
    y_seq = einops.rearrange(x_seq, "(T N) ... -> T N ...", T=T)
    return y_seq


@_conditional_compile()
def conv1d_forward_compiled(
    x_seq, weight, bias, stride, padding, dilation, groups
):
    T = x_seq.size(0)
    x_seq = einops.rearrange(x_seq, "T N C L -> (T N) C L")
    x_seq = F.conv1d(x_seq, weight, bias, stride, padding, dilation, groups)
    x_seq = einops.rearrange(x_seq, "(T N) C L -> T N C L", T=T)
    return x_seq


@_conditional_compile()
def conv1d_bn_forward_compiled(
    x_seq, weight, bias, stride, padding, dilation, groups, bn_weight, bn_bias,
    bn_running_mean, bn_running_var, training
):
    T = x_seq.size(0)
    x_seq = einops.rearrange(x_seq, "T N C L -> (T N) C L")
    x_seq = F.conv1d(x_seq, weight, bias, stride, padding, dilation, groups)
    x_seq = F.batch_norm(
        x_seq,
        bn_running_mean,
        bn_running_var,
        bn_weight,
        bn_bias,
        training=training
    )
    y_seq = einops.rearrange(x_seq, "(T N) C L -> T N C L", T=T)
    return y_seq


@_conditional_compile()
def conv2d_forward_compiled(
    x_seq, weight, bias, stride, padding, dilation, groups
):
    T = x_seq.size(0)
    x_seq = einops.rearrange(x_seq, "T N C H W -> (T N) C H W")
    x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
    x_seq = einops.rearrange(x_seq, "(T N) C H W -> T N C H W", T=T)
    return x_seq


@_conditional_compile()
def conv2d_bn_forward_compiled(
    x_seq, weight, bias, stride, padding, dilation, groups, bn_weight, bn_bias,
    bn_running_mean, bn_running_var, training
):
    T = x_seq.size(0)
    x_seq = einops.rearrange(x_seq, "T N C H W -> (T N) C H W")
    x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
    x_seq = F.batch_norm(
        x_seq,
        bn_running_mean,
        bn_running_var,
        bn_weight,
        bn_bias,
        training=training
    )
    y_seq = einops.rearrange(x_seq, "(T N) C H W -> T N C H W", T=T)
    return y_seq
