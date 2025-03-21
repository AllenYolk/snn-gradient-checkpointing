import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import einops

from .compress import *


class _LinearLIFAutogradFunction(autograd.Function):

    @staticmethod
    def forward(ctx, x_seq, weight, bias, neuron, spike_compressor):
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(
                spike_compressor.compress(x_seq), weight, bias
            )
            ctx.neuron = neuron
            ctx.spike_compressor = spike_compressor
            ctx.x_seq_shape = x_seq.shape
        return neuron(F.linear(x_seq, weight, bias))

    @staticmethod
    def backward(ctx, grad_s_seq):
        grad_x_seq, grad_weight, grad_bias = None, None, None

        if any(ctx.needs_input_grad):
            neuron = ctx.neuron
            spike_compressor = ctx.spike_compressor
            x_seq_shape = ctx.x_seq_shape
            x_seq, weight, bias = ctx.saved_tensors
            x_seq = spike_compressor.decompress(x_seq, x_seq_shape)

            with torch.set_grad_enabled(True):
                #! y = x.detach() => y is just a new "pointer" to x's data.
                #! No extra memory is allocated.
                x_seq = x_seq.detach().requires_grad_(True)
                weight = weight.detach().requires_grad_(True)
                bias = bias.detach().requires_grad_(True)
                y_seq = F.linear(x_seq, weight, bias)
                s_seq = neuron(y_seq)
                s_seq.backward(grad_s_seq)

            if ctx.needs_input_grad[0]:
                grad_x_seq = x_seq.grad
            if ctx.needs_input_grad[1]:
                grad_weight = weight.grad
            if ctx.needs_input_grad[2]:
                grad_bias = bias.grad
        return grad_x_seq, grad_weight, grad_bias, None, None


class LinearLIF(nn.Module):

    def __init__(
        self,
        proj: nn.Linear,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BooleanSpikeCompressor(),
    ):
        super().__init__()
        self.proj = proj
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    def forward(self, x_seq: torch.Tensor):
        return _LinearLIFAutogradFunction.apply(
            x_seq,
            self.proj.weight,
            self.proj.bias,
            self.neuron,
            self.spike_compressor,
        )


class _LinearBNLIFAutogradFunction(autograd.Function):

    @staticmethod
    def forward(
        ctx, x_seq, weight, bias, bn_weight, bn_bias, bn_running_mean,
        bn_running_var, training, neuron, spike_compressor
    ):
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(
                spike_compressor.compress(x_seq),
                weight,
                bias,
                bn_weight,
                bn_bias,
                bn_running_mean,
                bn_running_var,
            )
            ctx.neuron = neuron
            ctx.spike_compressor = spike_compressor
            ctx.x_seq_shape = x_seq.shape
            ctx.training = training

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
        x_seq = einops.rearrange(x_seq, "(T N) ... -> T N ...", T=T)
        return neuron(x_seq)

    @staticmethod
    def backward(ctx, grad_s_seq):
        grad_x_seq, grad_weight, grad_bias = None, None, None
        grad_bn_weight, grad_bn_bias = None, None

        if any(ctx.needs_input_grad):
            training = ctx.training
            neuron = ctx.neuron
            spike_compressor = ctx.spike_compressor
            x_seq_shape = ctx.x_seq_shape
            x_seq, weight, bias = ctx.saved_tensors[:3]
            bn_weight, bn_bias = ctx.saved_tensors[3:5]
            bn_running_mean, bn_running_var = ctx.saved_tensors[5:]
            x_seq = spike_compressor.decompress(x_seq, x_seq_shape)

            with torch.set_grad_enabled(True):
                #! y = x.detach() => y is just a new "pointer" to x's data.
                #! No extra memory is allocated.
                x_seq = x_seq.detach().requires_grad_(True)
                weight = weight.detach().requires_grad_(True)
                bias = bias.detach().requires_grad_(True)
                bn_weight = bn_weight.detach().requires_grad_(True)
                bn_bias = bn_bias.detach().requires_grad_(True)
                T = x_seq.size(0)
                y_seq = F.linear(x_seq, weight, bias)
                y_seq = einops.rearrange(y_seq, "T N ... -> (T N) ...")
                y_seq = F.batch_norm(
                    y_seq,
                    bn_running_mean,
                    bn_running_var,
                    bn_weight,
                    bn_bias,
                    training=training
                )
                y_seq = einops.rearrange(y_seq, "(T N) ... -> T N ...", T=T)
                s_seq = neuron(y_seq)
                s_seq.backward(grad_s_seq)

            if ctx.needs_input_grad[0]:
                grad_x_seq = x_seq.grad
            if ctx.needs_input_grad[1]:
                grad_weight = weight.grad
            if ctx.needs_input_grad[2]:
                grad_bias = bias.grad
            if ctx.needs_input_grad[3]:
                grad_bn_weight = bn_weight.grad
            if ctx.needs_input_grad[4]:
                grad_bn_bias = bn_bias.grad
        return (
            grad_x_seq, grad_weight, grad_bias, grad_bn_weight, grad_bn_bias,
            None, None, None, None, None
        )


class LinearBNLIF(nn.Module):

    def __init__(
        self,
        proj: nn.Linear,
        bn: nn.BatchNorm1d,  # actually, not necessarily 1d
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BooleanSpikeCompressor(),
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    def forward(self, x_seq: torch.Tensor):
        return _LinearBNLIFAutogradFunction.apply(
            x_seq,
            self.proj.weight,
            self.proj.bias,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
            self.neuron,
            self.spike_compressor,
        )


class _Conv2dLIFAutogradFunction(autograd.Function):

    @staticmethod
    def forward(
        ctx, x_seq, weight, bias, stride, padding, dilation, groups, neuron,
        spike_compressor
    ):
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(
                spike_compressor.compress(x_seq), weight, bias
            )
            ctx.stride = stride
            ctx.padding = padding
            ctx.dilation = dilation
            ctx.groups = groups
            ctx.neuron = neuron
            ctx.spike_compressor = spike_compressor
            ctx.x_seq_shape = x_seq.shape
        T = x_seq.size(0)
        x_seq = einops.rearrange(x_seq, "T N C H W -> (T N) C H W")
        x_seq = F.conv2d(x_seq, weight, bias, stride, padding, dilation, groups)
        x_seq = einops.rearrange(x_seq, "(T N) C H W -> T N C H W", T=T)
        return neuron(x_seq)

    @staticmethod
    def backward(ctx, grad_s_seq):
        grad_x_seq, grad_weight, grad_bias = None, None, None

        if any(ctx.needs_input_grad):
            stride, padding, dilation, groups = (
                ctx.stride, ctx.padding, ctx.dilation, ctx.groups
            )
            neuron = ctx.neuron
            spike_compressor = ctx.spike_compressor
            x_seq_shape = ctx.x_seq_shape
            x_seq, weight, bias = ctx.saved_tensors
            x_seq = spike_compressor.decompress(x_seq, x_seq_shape)

            with torch.set_grad_enabled(True):
                #! y = x.detach() => y is just a new "pointer" to x's data.
                #! No extra memory is allocated.
                x_seq = x_seq.detach().requires_grad_(True)
                weight = weight.detach().requires_grad_(True)
                bias = bias.detach().requires_grad_(True)
                T = x_seq.size(0)
                y_seq = einops.rearrange(x_seq, "T N C H W -> (T N) C H W")
                y_seq = F.conv2d(
                    y_seq, weight, bias, stride, padding, dilation, groups
                )
                y_seq = einops.rearrange(y_seq, "(T N) C H W -> T N C H W", T=T)
                s_seq = neuron(y_seq)
                s_seq.backward(grad_s_seq)

            if ctx.needs_input_grad[0]:
                grad_x_seq = x_seq.grad
            if ctx.needs_input_grad[1]:
                grad_weight = weight.grad
            if ctx.needs_input_grad[2]:
                grad_bias = bias.grad
        return (
            grad_x_seq, grad_weight, grad_bias, None, None, None, None, None,
            None
        )


class Conv2dLIF(nn.Module):

    def __init__(
        self,
        proj: nn.Linear,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BooleanSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    def forward(self, x_seq: torch.Tensor):
        return _Conv2dLIFAutogradFunction.apply(
            x_seq,
            self.proj.weight,
            self.proj.bias,
            self.proj.stride,
            self.proj.padding,
            self.proj.dilation,
            self.proj.groups,
            self.neuron,
            self.spike_compressor,
        )


class _Conv2dBNLIFAutogradFunction(autograd.Function):

    @staticmethod
    def forward(
        ctx, x_seq, weight, bias, stride, padding, dilation, groups, bn_weight,
        bn_bias, bn_running_mean, bn_running_var, training, neuron,
        spike_compressor
    ):
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(
                spike_compressor.compress(x_seq),
                weight,
                bias,
                bn_weight,
                bn_bias,
                bn_running_mean,
                bn_running_var,
            )
            ctx.stride = stride
            ctx.padding = padding
            ctx.dilation = dilation
            ctx.groups = groups
            ctx.training = training
            ctx.neuron = neuron
            ctx.spike_compressor = spike_compressor
            ctx.x_seq_shape = x_seq.shape
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
        x_seq = einops.rearrange(x_seq, "(T N) C H W -> T N C H W", T=T)
        return neuron(x_seq)

    @staticmethod
    def backward(ctx, grad_s_seq):
        grad_x_seq, grad_weight, grad_bias = None, None, None
        grad_bn_weight, grad_bn_bias = None, None

        if any(ctx.needs_input_grad):
            stride, padding, dilation, groups = (
                ctx.stride, ctx.padding, ctx.dilation, ctx.groups
            )
            training = ctx.training
            neuron = ctx.neuron
            spike_compressor = ctx.spike_compressor
            x_seq_shape = ctx.x_seq_shape
            x_seq, weight, bias = ctx.saved_tensors[:3]
            bn_weight, bn_bias = ctx.saved_tensors[3:5]
            bn_running_mean, bn_running_var = ctx.saved_tensors[5:]
            x_seq = spike_compressor.decompress(x_seq, x_seq_shape)

            with torch.set_grad_enabled(True):
                #! y = x.detach() => y is just a new "pointer" to x's data.
                #! No extra memory is allocated.
                x_seq = x_seq.detach().requires_grad_(True)
                weight = weight.detach().requires_grad_(True)
                bias = bias.detach().requires_grad_(True)
                bn_weight = bn_weight.detach().requires_grad_(True)
                bn_bias = bn_bias.detach().requires_grad_(True)
                T = x_seq.size(0)
                y_seq = einops.rearrange(x_seq, "T N C H W -> (T N) C H W")
                y_seq = F.conv2d(
                    y_seq, weight, bias, stride, padding, dilation, groups
                )
                y_seq = F.batch_norm(
                    y_seq,
                    bn_running_mean,
                    bn_running_var,
                    bn_weight,
                    bn_bias,
                    training=training
                )
                y_seq = einops.rearrange(y_seq, "(T N) C H W -> T N C H W", T=T)
                s_seq = neuron(y_seq)
                s_seq.backward(grad_s_seq)

            if ctx.needs_input_grad[0]:
                grad_x_seq = x_seq.grad
            if ctx.needs_input_grad[1]:
                grad_weight = weight.grad
            if ctx.needs_input_grad[2]:
                grad_bias = bias.grad
            if ctx.needs_input_grad[3]:
                grad_bn_weight = bn_weight.grad
            if ctx.needs_input_grad[4]:
                grad_bn_bias = bn_bias.grad
        return (
            grad_x_seq, grad_weight, grad_bias, None, None, None, None,
            grad_bn_weight, grad_bn_bias, None, None, None, None, None
        )


class Conv2dBNLIF(nn.Module):

    def __init__(
        self,
        proj: nn.Linear,
        bn: nn.BatchNorm2d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BooleanSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.neuron = neuron
        self.bn = bn
        self.spike_compressor = spike_compressor

    def forward(self, x_seq: torch.Tensor):
        return _Conv2dBNLIFAutogradFunction.apply(
            x_seq,
            self.proj.weight,
            self.proj.bias,
            self.proj.stride,
            self.proj.padding,
            self.proj.dilation,
            self.proj.groups,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
            self.neuron,
            self.spike_compressor,
        )
