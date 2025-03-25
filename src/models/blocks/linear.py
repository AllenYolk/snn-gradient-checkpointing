import torch
import torch.nn as nn
import torch.autograd as autograd

from ..compress import *
from ..neuron import SJSlidingPSN, SJPSN
from ..compiled import *


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
        return neuron(linear_forward_compiled(x_seq, weight, bias))

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
                y_seq = linear_forward_compiled(x_seq, weight, bias)
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


class _LinearPSNAutogradFunction(autograd.Function):

    @staticmethod
    def forward(
        ctx, x_seq, weight, bias, neuron_weight, neuron_bias, spike_compressor
    ):
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(
                spike_compressor.compress(x_seq), weight, bias, neuron_weight,
                neuron_bias
            )
            ctx.spike_compressor = spike_compressor
            ctx.x_seq_shape = x_seq.shape
        x_seq = linear_forward_compiled(x_seq, weight, bias)
        s_seq = SJPSN.forward_function(x_seq, neuron_weight, neuron_bias)
        return s_seq

    @staticmethod
    def backward(ctx, grad_s_seq):
        grad_x_seq, grad_weight, grad_bias = None, None, None
        grad_neuron_weight, grad_neuron_bias = None, None

        if any(ctx.needs_input_grad):
            spike_compressor = ctx.spike_compressor
            x_seq_shape = ctx.x_seq_shape
            x_seq, weight, bias, neuron_weight, neuron_bias = ctx.saved_tensors
            x_seq = spike_compressor.decompress(x_seq, x_seq_shape)

            with torch.set_grad_enabled(True):
                #! y = x.detach() => y is just a new "pointer" to x's data.
                #! No extra memory is allocated.
                x_seq = x_seq.detach().requires_grad_(True)
                weight = weight.detach().requires_grad_(True)
                bias = bias.detach().requires_grad_(True)
                neuron_weight = neuron_weight.detach().requires_grad_(True)
                neuron_bias = neuron_bias.detach().requires_grad_(True)
                y_seq = linear_forward_compiled(x_seq, weight, bias)
                s_seq = SJPSN.forward_function(
                    y_seq, neuron_weight, neuron_bias
                )
                s_seq.backward(grad_s_seq)

            if ctx.needs_input_grad[0]:
                grad_x_seq = x_seq.grad
            if ctx.needs_input_grad[1]:
                grad_weight = weight.grad
            if ctx.needs_input_grad[2]:
                grad_bias = bias.grad
            if ctx.needs_input_grad[3]:
                grad_neuron_weight = neuron_weight.grad
            if ctx.needs_input_grad[4]:
                grad_neuron_bias = neuron_bias.grad
        return (
            grad_x_seq, grad_weight, grad_bias, grad_neuron_weight,
            grad_neuron_bias, None
        )


class LinearPSN(nn.Module):

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
        return _LinearPSNAutogradFunction.apply(
            x_seq,
            self.proj.weight,
            self.proj.bias,
            self.neuron.weight,
            self.neuron.bias,
            self.spike_compressor,
        )


class _LinearSlidingPSNAutogradFunction(autograd.Function):

    @staticmethod
    def forward(
        ctx, x_seq, weight, bias, neuron_weight, neuron_bias, neuron_k,
        spike_compressor
    ):
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(
                spike_compressor.compress(x_seq), weight, bias, neuron_weight,
                neuron_bias
            )
            ctx.neuron_k = neuron_k
            ctx.spike_compressor = spike_compressor
            ctx.x_seq_shape = x_seq.shape
        x_seq = linear_forward_compiled(x_seq, weight, bias)
        s_seq = SJSlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )
        return s_seq

    @staticmethod
    def backward(ctx, grad_s_seq):
        grad_x_seq, grad_weight, grad_bias = None, None, None
        grad_neuron_weight, grad_neuron_bias = None, None

        if any(ctx.needs_input_grad):
            neuron_k = ctx.neuron_k
            spike_compressor = ctx.spike_compressor
            x_seq_shape = ctx.x_seq_shape
            x_seq, weight, bias, neuron_weight, neuron_bias = ctx.saved_tensors
            x_seq = spike_compressor.decompress(x_seq, x_seq_shape)

            with torch.set_grad_enabled(True):
                #! y = x.detach() => y is just a new "pointer" to x's data.
                #! No extra memory is allocated.
                x_seq = x_seq.detach().requires_grad_(True)
                weight = weight.detach().requires_grad_(True)
                bias = bias.detach().requires_grad_(True)
                neuron_weight = neuron_weight.detach().requires_grad_(True)
                neuron_bias = neuron_bias.detach().requires_grad_(True)
                y_seq = linear_forward_compiled(x_seq, weight, bias)
                s_seq = SJSlidingPSN.forward_function(
                    y_seq, neuron_weight, neuron_bias, neuron_k
                )
                s_seq.backward(grad_s_seq)

            if ctx.needs_input_grad[0]:
                grad_x_seq = x_seq.grad
            if ctx.needs_input_grad[1]:
                grad_weight = weight.grad
            if ctx.needs_input_grad[2]:
                grad_bias = bias.grad
            if ctx.needs_input_grad[3]:
                grad_neuron_weight = neuron_weight.grad
            if ctx.needs_input_grad[4]:
                grad_neuron_bias = neuron_bias.grad
        return (
            grad_x_seq, grad_weight, grad_bias, grad_neuron_weight,
            grad_neuron_bias, None, None
        )


class LinearSlidingPSN(nn.Module):

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
        return _LinearSlidingPSNAutogradFunction.apply(
            x_seq,
            self.proj.weight,
            self.proj.bias,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
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

        x_seq = linear_bn_forward_compiled(
            x_seq,
            weight,
            bias,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.  # disable running stats update
        )
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

                y_seq = linear_bn_forward_compiled(
                    x_seq,
                    weight,
                    bias,
                    bn_weight,
                    bn_bias,
                    bn_running_mean,
                    bn_running_var,
                    training,
                    momentum=0.1
                )
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


class _LinearBNPSNAutogradFunction(autograd.Function):

    @staticmethod
    def forward(
        ctx, x_seq, weight, bias, bn_weight, bn_bias, bn_running_mean,
        bn_running_var, training, neuron_weight, neuron_bias, spike_compressor
    ):
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(
                spike_compressor.compress(x_seq), weight, bias, bn_weight,
                bn_bias, bn_running_mean, bn_running_var, neuron_weight,
                neuron_bias
            )
            ctx.spike_compressor = spike_compressor
            ctx.x_seq_shape = x_seq.shape
            ctx.training = training
        x_seq = linear_bn_forward_compiled(
            x_seq,
            weight,
            bias,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.  # disable running stats update
        )
        s_seq = SJPSN.forward_function(x_seq, neuron_weight, neuron_bias)
        return s_seq

    @staticmethod
    def backward(ctx, grad_s_seq):
        grad_x_seq, grad_weight, grad_bias = None, None, None
        grad_bn_weight, grad_bn_bias = None, None
        grad_neuron_weight, grad_neuron_bias = None, None

        if any(ctx.needs_input_grad):
            training = ctx.training
            spike_compressor = ctx.spike_compressor
            x_seq_shape = ctx.x_seq_shape
            x_seq, weight, bias = ctx.saved_tensors[:3]
            bn_weight, bn_bias = ctx.saved_tensors[3:5]
            bn_running_mean, bn_running_var = ctx.saved_tensors[5:7]
            neuron_weight, neuron_bias = ctx.saved_tensors[7:]
            x_seq = spike_compressor.decompress(x_seq, x_seq_shape)

            with torch.set_grad_enabled(True):
                #! y = x.detach() => y is just a new "pointer" to x's data.
                #! No extra memory is allocated.
                x_seq = x_seq.detach().requires_grad_(True)
                weight = weight.detach().requires_grad_(True)
                bias = bias.detach().requires_grad_(True)
                neuron_weight = neuron_weight.detach().requires_grad_(True)
                neuron_bias = neuron_bias.detach().requires_grad_(True)
                bn_weight = bn_weight.detach().requires_grad_(True)
                bn_bias = bn_bias.detach().requires_grad_(True)
                y_seq = linear_bn_forward_compiled(
                    x_seq,
                    weight,
                    bias,
                    bn_weight,
                    bn_bias,
                    bn_running_mean,
                    bn_running_var,
                    training,
                    momentum=0.1
                )
                s_seq = SJPSN.forward_function(
                    y_seq, neuron_weight, neuron_bias
                )
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
            if ctx.needs_input_grad[8]:
                grad_neuron_weight = neuron_weight.grad
            if ctx.needs_input_grad[9]:
                grad_neuron_bias = neuron_bias.grad
        return (
            grad_x_seq, grad_weight, grad_bias, grad_bn_weight, grad_bn_bias,
            None, None, None, grad_neuron_weight, grad_neuron_bias, None
        )


class LinearBNPSN(nn.Module):

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
        return _LinearBNPSNAutogradFunction.apply(
            x_seq,
            self.proj.weight,
            self.proj.bias,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
            self.neuron.weight,
            self.neuron.bias,
            self.spike_compressor,
        )


class _LinearBNSlidingPSNAutogradFunction(autograd.Function):

    @staticmethod
    def forward(
        ctx, x_seq, weight, bias, bn_weight, bn_bias, bn_running_mean,
        bn_running_var, training, neuron_weight, neuron_bias, neuron_k,
        spike_compressor
    ):
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(
                spike_compressor.compress(x_seq), weight, bias, bn_weight,
                bn_bias, bn_running_mean, bn_running_var, neuron_weight,
                neuron_bias
            )
            ctx.spike_compressor = spike_compressor
            ctx.x_seq_shape = x_seq.shape
            ctx.training = training
            ctx.neuron_k = neuron_k
        x_seq = linear_bn_forward_compiled(
            x_seq,
            weight,
            bias,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.  # disable running stats update
        )
        s_seq = SJSlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )
        return s_seq

    @staticmethod
    def backward(ctx, grad_s_seq):
        grad_x_seq, grad_weight, grad_bias = None, None, None
        grad_bn_weight, grad_bn_bias = None, None
        grad_neuron_weight, grad_neuron_bias = None, None

        if any(ctx.needs_input_grad):
            training = ctx.training
            spike_compressor = ctx.spike_compressor
            x_seq_shape = ctx.x_seq_shape
            neuron_k = ctx.neuron_k
            x_seq, weight, bias = ctx.saved_tensors[:3]
            bn_weight, bn_bias = ctx.saved_tensors[3:5]
            bn_running_mean, bn_running_var = ctx.saved_tensors[5:7]
            neuron_weight, neuron_bias = ctx.saved_tensors[7:]
            x_seq = spike_compressor.decompress(x_seq, x_seq_shape)

            with torch.set_grad_enabled(True):
                #! y = x.detach() => y is just a new "pointer" to x's data.
                #! No extra memory is allocated.
                x_seq = x_seq.detach().requires_grad_(True)
                weight = weight.detach().requires_grad_(True)
                bias = bias.detach().requires_grad_(True)
                neuron_weight = neuron_weight.detach().requires_grad_(True)
                neuron_bias = neuron_bias.detach().requires_grad_(True)
                bn_weight = bn_weight.detach().requires_grad_(True)
                bn_bias = bn_bias.detach().requires_grad_(True)
                y_seq = linear_bn_forward_compiled(
                    x_seq,
                    weight,
                    bias,
                    bn_weight,
                    bn_bias,
                    bn_running_mean,
                    bn_running_var,
                    training,
                    momentum=0.1
                )
                s_seq = SJSlidingPSN.forward_function(
                    y_seq, neuron_weight, neuron_bias, neuron_k
                )
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
            if ctx.needs_input_grad[8]:
                grad_neuron_weight = neuron_weight.grad
            if ctx.needs_input_grad[9]:
                grad_neuron_bias = neuron_bias.grad
        return (
            grad_x_seq, grad_weight, grad_bias, grad_bn_weight, grad_bn_bias,
            None, None, None, grad_neuron_weight, grad_neuron_bias, None, None
        )


class LinearBNSlidingPSN(nn.Module):

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
        return _LinearBNSlidingPSNAutogradFunction.apply(
            x_seq,
            self.proj.weight,
            self.proj.bias,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.training,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
            self.spike_compressor,
        )
