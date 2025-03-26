import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from ..compress import *
from ..neuron import SJSlidingPSN, SJPSN
from ..kernels import *
from .checkpointing import SNNCheckpointingBlock


class Conv1dLIF(nn.Module):

    def __init__(
        self,
        proj: nn.Conv1d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq, weight, bias, stride, padding, dilation, groups, neuron,
        in_backward
    ):
        y_seq = conv1d_forward(
            x_seq, weight, bias, stride, padding, dilation, groups
        )
        return neuron(y_seq)

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlock.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.proj.weight,
            self.proj.bias,
            self.proj.stride,
            self.proj.padding,
            self.proj.dilation,
            self.proj.groups,
            self.neuron,
        )


class Conv1dPSN(nn.Module):

    def __init__(
        self,
        proj: nn.Conv1d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        neuron_weight,
        neuron_bias,
        in_backward=False
    ):
        x_seq = conv1d_forward(
            x_seq, weight, bias, stride, padding, dilation, groups
        )
        return SJPSN.forward_function(x_seq, neuron_weight, neuron_bias)

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlock.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.proj.weight,
            self.proj.bias,
            self.proj.stride,
            self.proj.padding,
            self.proj.dilation,
            self.proj.groups,
            self.neuron.weight,
            self.neuron.bias,
        )


class Conv1dSlidingPSN(nn.Module):

    def __init__(
        self,
        proj: nn.Conv1d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        neuron_weight,
        neuron_bias,
        neuron_k,
        in_backward=False
    ):
        x_seq = conv1d_forward(
            x_seq, weight, bias, stride, padding, dilation, groups
        )
        return SJSlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlock.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.proj.weight,
            self.proj.bias,
            self.proj.stride,
            self.proj.padding,
            self.proj.dilation,
            self.proj.groups,
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
        )


class Conv1dBNLIF(nn.Module):

    def __init__(
        self,
        proj: nn.Conv1d,
        bn: nn.BatchNorm1d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        training,
        neuron,
        in_backward=False
    ):
        x_seq = conv1d_bn_forward(
            x_seq,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.1 if in_backward else 0.
        )
        return neuron(x_seq)

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlock.apply(
            self.conventional_forward,
            self.spike_compressor,
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
        )


class Conv1dBNPSN(nn.Module):

    def __init__(
        self,
        proj: nn.Conv1d,
        bn: nn.BatchNorm1d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        training,
        neuron_weight,
        neuron_bias,
        in_backward=False
    ):
        x_seq = conv1d_bn_forward(
            x_seq,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.1 if in_backward else 0.
        )
        return SJPSN.forward_function(x_seq, neuron_weight, neuron_bias)

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlock.apply(
            self.conventional_forward,
            self.spike_compressor,
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
            self.neuron.weight,
            self.neuron.bias,
        )


class _Conv1dBNSlidingPSNAutogradFunction(autograd.Function):

    @staticmethod
    def forward(
        ctx, x_seq, weight, bias, stride, padding, dilation, groups, bn_weight,
        bn_bias, bn_running_mean, bn_running_var, training, neuron_weight,
        neuron_bias, neuron_k, spike_compressor
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
                neuron_weight,
                neuron_bias,
            )
            ctx.stride = stride
            ctx.padding = padding
            ctx.dilation = dilation
            ctx.groups = groups
            ctx.training = training
            ctx.spike_compressor = spike_compressor
            ctx.x_seq_shape = x_seq.shape
            ctx.neuron_k = neuron_k

    @staticmethod
    def backward(ctx, grad_s_seq):
        grad_x_seq, grad_weight, grad_bias = None, None, None
        grad_bn_weight, grad_bn_bias = None, None
        grad_neuron_weight, grad_neuron_bias = None, None

        if any(ctx.needs_input_grad):
            stride, padding, dilation, groups = (
                ctx.stride, ctx.padding, ctx.dilation, ctx.groups
            )
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
                bn_weight = bn_weight.detach().requires_grad_(True)
                bn_bias = bn_bias.detach().requires_grad_(True)
                neuron_weight = neuron_weight.detach().requires_grad_(True)
                neuron_bias = neuron_bias.detach().requires_grad_(True)
                y_seq = conv1d_bn_forward(
                    x_seq,
                    weight,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups,
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
            if ctx.needs_input_grad[7]:
                grad_bn_weight = bn_weight.grad
            if ctx.needs_input_grad[8]:
                grad_bn_bias = bn_bias.grad
            if ctx.needs_input_grad[12]:
                grad_neuron_weight = neuron_weight.grad
            if ctx.needs_input_grad[13]:
                grad_neuron_bias = neuron_bias.grad
        return (
            grad_x_seq, grad_weight, grad_bias, None, None, None, None,
            grad_bn_weight, grad_bn_bias, None, None, None, grad_neuron_weight,
            grad_neuron_bias, None, None
        )


class Conv1dBNSlidingPSN(nn.Module):

    def __init__(
        self,
        proj: nn.Conv1d,
        bn: nn.BatchNorm1d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.proj = proj
        self.bn = bn
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        training,
        neuron_weight,
        neuron_bias,
        neuron_k,
        in_backward=False
    ):
        x_seq = conv1d_bn_forward(
            x_seq,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.1 if in_backward else 0.
        )
        return SJSlidingPSN.forward_function(
            x_seq, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlock.apply(
            self.conventional_forward,
            self.spike_compressor,
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
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
        )


class AvgPool1dConv1dBNLIF(nn.Module):

    def __init__(
        self,
        pool: nn.AvgPool1d,
        proj: nn.Conv1d,
        bn: nn.BatchNorm1d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.pool = pool
        self.proj = proj
        self.bn = bn
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        training,
        neuron,
        in_backward=False
    ):
        y_seq = avgpool1d_conv1d_bn_forward(
            x_seq,
            pool_kernel_size,
            pool_stride,
            pool_padding,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.1 if in_backward else 0.
        )
        return neuron(y_seq)

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlock.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.pool.kernel_size[0],
            self.pool.stride[0],
            self.pool.padding[0],
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
        )


class AvgPool1dConv1dBNPSN(nn.Module):

    def __init__(
        self,
        pool: nn.AvgPool1d,
        proj: nn.Conv1d,
        bn: nn.BatchNorm1d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.pool = pool
        self.proj = proj
        self.bn = bn
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        training,
        neuron_weight,
        neuron_bias,
        in_backward=False
    ):
        y_seq = avgpool1d_conv1d_bn_forward(
            x_seq,
            pool_kernel_size,
            pool_stride,
            pool_padding,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.1 if in_backward else 0.
        )
        return SJPSN.forward_function(y_seq, neuron_weight, neuron_bias)

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlock.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.pool.kernel_size[0],
            self.pool.stride[0],
            self.pool.padding[0],
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
            self.neuron.weight,
            self.neuron.bias,
        )


class AvgPool1dConv1dBNSlidingPSN(nn.Module):

    def __init__(
        self,
        pool: nn.AvgPool1d,
        proj: nn.Conv1d,
        bn: nn.BatchNorm1d,
        neuron: nn.Module,
        spike_compressor: BaseSpikeCompressor = BitSpikeCompressor()
    ):
        super().__init__()
        self.pool = pool
        self.proj = proj
        self.bn = bn
        self.neuron = neuron
        self.spike_compressor = spike_compressor

    @staticmethod
    def conventional_forward(
        x_seq,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        training,
        neuron_weight,
        neuron_bias,
        neuron_k,
        in_backward=False
    ):

        y_seq = avgpool1d_conv1d_bn_forward(
            x_seq,
            pool_kernel_size,
            pool_stride,
            pool_padding,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum=0.1 if in_backward else 0.
        )
        return SJSlidingPSN.forward_function(
            y_seq, neuron_weight, neuron_bias, neuron_k
        )

    def forward(self, x_seq: torch.Tensor):
        return SNNCheckpointingBlock.apply(
            self.conventional_forward,
            self.spike_compressor,
            x_seq,
            self.pool.kernel_size[0],
            self.pool.stride[0],
            self.pool.padding[0],
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
            self.neuron.weight,
            self.neuron.bias,
            self.neuron.k,
        )
