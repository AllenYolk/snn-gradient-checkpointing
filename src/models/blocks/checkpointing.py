import torch
import torch.autograd as autograd
from nvidia import nvcomp

from ..amp import get_autocast_context, is_autocast_enabled


# TODO: make it compatible with nvcomp.Array as x_seq
class SNNCheckpointingBlock(autograd.Function):
    """Reference:
    https://github.com/pytorch/pytorch/blob/v2.6.0/torch/utils/checkpoint.py
    """

    @staticmethod
    def forward(ctx, forward_function, x_compressor, x_seq, *args):
        if any(ctx.needs_input_grad):
            ctx.forward_function = forward_function
            ctx.x_compressor = x_compressor
            ctx.x_seq_shape = x_seq.shape
            ctx.is_autocast_enabled = is_autocast_enabled()

            input_args = []  # (x_seq, *args), with tensors replaced by None
            tensor_args = []  # tensors in (x_seq, *args)
            tensor_args_indices = []  # indices of the tensors in (x_seq, *args)
            x_seq_compressed = x_compressor.compress(x_seq)
            if torch.is_tensor(x_seq_compressed):
                tensor_args.append(x_seq_compressed)
                tensor_args_indices.append(0)
                input_args.append(None)
            else:
                input_args.append(x_seq_compressed)
            for i, arg in enumerate(args):
                if torch.is_tensor(arg):
                    tensor_args.append(arg)
                    tensor_args_indices.append(i + 1)
                    input_args.append(None)
                else:
                    input_args.append(arg)
            ctx.save_for_backward(*tensor_args)
            ctx.input_args = input_args
            ctx.tensor_args_indices = tensor_args_indices

        # depend on external autocast context
        with torch.no_grad():
            y_seq = forward_function(x_seq, *args, in_backward=False)
        return y_seq

    @staticmethod
    def backward(ctx, grad_y_seq):
        cnt_input = len(ctx.input_args) + 2
        grads = [None] * cnt_input

        if any(ctx.needs_input_grad):
            if ctx.input_args[0] is None:  # x_seq_compressed is a tensor
                x_seq_compressed, *tensor_args = ctx.saved_tensors
            else:
                x_seq_compressed = ctx.input_args[0]
                tensor_args = ctx.saved_tensors
            tensor_args_indices = ctx.tensor_args_indices
            args = ctx.input_args
            x_seq_shape = ctx.x_seq_shape

            with torch.set_grad_enabled(True):
                with get_autocast_context(ctx.is_autocast_enabled):
                    x_seq = ctx.x_compressor.decompress(
                        x_seq_compressed, x_seq_shape
                    )
                    x_seq = x_seq.detach().requires_grad_(True)
                    for i, idx in enumerate(tensor_args_indices):
                        if idx == 0:  # x_seq_compressed, skip
                            continue
                        rg = (
                            ctx.needs_input_grad[idx + 3] and
                            tensor_args[i].requires_grad
                        )
                        args[idx -
                             1] = tensor_args[i].detach().requires_grad_(rg)
                    y_seq = ctx.forward_function(x_seq, *args, in_backward=True)
                y_seq.backward(grad_y_seq)

            if ctx.needs_input_grad[2]:
                grads[2] = x_seq.grad
            for idx in tensor_args_indices:
                if ctx.needs_input_grad[idx + 3]:
                    grads[idx + 3] = args[idx].grad

        return tuple(grads)
