import torch
import torch.autograd as autograd
from nvidia import nvcomp

from ..amp import get_autocast_context, is_autocast_enabled

CODEC = nvcomp.Codec(algorithm="Zstd", bitstream_kind=nvcomp.BitstreamKind.RAW)


class SNNCheckpointingBlock(autograd.Function):
    """Reference:
    https://github.com/pytorch/pytorch/blob/v2.6.0/torch/utils/checkpoint.py
    """

    @staticmethod
    def forward(ctx, forward_function, x_compressor, x_seq, *args):
        if any(ctx.needs_input_grad):
            ctx.forward_function = forward_function
            ctx.x_compressor = x_compressor
            input_args = []  # *args, with tensors replaced by None
            tensor_args = []  # tensors in *args
            tensor_args_indices = []  # indices of the tensors in *args
            for i, arg in enumerate(args):
                if torch.is_tensor(arg):
                    tensor_args.append(arg)
                    tensor_args_indices.append(i)
                    input_args.append(None)
                else:
                    input_args.append(arg)
            ctx.save_for_backward(x_compressor.compress(x_seq), *tensor_args)
            ctx.input_args = input_args
            ctx.tensor_args_indices = tensor_args_indices
            ctx.x_seq_shape = x_seq.shape
            ctx.is_autocast_enabled = is_autocast_enabled()

        # depend on external autocast context
        with torch.no_grad():
            y_seq = forward_function(x_seq, *args, in_backward=False)
        return y_seq

    @staticmethod
    def backward(ctx, grad_y_seq):
        cnt_input = len(ctx.input_args) + 3
        grads = [None] * cnt_input

        if any(ctx.needs_input_grad):
            x_seq_compressed, *tensor_args = ctx.saved_tensors
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
                        rg = (
                            ctx.needs_input_grad[idx + 3] and
                            tensor_args[i].requires_grad
                        )
                        args[idx] = tensor_args[i].detach().requires_grad_(rg)
                    y_seq = ctx.forward_function(x_seq, *args, in_backward=True)
                y_seq.backward(grad_y_seq)

            if ctx.needs_input_grad[2]:
                grads[2] = x_seq.grad
            for idx in tensor_args_indices:
                if ctx.needs_input_grad[idx + 3]:
                    grads[idx + 3] = args[idx].grad

        return tuple(grads)
