import torch
import torch.nn as nn
import torch.autograd as autograd

from ..amp import get_autocast_context, is_autocast_enabled


class InputCompressedGCFunction(autograd.Function):
    """Gradient checkpointing with input compression.

    Args:
        forward_function (callable): the forward function whose arguments will
            be checkpointed.
        x_compressor (BaseSpikeCompressor): the compressor for x_seq
        x_seq (Tensor): the input to be compressed and checkpointed.
        *args: other arguments that will be checkpointed without compression.

    Returns:
        a Tensor or a tuple

    Reference:
    https://github.com/pytorch/pytorch/blob/v2.6.0/torch/utils/checkpoint.py
    """

    @staticmethod
    def forward(ctx, forward_function, x_compressor, x_seq, *args):
        if any(ctx.needs_input_grad):
            ctx.forward_function = forward_function
            ctx.x_compressor = x_compressor
            ctx.x_seq_shape = x_seq.shape
            ctx.is_autocast_enabled = is_autocast_enabled()

            input_args = []  # (x_seq_compressed, *args); tensors -> None
            tensor_args = []  # tensors in (x_seq_compressed, *args)
            tensor_args_indices = []  # indices of the tensors in (*args,)
            x_seq_compressed = x_compressor.compress(x_seq)
            if torch.is_tensor(x_seq_compressed):
                tensor_args.append(x_seq_compressed)
                input_args.append(None)
            else:
                input_args.append(x_seq_compressed)
            for i, arg in enumerate(args):
                if torch.is_tensor(arg):
                    tensor_args.append(arg)
                    tensor_args_indices.append(i)
                    input_args.append(None)
                else:
                    input_args.append(arg)
            ctx.save_for_backward(*tensor_args)
            ctx.input_args = input_args
            ctx.tensor_args_indices = tensor_args_indices

        # depend on external autocast context
        with torch.no_grad():
            outputs = forward_function(x_seq, *args, in_backward=False)
        return outputs  # tensor or tuple

    @staticmethod
    def backward(ctx, *grad_outputs):
        cnt_input = len(ctx.input_args) + 2
        grads = [None] * cnt_input

        if any(ctx.needs_input_grad):
            x_seq_compressed, *args = ctx.input_args
            if x_seq_compressed is None:  # x_seq_compressed is a tensor
                x_seq_compressed, *tensor_args = ctx.saved_tensors
            else:
                tensor_args = ctx.saved_tensors  # tensors in (*args,)
            tensor_args_indices = ctx.tensor_args_indices  # idx of the tensors in (*args,)
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
                    outputs = ctx.forward_function(
                        x_seq, *args, in_backward=False
                    )
                # grad_outputs is a tuple, while outputs can be a tensor or a tuple
                if isinstance(outputs, torch.Tensor):
                    outputs = (outputs,)
                torch.autograd.backward(outputs, grad_outputs)

            if ctx.needs_input_grad[2]:
                grads[2] = x_seq.grad
            for idx in tensor_args_indices:
                if ctx.needs_input_grad[idx + 3]:
                    grads[idx + 3] = args[idx].grad

        return tuple(grads)


class BaseGCBlock(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def conventional_forward(*args, **kwargs):
        raise NotImplementedError(
            "The conventional forward function is not implemented."
        )

    def forward(self, x_seq: torch.Tensor):
        raise NotImplementedError("The forward function is not implemented.")

    def extra_repr(self) -> str:
        if hasattr(self, "spike_compressor"):
            return (
                f"spike_compressor={self.spike_compressor.__class__.__name__}"
            )
        else:
            return ""


class BaseTCGCBlock(BaseGCBlock):

    def __init__(self, n_chunk: int):
        super().__init__()
        self.n_chunk = n_chunk

    @staticmethod
    def conventional_forward(*args, **kwargs):
        # RNN-style forward
        raise NotImplementedError(
            "The temporally chunked conventional forward function is not implemented."
        )

    def forward(self, x_seq: torch.Tensor):
        # 1. temporally chunk x_seqs = torch.chunk(x_seq, self.n_chunk, dim=0)
        # 2. state initialization
        # 3. chunked forward
        # 4. stack chunked outputs
        raise NotImplementedError("The forward function is not implemented.")

    def extra_repr(self) -> str:
        if hasattr(self, "spike_compressor"):
            return (
                f"n_chunk={self.n_chunk},"
                f"spike_compressor={self.spike_compressor.__class__.__name__}"
            )
        else:
            return f"n_chunk={self.n_chunk}"
