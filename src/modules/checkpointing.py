from typing import Optional, Callable, Tuple, Union, Dict
import threading
import contextlib
import functools
import copy

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.multiprocessing as mp

from .amp import get_autocast_context, is_autocast_enabled
from .compress import *

import sys

sys.path.append("./src")

from utils.profiler import *

_thread_local = threading.local()


@contextlib.contextmanager
def gc_1st_forward():
    _thread_local.in_gc_1st_forward = True
    try:
        yield
    finally:
        _thread_local.in_gc_1st_forward = False


def in_gc_1st_forward():
    return getattr(_thread_local, "in_gc_1st_forward", False)


class InputCompressedGC(autograd.Function):
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
    def forward(
        ctx, forward_function, x_compressor: BaseSpikeCompressor, x_seq, *args
    ):
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

        # save RNG states
        ctx.fwd_rng_state_cpu = torch.get_rng_state()
        if torch.cuda._initialized:
            ctx.fwd_rng_state_cuda = torch.cuda.get_rng_state_all()
        else:
            ctx.fwd_rng_state_cuda = []

        # depend on external autocast context
        with gc_1st_forward(), torch.no_grad():
            outputs = forward_function(x_seq, *args)
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

                    devices = range(torch.cuda.device_count())
                    with torch.random.fork_rng(devices):
                        torch.set_rng_state(ctx.fwd_rng_state_cpu)
                        torch.cuda.set_rng_state_all(ctx.fwd_rng_state_cuda)
                        outputs = ctx.forward_function(x_seq, *args)

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


def input_compressed_gc(
    forward_function, x_compressor: BaseSpikeCompressor, x_seq, *args
):
    if torch.is_grad_enabled():
        x_seq.requires_grad_(True)  # make sure the retval requires grad
        return InputCompressedGC.apply(
            forward_function, x_compressor, x_seq, *args
        )
    else:
        # If gradients are not enabled, call the forward function directly
        return forward_function(x_seq, *args)


def to_gc_function(
    x_compressor: BaseSpikeCompressor,
    forward_function: Optional[Callable] = None
):
    """Convert a forward function to a GC-blocked forward function.

    Usage 1. as a decorator:
    ```
    @to_gc_block(x_compressor)
    def forward_function(x_seq, *args):
        ...
    ```

    Usage 2. as a conversion function:
    ```
    forward_function = to_gc_block(x_compressor, forward_function)
    ```

    Args:
        x_compressor
        forward_function (Callable, optional): if None, use the decorator mode;
            otherwise, use the conversion function mode. Defaults to None.

    Returns:
        Callable: the GC-blocked forward function
    """

    if forward_function is None:  # as a decorator

        def decorator_function(forward_function):

            @functools.wraps(forward_function)
            def wrapped_forward_function(x_seq, *args):
                return input_compressed_gc(
                    forward_function, x_compressor, x_seq, *args
                )

            return wrapped_forward_function

        return decorator_function

    else:  # as a conversion function

        @functools.wraps(forward_function)
        def wrapped_forward_function(x_seq, *args):
            return input_compressed_gc(
                forward_function, x_compressor, x_seq, *args
            )

        return wrapped_forward_function


class GCContainer(nn.Sequential):
    """A GC block module that can be defined just as nn.Sequential."""

    def __init__(self, x_compressor: BaseSpikeCompressor, *args):
        """Construct a GC block module in nn.Sequential style.

        Args:
            x_compressor
            *args: multiple nn.Module
        """
        super().__init__(*args)
        self.x_compressor = x_compressor

    def forward(self, x, *args):
        return input_compressed_gc(super().forward, self.x_compressor, x, *args)

    def extra_repr(self) -> str:
        return f"x_compressor={self.x_compressor.__class__.__name__},"


def _probe_binary_inputs(
    net: nn.Module,
    instance: Union[type, Tuple[type]],
    dummy_input: torch.Tensor,
) -> Dict[nn.Module, bool]:
    """Run dummy forward and record whether target modules receive binary inputs."""
    is_binary = {}
    hooks = []
    net = net.cuda()
    dummy_input = dummy_input.cuda()

    def hook_fn(m, inputs: tuple, out):
        x = inputs[0]  # assume the first input is the one to be checked
        binary = torch.all((x == 0) | (x == 1)).item()
        is_binary[m] = binary

    # register hooks
    for m in net.modules():
        if isinstance(m, instance):
            hooks.append(m.register_forward_hook(hook_fn))

    # run forward
    is_training = net.training
    net.eval()
    with torch.no_grad():
        _ = net(dummy_input)
    if is_training:
        net.train()

    # remove hooks
    for h in hooks:
        h.remove()

    return is_binary


def apply_gc(
    net: nn.Module,
    instance: Union[type, Tuple[type]],
    dummy_input: Optional[torch.Tensor] = None,
    compress_x: bool = True
) -> nn.Module:
    is_binary_input = {}
    if compress_x and dummy_input is not None:
        is_binary_input = _probe_binary_inputs(net, instance, dummy_input)

    def _replace(subnet: nn.Module):
        for name, child in list(subnet.named_children()):
            if isinstance(child, instance):
                b = is_binary_input.get(child, False)
                d = getattr(child, "disable_x_compressor", False)
                x_compressor = (
                    BitSpikeCompressor() if
                    (b and not d) else NullSpikeCompressor()
                )
                setattr(subnet, name, GCContainer(x_compressor, child))
            elif not isinstance(child, GCContainer):
                _replace(child)

    _replace(net)
    return net


def _dummy_train_step(net: nn.Module, dummy_input: torch.Tensor):
    net.train()

    dummy_input = dummy_input.clone().detach()
    # compute input's grad to avoid backward_peak == backward_start @ the 1st layer
    dummy_input.requires_grad = True
    out = net(dummy_input)

    # loss calculation
    if isinstance(out, (tuple, list)):
        loss_terms = [t for t in out if torch.is_tensor(t) and t.requires_grad]
        if not loss_terms:
            raise RuntimeError(
                "No tensor requiring grad found in model outputs."
            )
        loss = torch.stack([t.float().sum() for t in loss_terms]).sum()
    elif torch.is_tensor(out):
        loss = out.sum()
    else:
        raise RuntimeError("Model output is not a tensor/sequence of tensors.")

    loss.backward()


def _train_memory_profile_worker(net, dummy_input, q):
    """`net` and `dummy_input` should be a deep copy of the original model and
    should be located on CPU, since they must be pickle-able.
    """
    net = net.cuda()
    dummy_input = dummy_input.cuda()

    prof = LayerWiseMemoryProfiler(
        (net,),
        model_names=("net",),
        search_mode=("submodules",),
        instances=(GCContainer,),
    )
    _dummy_train_step(net, dummy_input)
    results = prof.export(output=False)
    prof.close()

    q.put(results)


def _inference_time_profile_worker(net, dummy_input, q, N=50):
    """`net` and `dummy_input` should be a deep copy of the original model and
    should be located on CPU, since they must be pickle-able.
    """
    net = net.cuda()
    dummy_input = dummy_input.cuda()

    prof = LayerWiseFPCUDATimeProfiler(
        (net,),
        model_names=("net",),
        search_mode=("submodules",),
        instances=(GCContainer,),
    )

    net.eval()
    with torch.no_grad():
        for _ in range(N):
            _ = net(dummy_input)
    results = prof.export(output=False)
    prof.close()

    q.put(results)


def memory_optimization(
    net: nn.Module,
    instance: Union[type, Tuple[type]],
    dummy_input: torch.Tensor = None,
    compress_x: bool = True,
    level: int = 0,
):
    module_obj = {}  # mapping from module name to the module itself
    for n, m in net.named_modules():
        mname = f"net's {n}"
        if isinstance(m, instance):
            module_obj[mname] = m

    ctx = mp.get_context("spawn")

    if level > 0:  # layer-wise GC with input spike compression
        net = apply_gc(net, instance, dummy_input, compress_x)

    if level > 1:  # spatial split
        if dummy_input is None:
            raise ValueError(
                "dummy_input must be provided for memory profiling."
            )

        q = ctx.Queue(maxsize=1)
        p = ctx.Process(
            target=_train_memory_profile_worker,
            args=(copy.deepcopy(net).cpu(), dummy_input.cpu(), q),
        )
        p.start()
        results = q.get()
        p.join()

        print(list(module_obj.keys()))
        print("Layer-wise GCContainer profiling results:", results)

    if level > 3:
        q = ctx.Queue(maxsize=1)
        p = ctx.Process(
            target=_inference_time_profile_worker,
            args=(copy.deepcopy(net).cpu(), dummy_input.cpu(), q),
            kwargs={"N": 50},
        )
        p.start()
        results = q.get()
        p.join()
        print("Layer-wise inference time profiling results:", results)

    return net


class BaseTCGCBlock(nn.Module):

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
