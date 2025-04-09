import abc
import gc
import inspect
from typing import DefaultDict, Tuple
import time
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.optim as optim

KB = 1024.
MB = 1024. * 1024.


def _get_caller_info(depth=1):
    caller_frame = inspect.currentframe().f_back
    for _ in range(depth - 1):
        caller_frame = caller_frame.f_back
    caller_file = caller_frame.f_code.co_filename
    caller_lineno = caller_frame.f_lineno
    caller_func = caller_frame.f_code.co_name
    caller_str = f"{caller_file}:line{caller_lineno}, {caller_func}"
    return caller_str


def get_all_addresses_referenced_by_tensor(depth=float("inf"), verbose=False):
    id_to_name = {}
    current_frame = inspect.currentframe().f_back

    d = 0
    while current_frame and d < depth:
        frame_name = current_frame.f_code.co_name
        locals_dict = current_frame.f_locals
        for obj_name, obj in locals_dict.items():
            if isinstance(obj, torch.Tensor):
                id_to_name[id(obj)] = f"{frame_name}.{obj_name}"
        current_frame = current_frame.f_back
        d += 1

    addr_to_tensor = DefaultDict(list)
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor):
            d = {"tensor": obj, "name": id_to_name.get(id(obj), None)}
            addr_to_tensor[obj.data_ptr()].append(d)

    if verbose:
        title = "Addresses Referenced by Tensors"
        print("=" * 20, title, "=" * 20)
        for addr, tensor_info_list in addr_to_tensor.items():
            print(f"<{addr}>", end="")
            n_indent = len(str(addr)) + 2
            for i, tensor_info in enumerate(tensor_info_list):
                real_indent = 0 if i == 0 else n_indent
                x = tensor_info["tensor"]
                obj_name = tensor_info["name"]
                info_str = f'-> "{obj_name}" ' if obj_name else "-> "
                info_str += f"{list(x.size())} ({x.dtype}, {x.device})"
                print(" " * real_indent, info_str)
        print("=" * (42 + len(title)))

    return addr_to_tensor


class BaseMemoryProfiler(abc.ABC):

    def __init__(self, models, filename):
        if isinstance(models, nn.Module):
            models = (models,)
        elif not isinstance(models, tuple):
            models = tuple(models)
        self.models = models
        self.filename = Path(filename)
        if self.filename.exists():
            os.remove(self.filename)

    @staticmethod
    def cuda_tensors():
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    tensor = obj
                elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                    tensor = obj.data
                else:
                    continue

                if tensor.is_cuda:
                    yield tensor
            except Exception:
                pass

    @abc.abstractmethod
    def profile(self):
        pass


class CategoryMemoryProfiler(BaseMemoryProfiler):

    def __init__(
        self,
        models: Tuple[nn.Module],
        optimizers: Tuple[optim.Optimizer],
        filename='snn_memory.prof'
    ):
        super().__init__(models, filename)
        if isinstance(optimizers, optim.Optimizer):
            optimizers = (optimizers,)
        elif not isinstance(optimizers, tuple):
            optimizers = tuple(optimizers)
        self.optimizers = optimizers
        self.device_count = torch.cuda.device_count()

    def _get_memory_stats(self):
        memory_usage = DefaultDict(float)  # KB

        # model weights
        weight_tensors = set()
        for model in self.models:
            for param in model.parameters():
                if param.is_cuda:
                    nbytes = param.element_size() * param.numel()
                    memory_usage['weight'] += nbytes
                    weight_tensors.add(param.data_ptr())

        # gradients
        gradient_tensors = set()
        for model in self.models:
            for param in model.parameters():
                if param.grad is not None and param.grad.is_cuda:
                    nbytes = param.grad.element_size() * param.grad.numel()
                    memory_usage['gradient'] += nbytes
                    gradient_tensors.add(param.grad.data_ptr())

        # optimizer state
        optimizer_state_tensors = set()
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param in optimizer.state:
                        state = optimizer.state[param]
                        for key, value in state.items():
                            if torch.is_tensor(value) and value.is_cuda:
                                nbytes = value.element_size() * value.numel()
                                memory_usage['optimizer_state'] += nbytes
                                optimizer_state_tensors.add(value.data_ptr())

        classified_tensors = weight_tensors | gradient_tensors | optimizer_state_tensors
        for x in self.cuda_tensors():
            if x.data_ptr() not in classified_tensors:
                nbytes = x.element_size() * x.numel()
                memory_usage["input_or_state"] += nbytes
                classified_tensors.add(x.data_ptr())

        return memory_usage

    def profile(self, depth=2, *args, **kwargs):
        memory_usage = self._get_memory_stats()
        caller_str = _get_caller_info(depth)

        total_mem = {}
        for device_id in range(self.device_count):
            with torch.cuda.device(device_id):
                total_mem[device_id] = {
                    'allocated': torch.cuda.memory_allocated() / MB,
                    'reserved': torch.cuda.memory_reserved() / MB,
                }

        header_str = (
            f"=== Category-wise Memory Stats ({time.ctime()}; {caller_str}) ==="
        )
        with open(self.filename, 'a') as f:
            f.write("=" * len(header_str) + "\n")
            f.write(header_str + "\n")
            f.write("=" * len(header_str) + "\n")
            for device_id in range(self.device_count):
                f.write(
                    f"cuda:{device_id} - "
                    f"Allocated: {total_mem[device_id]['allocated']:.2f} MB, "
                    f"Reserved: {total_mem[device_id]['reserved']:.2f} MB\n"
                )

            f.write("Memory Usage by Category:\n")
            for category, usage in memory_usage.items():
                f.write(f"  {category}: {usage / MB:.2f} MB\n")
            f.write(
                f"  Total Tracked: {sum(memory_usage.values()) / MB:.2f} MB\n"
            )
            f.write("=" * len(header_str) + "\n")
            f.write("=" * len(header_str) + "\n"*3)
            f.flush()


class LayerWiseMemoryProfiler(BaseMemoryProfiler):

    field_idx = {
        "name": 0,
        "forward_start_memory": 1,
        "forward_peak_memory": 2,
        "forward_computation_memory": 3,
        "backward_start_memory": 4,
        "backward_peak_memory": 5,
        "backward_computation_memory": 6
    }

    def __init__(
        self,
        models: Tuple[nn.Module],
        instances: Tuple[nn.Module],
        filename='layer_memory.prof',
        direct_children_only: Tuple[bool] = (False,),
    ):
        super().__init__(models, filename)
        if isinstance(instances, nn.Module):
            instances = (instances,)
        elif not isinstance(instances, tuple):
            instances = tuple(instances)
        self.instances = instances

        if isinstance(direct_children_only, bool):
            direct_children_only = (direct_children_only,)
        elif not isinstance(direct_children_only, tuple):
            direct_children_only = tuple(direct_children_only)
        if len(direct_children_only) != len(self.models):
            raise ValueError(
                "direct_children_only should have the same length as models"
            )

        self.forward_start_memory = DefaultDict(float)
        self.forward_peak_memory = DefaultDict(float)
        self.backward_start_memory = DefaultDict(float)
        self.backward_peak_memory = DefaultDict(float)
        self.module_str = {}

        def pre_hook_generator(name):

            def pre_hook(module, input):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                self.forward_start_memory[name] = torch.cuda.memory_allocated()
                self.forward_peak_memory[name] = 0

            return pre_hook

        def post_hook_generator(name):

            def post_hook(module, input, output):
                torch.cuda.synchronize()
                self.forward_peak_memory[name] = max(
                    torch.cuda.max_memory_allocated(),
                    self.forward_peak_memory[name]
                )

            return post_hook

        def backward_pre_hook_generator(name):

            def backward_pre_hook(module, grad_output):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                self.backward_start_memory[name] = torch.cuda.memory_allocated()
                self.backward_peak_memory[name] = 0

            return backward_pre_hook

        def backward_post_hook_generator(name):

            def backward_post_hook(module, grad_input, grad_output):
                torch.cuda.synchronize()
                self.backward_peak_memory[name] = max(
                    torch.cuda.max_memory_allocated(),
                    self.backward_peak_memory[name]
                )

            return backward_post_hook

        for i, model in enumerate(self.models):
            it = (
                model.named_children()
                if direct_children_only[i] else model.named_modules()
            )
            for name, m in it:
                if isinstance(m, instances):
                    mname = f"net{i}-{name}"
                    self.module_str[mname] = str(m)
                    m.register_forward_pre_hook(pre_hook_generator(mname))
                    m.register_forward_hook(post_hook_generator(mname))
                    m.register_full_backward_pre_hook(
                        backward_pre_hook_generator(mname)
                    )
                    m.register_full_backward_hook(
                        backward_post_hook_generator(mname)
                    )

    def profile(self, depth=2, sort_by="peak_memory", *args, **kwargs):
        results = []
        for name in self.forward_peak_memory.keys():
            forward_start_memory = self.forward_start_memory[name]
            forward_peak_memory = self.forward_peak_memory[name]
            backward_start_memory = self.backward_start_memory[name]
            backward_peak_memory = self.backward_peak_memory[name]
            forward_computation_memory = (
                forward_peak_memory - forward_start_memory
            )
            backward_computation_memory = (
                backward_peak_memory - backward_start_memory
            )
            results.append((
                name, forward_start_memory, forward_peak_memory,
                forward_computation_memory, backward_start_memory,
                backward_peak_memory, backward_computation_memory
            ))

        results = sorted(
            results, key=lambda x: x[self.field_idx[sort_by]], reverse=True
        )

        caller_str = _get_caller_info(depth)
        header_str = (
            f"=== Layer-wise Memory Stats ({time.ctime()}; {caller_str}) ==="
        )

        with open(self.filename, 'a') as f:
            f.write("=" * len(header_str) + "\n")
            f.write(header_str + "\n")
            f.write("=" * len(header_str) + "\n")
            for (
                name, forward_start_memory, forward_peak_memory,
                forward_computation_memory, backward_start_memory,
                backward_peak_memory, backward_computation_memory
            ) in results:
                f.write(
                    f"{name}:{self.module_str[name]}\n"
                    f"  forward_start_memory: {forward_start_memory / MB:.2f} MB, "
                    f"forward_peak_memory: {forward_peak_memory / MB:.2f} MB, "
                    f"forward_computation: {forward_computation_memory / MB:.2f} MB\n"
                    f"  backward_start_memory: {backward_start_memory / MB:.2f} MB, "
                    f"backward_peak_memory: {backward_peak_memory / MB:.2f} MB, "
                    f"backward_computation: {backward_computation_memory / MB:.2f} MB\n"
                )
            f.write("=" * len(header_str) + "\n")
            f.write("=" * len(header_str) + "\n"*3)
            f.flush()


class MemoryProfilerList(list):

    def __init__(self, *args):
        super().__init__()
        for e in args:
            self.append(e)

    def profile(self, depth=3, *args, **kwargs):
        for p in self:
            p.profile(depth, *args, **kwargs)
