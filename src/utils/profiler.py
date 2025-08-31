import abc
import gc
import inspect
from typing import Tuple
from collections import defaultdict
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


def _get_all_addresses_referenced_by_tensor(depth=float("inf"), verbose=False):
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

    addr_to_tensor = defaultdict(list)
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


def _cuda_tensors():
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


class BaseProfiler(abc.ABC):
    """For profiling memory / time usage.
    """

    def __init__(self, models: Tuple[nn.Module]):
        """Constructing a profiler.

        Args:
            models (Tuple[Module]): a tuple of targeting nn.Module
        """
        if isinstance(models, nn.Module):
            models = (models,)
        elif not isinstance(models, tuple):
            models = tuple(models)
        self.models = models

    @abc.abstractmethod
    def export(self):
        pass

    @abc.abstractmethod
    def close(self):
        pass


class CategoryMemoryProfiler(BaseProfiler):

    def __init__(
        self,
        models: Tuple[nn.Module],
        optimizers: Tuple[optim.Optimizer],
        log_path='snn_memory.prof.txt',
    ):
        super().__init__(models)

        if isinstance(optimizers, optim.Optimizer):
            optimizers = (optimizers,)
        elif not isinstance(optimizers, tuple):
            optimizers = tuple(optimizers)
        self.optimizers = optimizers

        self.log_path = Path(log_path)
        if self.log_path.exists():
            os.remove(self.log_path)

        self.device_count = torch.cuda.device_count()

    def _get_memory_stats(self):
        memory_usage = defaultdict(float)  # KB

        # model weights
        weight_tensors = set()
        for model in self.models:
            for param in model.parameters():
                if param.is_cuda:
                    nbytes = param.element_size() * param.numel()
                    memory_usage['weight'] += nbytes
                    weight_tensors.add(param.data_ptr())
            for buffer in model.buffers():
                if buffer.is_cuda:
                    nbytes = buffer.element_size() * buffer.numel()
                    memory_usage["buffer"] += nbytes
                    weight_tensors.add(buffer.data_ptr())

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
        for x in _cuda_tensors():
            if x.data_ptr() not in classified_tensors:
                nbytes = x.element_size() * x.numel()
                memory_usage["input_or_state"] += nbytes
                classified_tensors.add(x.data_ptr())

        return memory_usage

    def export(self, depth=2, output: bool = True, *args, **kwargs):
        memory_usage = self._get_memory_stats()

        total_mem = {}
        for device_id in range(self.device_count):
            with torch.cuda.device(device_id):
                total_mem[device_id] = {
                    'allocated': torch.cuda.memory_allocated() / MB,
                    'reserved': torch.cuda.memory_reserved() / MB,
                }

        if output:
            caller_str = _get_caller_info(depth)
            header_str = (
                f"=== Category-wise Memory ({time.ctime()}; {caller_str}) ==="
            )

            out_str = "=" * len(header_str) + "\n"
            out_str += header_str + "\n"
            out_str += "=" * len(header_str) + "\n"
            for device_id in range(self.device_count):
                out_str += (
                    f"cuda:{device_id} - "
                    f"Allocated: {total_mem[device_id]['allocated']:.2f} MB, "
                    f"Reserved: {total_mem[device_id]['reserved']:.2f} MB\n"
                )
            out_str += "Memory Usage by Category:\n"
            for category, usage in memory_usage.items():
                out_str += f"  {category}: {usage / MB:.2f} MB\n"
            out_str += f"  Total: {sum(memory_usage.values()) / MB:.2f} MB\n"
            out_str += "=" * len(header_str) + "\n"
            out_str += "=" * len(header_str) + "\n"*3

            print(out_str)
            with open(self.log_path, 'a') as f:
                f.write(out_str)

        return memory_usage, total_mem

    def close(self):
        pass


class LayerWiseMemoryProfiler(BaseProfiler):

    field_idx = {
        "name": 0,
        "forward_start_memory": 1,
        "forward_end_memory": 2,
        "forward_peak_memory": 3,
        "forward_delta_memory": 4,
        "backward_start_memory": 5,
        "backward_end_memory": 6,
        "backward_peak_memory": 7,
        "backward_delta_memory": 8,
    }

    def __init__(
        self,
        models: Tuple[nn.Module],
        model_names: Tuple[str] = None,
        search_mode: Tuple[str] = ("direct_children",),
        instances: Tuple[nn.Module] = (nn.Module,),
        log_path='layer_memory.prof.txt',
        data_path='layer_memory.prof.pt',
    ):
        super().__init__(models)

        if model_names is None:
            model_names = tuple([f"net{i}" for i in range(len(models))])
        if not isinstance(model_names, (tuple, list)):
            raise ValueError("model_names should be a tuple of strings")
        if len(model_names) != len(models):
            raise ValueError(
                "model_names should have the same length as models"
            )

        if isinstance(search_mode, str):
            search_mode = (search_mode,)
        elif not isinstance(search_mode, tuple):
            search_mode = tuple(search_mode)
        if len(search_mode) != len(self.models):
            raise ValueError(
                "search_mode should have the same length as models"
            )

        if isinstance(instances, nn.Module):
            instances = (instances,)
        elif not isinstance(instances, tuple):
            instances = tuple(instances)
        self.instances = instances

        self.log_path = Path(log_path)
        if self.log_path.exists():
            os.remove(self.log_path)
        self.data_path = Path(data_path)
        if self.data_path.exists():
            os.remove(self.data_path)

        self.forward_start_memory = defaultdict(float)
        self.forward_end_memory = defaultdict(float)
        self.forward_peak_memory = defaultdict(float)
        self.backward_start_memory = defaultdict(float)
        self.backward_end_memory = defaultdict(float)
        self.backward_peak_memory = defaultdict(float)
        self.hooks = []
        self.module_obj = {}  # mapping from module name to the module itself

        def pre_hook_generator(name):

            def pre_hook(module, input):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
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
                self.forward_end_memory[name] = torch.cuda.memory_allocated()

            return post_hook

        def backward_pre_hook_generator(name):

            def backward_pre_hook(module, grad_output):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
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
                self.backward_end_memory[name] = torch.cuda.memory_allocated()

            return backward_post_hook

        for i, model in enumerate(self.models):
            if search_mode[i] == "self":
                it = (("self", model),)
            elif search_mode[i] == "submodules":
                it = model.named_modules()
            elif search_mode[i] == "direct_children":
                it = model.named_children()
            for name, m in it:
                if isinstance(m, instances):
                    mname = f"{model_names[i]}'s {name}"
                    self.module_obj[mname] = m
                    h1 = m.register_forward_pre_hook(pre_hook_generator(mname))
                    h2 = m.register_forward_hook(post_hook_generator(mname))
                    h3 = m.register_full_backward_pre_hook(
                        backward_pre_hook_generator(mname)
                    )
                    h4 = m.register_full_backward_hook(
                        backward_post_hook_generator(mname)
                    )
                    self.hooks += [h1, h2, h3, h4]

    def export(
        self, depth=2, sort_by=None, output: bool = True, *args, **kwargs
    ):
        results = []
        for name in self.forward_peak_memory.keys():
            f_start = self.forward_start_memory[name]
            f_end = self.forward_end_memory[name]
            f_peak = self.forward_peak_memory[name]
            f_delta = f_peak - f_start
            b_start = self.backward_start_memory[name]
            b_end = self.backward_end_memory[name]
            b_peak = self.backward_peak_memory[name]
            b_delta = b_peak - b_start

            results.append((
                name, f_start, f_end, f_peak, f_delta, b_start, b_end, b_peak,
                b_delta
            ))

        if sort_by is not None:
            results = sorted(
                results, key=lambda x: x[self.field_idx[sort_by]], reverse=True
            )
        else:
            results = sorted(results, key=lambda x: max(x[1:]), reverse=True)

        if output:
            caller_str = _get_caller_info(depth)
            header_str = (
                f"=== Layer-wise Memory Stats ({time.ctime()}; {caller_str}) ==="
            )

            out_str = "=" * len(header_str) + "\n"
            out_str += header_str + "\n"
            out_str += "=" * len(header_str) + "\n"
            for (
                name, f_start, f_end, f_peak, f_delta, b_start, b_end, b_peak,
                b_delta
            ) in results:
                out_str += f"{name}:{str(self.module_obj[name])}\n"
                out_str += f" forward_start_memory: {f_start / MB:.2f} MB, "
                out_str += f"forward_end_memory: {f_end / MB:.2f} MB, "
                out_str += f"forward_peak_memory: {f_peak / MB:.2f} MB, "
                out_str += f"forward_delta_memory: {f_delta / MB:.2f} MB, "
                out_str += f"\n backward_start_memory: {b_start / MB:.2f} MB, "
                out_str += f"backward_end_memory: {b_end / MB:.2f} MB, "
                out_str += f"backward_peak_memory: {b_peak / MB:.2f} MB, "
                out_str += f"backward_delta_memory: {b_delta / MB:.2f} MB\n"
            out_str += "=" * len(header_str) + "\n"
            out_str += "=" * len(header_str) + "\n"*3

            print(out_str)
            with open(self.log_path, 'a') as f:
                f.write(out_str)

            torch.save({
                "forward_start_memory": self.forward_start_memory,
                "forward_end_memory": self.forward_end_memory,
                "forward_peak_memory": self.forward_peak_memory,
                "backward_start_memory": self.backward_start_memory,
                "backward_end_memory": self.backward_end_memory,
                "backward_peak_memory": self.backward_peak_memory,
            }, self.data_path)

        return results

    def close(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def __del__(self):
        self.close()


class LayerWiseFPCUDATimeProfiler(BaseProfiler):

    def __init__(
        self,
        models: Tuple[nn.Module],
        model_names: Tuple[str] = None,
        search_mode: Tuple[str] = ("direct_children",),
        instances: Tuple[nn.Module] = (nn.Module,),
        warmup: int = 10,
        log_path='layer_time.prof.txt',
    ):
        super().__init__(models)

        if model_names is None:
            model_names = tuple([f"net{i}" for i in range(len(models))])
        if not isinstance(model_names, (tuple, list)):
            raise ValueError("model_names should be a tuple of strings")
        if len(model_names) != len(models):
            raise ValueError(
                "model_names should have the same length as models"
            )

        if isinstance(search_mode, str):
            search_mode = (search_mode,)
        elif not isinstance(search_mode, tuple):
            search_mode = tuple(search_mode)
        if len(search_mode) != len(self.models):
            raise ValueError(
                "search_mode should have the same length as models"
            )

        if isinstance(instances, nn.Module):
            instances = (instances,)
        elif not isinstance(instances, tuple):
            instances = tuple(instances)
        self.instances = instances

        self.warmup = warmup

        self.log_path = Path(log_path)
        if self.log_path.exists():
            os.remove(self.log_path)

        self.result = defaultdict(list)
        self.module_obj = {}
        self.start_events = {}
        self.hooks = []

        def pre_hook_generator(name):

            def pre_hook(module, input):
                start_event = torch.cuda.Event(enable_timing=True)
                # make sure that previous modules have been FPed
                torch.cuda.synchronize()
                start_event.record()
                self.start_events[name] = start_event

            return pre_hook

        def post_hook_generator(name):

            def post_hook(module, input, output):
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                # make sure that the current module have been FPed
                torch.cuda.synchronize()
                start_event = self.start_events.get(name, None)
                if start_event is not None:
                    elapsed_time = start_event.elapsed_time(end_event)
                    self.result[name].append(elapsed_time)
                    self.start_events[name] = None

            return post_hook

        for i, model in enumerate(self.models):
            if search_mode[i] == "self":
                it = (("self", model),)
            elif search_mode[i] == "submodules":
                it = model.named_modules()
            elif search_mode[i] == "direct_children":
                it = model.named_children()
            for name, m in it:
                if isinstance(m, instances):
                    mname = f"{model_names[i]}'s {name}"
                    self.module_obj[mname] = m
                    pre_handle = m.register_forward_pre_hook(
                        pre_hook_generator(mname)
                    )
                    post_handle = m.register_forward_hook(
                        post_hook_generator(mname)
                    )
                    self.start_events[mname] = None
                    self.hooks += [pre_handle, post_handle]

    def export(self, output: bool = True, *args, **kwargs):
        table = []
        for name in self.result.keys():
            forward_time = self.result[name][self.warmup:]
            avg_t = sum(forward_time) / len(forward_time)
            table.append((name, avg_t))

        table = sorted(table, key=lambda x: x[1], reverse=True)

        if output:
            out_str = ""
            for name, avg_t in table:
                out_str += f"{name}:{str(self.module_obj[name])} => "
                out_str += f"{avg_t:.2f} ms\n\n"

            print(out_str)
            with open(self.log_path, "a") as f:
                f.write(out_str)

        return table

    def close(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def __del__(self):
        self.close()


class ProfilerList(list):

    def __init__(self, *args):
        super().__init__()
        for e in args:
            self.append(e)

    def export(self, *args, **kwargs):
        for p in self:
            p.export(*args, **kwargs)

    def close(self):
        for p in self:
            p.close()

    def __del__(self):
        self.close()
