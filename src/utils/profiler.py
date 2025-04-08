"""Memory profiler.
"""
import gc
import inspect
import traceback as tb
from typing import DefaultDict
import time
from pathlib import Path
import os

import torch
import torch.nn as nn

KB = 1024.
MB = 1024. * 1024.


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


class CategoryMemoryProfiler:

    def __init__(self, model, optimizer, filename='snn_memory.prof'):
        self.model = model
        self.optimizer = optimizer
        self.filename = Path(filename)
        self.device_count = torch.cuda.device_count()

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

    def get_memory_stats(self):
        memory_usage = DefaultDict(float)  # KB

        # model weights
        weight_tensors = set()
        for param in self.model.parameters():
            if param.is_cuda:
                nbytes = param.element_size() * param.numel()
                memory_usage['weight'] += nbytes
                weight_tensors.add(param.data_ptr())

        # gradients
        gradient_tensors = set()
        for param in self.model.parameters():
            if param.grad is not None and param.grad.is_cuda:
                nbytes = param.grad.element_size() * param.grad.numel()
                memory_usage['gradient'] += nbytes
                gradient_tensors.add(param.grad.data_ptr())

        # optimizer state
        optimizer_state_tensors = set()
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param in self.optimizer.state:
                    state = self.optimizer.state[param]
                    for key, value in state.items():
                        if torch.is_tensor(value) and value.is_cuda:
                            nbytes = value.element_size() * value.numel()
                            memory_usage['optimizer_state'] += nbytes
                            optimizer_state_tensors.add(value.data_ptr())

        classified_tensors = weight_tensors | gradient_tensors | optimizer_state_tensors
        for x in self.cuda_tensors():
            if x.data_ptr() not in classified_tensors:
                nbytes = x.element_size() * x.numel()
                memory_usage["input_or_state"] = nbytes

        return memory_usage

    def profile(self):
        memory_usage = self.get_memory_stats()

        total_mem = {}
        for device_id in range(self.device_count):
            with torch.cuda.device(device_id):
                total_mem[device_id] = {
                    'allocated': torch.cuda.memory_allocated() / MB,
                    'reserved': torch.cuda.memory_reserved() / MB,
                }

        with open(self.filename, 'a') as f:
            f.write(f"\n=== SNN Memory Stats ({time.ctime()}) ===\n")
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


def flatten_tuple_generator(nested_tuple):
    for item in nested_tuple:
        if isinstance(item, tuple):
            yield from flatten_tuple_generator(item)
        else:
            yield item


def tensor_memory(x: torch.Tensor):
    storage_size = x.untyped_storage().size()
    element_size = x.element_size()
    return storage_size * element_size


class LayerWiseMemoryProfiler:

    def __init__(self, model: nn.Module, instances):
        self.start_memory = {}
        self.peak_memory = {}
        self.module_str = {}

        def pre_hook_generator(name):

            def pre_hook(module, input):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                self.start_memory[name] = torch.cuda.memory_allocated()
                self.peak_memory[name] = 0

            return pre_hook

        def post_hook_generator(name):

            def post_hook(module, input, output):
                torch.cuda.synchronize()
                self.peak_memory[name] = max(
                    torch.cuda.max_memory_allocated(), self.peak_memory[name]
                )

            return post_hook

        for name, m in model.named_modules():
            if isinstance(m, instances):
                self.module_str[name] = str(m)
                m.register_forward_pre_hook(pre_hook_generator(name))
                m.register_forward_hook(post_hook_generator(name))

    def memory_info(self):
        results = []
        for name in self.peak_memory.keys():
            start_memory = self.start_memory[name]
            peak_memory = self.peak_memory[name]
            computation_memory = peak_memory - start_memory
            results.append(
                (name, start_memory, computation_memory, peak_memory)
            )

        results = sorted(results, key=lambda x: x[-1], reverse=True)

        print('-----Memory Info (MB)-----')
        for name, start_memory, computation_memory, peak_memory in results:
            print(
                f"{name}:{self.module_str[name]}\nstart_memory: {start_memory / MB:.3f}, computation: {computation_memory / MB:.3f}, peak_memory: {peak_memory / MB:.3f}"
            )
