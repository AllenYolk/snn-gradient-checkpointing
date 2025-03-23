import gc
import torch


def is_tensor_memory_referenced(addr):
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor) and obj.data_ptr() == addr:
            return True
    return False


def get_all_referenced_tensor_memory():
    tensor_memory = set()
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor):
            tensor_memory.add(obj.data_ptr())
    return list(tensor_memory)


def print_all_active_tensors():
    print("Active tensors:")
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor):
            print(f"\t{obj.data_ptr()} -> {obj.shape}, {obj.dtype}")


def garbage_collection():
    collected = gc.collect()
    print(f"Garbage collection: collected {collected} objects.")
