from typing import DefaultDict
import inspect
import gc
import torch


def garbage_collection():
    collected = gc.collect()
    print(f"Garbage collection: collected {collected} objects.")


def is_address_referenced_by_tensor(addr):
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor) and obj.data_ptr() == addr:
            return True
    return False


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
