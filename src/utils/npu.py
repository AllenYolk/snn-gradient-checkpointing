import multiprocessing as mp


def use_torch_npu():
    """
    Returns:
        True -> NPU is available, and all "cuda" related code will be automatically replaced with "npu".
        False -> NPU is not available, and "cuda" related code will remain unchanged.
    """
    try:
        import torch_npu

        npu_available = torch_npu.npu.is_available()
    except:
        npu_available = False
    # if we have torch_npu package and npu is available,
    # we can replace all "cuda" related code with "npu" by the following line
    if npu_available:
        from torch_npu.contrib import transfer_to_npu

    if mp.current_process().name == "MainProcess":
        if npu_available:
            print("NPU is available.")
        else:
            print("NPU is not available.")

    return npu_available
