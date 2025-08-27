import multiprocessing as mp

import torch

try:
    from torch.amp import autocast, GradScaler
    USE_CUDA_DOT_AMP = False
    if mp.current_process().name == "MainProcess":
        print("Use torch.amp")
except Exception:
    from torch.cuda.amp import autocast, GradScaler
    USE_CUDA_DOT_AMP = True
    if mp.current_process().name == "MainProcess":
        print("torch.amp is not available. Use torch.cuda.amp instead.")

AUTOCAST_DTYPE = torch.float16
CACHE_ENABLED = False


def get_autocast_context(enabled: bool):
    """A wrapper for torch.amp.autocast or torch.cuda.amp.autocast context.
    """
    autocast_params = {
        "enabled": enabled,
        "dtype": AUTOCAST_DTYPE,
        "cache_enabled": CACHE_ENABLED,
    }
    if not USE_CUDA_DOT_AMP:
        autocast_params["device_type"] = "cuda"

    return autocast(**autocast_params)


def is_autocast_enabled():
    if USE_CUDA_DOT_AMP:
        return torch.is_autocast_enabled()
    else:
        return (
            torch.is_autocast_enabled("cpu") or
            torch.is_autocast_enabled("cuda")
        )
