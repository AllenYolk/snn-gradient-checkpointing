import torch

try:
    from torch.amp import autocast, GradScaler
    USE_CUDA_DOT_AMP = False
    print("Use torch.amp")
except Exception:
    from torch.cuda.amp import autocast, GradScaler
    USE_CUDA_DOT_AMP = True
    print("torch.amp is not available. Use torch.cuda.amp instead.")

AUTOCAST_DTYPE = torch.bfloat16
CACHE_ENABLED = False


def get_autocast_context(enabled: bool):
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
