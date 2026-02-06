from typing import Dict, List
from pathlib import Path
import os

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
from torch.utils import data
import torch.distributed as dist
import torchvision.ops.boxes as box_ops


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = trainloader = data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2
    )

    mean = torch.zeros(3)  # RGB, 3 channels
    std = torch.zeros(3)
    print("==> Computing mean and std..")
    for inputs, _ in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    """Initialize layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode="fan_out")
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def get_one_hot(x: torch.Tensor, l: int):
    x_one_hot = torch.zeros(*x.shape, l, device=x.device)
    x_one_hot.scatter_(1, x.unsqueeze(-1), 1.0)
    return x_one_hot


def txt2list(filename):
    lines_list = []
    with open(filename, "r") as txt:
        for line in txt:
            lines_list.append(line.rstrip("\n"))
    return lines_list


def count_learnable_parameters(model):
    cnt = 0
    for param in model.parameters():
        if param.requires_grad:
            cnt += param.numel()
    return cnt


def move_state_dict_to_cpu(path):
    """Move the state dict saved in the specified path to CPU."""
    path = Path(path)
    sd = torch.load(path)
    if not isinstance(sd, dict):
        raise TypeError("`path` should point to a file containing a model state dict")

    cpu_sd = {k: v.cpu() for k, v in sd.items()}
    torch.save(cpu_sd, path)


def tensor_size(x: torch.Tensor):  # in bytes
    return x.element_size() * x.numel()


def resolve_device() -> str:
    """Resolve the logical device for the current process.

    Priority:
      1) If no cuda available -> cpu
      2) LOCAL_RANK / SLURM_LOCALID / OMPI_COMM_WORLD_LOCAL_RANK env
      3) If torch.distributed initialized -> use rank % ngpus
      4) torch.cuda.current_device()
      5) fallback to cuda
    """
    if not torch.cuda.is_available():
        return "cpu"

    # common env vars
    for k in (
        "LOCAL_RANK",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
    ):
        v = os.environ.get(k)
        if v is not None:
            try:
                return f"cuda:{int(v)}"
            except Exception:
                pass

    # if dist inited, use rank % n_gpus
    try:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            n_gpu = torch.cuda.device_count()
            if n_gpu > 0:
                return f"cuda:{rank % n_gpu}"
    except Exception:
        pass

    # fallback to current_device (logical ID after CUDA_VISIBLE_DEVICES)
    try:
        return f"cuda:{torch.cuda.current_device()}"
    except Exception:
        return "cuda"
