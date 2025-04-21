import torch
import sys

sys.path.append("./src")

from utils.profiler import get_all_addresses_referenced_by_tensor

T, B, C, H, W = 10, 32, 128, 48, 48
x = torch.randn(T, B, C, H, W)
x.requires_grad_(True)
f = torch.nn.Conv2d(C, C, 3, 1, 1)
g = torch.nn.BatchNorm2d(C)
w = torch.randn(T, 1, 1, 1, 1)
w.requires_grad_(True)

y = g(f(x.flatten(0, 1)))
y = y.reshape(T, -1, *y.shape[1:])
y = y * w
l = y.sum()

get_all_addresses_referenced_by_tensor(verbose=True)
