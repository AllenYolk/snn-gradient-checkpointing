import sys

sys.path.insert(0, "./src")

import torch
import torch.nn as nn

from models import HandWrittenLIFNode
from utils import *


def test_lif_memory_referenced():
    T = 1000
    N = 64
    C = 200
    x = torch.randn(T, N, C) + 0.6
    print(f"network input: {x.data_ptr()}")

    net = nn.Sequential(
        nn.ReLU(),
        HandWrittenLIFNode(),
        nn.ReLU(),
    )
    x.requires_grad = True

    s = net(x)
    loss = s.sum()
    loss.backward()

    print(f"network output: {s.data_ptr()}")
    print(f"loss: {loss.data_ptr()}")


if __name__ == "__main__":
    test_lif_memory_referenced()
