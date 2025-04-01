import sys

sys.path.insert(0, "./src")

import torch
import torch.nn as nn

from models import *
from utils import *


def test_lif_address_referenced():
    T = 1000
    N = 64
    C = 200
    x = torch.randn(T, N, C) + 0.6
    print(f"network input: {x.data_ptr()}")

    net = nn.Sequential(
        nn.ReLU(),
        HandWrittenLIF(),
        nn.ReLU(),
    )
    x.requires_grad = True

    s = net(x)
    loss = s.sum()
    loss.backward()

    print(f"network output: {s.data_ptr()}")
    print(f"loss: {loss.data_ptr()}")


def test_linear_lif_address_referenced():
    T = 1000
    N = 64
    C = 200
    x = torch.randn(T, N, C) + 0.6
    print(f"network input: {x.data_ptr()}")

    h_quantizer_type = "IdentityHQuantizer"
    net = nn.Sequential(
        LinearLIF(
            proj=nn.Linear(C, C),
            neuron=HandWrittenLIF(
                h_quantizer=get_h_quantizer(h_quantizer_type)
            ),
            spike_compressor=Uint8SpikeCompressor(),
        ),
        LinearLIF(
            proj=nn.Linear(C, C),
            neuron=HandWrittenLIF(
                h_quantizer=get_h_quantizer(h_quantizer_type)
            ),
            spike_compressor=BooleanSpikeCompressor(),
        ),
        LinearLIF(
            proj=nn.Linear(C, C),
            neuron=HandWrittenLIF(
                h_quantizer=get_h_quantizer(h_quantizer_type)
            ),
            spike_compressor=BitSpikeCompressor(),
        ),
        LinearLIF(
            proj=nn.Linear(C, C),
            neuron=HandWrittenLIF(
                h_quantizer=get_h_quantizer(h_quantizer_type)
            ),
            spike_compressor=SparseSpikeCompressor(),
        )
    )
    x.requires_grad = True

    s = net(x)
    loss = s.sum()
    print("End of FP")
    get_all_addresses_referenced_by_tensor(verbose=True)
    print("Begin of BP")
    loss.backward()

    print(f"network output: {s.data_ptr()}")
    print(f"loss: {loss.data_ptr()}")


if __name__ == "__main__":
    # test_lif_address_referenced()
    test_linear_lif_address_referenced()
