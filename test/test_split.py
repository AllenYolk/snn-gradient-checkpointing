import sys

sys.path.append("./src")

from modules import get_block, get_spike_compressor, get_neuron
from utils import LayerWiseMemoryProfiler
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

net = nn.Sequential(
    layer.Conv2d(3, 64, 1, 1, 0, step_mode="m"),
    get_block(
        "Conv2d",
        proj=nn.Conv2d(64, 128, 3, 1, 1),
        spike_compressor=get_spike_compressor("BitSpikeCompressor")
    ),
    get_block(
        "BNSlidingPSNAvgPool2d",
        bn=nn.BatchNorm2d(128),
        neuron=get_neuron("SlidingPSN", detach_reset=True, T=10, k=8),
        pool=nn.AvgPool2d(2),
    )
).to("cuda")

p = LayerWiseMemoryProfiler(
    (net,),
    ("net",),
    ("direct_children",),
)

x = torch.randn(10, 32, 3, 48, 48).to("cuda")
y = net(x)
l = y.sum()
l.backward()

p.profile()
