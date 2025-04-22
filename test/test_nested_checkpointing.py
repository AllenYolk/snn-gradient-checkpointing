import sys

sys.path.append("./src")

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from models.checkpointing import NestedTEBNLIF, TEBNLIF
from models.tebn import TEBNProjection
from models.neuron import HandWrittenLIF
from models.compress import get_spike_compressor
from utils.profiler import LayerWiseMemoryProfiler

T, B, C, H, W = 10, 32, 128, 48, 48
x_seq = torch.randn(T, B, C, H, W).to("cuda")
f = nn.Sequential(
    layer.Conv2d(C, C, 1, 1, step_mode="m"),
    NestedTEBNLIF(
        bn=nn.BatchNorm2d(C),
        tebn_proj=TEBNProjection(T),
        neuron=HandWrittenLIF(),
    )
).to("cuda")

profiler = LayerWiseMemoryProfiler(
    (f,),
    search_mode="direct_children",
    instances=(nn.Module,),
    filename="test_nested_checkpointing.prof.txt",
)

torch.cuda.reset_peak_memory_stats("cuda")

y_seq = f(x_seq)
print(y_seq.shape)
l = y_seq.sum()
l.backward()

profiler.profile()

mem_stats = torch.cuda.memory_stats("cuda")
peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)

print(
    f"peak_allocated={peak_allocated:.2f} MB, "
    f"peak_reserved={peak_reserved:.2f} MB"
)
