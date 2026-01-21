import sys

sys.path.append("./src")

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from modules.neuron import get_neuron
from modules.compress import *
from modules.bn import TEBNProjection, BatchNorm2d_
from modules.checkpointing import memory_optimization, first_l_memory_optimization


class VGGBlock(nn.Module):

    def __init__(
        self,
        in_plane,
        out_plane,
        kernel_size,
        stride,
        padding,
        T,
        neuron_type,
        preceding_avg_pool=False,
        **kwargs
    ):
        super().__init__()
        proj_bn = []
        if preceding_avg_pool:
            proj_bn.append(nn.AvgPool2d(2))
        proj_bn += [
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            BatchNorm2d_(out_plane),
        ]
        self.proj_bn = layer.SeqToANNContainer(*proj_bn)  # not split-able

        kwargs["T"] = T
        if not neuron_type.endswith("PSN"):
            self.neuron = nn.Sequential(
                TEBNProjection(T),
                get_neuron(neuron_type, **kwargs),
            )  # not split-able
        else:
            self.neuron = get_neuron(neuron_type, **kwargs)  # not split-able

    def forward(self, x_seq):
        return self.neuron(self.proj_bn(x_seq))

    def __spatial_split__(self):
        return self.proj_bn, self.neuron


class CIFAR10DVSVGG(nn.Module):

    def __init__(self, T, neuron_type, dropout=0.25, **kwargs):
        super().__init__()

        self.features = nn.Sequential(
            VGGBlock(2, 64, 3, 1, 1, T, neuron_type, False, **kwargs),
            VGGBlock(64, 128, 3, 1, 1, T, neuron_type, False, **kwargs),
            VGGBlock(128, 256, 3, 1, 1, T, neuron_type, True, **kwargs),
            VGGBlock(256, 256, 3, 1, 1, T, neuron_type, False, **kwargs),
            VGGBlock(256, 512, 3, 1, 1, T, neuron_type, True, **kwargs),
            VGGBlock(512, 512, 3, 1, 1, T, neuron_type, False, **kwargs),
            VGGBlock(512, 512, 3, 1, 1, T, neuron_type, True, **kwargs),
            VGGBlock(512, 512, 3, 1, 1, T, neuron_type, False, **kwargs),
            layer.AvgPool2d(2, step_mode="m"),
        )
        self.features[0].x_compressor = "NullSpikeCompressor"
        d = int(48 / 2 / 2 / 2 / 2)
        l = [nn.Dropout(dropout)] if dropout > 0 else []
        l.append(nn.Linear(512 * d * d, 10))
        self.classifier = nn.Sequential(*l)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )

    def forward(self, input):
        # input.shape = [N, T, C, H, W]
        input = input.transpose(0, 1).contiguous()  # [T, N, C, H, W]
        x = self.features(input)
        x = torch.flatten(x, 2)  # [T, N, D]
        x = self.classifier(x)
        return x


def GCCIFAR10DVSVGG(
    T, neuron_type, compress_x: bool, level: int = 1, dropout=0.25, **kwargs
):
    net = CIFAR10DVSVGG(T, neuron_type, dropout, **kwargs)
    return memory_optimization(
        net, (VGGBlock,),
        dummy_input=torch.zeros(32, T, 2, 48, 48) + 0.9,
        compress_x=compress_x,
        level=level,
        verbose=True
    )


def FLGCCIFAR10DVSVGG(
    T, neuron_type, compress_x: bool, level: int = 1, dropout=0.25, **kwargs
):
    net = CIFAR10DVSVGG(T, neuron_type, dropout, **kwargs)
    return first_l_memory_optimization(
        net,
        (VGGBlock,),
        dummy_input=torch.zeros(32, T, 2, 48, 48) + 0.9,
        compress_x=compress_x,
        L=level,
        verbose=True,
    )
