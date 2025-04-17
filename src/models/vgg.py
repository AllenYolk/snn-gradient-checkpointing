import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import layer

from .blocks import get_block, neuron_type_to_str
from .neuron import get_neuron
from .compress import *


class TEBNProjection(nn.Module):

    def __init__(self, T, input_ndim: int = 5):
        super().__init__()
        self.p = nn.Parameter(
            torch.ones(T, *[1 for _ in range(input_ndim - 1)])
        )

    def forward(self, x_seq):
        return x_seq * self.p


def vgg_block(
    in_plane, out_plane, kernel_size, stride, padding, T, neuron_type, **kwargs
):
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            nn.BatchNorm2d(out_plane),
        ),
        TEBNProjection(T),
        get_neuron(neuron_type, **kwargs),
    )


class CIFAR10DVSVGG(nn.Module):

    def __init__(self, T, neuron_type, **kwargs):
        super().__init__()

        self.features = nn.Sequential(
            vgg_block(2, 64, 3, 1, 1, T, neuron_type, **kwargs),
            vgg_block(64, 128, 3, 1, 1, T, neuron_type, **kwargs),
            layer.AvgPool2d(2),
            vgg_block(128, 256, 3, 1, 1, T, neuron_type, **kwargs),
            vgg_block(256, 256, 3, 1, 1, T, neuron_type, **kwargs),
            layer.AvgPool2d(2),
            vgg_block(256, 512, 3, 1, 1, T, neuron_type, **kwargs),
            vgg_block(512, 512, 3, 1, 1, T, neuron_type, **kwargs),
            layer.AvgPool2d(2),
            vgg_block(512, 512, 3, 1, 1, T, neuron_type, **kwargs),
            vgg_block(512, 512, 3, 1, 1, T, neuron_type, **kwargs),
            layer.AvgPool2d(2),
        )
        self.dropout = nn.Dropout(0.25)
        d = int(48 / 2 / 2 / 2 / 2)
        self.classifier = nn.Linear(512 * d * d, 10)

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
        x = self.dropout(x)
        x = x.mean(dim=0)
        x = self.classifier(x)
        return x
