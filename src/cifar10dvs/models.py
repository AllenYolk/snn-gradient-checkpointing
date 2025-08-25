import sys

sys.path.append("./src")

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from modules.neuron import get_neuron
from modules.compress import *
from modules.bn import TEBNProjection, BatchNorm2d_
from modules.checkpointing import GCContainer, memory_optimization


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
        kwargs["T"] = T
        l = []
        if preceding_avg_pool:
            l.append(nn.AvgPool2d(2))
        l += [
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            BatchNorm2d_(out_plane),
        ]
        self.proj_bn = layer.SeqToANNContainer(*l)
        if neuron_type != "PSN":
            self.neuron = nn.Sequential(
                TEBNProjection(T),
                get_neuron(neuron_type, **kwargs),
            )
        else:
            self.neuron = get_neuron(neuron_type, **kwargs)

    def forward(self, x_seq):
        return self.neuron(self.proj_bn(x_seq))


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
        self.dropout = layer.Dropout(dropout)
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
        x = self.classifier(x)
        return x


def vgg_block_checkpointing(
    in_plane, out_plane, kernel_size, stride, padding, T, neuron_type,
    spike_compressor: str, preceding_avg_pool: bool, **kwargs
):
    original_block = VGGBlock(
        in_plane, out_plane, kernel_size, stride, padding, T, neuron_type,
        preceding_avg_pool, **kwargs
    )
    return GCContainer(get_spike_compressor(spike_compressor), original_block)


def vgg_critical_block_checkpointing(
    in_plane, out_plane, kernel_size, stride, padding, T, neuron_type,
    spike_compressor: str, **kwargs
):
    kwargs["T"] = T
    if neuron_type.endswith("PSN"):
        proj_bn = [
            layer.SeqToANNContainer(
                nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
                nn.BatchNorm2d(out_plane),
            )
        ]
        if neuron_type == "SlidingPSN":
            proj_bn.append(TEBNProjection(T))
        return [
            GCContainer(get_spike_compressor(spike_compressor), *proj_bn),
            GCContainer(
                get_spike_compressor(spike_compressor),
                get_neuron(neuron_type, **kwargs),
                layer.SeqToANNContainer(nn.AvgPool2d(2)),
            )
        ]
    else:
        return [
            GCContainer(
                get_spike_compressor(spike_compressor),
                layer.SeqToANNContainer(
                    nn.Conv2d(
                        in_plane, out_plane, kernel_size, stride, padding
                    ),
                ),
            ),
            GCContainer(
                get_spike_compressor(spike_compressor),
                layer.SeqToANNContainer(nn.BatchNorm2d(out_plane)),
                TEBNProjection(T),
                get_neuron(neuron_type, **kwargs),
                layer.SeqToANNContainer(nn.AvgPool2d(2)),
            )
        ]


def AutoGCCIFAR10DVSVGG(
    T, neuron_type, spike_compressor: str, dropout=0.25, **kwargs
):
    net = CIFAR10DVSVGG(T, neuron_type, dropout, **kwargs)
    return memory_optimization(net, (VGGBlock,), spike_compressor, level=1)


class GCCIFAR10DVSVGG(nn.Module):

    def __init__(
        self, T, neuron_type, spike_compressor: str, dropout=0.25, **kwargs
    ):
        super().__init__()

        self.features = nn.Sequential(
            vgg_block_checkpointing(
                2, 64, 3, 1, 1, T, neuron_type, "NullSpikeCompressor", False,
                **kwargs
            ),
            vgg_block_checkpointing(
                64, 128, 3, 1, 1, T, neuron_type, spike_compressor, False,
                **kwargs
            ),
            vgg_block_checkpointing(
                128, 256, 3, 1, 1, T, neuron_type, spike_compressor, True,
                **kwargs
            ),
            vgg_block_checkpointing(
                256, 256, 3, 1, 1, T, neuron_type, spike_compressor, False,
                **kwargs
            ),
            vgg_block_checkpointing(
                256, 512, 3, 1, 1, T, neuron_type, spike_compressor, True,
                **kwargs
            ),
            vgg_block_checkpointing(
                512, 512, 3, 1, 1, T, neuron_type, spike_compressor, False,
                **kwargs
            ),
            vgg_block_checkpointing(
                512, 512, 3, 1, 1, T, neuron_type, spike_compressor, True,
                **kwargs
            ),
            vgg_block_checkpointing(
                512, 512, 3, 1, 1, T, neuron_type, spike_compressor, False,
                **kwargs
            ),
            layer.AvgPool2d(2, step_mode="m"),
        )
        self.dropout = layer.Dropout(dropout)
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
        x = self.classifier(x)
        return x


class FGCCIFAR10DVSVGG(nn.Module):

    def __init__(
        self, T, neuron_type, spike_compressor: str, dropout=0.25, **kwargs
    ):
        super().__init__()

        self.features = nn.Sequential(
            vgg_block_checkpointing(
                2, 64, 3, 1, 1, T, neuron_type, "NullSpikeCompressor", False,
                **kwargs
            ),
            *vgg_critical_block_checkpointing(
                64, 128, 3, 1, 1, T, neuron_type, spike_compressor, **kwargs
            ),
            vgg_block_checkpointing(
                128, 256, 3, 1, 1, T, neuron_type, "NullSpikeCompressor", False,
                **kwargs
            ),
            vgg_block_checkpointing(
                256, 256, 3, 1, 1, T, neuron_type, spike_compressor, False,
                **kwargs
            ),
            vgg_block_checkpointing(
                256, 512, 3, 1, 1, T, neuron_type, spike_compressor, True,
                **kwargs
            ),
            vgg_block_checkpointing(
                512, 512, 3, 1, 1, T, neuron_type, spike_compressor, False,
                **kwargs
            ),
            vgg_block_checkpointing(
                512, 512, 3, 1, 1, T, neuron_type, spike_compressor, True,
                **kwargs
            ),
            vgg_block_checkpointing(
                512, 512, 3, 1, 1, T, neuron_type, spike_compressor, False,
                **kwargs
            ),
            layer.AvgPool2d(2, step_mode="m"),
        )
        self.dropout = layer.Dropout(dropout)
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
        x = self.classifier(x)
        return x


def vgg_block_partial_checkpointing(
    in_plane, out_plane, kernel_size, stride, padding, T, neuron_type,
    spike_compressor: str, preceding_avg_pool: bool, **kwargs
):
    kwargs["T"] = T
    if preceding_avg_pool:
        raise NotImplementedError
    else:
        proj_bn = [
            layer.SeqToANNContainer(
                nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
                nn.BatchNorm2d(out_plane),
            )
        ]
        if neuron_type != "PSN":
            proj_bn.append(TEBNProjection(T))
        return nn.Sequential(
            GCContainer(get_spike_compressor(spike_compressor), *proj_bn),
            get_neuron(neuron_type, **kwargs),
        )


class PGCCIFAR10DVSVGG(nn.Module):

    def __init__(
        self, T, neuron_type, spike_compressor: str, dropout=0.25, **kwargs
    ):
        super().__init__()

        self.features = nn.Sequential(
            vgg_block_checkpointing(
                2, 64, 3, 1, 1, T, neuron_type, "NullSpikeCompressor", False,
                **kwargs
            ),
            *vgg_critical_block_checkpointing(
                64, 128, 3, 1, 1, T, neuron_type, spike_compressor, **kwargs
            ),
            vgg_block_checkpointing(
                128, 256, 3, 1, 1, T, neuron_type, "NullSpikeCompressor", False,
                **kwargs
            ),
            vgg_block_partial_checkpointing(
                256, 256, 3, 1, 1, T, neuron_type, spike_compressor, False,
                **kwargs
            ),
            VGGBlock(256, 512, 3, 1, 1, T, neuron_type, True, **kwargs),
            VGGBlock(512, 512, 3, 1, 1, T, neuron_type, False, **kwargs),
            VGGBlock(512, 512, 3, 1, 1, T, neuron_type, True, **kwargs),
            VGGBlock(512, 512, 3, 1, 1, T, neuron_type, False, **kwargs),
            layer.AvgPool2d(2, step_mode="m"),
        )
        self.dropout = layer.Dropout(dropout)
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
        x = self.classifier(x)
        return x
