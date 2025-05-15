import sys

sys.path.append("./src")

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from modules.blocks import get_block, neuron_type_to_str
from modules.neuron import get_neuron
from modules.compress import *
from modules.tebn import TEBNProjection


def vgg_block(
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
    kwargs["T"] = T
    l = []
    if preceding_avg_pool:
        l.append(layer.AvgPool2d(2, step_mode="m"))
    if neuron_type_to_str(neuron_type) != "PSN":
        l += [
            layer.SeqToANNContainer(
                nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
                nn.BatchNorm2d(out_plane),
            ),
            TEBNProjection(T),
            get_neuron(neuron_type, **kwargs),
        ]
    else:
        l += [
            layer.SeqToANNContainer(
                nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
                nn.BatchNorm2d(out_plane),
            ),
            get_neuron(neuron_type, **kwargs),
        ]
    return nn.Sequential(*l)


class CIFAR10DVSVGG(nn.Module):

    def __init__(self, T, neuron_type, dropout=0.25, **kwargs):
        super().__init__()

        self.features = nn.Sequential(
            vgg_block(2, 64, 3, 1, 1, T, neuron_type, False, **kwargs),
            vgg_block(64, 128, 3, 1, 1, T, neuron_type, False, **kwargs),
            vgg_block(128, 256, 3, 1, 1, T, neuron_type, True, **kwargs),
            vgg_block(256, 256, 3, 1, 1, T, neuron_type, False, **kwargs),
            vgg_block(256, 512, 3, 1, 1, T, neuron_type, True, **kwargs),
            vgg_block(512, 512, 3, 1, 1, T, neuron_type, False, **kwargs),
            vgg_block(512, 512, 3, 1, 1, T, neuron_type, True, **kwargs),
            vgg_block(512, 512, 3, 1, 1, T, neuron_type, False, **kwargs),
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
    kwargs["T"] = T
    if preceding_avg_pool:
        if neuron_type_to_str(neuron_type) != "PSN":
            return get_block(
                f"AvgPool2dConv2dTEBN{neuron_type_to_str(neuron_type)}",
                pool=nn.AvgPool2d(2),
                proj=nn.Conv2d(
                    in_plane, out_plane, kernel_size, stride, padding
                ),
                bn=nn.BatchNorm2d(out_plane),
                tebn_proj=TEBNProjection(T),
                neuron=get_neuron(neuron_type, **kwargs),
                spike_compressor=get_spike_compressor(spike_compressor)
            )
        else:
            return get_block(
                f"AvgPool2dConv2dBN{neuron_type_to_str(neuron_type)}",
                pool=nn.AvgPool2d(2),
                proj=nn.Conv2d(
                    in_plane, out_plane, kernel_size, stride, padding
                ),
                bn=nn.BatchNorm2d(out_plane),
                neuron=get_neuron(neuron_type, **kwargs),
                spike_compressor=get_spike_compressor(spike_compressor)
            )
    else:
        if neuron_type_to_str(neuron_type) != "PSN":
            return get_block(
                f"Conv2dTEBN{neuron_type_to_str(neuron_type)}",
                proj=nn.Conv2d(
                    in_plane, out_plane, kernel_size, stride, padding
                ),
                bn=nn.BatchNorm2d(out_plane),
                tebn_proj=TEBNProjection(T),
                neuron=get_neuron(neuron_type, **kwargs),
                spike_compressor=get_spike_compressor(spike_compressor)
            )
        else:
            return get_block(
                f"Conv2dBN{neuron_type_to_str(neuron_type)}",
                proj=nn.Conv2d(
                    in_plane, out_plane, kernel_size, stride, padding
                ),
                bn=nn.BatchNorm2d(out_plane),
                neuron=get_neuron(neuron_type, **kwargs),
                spike_compressor=get_spike_compressor(spike_compressor)
            )


def vgg_critical_block_checkpointing(
    in_plane, out_plane, kernel_size, stride, padding, T, neuron_type,
    spike_compressor: str, **kwargs
):
    kwargs["T"] = T
    neuron_type_extracted = neuron_type_to_str(neuron_type)
    if neuron_type_extracted == "SlidingPSN":
        return [
            get_block(
                f"Conv2dTEBN",
                proj=nn.Conv2d(
                    in_plane, out_plane, kernel_size, stride, padding
                ),
                bn=nn.BatchNorm2d(out_plane),
                tebn_proj=TEBNProjection(T),
                spike_compressor=get_spike_compressor(spike_compressor)
            ),
            get_block(
                f"SlidingPSNAvgPool2d",
                neuron=get_neuron(neuron_type, **kwargs),
                pool=nn.AvgPool2d(2),
            )
        ]
    elif neuron_type_extracted == "PSN":
        return [
            get_block(
                f"Conv2dBN",
                proj=nn.Conv2d(
                    in_plane, out_plane, kernel_size, stride, padding
                ),
                bn=nn.BatchNorm2d(out_plane),
                spike_compressor=get_spike_compressor(spike_compressor)
            ),
            get_block(
                f"PSNAvgPool2d",
                neuron=get_neuron(neuron_type, **kwargs),
                pool=nn.AvgPool2d(2)
            )
        ]
    else:
        # In this case, peak allocated memory is achieved when backwarding on Conv2d!
        return [
            get_block(
                f"Conv2d",
                proj=nn.Conv2d(
                    in_plane, out_plane, kernel_size, stride, padding
                ),
                spike_compressor=get_spike_compressor(spike_compressor)
            ),
            get_block(
                f"TEBN{neuron_type_extracted}AvgPool2d",
                bn=nn.BatchNorm2d(out_plane),
                tebn_proj=TEBNProjection(T),
                neuron=get_neuron(neuron_type, **kwargs),
                pool=nn.AvgPool2d(2),
            )
        ]


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
        if neuron_type_to_str(neuron_type) != "PSN":
            return nn.Sequential(
                get_block(
                    f"Conv2dTEBN",
                    proj=nn.Conv2d(
                        in_plane, out_plane, kernel_size, stride, padding
                    ),
                    bn=nn.BatchNorm2d(out_plane),
                    tebn_proj=TEBNProjection(T),
                    spike_compressor=get_spike_compressor(spike_compressor)
                ),
                get_neuron(neuron_type, **kwargs),
            )
        else:
            return nn.Sequential(
                get_block(
                    f"Conv2dBN",
                    proj=nn.Conv2d(
                        in_plane, out_plane, kernel_size, stride, padding
                    ),
                    bn=nn.BatchNorm2d(out_plane),
                    spike_compressor=get_spike_compressor(spike_compressor)
                ),
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
            vgg_block(256, 512, 3, 1, 1, T, neuron_type, True, **kwargs),
            vgg_block(512, 512, 3, 1, 1, T, neuron_type, False, **kwargs),
            vgg_block(512, 512, 3, 1, 1, T, neuron_type, True, **kwargs),
            vgg_block(512, 512, 3, 1, 1, T, neuron_type, False, **kwargs),
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
