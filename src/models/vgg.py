import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from .checkpointing import get_block, neuron_type_to_str
from .neuron import get_neuron
from .compress import *
from .tebn import TEBNProjection


def vgg_block(
    in_plane, out_plane, kernel_size, stride, padding, use_tebn: bool, T,
    neuron_type, **kwargs
):
    if use_tebn:
        return nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
                nn.BatchNorm2d(out_plane),
            ),
            TEBNProjection(T),
            get_neuron(neuron_type, T=T, **kwargs),
        )
    else:
        return nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
                nn.BatchNorm2d(out_plane),
            ),
            get_neuron(neuron_type, T=T, **kwargs),
        )


class CIFAR10DVSVGG(nn.Module):

    def __init__(self, T, neuron_type, dropout=0.25, allow_tebn=True, **kwargs):
        super().__init__()

        use_tebn = (neuron_type != "PSN") and allow_tebn
        self.features = nn.Sequential(
            vgg_block(2, 64, 3, 1, 1, use_tebn, T, neuron_type, **kwargs),
            vgg_block(64, 128, 3, 1, 1, use_tebn, T, neuron_type, **kwargs),
            layer.AvgPool2d(2, step_mode="m"),
            vgg_block(128, 256, 3, 1, 1, use_tebn, T, neuron_type, **kwargs),
            vgg_block(256, 256, 3, 1, 1, use_tebn, T, neuron_type, **kwargs),
            layer.AvgPool2d(2, step_mode="m"),
            vgg_block(256, 512, 3, 1, 1, use_tebn, T, neuron_type, **kwargs),
            vgg_block(512, 512, 3, 1, 1, use_tebn, T, neuron_type, **kwargs),
            layer.AvgPool2d(2, step_mode="m"),
            vgg_block(512, 512, 3, 1, 1, use_tebn, T, neuron_type, **kwargs),
            vgg_block(512, 512, 3, 1, 1, use_tebn, T, neuron_type, **kwargs),
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
    in_plane, out_plane, kernel_size, stride, padding, use_tebn: bool, T,
    neuron_type, spike_compressor: str, preceding_avg_pool: bool, **kwargs
):
    if preceding_avg_pool:
        if use_tebn:
            return get_block(
                f"AvgPool2dConv2dTEBN{neuron_type_to_str(neuron_type)}",
                pool=nn.AvgPool2d(2),
                proj=nn.Conv2d(
                    in_plane, out_plane, kernel_size, stride, padding
                ),
                bn=nn.BatchNorm2d(out_plane),
                tebn_proj=TEBNProjection(T),
                neuron=get_neuron(neuron_type, T=T, **kwargs),
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
                neuron=get_neuron(neuron_type, T=T, **kwargs),
                spike_compressor=get_spike_compressor(spike_compressor)
            )
    else:
        if use_tebn:
            return get_block(
                f"Conv2dTEBN{neuron_type_to_str(neuron_type)}",
                proj=nn.Conv2d(
                    in_plane, out_plane, kernel_size, stride, padding
                ),
                bn=nn.BatchNorm2d(out_plane),
                tebn_proj=TEBNProjection(T),
                neuron=get_neuron(neuron_type, T=T, **kwargs),
                spike_compressor=get_spike_compressor(spike_compressor)
            )
        else:
            return get_block(
                f"Conv2dBN{neuron_type_to_str(neuron_type)}",
                proj=nn.Conv2d(
                    in_plane, out_plane, kernel_size, stride, padding
                ),
                bn=nn.BatchNorm2d(out_plane),
                neuron=get_neuron(neuron_type, T=T, **kwargs),
                spike_compressor=get_spike_compressor(spike_compressor)
            )


def vgg_critical_block_checkpointing(
    in_plane, out_plane, kernel_size, stride, padding, use_tebn: bool, T,
    neuron_type, spike_compressor: str, **kwargs
):
    if use_tebn:
        return nn.Sequential(
            get_block(
                f"Conv2d",
                proj=nn.Conv2d(
                    in_plane, out_plane, kernel_size, stride, padding
                ),
                spike_compressor=get_spike_compressor(spike_compressor)
            ),
            get_block(
                f"NestedTEBN{neuron_type_to_str(neuron_type)}",
                bn=nn.BatchNorm2d(out_plane),
                tebn_proj=TEBNProjection(T),
                neuron=get_neuron(neuron_type, T=T, **kwargs),
            )
        )
    else:
        return nn.Sequential(
            get_block(
                f"Conv2d",
                proj=nn.Conv2d(
                    in_plane, out_plane, kernel_size, stride, padding
                ),
                spike_compressor=get_spike_compressor(spike_compressor)
            ),
            get_block(
                f"BN{neuron_type_to_str(neuron_type)}",
                bn=nn.BatchNorm2d(out_plane),
                neuron=get_neuron(neuron_type, T=T, **kwargs),
                spike_compressor=get_spike_compressor("NullSpikeCompressor")
            )
        )


class MECIFAR10DVSVGG(nn.Module):

    def __init__(
        self,
        T,
        neuron_type,
        spike_compressor: str,
        dropout=0.25,
        allow_tebn=True,
        **kwargs
    ):
        super().__init__()

        use_tebn = (neuron_type != "PSN") and allow_tebn
        self.features = nn.Sequential(
            vgg_block_checkpointing(
                2, 64, 3, 1, 1, use_tebn, T, neuron_type, "NullSpikeCompressor",
                False, **kwargs
            ),
            vgg_critical_block_checkpointing(
                64, 128, 3, 1, 1, use_tebn, T, neuron_type, spike_compressor,
                **kwargs
            ),
            vgg_block_checkpointing(
                128, 256, 3, 1, 1, use_tebn, T, neuron_type, spike_compressor,
                True, **kwargs
            ),
            vgg_block_checkpointing(
                256, 256, 3, 1, 1, use_tebn, T, neuron_type, spike_compressor,
                False, **kwargs
            ),
            vgg_block_checkpointing(
                256, 512, 3, 1, 1, use_tebn, T, neuron_type, spike_compressor,
                True, **kwargs
            ),
            vgg_block_checkpointing(
                512, 512, 3, 1, 1, use_tebn, T, neuron_type, spike_compressor,
                False, **kwargs
            ),
            vgg_block_checkpointing(
                512, 512, 3, 1, 1, use_tebn, T, neuron_type, spike_compressor,
                True, **kwargs
            ),
            vgg_block_checkpointing(
                512, 512, 3, 1, 1, use_tebn, T, neuron_type, spike_compressor,
                False, **kwargs
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
