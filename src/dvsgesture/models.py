import sys

sys.path.append("./src")

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional

from modules.neuron import get_neuron
from modules.blocks import get_block, neuron_type_to_str
from modules.compress import get_spike_compressor


def conv3x3(in_channels, out_channels, neuron_type, **kwargs):
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
        ),
        get_neuron(neuron_type, **kwargs),
    )


def conv3x3_checkpointing(
    in_channels,
    out_channels,
    neuron_type,
    spike_compressor,
    input_non_binary_int,
    **kwargs,
):
    sc_class = get_spike_compressor(spike_compressor)
    forced_uint8 = sc_class.requires_strictly_binary and input_non_binary_int
    return get_block(
        block_type=f"Conv2dBN{neuron_type_to_str(neuron_type)}",
        proj=nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False,
        ),
        bn=nn.BatchNorm2d(out_channels),
        neuron=get_neuron(neuron_type, **kwargs),
        spike_compressor=get_spike_compressor(
            "Uint8SpikeCompressor" if forced_uint8 else spike_compressor
        ),
    )


def conv1x1(in_channels, out_channels, neuron_type, **kwargs):
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        ),
        get_neuron(neuron_type, **kwargs),
    )


def conv1x1_checkpointing(
    in_channels,
    out_channels,
    neuron_type,
    spike_compressor,
    input_non_binary_int,
    **kwargs,
):
    sc_class = get_spike_compressor(spike_compressor)
    forced_uint8 = sc_class.requires_strictly_binary and input_non_binary_int
    return get_block(
        block_type=f"Conv2dBN{neuron_type_to_str(neuron_type)}",
        proj=nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        ),
        bn=nn.BatchNorm2d(out_channels),
        neuron=get_neuron(neuron_type, **kwargs),
        spike_compressor=get_spike_compressor(
            "Uint8SpikeCompressor" if forced_uint8 else spike_compressor
        ),
    )


class SEWBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, neuron_type, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            conv3x3(in_channels, mid_channels, neuron_type, **kwargs),
            conv3x3(mid_channels, in_channels, neuron_type, **kwargs),
        )

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        out = out + x
        return out


class SEWBlockCheckpointing(nn.Module):

    def __init__(
        self,
        in_channels,
        mid_channels,
        neuron_type,
        spike_compressor,
        input_non_binary_int,
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            conv3x3_checkpointing(
                in_channels,
                mid_channels,
                neuron_type,
                spike_compressor,
                input_non_binary_int,
                **kwargs,
            ),
            conv3x3_checkpointing(
                mid_channels,
                in_channels,
                neuron_type,
                spike_compressor,
                input_non_binary_int=False,
                **kwargs,
            ),
        )

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        out = out + x
        return out


class PlainBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, neuron_type, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            conv3x3(in_channels, mid_channels, neuron_type, **kwargs),
            conv3x3(mid_channels, in_channels, neuron_type, **kwargs),
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class PlainBlockCheckpointing(nn.Module):

    def __init__(
        self,
        in_channels,
        mid_channels,
        neuron_type,
        spike_compressor,
        input_non_binary_int,
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            conv3x3_checkpointing(
                in_channels,
                mid_channels,
                neuron_type,
                spike_compressor,
                input_non_binary_int,
                **kwargs,
            ),
            conv3x3_checkpointing(
                mid_channels,
                in_channels,
                neuron_type,
                spike_compressor,
                input_non_binary_int=False,
                **kwargs,
            ),
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class ResNetN(nn.Module):

    def __init__(self, layer_list, num_classes, neuron_type, **kwargs):
        super().__init__()
        in_channels = 2
        conv = []

        for cfg_dict in layer_list:
            channels = cfg_dict['channels']

            if 'mid_channels' in cfg_dict:
                mid_channels = cfg_dict['mid_channels']
            else:
                mid_channels = channels

            if in_channels != channels:
                if cfg_dict['up_kernel_size'] == 3:
                    conv.append(
                        conv3x3(in_channels, channels, neuron_type, **kwargs)
                    )
                elif cfg_dict['up_kernel_size'] == 1:
                    conv.append(
                        conv1x1(in_channels, channels, neuron_type, **kwargs)
                    )
                else:
                    raise NotImplementedError

            in_channels = channels

            if 'num_blocks' in cfg_dict:
                num_blocks = cfg_dict['num_blocks']
                if cfg_dict['block_type'] == 'sew':
                    for _ in range(num_blocks):
                        conv.append(
                            SEWBlock(
                                in_channels, mid_channels, neuron_type, **kwargs
                            )
                        )
                elif cfg_dict['block_type'] == 'plain':
                    for _ in range(num_blocks):
                        conv.append(
                            PlainBlock(
                                in_channels, mid_channels, neuron_type, **kwargs
                            )
                        )
                else:
                    raise NotImplementedError

            if 'k_pool' in cfg_dict:
                k_pool = cfg_dict['k_pool']
                conv.append(
                    layer.SeqToANNContainer(nn.MaxPool2d(k_pool, k_pool))
                )

        conv.append(nn.Flatten(2))

        self.conv = nn.Sequential(*conv)

        with torch.no_grad():
            x = torch.zeros([1, 1, 128, 128])
            for m in self.conv.modules():
                if isinstance(m, nn.MaxPool2d):
                    x = m(x)
            out_features = x.numel() * in_channels

        self.out = nn.Linear(out_features, num_classes)

    def forward(self, x_seq: torch.Tensor):
        functional.reset_net(self)
        # x_seq.shape = [N, T, 2, H, W]
        x_seq = x_seq.permute(1, 0, 2, 3, 4)  # [T, N, 2, H, W]
        x_seq = self.conv(x_seq)
        return self.out(x_seq.mean(0))


class ResNetNCheckpointing(nn.Module):

    def __init__(
        self,
        layer_list,
        num_classes,
        neuron_type,
        spike_compressor,
        **kwargs,
    ):
        super().__init__()
        in_channels = 2
        conv = []
        input_non_binary_int = False

        for cfg_dict in layer_list:
            gc = cfg_dict['gc']
            channels = cfg_dict['channels']

            if 'mid_channels' in cfg_dict:
                mid_channels = cfg_dict['mid_channels']
            else:
                mid_channels = channels

            if in_channels != channels:  # scale up #channels
                sc = "NullSpikeCompressor" if len(
                    conv
                ) == 0 else spike_compressor
                if cfg_dict['up_kernel_size'] == 3:
                    if gc:
                        conv.append(
                            conv3x3_checkpointing(
                                in_channels,
                                channels,
                                neuron_type,
                                sc,
                                input_non_binary_int,
                                **kwargs,
                            )
                        )
                    else:
                        conv.append(
                            conv3x3(
                                in_channels, channels, neuron_type, **kwargs
                            )
                        )
                elif cfg_dict['up_kernel_size'] == 1:
                    if gc:
                        conv.append(
                            conv1x1_checkpointing(
                                in_channels,
                                channels,
                                neuron_type,
                                sc,
                                input_non_binary_int,
                                **kwargs,
                            )
                        )
                    else:
                        conv.append(
                            conv1x1(
                                in_channels, channels, neuron_type, **kwargs
                            )
                        )
                else:
                    raise NotImplementedError
                input_non_binary_int = False

            in_channels = channels

            if 'num_blocks' in cfg_dict:
                num_blocks = cfg_dict['num_blocks']
                if cfg_dict['block_type'] == 'sew':
                    for _ in range(num_blocks):
                        if gc:
                            conv.append(
                                SEWBlockCheckpointing(
                                    in_channels,
                                    mid_channels,
                                    neuron_type,
                                    spike_compressor,
                                    input_non_binary_int,
                                    **kwargs,
                                )
                            )
                        else:
                            conv.append(
                                SEWBlock(
                                    in_channels, mid_channels, neuron_type,
                                    **kwargs
                                )
                            )
                        input_non_binary_int = True
                elif cfg_dict['block_type'] == 'plain':
                    for _ in range(num_blocks):
                        if gc:
                            conv.append(
                                PlainBlockCheckpointing(
                                    in_channels,
                                    mid_channels,
                                    neuron_type,
                                    spike_compressor,
                                    input_non_binary_int,
                                    **kwargs,
                                )
                            )
                        else:
                            conv.append(
                                PlainBlock(
                                    in_channels, mid_channels, neuron_type,
                                    **kwargs
                                )
                            )
                        input_non_binary_int = False
                else:
                    raise NotImplementedError

            if 'k_pool' in cfg_dict:
                k_pool = cfg_dict['k_pool']
                conv.append(
                    layer.SeqToANNContainer(nn.MaxPool2d(k_pool, k_pool))
                )

        conv.append(nn.Flatten(2))

        self.conv = nn.Sequential(*conv)

        with torch.no_grad():
            x = torch.zeros([1, 1, 128, 128])
            for m in self.conv.modules():
                if isinstance(m, nn.MaxPool2d):
                    x = m(x)
            out_features = x.numel() * in_channels

        self.out = nn.Linear(out_features, num_classes)

    def forward(self, x_seq: torch.Tensor):
        functional.reset_net(self)
        # x_seq.shape = [N, T, 2, H, W]
        x_seq = x_seq.permute(1, 0, 2, 3, 4)  # [T, N, 2, H, W]
        x_seq = self.conv(x_seq)
        return self.out(x_seq.mean(0))


def SEWResNet(neuron_type, **kwargs):
    layer_list = [
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
    ]
    num_classes = 11
    return ResNetN(layer_list, num_classes, neuron_type, **kwargs)


def FGCSEWResNet(neuron_type, spike_compressor, **kwargs):
    layer_list = [
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'gc': True,
            'k_pool': 2,
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'gc': True,
            'k_pool': 2,
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'gc': True,
            'k_pool': 2
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'gc': True,
            'k_pool': 2
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'gc': True,
            'k_pool': 2
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'gc': True,
            'k_pool': 2
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'gc': True,
            'k_pool': 2
        },
    ]
    num_classes = 11
    return ResNetNCheckpointing(
        layer_list, num_classes, neuron_type, spike_compressor, **kwargs
    )


def PGCSEWResNet(neuron_type, spike_compressor, **kwargs):
    layer_list = [
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'gc': True,
            'k_pool': 2,
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'gc': False,
            'k_pool': 2,
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'gc': False,
            'k_pool': 2
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'gc': False,
            'k_pool': 2
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'gc': False,
            'k_pool': 2
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'gc': False,
            'k_pool': 2
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'gc': False,
            'k_pool': 2
        },
    ]
    num_classes = 11
    return ResNetNCheckpointing(
        layer_list, num_classes, neuron_type, spike_compressor, **kwargs
    )
