import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from ..blocks import get_block, neuron_type_to_str
from ..neuron import get_neuron
from ..compress import *


def _conv3x3(in_channels, out_channels, neuron_type, **kwargs):
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
        ), get_neuron(neuron_type, **kwargs)
    )


def _conv3x3_checkpointing(
    in_channels, out_channels, neuron_type, spike_compressor: str, **kwargs
):
    return get_block(
        block_type=f"Conv2dBN{neuron_type_to_str(neuron_type)}",
        proj=nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False
        ),
        bn=nn.BatchNorm2d(out_channels),
        neuron=get_neuron(neuron_type, **kwargs),
        spike_compressor=get_spike_compressor(spike_compressor),
    )


def _conv1x1(in_channels, out_channels, neuron_type, **kwargs):
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        ), get_neuron(neuron_type, **kwargs)
    )


def _conv1x1_checkpointing(
    in_channels, out_channels, neuron_type, spike_compressor: str, **kwargs
):
    return get_block(
        block_type=f"Conv2dBN{neuron_type_to_str(neuron_type)}",
        proj=nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        ),
        bn=nn.BatchNorm2d(out_channels),
        neuron=get_neuron(neuron_type, **kwargs),
        spike_compressor=get_spike_compressor(spike_compressor),
    )


class DVSSEWBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, neuron_type, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            _conv3x3(in_channels, mid_channels, neuron_type, **kwargs),
            _conv3x3(mid_channels, in_channels, neuron_type, **kwargs),
        )

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        out = out + x
        return out


class DVSSEWBlockCheckpointing(nn.Module):

    def __init__(
        self,
        in_channels,
        mid_channels,
        neuron_type,
        spike_compressor,
        forced_uint8: bool = True,
        **kwargs
    ):
        super().__init__()
        spike_compressor_class = get_spike_compressor(spike_compressor)
        forced_uint8 = (
            spike_compressor_class.requires_strictly_binary and forced_uint8
        )
        self.conv = nn.Sequential(
            _conv3x3_checkpointing(
                in_channels,
                mid_channels,
                neuron_type,
                "Uint8SpikeCompressor" if forced_uint8 else spike_compressor,
                **kwargs,
            ),
            _conv3x3_checkpointing(
                mid_channels, in_channels, neuron_type, spike_compressor,
                **kwargs
            ),
        )

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        out = out + x
        return out


class DVSPlainBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, neuron_type, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            _conv3x3(in_channels, mid_channels, neuron_type, **kwargs),
            _conv3x3(mid_channels, in_channels, neuron_type, **kwargs),
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class DVSPlainBlockCheckpointing(nn.Module):

    def __init__(
        self, in_channels, mid_channels, neuron_type, spike_compressor, **kwargs
    ):
        super().__init__()
        self.conv = nn.Sequential(
            _conv3x3_checkpointing(
                in_channels, mid_channels, neuron_type, spike_compressor,
                **kwargs
            ),
            _conv3x3_checkpointing(
                mid_channels, in_channels, neuron_type, spike_compressor,
                **kwargs
            ),
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class DVSResNet(nn.Module):

    def __init__(
        self,
        neuron_type,
        layer_list,
        num_classes,
        checkpointing=False,
        spike_compressor="IdentitySpikeCompressor",
        **kwargs
    ):
        super().__init__()
        in_channels = 2
        conv = []

        for i, cfg_dict in enumerate(layer_list):
            channels = cfg_dict['channels']

            if 'mid_channels' in cfg_dict:
                mid_channels = cfg_dict['mid_channels']
            else:
                mid_channels = channels

            if in_channels != channels:  # first layer
                if cfg_dict['up_kernel_size'] == 3:
                    conv.append(
                        _conv3x3_checkpointing(
                            in_channels, channels, neuron_type,
                            "NullSpikeCompressor", **kwargs
                        ) if checkpointing else
                        _conv3x3(in_channels, channels, neuron_type, **kwargs)
                    )
                elif cfg_dict['up_kernel_size'] == 1:
                    conv.append(
                        _conv1x1_checkpointing(
                            in_channels, channels, neuron_type,
                            "NullSpikeCompressor", **kwargs
                        ) if checkpointing else
                        _conv1x1(in_channels, channels, neuron_type, **kwargs)
                    )
                else:
                    raise NotImplementedError

            in_channels = channels

            if 'num_blocks' in cfg_dict:
                num_blocks = cfg_dict['num_blocks']
                if cfg_dict['block_type'] == 'sew':
                    for j in range(num_blocks):
                        conv.append(
                            DVSSEWBlockCheckpointing(
                                in_channels,
                                mid_channels,
                                neuron_type,
                                spike_compressor,
                                forced_uint8=not (
                                    i == 0 and j == 0
                                ),  # input to the first res block is binary
                                **kwargs,
                            ) if checkpointing else DVSSEWBlock(
                                in_channels,
                                mid_channels,
                                neuron_type,
                                **kwargs,
                            )
                        )
                elif cfg_dict['block_type'] == 'plain':
                    for _ in range(num_blocks):
                        conv.append(
                            DVSPlainBlockCheckpointing(
                                in_channels,
                                mid_channels,
                                neuron_type,
                                spike_compressor,
                                **kwargs,
                            ) if checkpointing else DVSPlainBlock(
                                in_channels,
                                mid_channels,
                                neuron_type,
                                **kwargs,
                            )
                        )
                else:
                    raise NotImplementedError

            if 'k_pool' in cfg_dict:
                k_pool = cfg_dict['k_pool']
                conv.append(layer.MaxPool2d(k_pool, k_pool, step_mode="m"))

        conv.append(nn.Flatten(2))  # [T, N, D]

        self.conv = nn.Sequential(*conv)

        with torch.no_grad():
            x = torch.zeros([1, 1, 1, 128, 128])
            for m in self.conv.modules():
                if isinstance(m, layer.MaxPool2d):
                    x = m(x)
            out_features = x.numel() * in_channels

        self.out = nn.Linear(out_features, num_classes, bias=True)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, T, 2, H, W]
        x = x.permute(1, 0, 2, 3, 4).contiguous()  # [T, N, 2, *, *]
        x = self.conv(x)
        return self.out(x.mean(0))


class CIFAR10DVSSEWResNet(DVSResNet):

    layer_list = [
        {
            'channels': 64,
            'up_kernel_size': 1,
            'mid_channels': 64,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
        {
            'channels': 64,
            'up_kernel_size': 1,
            'mid_channels': 64,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
        {
            'channels': 64,
            'up_kernel_size': 1,
            'mid_channels': 64,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
        {
            'channels': 64,
            'up_kernel_size': 1,
            'mid_channels': 64,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
        {
            'channels': 128,
            'up_kernel_size': 1,
            'mid_channels': 128,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
        {
            'channels': 128,
            'up_kernel_size': 1,
            'mid_channels': 128,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
        {
            'channels': 128,
            'up_kernel_size': 1,
            'mid_channels': 128,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
    ]
    num_classes = 10

    def __init__(self, neuron_type, **kwargs):
        super().__init__(
            neuron_type,
            self.layer_list,
            self.num_classes,
            checkpointing=False,
            spike_compressor="IdentitySpikeCompressor",
            **kwargs
        )


class MECIFAR10DVSSEWResNet(DVSResNet):

    def __init__(self, neuron_type, spike_compressor, **kwargs):
        super().__init__(
            neuron_type,
            CIFAR10DVSSEWResNet.layer_list,
            num_classes=CIFAR10DVSSEWResNet.num_classes,
            checkpointing=True,
            spike_compressor=spike_compressor,
            **kwargs
        )


class DVSGestureSEWResNet(DVSResNet):

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

    def __init__(self, neuron_type, **kwargs):
        super().__init__(
            neuron_type,
            self.layer_list,
            self.num_classes,
            checkpointing=False,
            spike_compressor="IdentitySpikeCompressor",
            **kwargs
        )


class MEDVSGestureSEWResNet(DVSResNet):

    def __init__(self, neuron_type, spike_compressor, **kwargs):
        super().__init__(
            neuron_type,
            DVSGestureSEWResNet.layer_list,
            num_classes=DVSGestureSEWResNet.num_classes,
            checkpointing=True,
            spike_compressor=spike_compressor,
            **kwargs
        )
