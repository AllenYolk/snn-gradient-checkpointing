import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional

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


def _conv1x1(in_channels, out_channels, neuron_type, **kwargs):
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        ), get_neuron(neuron_type, **kwargs)
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


class DVSPlainBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, neuron_type, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            _conv3x3(in_channels, mid_channels, neuron_type, **kwargs),
            _conv3x3(mid_channels, in_channels, neuron_type, **kwargs),
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class DVSResBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, neuron_type, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            _conv3x3(in_channels, mid_channels, neuron_type, **kwargs),
            layer.SeqToANNContainer(
                nn.Conv2d(
                    mid_channels,
                    in_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=False
                ),
                nn.BatchNorm2d(in_channels),
            ),
        )
        self.sn = get_neuron(neuron_type, **kwargs)

    def forward(self, x: torch.Tensor):
        return self.sn(x + self.conv(x))


class DVSResNet(nn.Module):

    def __init__(
        self, neuron_type, layer_list, num_classes, connect_f=None, **kwargs
    ):
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
                    conv.append(_conv3x3(in_channels, channels))
                elif cfg_dict['up_kernel_size'] == 1:
                    conv.append(_conv1x1(in_channels, channels))
                else:
                    raise NotImplementedError

            in_channels = channels

            if 'num_blocks' in cfg_dict:
                num_blocks = cfg_dict['num_blocks']
                if cfg_dict['block_type'] == 'sew':
                    for _ in range(num_blocks):
                        conv.append(
                            DVSSEWBlock(
                                in_channels,
                                mid_channels,
                                neuron_type,
                                **kwargs,
                            )
                        )
                elif cfg_dict['block_type'] == 'plain':
                    for _ in range(num_blocks):
                        conv.append(
                            DVSPlainBlock(
                                in_channels,
                                mid_channels,
                                neuron_type,
                                **kwargs,
                            )
                        )
                elif cfg_dict['block_type'] == 'basic':
                    for _ in range(num_blocks):
                        conv.append(
                            DVSResBlock(
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
            x = torch.zeros([1, 1, 128, 128])
            for m in self.conv.modules():
                if isinstance(m, nn.MaxPool2d):
                    x = m(x)
            out_features = x.numel() * in_channels

        self.out = nn.Linear(out_features, num_classes, bias=True)

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        x = self.conv(x)
        return self.out(x.mean(0))


# TODO:
def SEWResNet(connect_f):
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
    return DVSResNet(layer_list, num_classes, connect_f)
