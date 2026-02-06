import sys

sys.path.append("./src")

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from modules.neuron import get_neuron
from modules.checkpointing import memory_optimization
from modules.bn import BatchNorm2d_


class SeqToANNContainer(layer.SeqToANNContainer):
    """Stateless layer container that supports temporal chunkingt"""

    def __tc_init_states__(self, x_seq):
        return []

    def __tc_forward__(self, xc):
        return (self.forward(xc),)


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, neuron_type, **kwargs):
        super().__init__()
        self.conv = SeqToANNContainer(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        )
        self.bn_neuron = nn.Sequential(
            SeqToANNContainer(BatchNorm2d_(out_channels)),
            get_neuron(neuron_type, **kwargs),
        )

    def forward(self, x_seq):
        return self.bn_neuron(self.conv(x_seq))

    def __spatial_split__(self):
        return self.conv, self.bn_neuron


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, neuron_type, **kwargs):
        super().__init__()
        self.conv = SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        )
        self.bn_neuron = nn.Sequential(
            SeqToANNContainer(BatchNorm2d_(out_channels)),
            get_neuron(neuron_type, **kwargs),
        )

    def forward(self, x_seq):
        return self.bn_neuron(self.conv(x_seq))

    def __spatial_split__(self):
        return self.conv, self.bn_neuron


class SEWBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, neuron_type, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            Conv3x3(in_channels, mid_channels, neuron_type, **kwargs),
            Conv3x3(mid_channels, in_channels, neuron_type, **kwargs),
        )
        self.conv[0].x_compressor = "Uint8SpikeCompressor"

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        out = out + x
        return out


class ResNetN(nn.Module):
    def __init__(self, layer_list, num_classes, neuron_type, **kwargs):
        super().__init__()
        in_channels = 2
        conv = []

        for cfg_dict in layer_list:
            channels = cfg_dict["channels"]

            if "mid_channels" in cfg_dict:
                mid_channels = cfg_dict["mid_channels"]
            else:
                mid_channels = channels

            if in_channels != channels:
                if cfg_dict["up_kernel_size"] == 3:
                    conv.append(Conv3x3(in_channels, channels, neuron_type, **kwargs))
                elif cfg_dict["up_kernel_size"] == 1:
                    conv.append(Conv1x1(in_channels, channels, neuron_type, **kwargs))
                else:
                    raise NotImplementedError

            in_channels = channels

            if "num_blocks" in cfg_dict:
                num_blocks = cfg_dict["num_blocks"]
                if cfg_dict["block_type"] == "sew":
                    for _ in range(num_blocks):
                        conv.append(
                            SEWBlock(in_channels, mid_channels, neuron_type, **kwargs)
                        )
                else:
                    raise NotImplementedError

            if "k_pool" in cfg_dict:
                k_pool = cfg_dict["k_pool"]
                conv.append(layer.SeqToANNContainer(nn.MaxPool2d(k_pool, k_pool)))

        conv.append(nn.Flatten(2))

        self.conv = nn.Sequential(*conv)
        self.conv[1].conv[0].x_compressor = "NullSpikeCompressor"

        with torch.no_grad():
            x = torch.zeros([1, 1, 128, 128])
            for m in self.conv.modules():
                if isinstance(m, nn.MaxPool2d):
                    x = m(x)
            out_features = x.numel() * in_channels

        self.out = nn.Linear(out_features, num_classes)

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [N, T, 2, H, W]
        x_seq = x_seq.permute(1, 0, 2, 3, 4)  # [T, N, 2, H, W]
        x_seq = self.conv(x_seq)
        return self.out(x_seq.mean(0))


def SEWResNet(neuron_type, **kwargs):
    layer_list = [
        {
            "channels": 32,
            "up_kernel_size": 1,
            "mid_channels": 32,
            "num_blocks": 1,
            "block_type": "sew",
            "k_pool": 2,
        },
        {
            "channels": 32,
            "up_kernel_size": 1,
            "mid_channels": 32,
            "num_blocks": 1,
            "block_type": "sew",
            "k_pool": 2,
        },
        {
            "channels": 32,
            "up_kernel_size": 1,
            "mid_channels": 32,
            "num_blocks": 1,
            "block_type": "sew",
            "k_pool": 2,
        },
        {
            "channels": 32,
            "up_kernel_size": 1,
            "mid_channels": 32,
            "num_blocks": 1,
            "block_type": "sew",
            "k_pool": 2,
        },
        {
            "channels": 32,
            "up_kernel_size": 1,
            "mid_channels": 32,
            "num_blocks": 1,
            "block_type": "sew",
            "k_pool": 2,
        },
        {
            "channels": 32,
            "up_kernel_size": 1,
            "mid_channels": 32,
            "num_blocks": 1,
            "block_type": "sew",
            "k_pool": 2,
        },
        {
            "channels": 32,
            "up_kernel_size": 1,
            "mid_channels": 32,
            "num_blocks": 1,
            "block_type": "sew",
            "k_pool": 2,
        },
    ]
    num_classes = 11
    return ResNetN(layer_list, num_classes, neuron_type, **kwargs)


def GCSEWResNet(neuron_type, compress_x, level, **kwargs):
    net = SEWResNet(neuron_type, **kwargs)
    return memory_optimization(
        net,
        (Conv1x1, Conv3x3),
        dummy_input=torch.zeros(16, 16, 2, 128, 128) + 0.9,
        compress_x=compress_x,
        level=level,
        verbose=True,
    )
