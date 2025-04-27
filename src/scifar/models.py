import sys

sys.path.append("./src")

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional

from modules.blocks import get_block, neuron_type_to_str
from modules.neuron import get_neuron
from modules.compress import *


class MESequentialCIFARNet(nn.Module):

    def __init__(
        self,
        channels: int,
        neuron_type: str,
        spike_compressor: str,
        num_classes=100,
        **kwargs
    ):
        """A Conv1d-based network for Sequential CIFAR-10/100 classification.

        Args:
            channels (int)
            neuron_type (str)
            spike_compressor (str)
            num_classes (int, optional): Defaults to 100.
            **kwargs: Additional arguments for `get_neuron(...)`. See 
                `src/models/neuron.py` for details.
        """
        super().__init__()
        neuron_str = neuron_type_to_str(neuron_type)

        conv = []
        for i in range(2):
            for j in range(3):
                if len(conv) == 0:
                    in_channels = 3
                else:
                    in_channels = channels

                if i == 0 and j == 0:
                    conv_block = [
                        get_block(
                            block_type=f"Conv1dBN{neuron_str}",
                            proj=nn.Conv1d(
                                in_channels,
                                channels,
                                kernel_size=3,
                                padding=1,
                                bias=True
                            ),
                            bn=nn.BatchNorm1d(channels),
                            neuron=get_neuron(neuron_type, **kwargs),
                            spike_compressor=get_spike_compressor(
                                "NullSpikeCompressor"
                            ),
                        )
                    ]
                elif i == 0 and j == 2:  # critical layer
                    conv_block = [
                        get_block(
                            f"Conv1d",
                            proj=nn.Conv1d(
                                in_channels,
                                channels,
                                kernel_size=3,
                                padding=1,
                                bias=True
                            ),
                            spike_compressor=get_spike_compressor(
                                "BitSpikeCompressor"
                            ),
                        ),
                        get_block(
                            f"BN{neuron_str}",
                            bn=nn.BatchNorm1d(channels),
                            neuron=get_neuron(neuron_type, **kwargs),
                        )
                    ]
                elif j == 0:
                    conv_block = [
                        get_block(
                            block_type=f"AvgPool1dConv1dBN{neuron_str}",
                            pool=nn.AvgPool1d(2, 2),
                            proj=nn.Conv1d(
                                in_channels,
                                channels,
                                kernel_size=3,
                                padding=1,
                                bias=True
                            ),
                            bn=nn.BatchNorm1d(channels),
                            neuron=get_neuron(neuron_type, **kwargs),
                            spike_compressor=get_spike_compressor(
                                spike_compressor
                            ),
                        )
                    ]
                else:
                    conv_block = [
                        get_block(
                            block_type=f"Conv1dBN{neuron_str}",
                            proj=nn.Conv1d(
                                in_channels,
                                channels,
                                kernel_size=3,
                                padding=1,
                                bias=True
                            ),
                            bn=nn.BatchNorm1d(channels),
                            neuron=get_neuron(neuron_type, **kwargs),
                            spike_compressor=get_spike_compressor(
                                spike_compressor
                            ),
                        )
                    ]
                conv += conv_block

        self.conv = nn.Sequential(*conv)
        self.fc = get_block(
            block_type=f"AvgPool1dFlattenLinear{neuron_str}",
            pool=nn.AvgPool1d(2, 2),
            proj=nn.Linear(channels * 8, channels * 8 // 4),
            neuron=get_neuron(neuron_type, **kwargs),
            spike_compressor=get_spike_compressor(spike_compressor)
        )
        self.decode = nn.Linear(channels * 8 // 4, num_classes)

    def forward(self, x: torch.Tensor):
        # x.shape = [T, N, Cin, L]
        y = self.conv(x)
        y = self.fc(y)
        y = y.mean(dim=0)  # [N, C]
        y = self.decode(y)
        return y


class SequentialCIFARNet(nn.Module):

    def __init__(
        self, channels: int, neuron_type: str, num_classes=100, **kwargs
    ):
        """A Conv1d-based network for Sequential CIFAR-10/100 classification.

        Args:
            channels (int)
            neuron_type (str)
            num_classes (int, optional): Defaults to 100.
            **kwargs: Additional arguments for `get_neuron(...)`. See 
                `src/models/neuron.py` for details.
        """
        super().__init__()

        conv = []
        for _ in range(2):
            for _ in range(3):
                if len(conv) == 0:
                    in_channels = 3
                else:
                    in_channels = channels

                conv_block = nn.Sequential(
                    layer.Conv1d(
                        in_channels,
                        channels,
                        kernel_size=3,
                        padding=1,
                        bias=True
                    ),
                    layer.BatchNorm1d(channels),
                    get_neuron(neuron_type, **kwargs),
                )
                conv.append(conv_block)
            conv.append(layer.AvgPool1d(2, 2))

        self.conv = nn.Sequential(*conv)

        self.fc = nn.Sequential(
            layer.Linear(channels * 8, channels * 8 // 4),
            get_neuron(neuron_type, **kwargs),
        )

        self.decode = nn.Linear(channels * 8 // 4, num_classes)

        functional.set_step_mode(self, "m")

    def forward(self, x: torch.Tensor):
        # x.shape = [T, N, Cin, L]
        y = self.conv(x)
        y = y.flatten(start_dim=-2)  # [T, N, C*L]
        y = self.fc(y)  # [T, N, C']
        y = y.mean(dim=0)  # [N, C']
        y = self.decode(y)
        return y
