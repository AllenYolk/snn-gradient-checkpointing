import sys

sys.path.append("./src")

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from modules.neuron import get_neuron
from modules.compress import *
from modules.checkpointing import memory_optimization
from modules.bn import BatchNorm1d_


class SeqToANNContainer(layer.SeqToANNContainer):
    """Stateless layer container that supports temporal chunkingt"""

    def __tc_init_states__(self, x_seq):
        return []

    def __tc_forward__(self, xc):
        return (self.forward(xc),)


class ConvBNNeuron(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        neuron_type,
        preceding_avg_pool: bool = False,
        **kwargs
    ):
        super().__init__()
        conv = [nn.AvgPool1d(2, 2)] if preceding_avg_pool else []
        conv += [
            nn.Conv1d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=True
            )
        ]
        self.conv = SeqToANNContainer(*conv)

        self.bn_neuron = nn.Sequential(
            SeqToANNContainer(BatchNorm1d_(out_channels)),
            get_neuron(neuron_type, **kwargs),
        )

    def forward(self, x_seq):
        return self.bn_neuron(self.conv(x_seq))

    def __spatial_split__(self):
        return self.conv, self.bn_neuron


class AvgPoolFlattenLinearNeuron(nn.Module):

    def __init__(self, channels: int, neuron_type, **kwargs):
        super().__init__()
        self.fc = SeqToANNContainer(
            nn.AvgPool1d(2, 2),
            nn.Flatten(start_dim=-2),
            nn.Linear(channels * 8, channels * 8 // 4),
        )
        self.neuron = get_neuron(neuron_type, **kwargs)

    def forward(self, x_seq):
        return self.neuron(self.fc(x_seq))

    def __spatial_split__(self):
        return self.fc, self.neuron


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
        for i in range(2):
            for j in range(3):
                if len(conv) == 0:
                    in_channels = 3
                else:
                    in_channels = channels

                conv_block = ConvBNNeuron(
                    in_channels,
                    channels,
                    neuron_type,
                    preceding_avg_pool=(j == 0 and i != 0),
                    **kwargs,
                )
                conv.append(conv_block)

        self.conv = nn.Sequential(*conv)
        self.conv[0].disable_x_compressor = True

        self.fc = AvgPoolFlattenLinearNeuron(channels, neuron_type, **kwargs)
        self.decode = nn.Linear(channels * 8 // 4, num_classes)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x = x.permute(3, 0, 1, 2)
        # x.shape = [T, N, Cin, L]
        y = self.conv(x)
        y = self.fc(y)  # [T, N, C']
        y = y.mean(dim=0)  # [N, C']
        y = self.decode(y)
        return y


def GCSequentialCIFARNet(
    channels: int,
    neuron_type: str,
    num_classes=100,
    compress_x: bool = True,
    level: int = 0,
    **kwargs
):
    net = SequentialCIFARNet(channels, neuron_type, num_classes, **kwargs)
    return memory_optimization(
        net, (ConvBNNeuron, AvgPoolFlattenLinearNeuron),
        dummy_input=torch.zeros(128, 3, 32, 32) + 0.9,
        compress_x=compress_x,
        level=level,
        verbose=True
    )
