import torch
import torch.nn as nn

from .checkpointing import SNNCheckpointingBlockFunction
from ..neuron import *
from ..compress import get_spike_compressor


class CheckpointingModule(nn.Module):

    def __init__(self):
        super().__init__()


class LIFCheckpointing(CheckpointingModule):

    def __init__(self, neuron):
        super().__init__()
        self.neuron = neuron

    @staticmethod
    def conventional_forward(x_seq, neuron, in_backward):
        return neuron(x_seq)

    def forward(self, x_seq):
        return SNNCheckpointingBlockFunction.apply(
            self.conventional_forward,
            get_spike_compressor("IdentitySpikeCompressor"),
            x_seq,
            self.neuron,
        )
