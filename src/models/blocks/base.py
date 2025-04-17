import torch
import torch.nn as nn


class BaseCheckpointingBlock(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def conventional_forward(*args, **kwargs):
        raise NotImplementedError(
            "The conventional forward function is not implemented."
        )

    def forward(self, x_seq: torch.Tensor):
        raise NotImplementedError("The forward function is not implemented.")

    def extra_repr(self) -> str:
        if hasattr(self, "spike_compressor"):
            return (
                f"Spike compressor: {self.spike_compressor.__class__.__name__}"
            )
        else:
            return ""
