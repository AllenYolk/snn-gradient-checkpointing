from .blocks import *
from .checkpointing import *
from .checkpointing_module import *


def get_block(block_type, **kwargs):
    return globals()[block_type](**kwargs)


def get_checkpointing_module(module_type, **kwargs):
    module_type = module_type + "Checkpointing"
    return globals()[module_type](**kwargs)


def neuron_type_to_str(neuron_type):
    if "SlidingPSN" in neuron_type:
        return "SlidingPSN"
    elif "PSN" in neuron_type:
        return "PSN"
    elif "LIF" in neuron_type:
        return "LIF"

    raise ValueError(f"neuron_type {neuron_type} not supported")
