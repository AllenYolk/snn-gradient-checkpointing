from .conv1d import *
from .conv2d import *
from .linear import *
from .bn_neuron import *
from .attention import *


def get_block(block_type, n_chunk=1, **kwargs):
    if n_chunk > 1:
        return globals()["TC" + block_type](n_chunk=n_chunk, **kwargs)
    else:
        return globals()[block_type](**kwargs)


def neuron_type_to_str(neuron_type):
    if "SlidingPSN" in neuron_type:
        return "SlidingPSN"
    elif "PSN" in neuron_type:
        return "PSN"
    elif "LIF" in neuron_type:
        return "LIF"

    raise ValueError(f"neuron_type {neuron_type} not supported")
