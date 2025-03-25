from .linear import *
from .conv1d import *
from .conv2d import *


def get_block(proj_type: str, neuron_type: str, need_bn: bool, **kwargs):
    proj_type = proj_type[0].upper() + proj_type[1:].lower()

    if "SlidingPSN" in neuron_type:
        neuron_type = "SlidingPSN"
    elif "PSN" in neuron_type:
        neuron_type = "PSN"
    elif "LIF" in neuron_type:
        neuron_type = "LIF"

    bn_str = "BN" if need_bn else ""

    class_name = f"{proj_type}{bn_str}{neuron_type}"
    return globals()[class_name](**kwargs)
